import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.cuda.amp import GradScaler, autocast
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from nsf_pinn import NSFPINN
from physics_loss import NSFPINNLoss
from sam import SAM
from dataset import apply_mixup

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Computes Equal Error Rate (EER) in percentage."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer * 100.0)

def train_one_epoch(
    model: NSFPINN,
    dataloader,
    criterion: NSFPINNLoss,
    sam_optimizer: SAM,
    scaler: GradScaler,
    epoch: int,
    device: torch.device,
    use_mixup: bool = True
):
    model.train()
    running_loss = 0.0
    for waveforms, labels, _ in dataloader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            inputs, targets = apply_mixup(waveforms, labels, alpha=0.2)
        else:
            inputs, targets = waveforms, labels

       # --- Step 1: Forward & Ascent (Find e* in rho-ball) ---
        sam_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            outputs_1 = model(inputs)
            loss_dict_1 = criterion(outputs_1, targets, epoch)
            loss_1 = loss_dict_1["total"]

        scaler.scale(loss_1).backward()
        
        # Check for INFs before modifying gradients
        grads_finite = True
        for group in sam_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break
            if not grads_finite:
                break

        if not grads_finite:
            # Gradients blew up. Let GradScaler handle it officially to reduce the scale.
            scaler.unscale_(sam_optimizer.base_optimizer)
            scaler.step(sam_optimizer.base_optimizer)  # Skips update due to infs
            scaler.update()
            sam_optimizer.zero_grad(set_to_none=True)
            continue

        # Gradients are finite. Manually unscale them for SAM Pass 1.
        # This keeps GradScaler's state clean for Pass 2.
        inv_scale = 1.0 / scaler.get_scale()
        for group in sam_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.mul_(inv_scale)

        sam_optimizer.first_step(zero_grad=True)

        # --- Step 2: Forward at Perturbed Weights & Descent ---
        with torch.amp.autocast('cuda'):
            outputs_2 = model(inputs)
            loss_dict_2 = criterion(outputs_2, targets, epoch)
            loss_2 = loss_dict_2["total"]

        scaler.scale(loss_2).backward()
        
        # Restore theta <- theta_old 
        sam_optimizer.restore_weights()
        
        # GradScaler works normally for Pass 2
        scaler.step(sam_optimizer.base_optimizer)
        scaler.update()
        sam_optimizer.zero_grad(set_to_none=True)

        running_loss += loss_1.item()

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(model: NSFPINN, dataloader, device: torch.device):
    """Algorithm 4: Inference Procedure."""
    model.eval()
    all_scores, all_labels = [], []

    for waveforms, labels, _ in dataloader:
        waveforms = waveforms.to(device, non_blocking=True)
        with autocast():
            outputs = model(waveforms)
            probs = torch.softmax(outputs["logits"], dim=-1)
            spoof_scores = probs[:, 1]  # Class 1 = Spoof probability

        all_scores.extend(spoof_scores.cpu().numpy())
        all_labels.extend(labels.numpy())

    eer = compute_eer(np.array(all_labels), np.array(all_scores))
    return eer

import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset import ASVspoof2019Dataset

def main():
    parser = argparse.ArgumentParser(description="Train NSF-PINN on ASVspoof 2019 LA")
    parser.add_argument("--data_root", type=str, required=True, help="Path to LA/ directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size for 4GB VRAM")
    parser.add_argument("--accum_steps", type=int, default=4, help="8 * 4 = 32 effective batch size")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--frozen_layers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Enable WavLM gradient checkpointing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Begining Training")

    # Ampere (RTX 3050) hardware speedups
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Paths for ASVspoof 2019 LA
    train_audio = os.path.join(args.data_root, "ASVspoof2019_LA_train", "flac")
    train_proto = os.path.join(args.data_root, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
    dev_audio   = os.path.join(args.data_root, "ASVspoof2019_LA_dev", "flac")
    dev_proto   = os.path.join(args.data_root, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")

    train_ds = ASVspoof2019Dataset(train_audio, train_proto, is_train=True)
    dev_ds   = ASVspoof2019Dataset(dev_audio, dev_proto, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print("Data Loaded")

    # Initialize Model (Trial T14 disables SpecAugment and freezes 6 layers)
    model = NSFPINN(
        frozen_layers=args.frozen_layers,
        use_spec_augment=False
    ).to(device)

    if args.grad_checkpoint:
        model.wavlm.gradient_checkpointing_enable()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    sam_optimizer = SAM(
        trainable_params, torch.optim.AdamW,
        rho=args.rho, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        sam_optimizer.base_optimizer, T_max=args.epochs
    )
    criterion = NSFPINNLoss()
    scaler = GradScaler()

    print("Model Initialized")

    best_eer = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, sam_optimizer,
            scaler, epoch, device, use_mixup=True
        )
        scheduler.step()

        if epoch % args.val_every == 0 or epoch >= 30:
            dev_eer = evaluate(model, dev_loader, device)
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Dev EER: {dev_eer:.2f}%")

            if dev_eer < best_eer:
                best_eer = dev_eer
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "nsf_pinn_best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best Dev EER: {best_eer:.2f}%")
                    break

if __name__ == "__main__":
    main()
