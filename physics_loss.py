import torch
import torch.nn as nn
import torch.nn.functional as F

class NSFPINNLoss(nn.Module):
    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_rec: float = 0.1,
        lambda_smo: float = 0.1,
        lambda_orth: float = 0.1,
        warmup_epochs: float = 2.0,
        clamp_losses: bool = True
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        self.lambda_smo = lambda_smo
        self.lambda_orth = lambda_orth
        self.warmup_epochs = warmup_epochs
        self.clamp_losses = clamp_losses
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, targets: torch.Tensor, epoch: int) -> dict:
        logits = outputs["logits"]
        S, F_seq = outputs["S"], outputs["F"]
        s, f = outputs["s"], outputs["f"]
        H_tilde, H_hat = outputs["H_tilde"], outputs["H_hat"]

        # 1. Warmup factor alpha(e) = min(1, e / E_w)
        alpha = min(1.0, float(epoch) / self.warmup_epochs) if self.warmup_epochs > 0 else 1.0

        # 2. Classification Loss (supports standard integer labels or Mixup soft targets)
        l_cls = self.ce(logits, targets)

        # 3. Reconstruction Consistency Loss: ||H_tilde - H_hat||_2^2
        l_rec = F.mse_loss(H_hat, H_tilde.detach())

        # 4. Temporal Smoothness Loss: frame-to-frame finite differences
        diff_S = S[:, 1:, :] - S[:, :-1, :]
        diff_F = F_seq[:, 1:, :] - F_seq[:, :-1, :]
        l_smo = torch.mean(diff_S.pow(2)) + torch.mean(diff_F.pow(2))

        # 5. Source-Filter Orthogonality Loss: squared cosine similarity (Eq. 27)
        s_norm = F.normalize(s, p=2, dim=-1)
        f_norm = F.normalize(f, p=2, dim=-1)
        cos_sim = torch.sum(s_norm * f_norm, dim=-1)
        l_orth = torch.mean(cos_sim.pow(2))

        # Optional clamping from Algorithm 2
        if self.clamp_losses:
            l_cls = torch.clamp(l_cls, 0.0, 1.0)
            l_rec = torch.clamp(l_rec, 0.0, 10.0)
            l_smo = torch.clamp(l_smo, 0.0, 10.0)
            l_orth = torch.clamp(l_orth, 0.0, 10.0)

        # Total composite objective (Eq. 30)
        total_loss = (
            self.lambda_cls * l_cls
            + alpha * (
                self.lambda_rec * l_rec
                + self.lambda_smo * l_smo
                + self.lambda_orth * l_orth
            )
        )

        return {
            "total": total_loss,
            "cls": l_cls.detach(),
            "rec": l_rec.detach(),
            "smo": l_smo.detach(),
            "orth": l_orth.detach()
        }
