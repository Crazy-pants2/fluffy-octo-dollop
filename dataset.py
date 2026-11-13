import os
import torch
import torchaudio
from torch.utils.data import Dataset

class ASVspoof2019Dataset(Dataset):
    def __init__(self, audio_dir: str, protocol_path: str, max_len: int = 64000, is_train: bool = True):
        self.audio_dir = audio_dir
        self.max_len = max_len
        self.is_train = is_train
        self.samples = []

        # Parse ASVspoof 2019 LA protocol: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    utt_id = parts[1]
                    # Section 3.5.1: y = 0 for genuine (bonafide), y = 1 for spoof
                    label = 0 if parts[4] == "bonafide" else 1
                    self.samples.append((utt_id, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        utt_id, label = self.samples[idx]
        file_path = os.path.join(self.audio_dir, f"{utt_id}.flac")
        waveform, sr = torchaudio.load(file_path)

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        waveform = waveform.squeeze(0)  # Mono (T,)

        length = waveform.shape[0]
        if length > self.max_len:
            if self.is_train:
                start = torch.randint(0, length - self.max_len + 1, (1,)).item()
            else:
                start = (length - self.max_len) // 2
            waveform = waveform[start : start + self.max_len]
        elif length < self.max_len:
            pad_amount = self.max_len - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform, torch.tensor(label, dtype=torch.long), utt_id


def apply_mixup(waveforms: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2, num_classes: int = 2):
    """Applies waveform-level Mixup and returns soft one-hot targets."""
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    if alpha <= 0:
        return waveforms, one_hot

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = waveforms.size(0)
    index = torch.randperm(batch_size, device=waveforms.device)

    mixed_waveforms = lam * waveforms + (1.0 - lam) * waveforms[index]
    mixed_targets = lam * one_hot + (1.0 - lam) * one_hot[index]
    return mixed_waveforms, mixed_targets
