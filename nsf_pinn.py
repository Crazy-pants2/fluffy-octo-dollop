import torch
import torch.nn as nn
from transformers import WavLMModel

class SourceFilterBranch(nn.Module):
    """
    Disentangles WavLM frame-level states H (B, L, 768) into 
    glottal source S (B, L, 128) and vocal tract filter F (B, L, 128).
    """
    def __init__(self, in_dim: int = 768, shared_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        # Shared Temporal MLP: 2x (Linear -> GELU)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
            nn.GELU()
        )
        # Independent linear projection heads
        self.source_head = nn.Linear(shared_dim, latent_dim)
        self.filter_head = nn.Linear(shared_dim, latent_dim)

    def forward(self, H: torch.Tensor):
        Z = self.shared_mlp(H)          # (B, L, 256)
        S = self.source_head(Z)         # (B, L, 128)
        F = self.filter_head(Z)         # (B, L, 128)
        
        # Mean pooling across temporal frames L
        s = S.mean(dim=1)               # (B, 128)
        f = F.mean(dim=1)               # (B, 128)
        return S, F, s, f


class Reconstructor(nn.Module):
    """
    Two-layer MLP decoding concatenated [s || f] (B, 256) back to H_tilde (B, 768).
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, s: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        c = torch.cat([s, f], dim=-1)   # (B, 256)
        return self.net(c)              # (B, 768)


class NSFPINN(nn.Module):
    def __init__(
        self,
        pretrained_model: str = "microsoft/wavlm-base-plus",
        frozen_layers: int = 6,
        dropout: float = 0.2,
        use_spec_augment: bool = True
    ):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(pretrained_model)
        self.wavlm.config.apply_spec_augment = use_spec_augment

        # Freeze feature extractor CNN and the first `frozen_layers` transformer blocks
        self.wavlm.feature_extractor._freeze_parameters()
        for i, layer in enumerate(self.wavlm.encoder.layers):
            if i < frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # Holistic SSL Projection Head: 768 -> 768 -> 128
        self.ssl_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 128)
        )

        # Physics-Informed Source-Filter Branch & Reconstructor
        self.sf_branch = SourceFilterBranch(in_dim=768, shared_dim=256, latent_dim=128)
        self.reconstructor = Reconstructor(in_dim=256, hidden_dim=512, out_dim=768)

        # Binary Classifier MLP on fused g = [e_ssl || s || f] in R^384
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor):
        # 1. Extract WavLM contextual representations
        outputs = self.wavlm(x)
        H = outputs.last_hidden_state           # (B, L, 768), L ~ 199 for 4s audio
        
        # 2. Utterance-level holistic SSL embedding
        H_tilde = H.mean(dim=1)                 # (B, 768)
        e_ssl = self.ssl_proj(H_tilde)          # (B, 128)

        # 3. Source-Filter Decomposition & Reconstruction
        S, F, s, f = self.sf_branch(H)
        H_hat = self.reconstructor(s, f)        # (B, 768)

        # 4. Feature Fusion & Classification
        g = torch.cat([e_ssl, s, f], dim=-1)    # (B, 384)
        logits = self.classifier(g)             # (B, 2)

        return {
            "logits": logits,
            "S": S,
            "F": F,
            "s": s,
            "f": f,
            "H_tilde": H_tilde,
            "H_hat": H_hat
        }
