"""
DeepTrace Ensemble Model Architecture
EfficientNet-B0 + ViT-Small + FFT Feature Head → 5-class provenance classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import Dict, Tuple, Optional


CLASS_NAMES = ["stable_diffusion", "midjourney", "dalle3", "flux", "real"]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# FFT Feature Extractor
# Converts spatial image to frequency-domain descriptor.
# Directly motivated by EDA showing AI generators have unnatural HF patterns.
# ---------------------------------------------------------------------------

class FFTFeatureExtractor(nn.Module):
    """
    Extracts frequency-domain features from images.
    Computes 2D FFT magnitude spectrum, bins it into radial frequency bands,
    and learns a projection to a 64-dim embedding.
    """

    def __init__(self, out_dim: int = 64, num_bins: int = 32):
        super().__init__()
        self.num_bins = num_bins
        self.projection = nn.Sequential(
            nn.Linear(num_bins * 3, 128),   # 3 channels
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        B, C, H, W = x.shape
        bin_features = []

        for c in range(C):
            channel = x[:, c, :, :]                        # (B, H, W)
            fft = torch.fft.fft2(channel)
            magnitude = torch.abs(fft)
            magnitude = torch.log1p(magnitude)              # log-scale for stability

            # Radial binning: average magnitude in concentric rings
            cx, cy = H // 2, W // 2
            y_idx = torch.arange(H, device=x.device).float() - cx
            x_idx = torch.arange(W, device=x.device).float() - cy
            yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
            radius = torch.sqrt(yy ** 2 + xx ** 2)
            max_r = radius.max()

            bins = torch.zeros(B, self.num_bins, device=x.device)
            bin_edges = torch.linspace(0, max_r.item(), self.num_bins + 1, device=x.device)

            for i in range(self.num_bins):
                mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
                if mask.sum() > 0:
                    bins[:, i] = magnitude[:, mask].mean(dim=1)

            bin_features.append(bins)

        freq_features = torch.cat(bin_features, dim=1)     # (B, num_bins * 3)
        return self.projection(freq_features)               # (B, out_dim)


# ---------------------------------------------------------------------------
# Ensemble Model
# ---------------------------------------------------------------------------

class DeepTraceEnsemble(nn.Module):
    """
    Three-branch ensemble for AI image provenance detection.

    Branches:
      1. EfficientNet-B0  → spatial / texture features (1280-dim)
      2. ViT-Small/16     → global structure features  (384-dim)
      3. FFTFeatureExtractor → spectral artifact features (64-dim)

    Fusion: LayerNorm → MLP → 5-class softmax
    """

    EFFNET_DIM = 1280
    VIT_DIM = 384
    FFT_DIM = 64

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3,
                 use_fft: bool = True, pretrained: bool = True):
        super().__init__()
        self.use_fft = use_fft

        # ---- Backbones ----
        self.effnet = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.vit = timm.create_model(
            "vit_small_patch16_224", pretrained=pretrained, num_classes=0
        )

        # ---- FFT head ----
        if use_fft:
            self.fft_extractor = FFTFeatureExtractor(out_dim=self.FFT_DIM)

        # ---- Fusion classifier ----
        feat_dim = self.EFFNET_DIM + self.VIT_DIM + (self.FFT_DIM if use_fft else 0)

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.effnet(x)                        # (B, 1280)
        f2 = self.vit(x)                           # (B, 384)

        features = [f1, f2]
        if self.use_fft:
            f3 = self.fft_extractor(x)             # (B, 64)
            features.append(f3)

        fused = torch.cat(features, dim=1)
        return self.classifier(fused)              # (B, num_classes)

    def freeze_backbones(self):
        """Phase 1 training: train only the classifier head."""
        for param in self.effnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_top_blocks(self, effnet_blocks: int = 2, vit_blocks: int = 2):
        """Phase 2 training: unfreeze top N blocks of each backbone."""
        # EfficientNet: unfreeze last N blocks
        blocks = list(self.effnet.blocks.children())
        for block in blocks[-effnet_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # ViT: unfreeze last N transformer blocks
        for block in self.vit.blocks[-vit_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Phase 3 training: full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Temperature Scaling (Post-hoc calibration)
# ---------------------------------------------------------------------------

class TemperatureScaledModel(nn.Module):
    """
    Wraps a trained DeepTraceEnsemble and applies temperature scaling.
    Calibrated probabilities: a "73% Stable Diffusion" should be right 73% of the time.
    """

    def __init__(self, base_model: DeepTraceEnsemble):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        return logits / self.temperature

    def calibrate(self, val_loader, device: str = "cuda", lr: float = 0.01,
                  max_iter: int = 50) -> float:
        """
        Find optimal temperature using NLL minimization on validation set.
        Returns final NLL loss.
        """
        self.base_model.eval()
        self.temperature.requires_grad_(True)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits_list.append(self.base_model(images).cpu())
                labels_list.append(labels)

        logits_tensor = torch.cat(logits_list)
        labels_tensor = torch.cat(labels_list)

        def eval_fn():
            optimizer.zero_grad()
            scaled = logits_tensor / self.temperature.cpu()
            loss = F.cross_entropy(scaled, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_fn)

        with torch.no_grad():
            scaled = logits_tensor / self.temperature.cpu()
            final_nll = F.cross_entropy(scaled, labels_tensor).item()

        print(f"Calibration complete. Temperature: {self.temperature.item():.4f}, NLL: {final_nll:.4f}")
        return final_nll


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(pretrained: bool = True, dropout: float = 0.3,
                use_fft: bool = True) -> DeepTraceEnsemble:
    return DeepTraceEnsemble(
        num_classes=NUM_CLASSES,
        dropout=dropout,
        use_fft=use_fft,
        pretrained=pretrained,
    )


def load_model(checkpoint_path: str, device: str = "cpu",
               calibrated: bool = False) -> nn.Module:
    """Load a saved model checkpoint."""
    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)

    if calibrated:
        calibrated_model = TemperatureScaledModel(model)
        calibrated_model.load_state_dict(state["state_dict"])
        calibrated_model.eval()
        return calibrated_model
    else:
        model.load_state_dict(state["state_dict"])
        model.eval()
        return model


if __name__ == "__main__":
    # Quick sanity check
    model = build_model(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # expect (2, 5)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
