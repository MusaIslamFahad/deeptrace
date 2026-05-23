"""
DeepTrace XAI Pipeline
Grad-CAM + LIME + FFT spectrum → stored as queryable artifacts
Optionally generates natural-language explanations via Anthropic API.
"""

import io
import os
import base64
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model.architecture import CLASS_NAMES


# ---------------------------------------------------------------------------
# Grad-CAM (manual implementation — no extra dependency needed)
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM for EfficientNet and ViT branches of the ensemble.
    Hook-based: works on any layer that produces spatial feature maps.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, images: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Returns Grad-CAM heatmap normalized to [0, 1], shape (H, W).
        """
        self.model.eval()
        images.requires_grad_(True)

        logits = self.model(images)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Pool gradients over spatial dims
        gradients = self.gradients[0]        # (C, H, W)
        activations = self.activations[0]    # (C, H, W)

        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam = (weights * activations).sum(dim=0)             # (H, W)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.
    image_np: (H, W, 3) uint8, heatmap: (H, W) float [0,1]
    Returns (H, W, 3) uint8
    """
    h, w = image_np.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    colormap = cm.get_cmap("jet")
    colored = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    overlaid = (alpha * colored + (1 - alpha) * image_np).astype(np.uint8)
    return overlaid


# ---------------------------------------------------------------------------
# FFT spectrum visualization
# ---------------------------------------------------------------------------

def compute_fft_spectrum(image_np: np.ndarray) -> np.ndarray:
    """
    Compute the 2D FFT magnitude spectrum (log-scaled) of a grayscale image.
    image_np: (H, W, 3) float or uint8
    Returns: (H, W) float array, normalized to [0, 1]
    """
    gray = np.mean(image_np, axis=2).astype(float)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude


# ---------------------------------------------------------------------------
# LIME explanation (basic superpixel version)
# ---------------------------------------------------------------------------

def run_lime_explanation(model: torch.nn.Module, image_np: np.ndarray,
                          target_class: int, device: str,
                          num_samples: int = 200) -> np.ndarray:
    """
    Simplified LIME: perturb superpixel segments and fit a linear model
    to get per-segment importance scores.
    image_np: (H, W, 3) uint8
    Returns: (H, W) importance map, normalized [0, 1]
    """
    from skimage.segmentation import slic
    from torchvision import transforms

    segments = slic(image_np, n_segments=50, compactness=10, start_label=0)
    n_segments = segments.max() + 1

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Sample perturbations
    masks = np.random.randint(0, 2, (num_samples, n_segments))
    preds = []

    model.eval()
    with torch.no_grad():
        for mask in masks:
            perturbed = image_np.copy().astype(float)
            for seg_id in range(n_segments):
                if mask[seg_id] == 0:
                    perturbed[segments == seg_id] = 128  # grey out segment

            pil_img = Image.fromarray(perturbed.astype(np.uint8))
            tensor = preprocess(pil_img).unsqueeze(0).to(device)
            prob = torch.softmax(model(tensor), dim=1)[0, target_class].item()
            preds.append(prob)

    # Weighted linear model
    weights = np.exp(-0.5 * ((1 - np.array(preds)) ** 2) / 0.25)
    from numpy.linalg import lstsq
    coef, *_ = lstsq(masks * weights[:, None], np.array(preds) * weights, rcond=None)

    # Map coefficients back to image space
    importance_map = np.zeros(image_np.shape[:2], dtype=float)
    for seg_id in range(n_segments):
        importance_map[segments == seg_id] = max(coef[seg_id], 0)

    importance_map -= importance_map.min()
    if importance_map.max() > 0:
        importance_map /= importance_map.max()

    return importance_map


# ---------------------------------------------------------------------------
# XAI Report
# ---------------------------------------------------------------------------

class XAIReport:
    """Bundles all XAI outputs for a single prediction."""

    def __init__(
        self,
        image_np: np.ndarray,
        predicted_class: int,
        confidence: float,
        per_class_probs: Dict[str, float],
        gradcam_heatmap: Optional[np.ndarray] = None,
        lime_mask: Optional[np.ndarray] = None,
        fft_spectrum: Optional[np.ndarray] = None,
        explanation_text: Optional[str] = None,
    ):
        self.image_np = image_np
        self.predicted_class = predicted_class
        self.predicted_source = CLASS_NAMES[predicted_class]
        self.confidence = confidence
        self.per_class_probs = per_class_probs
        self.gradcam_heatmap = gradcam_heatmap
        self.lime_mask = lime_mask
        self.fft_spectrum = fft_spectrum
        self.explanation_text = explanation_text

    def to_composite_png(self, save_path: Optional[str] = None) -> bytes:
        """
        Renders a 2x2 grid: original | Grad-CAM | LIME | FFT spectrum
        Returns PNG bytes.
        """
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle(
            f"DeepTrace — {self.predicted_source.replace('_', ' ').title()} "
            f"({self.confidence * 100:.1f}%)",
            fontsize=14, fontweight="bold"
        )

        h, w = self.image_np.shape[:2]

        # Panel 1: Original
        axes[0].imshow(self.image_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Panel 2: Grad-CAM
        if self.gradcam_heatmap is not None:
            overlaid = overlay_heatmap(self.image_np, self.gradcam_heatmap)
            axes[1].imshow(overlaid)
            axes[1].set_title("Grad-CAM")
        else:
            axes[1].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[1].axis("off")

        # Panel 3: LIME
        if self.lime_mask is not None:
            lime_resized = np.array(
                Image.fromarray((self.lime_mask * 255).astype(np.uint8)).resize((w, h))
            ) / 255.0
            axes[2].imshow(self.image_np)
            axes[2].imshow(lime_resized, cmap="hot", alpha=0.5)
            axes[2].set_title("LIME")
        else:
            axes[2].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[2].axis("off")

        # Panel 4: FFT
        if self.fft_spectrum is not None:
            axes[3].imshow(self.fft_spectrum, cmap="inferno")
            axes[3].set_title("FFT Spectrum")
        else:
            axes[3].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[3].axis("off")

        # Add class probability bar annotation
        prob_text = "\n".join(
            [f"{k}: {v:.2%}" for k, v in sorted(self.per_class_probs.items(),
                                                  key=lambda x: -x[1])]
        )
        fig.text(0.01, 0.01, prob_text, fontsize=8, verticalalignment="bottom",
                 family="monospace")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.read()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(png_bytes)

        return png_bytes

    def to_base64(self) -> str:
        return base64.b64encode(self.to_composite_png()).decode()


# ---------------------------------------------------------------------------
# XAI Service (used by API)
# ---------------------------------------------------------------------------

class XAIService:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

        # Target the last conv block of EfficientNet for Grad-CAM
        try:
            target_layer = list(self.model.effnet.blocks.children())[-1][-1]
        except Exception:
            target_layer = self.model.effnet.conv_head

        self.gradcam = GradCAM(model, target_layer)

        # Optionally set up Anthropic client for NL explanations
        self.anthropic_client = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                pass

    def explain(
        self,
        image_np: np.ndarray,
        predicted_class: int,
        confidence: float,
        per_class_probs: Dict[str, float],
        include_lime: bool = False,
        include_nl_explanation: bool = False,
    ) -> XAIReport:
        """
        Full XAI pipeline for a single image.
        image_np: (H, W, 3) uint8 RGB
        """
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        pil_img = Image.fromarray(image_np)
        tensor = preprocess(pil_img).unsqueeze(0).to(self.device)

        # Grad-CAM
        gradcam_heatmap = self.gradcam(tensor.clone(), target_class=predicted_class)

        # FFT
        fft_spectrum = compute_fft_spectrum(image_np)

        # LIME (slow; optional)
        lime_mask = None
        if include_lime:
            lime_mask = run_lime_explanation(
                self.model, image_np, predicted_class, self.device
            )

        # Natural-language explanation
        explanation_text = None
        if include_nl_explanation and self.anthropic_client:
            explanation_text = self._generate_nl_explanation(
                predicted_class, confidence, gradcam_heatmap
            )

        return XAIReport(
            image_np=image_np,
            predicted_class=predicted_class,
            confidence=confidence,
            per_class_probs=per_class_probs,
            gradcam_heatmap=gradcam_heatmap,
            lime_mask=lime_mask,
            fft_spectrum=fft_spectrum,
            explanation_text=explanation_text,
        )

    def _describe_hot_region(self, heatmap: np.ndarray) -> str:
        """Describe the spatial location of the highest-activation region."""
        h, w = heatmap.shape
        # Find centroid of top 10% activations
        threshold = np.percentile(heatmap, 90)
        mask = heatmap > threshold
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return "the center of the image"
        cy, cx = ys.mean() / h, xs.mean() / w
        v = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "middle")
        h_pos = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
        return f"the {v}-{h_pos} region"

    def _generate_nl_explanation(self, predicted_class: int,
                                  confidence: float,
                                  heatmap: np.ndarray) -> str:
        """Generate one-sentence explanation using Anthropic API."""
        source_name = CLASS_NAMES[predicted_class].replace("_", " ").title()
        hot_region = self._describe_hot_region(heatmap)
        prompt = (
            f"A deep learning model classified this image as generated by {source_name} "
            f"with {confidence * 100:.0f}% confidence. "
            f"The Grad-CAM attention map highlights {hot_region} as most influential. "
            f"Write exactly one plain-English sentence explaining what visual artifact "
            f"in {hot_region} is characteristic of {source_name}-generated images. "
            f"Be specific about the type of artifact (texture, blending, noise pattern, etc.)."
        )
        try:
            msg = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            return f"Model focused on {hot_region} to identify {source_name} artifacts."
