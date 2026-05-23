"""
DeepTrace Inference Service
Handles model loading, image preprocessing, and prediction.
Designed as a singleton loaded at API startup.
"""

import time
import uuid
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from model.architecture import (
    build_model, TemperatureScaledModel, CLASS_NAMES, NUM_CLASSES
)
from data.dataset import get_inference_transforms
from api.config import get_settings


class InferenceService:
    """
    Singleton service that:
    - Loads the calibrated model once at startup
    - Preprocesses images
    - Returns structured prediction dicts
    """

    _instance: Optional["InferenceService"] = None

    def __init__(self):
        settings = get_settings()
        self.device = torch.device(settings.model_device
                                   if torch.cuda.is_available() or settings.model_device == "cpu"
                                   else "cpu")
        self.model_version = "unknown"
        self.model: Optional[torch.nn.Module] = None
        self.transform = get_inference_transforms(settings.image_size)
        self._load_model(settings.model_uri)

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self, model_uri: str):
        """
        Load model from:
          - Local .pt checkpoint path (development)
          - MLflow model URI (production): 'models:/DeepTrace/Production'
        """
        if model_uri.startswith("models:/"):
            self._load_from_mlflow(model_uri)
        elif Path(model_uri).exists():
            self._load_from_checkpoint(model_uri)
        else:
            print(f"[InferenceService] WARNING: model not found at {model_uri}. "
                  f"Using untrained model for development.")
            self.model = build_model(pretrained=False).to(self.device)
            self.model.eval()
            self.model_version = "dev-untrained"

    def _load_from_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        base_model = build_model(pretrained=False).to(self.device)

        if "temperature" in state:
            # Calibrated model
            calibrated = TemperatureScaledModel(base_model).to(self.device)
            calibrated.load_state_dict(state["state_dict"])
            self.model = calibrated
        else:
            base_model.load_state_dict(state["state_dict"])
            self.model = base_model

        self.model.eval()
        self.model_version = state.get("metrics", {}).get("version", Path(path).stem)
        print(f"[InferenceService] Loaded from checkpoint: {path} "
              f"(version={self.model_version})")

    def _load_from_mlflow(self, model_uri: str):
        try:
            import mlflow.pytorch
            from api.config import get_settings
            import mlflow
            mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
            self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            self.model.eval()
            self.model_version = model_uri.split("/")[-1]
            print(f"[InferenceService] Loaded from MLflow: {model_uri}")
        except Exception as e:
            print(f"[InferenceService] MLflow load failed ({e}), falling back to dev model.")
            self.model = build_model(pretrained=False).to(self.device)
            self.model.eval()
            self.model_version = "dev-fallback"

    def preprocess(self, image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Returns:
            tensor: (1, 3, 224, 224) ready for model
            image_np: (H, W, 3) uint8 original image for XAI
        """
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return tensor, image_np

    @torch.no_grad()
    def predict(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        include_xai: bool = False,
        include_lime: bool = False,
        include_nl: bool = False,
    ) -> Dict:
        """
        Full prediction pipeline for a single image.
        Returns a dict matching PredictionResponse schema.
        """
        t0 = time.time()
        image_id = f"img_{hashlib.md5(image_bytes).hexdigest()[:12]}"

        tensor, image_np = self.preprocess(image_bytes)

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_idx = int(probs.argmax())
        predicted_source = CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx])

        per_class_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

        gradcam_url = None
        explanation_text = None

        if include_xai:
            gradcam_url, explanation_text = self._run_xai(
                tensor, image_np, predicted_idx, confidence,
                per_class_probs, include_lime, include_nl
            )

        processing_ms = int((time.time() - t0) * 1000)

        return {
            "image_id": image_id,
            "filename": filename,
            "predicted_source": predicted_source,
            "confidence": confidence,
            "is_ai_generated": predicted_source != "real",
            "per_class_probs": per_class_probs,
            "gradcam_url": gradcam_url,
            "explanation_text": explanation_text,
            "processing_ms": processing_ms,
            "model_version": self.model_version,
        }

    def _run_xai(self, tensor, image_np, predicted_idx, confidence,
                  per_class_probs, include_lime, include_nl) -> Tuple[Optional[str], Optional[str]]:
        """Run XAI pipeline and return (gradcam_url, explanation_text)."""
        try:
            from model.xai_pipeline import XAIService
            xai = XAIService(
                model=self.model.base_model if hasattr(self.model, "base_model") else self.model,
                device=str(self.device)
            )
            report = xai.explain(
                image_np=image_np,
                predicted_class=predicted_idx,
                confidence=confidence,
                per_class_probs=per_class_probs,
                include_lime=include_lime,
                include_nl_explanation=include_nl,
            )

            # Save locally (in production, upload to S3)
            save_path = f"reports/xai/{report.image_id}.png"
            report.to_composite_png(save_path=save_path)

            return save_path, report.explanation_text
        except Exception as e:
            print(f"[InferenceService] XAI failed: {e}")
            return None, None

    def predict_batch(self, image_bytes_list: list, filenames: list = None) -> list:
        """Process a list of images and return list of prediction dicts."""
        filenames = filenames or [f"image_{i}.jpg" for i in range(len(image_bytes_list))]
        results = []
        for img_bytes, fname in zip(image_bytes_list, filenames):
            try:
                result = self.predict(img_bytes, filename=fname)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "filename": fname})
        return results

    def is_healthy(self) -> bool:
        """Quick health check: run a dummy forward pass."""
        try:
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                self.model(dummy)
            return True
        except Exception:
            return False
