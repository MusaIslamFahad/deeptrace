"""
api/services/inference_render.py

Drop-in replacement for inference.py on Render free tier.
Checks MODEL_DEMO_MODE env var — if true, uses demo predictions.
If false, loads the real model normally.

To use: copy this file over api/services/inference.py before deploying.
"""

import os
import time
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from model.architecture import CLASS_NAMES, NUM_CLASSES
from data.dataset import get_inference_transforms
from api.config import get_settings


def _is_demo_mode() -> bool:
    return os.getenv("MODEL_DEMO_MODE", "false").lower() in ("true", "1", "yes")


class InferenceService:
    _instance: Optional["InferenceService"] = None

    def __init__(self):
        settings = get_settings()
        self.device = torch.device("cpu")
        self.model_version = "demo-1.0" if _is_demo_mode() else "unknown"
        self.model = None
        self.transform = get_inference_transforms(settings.image_size)
        self._demo_mode = _is_demo_mode()

        if not self._demo_mode:
            self._load_model(settings.model_uri)
        else:
            print("[InferenceService] Running in DEMO MODE — mock predictions enabled.")

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self, model_uri: str):
        from model.architecture import build_model, TemperatureScaledModel

        if model_uri.startswith("models:/"):
            try:
                import mlflow.pytorch
                self.model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
                self.model.eval()
                self.model_version = model_uri.split("/")[-1]
                print(f"[InferenceService] Loaded from MLflow: {model_uri}")
                return
            except Exception as e:
                print(f"[InferenceService] MLflow load failed ({e}), switching to demo mode.")
                self._demo_mode = True
                return

        if Path(model_uri).exists():
            state = torch.load(model_uri, map_location="cpu")
            base = build_model(pretrained=False)
            if "temperature" in state:
                model = TemperatureScaledModel(base)
            else:
                model = base
            model.load_state_dict(state["state_dict"])
            model.eval()
            self.model = model
            self.model_version = Path(model_uri).stem
            print(f"[InferenceService] Loaded checkpoint: {model_uri}")
        else:
            print(f"[InferenceService] No checkpoint at {model_uri} — switching to demo mode.")
            self._demo_mode = True

    def predict(self, image_bytes: bytes, filename: str = "upload.jpg",
                include_xai: bool = False, include_lime: bool = False,
                include_nl: bool = False) -> Dict:

        if self._demo_mode:
            from api.services.demo_inference import demo_predict
            return demo_predict(image_bytes, filename)

        t0 = time.time()
        image_id = f"img_{hashlib.md5(image_bytes).hexdigest()[:12]}"

        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].numpy()

        predicted_idx = int(probs.argmax())
        confidence = float(probs[predicted_idx])
        per_class_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4)
                           for i in range(NUM_CLASSES)}

        return {
            "image_id": image_id,
            "filename": filename,
            "predicted_source": CLASS_NAMES[predicted_idx],
            "confidence": confidence,
            "is_ai_generated": CLASS_NAMES[predicted_idx] != "real",
            "per_class_probs": per_class_probs,
            "gradcam_url": None,
            "explanation_text": None,
            "processing_ms": int((time.time() - t0) * 1000),
            "model_version": self.model_version,
        }

    def predict_batch(self, image_bytes_list: list,
                      filenames: list = None) -> list:
        filenames = filenames or [f"image_{i}.jpg"
                                  for i in range(len(image_bytes_list))]
        return [
            self.predict(b, f)
            for b, f in zip(image_bytes_list, filenames)
        ]

    def is_healthy(self) -> bool:
        if self._demo_mode:
            return True
        try:
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                self.model(dummy)
            return True
        except Exception:
            return False
