"""
api/services/demo_inference.py

Demo mode for Render free tier deployment.
Returns realistic-looking predictions without a real model checkpoint.
Used when MODEL_DEMO_MODE=true in environment.

This is honest: the API structure, auth, rate limiting, XAI endpoints,
and response schemas are all real. Only the model weights are mocked.
Add a note in your portfolio that the live demo uses mock predictions —
the training code and architecture are what matter.
"""

import hashlib
import time
import random
from typing import Dict

from model.architecture import CLASS_NAMES, NUM_CLASSES


# Seeded so the same image always gets the same "prediction" (deterministic)
def _seed_from_bytes(image_bytes: bytes) -> int:
    return int(hashlib.md5(image_bytes).hexdigest(), 16) % (2**32)


def demo_predict(image_bytes: bytes, filename: str = "upload.jpg") -> Dict:
    """
    Returns a deterministic mock prediction based on image content hash.
    The predicted class and probabilities vary per image but are stable
    across repeated calls with the same image.
    """
    t0 = time.time()

    seed = _seed_from_bytes(image_bytes)
    rng = random.Random(seed)

    image_id = f"img_{hashlib.md5(image_bytes).hexdigest()[:12]}"

    # Pick a predicted class weighted toward AI sources (more interesting demo)
    weights = [0.28, 0.22, 0.18, 0.17, 0.15]   # SD, MJ, DALL-E, Flux, Real
    predicted_idx = rng.choices(range(NUM_CLASSES), weights=weights)[0]

    # Generate realistic-looking probabilities that sum to 1
    raw = [rng.uniform(0.01, 0.15) for _ in range(NUM_CLASSES)]
    raw[predicted_idx] = rng.uniform(0.65, 0.94)   # top class gets high prob
    total = sum(raw)
    probs = [p / total for p in raw]

    per_class_probs = {CLASS_NAMES[i]: round(probs[i], 4) for i in range(NUM_CLASSES)}
    predicted_source = CLASS_NAMES[predicted_idx]
    confidence = round(probs[predicted_idx], 4)

    processing_ms = int((time.time() - t0) * 1000) + rng.randint(80, 180)

    return {
        "image_id": image_id,
        "filename": filename,
        "predicted_source": predicted_source,
        "confidence": confidence,
        "is_ai_generated": predicted_source != "real",
        "per_class_probs": per_class_probs,
        "gradcam_url": None,
        "explanation_text": (
            f"[Demo mode] This deployment uses mock predictions. "
            f"The full trained model runs locally. "
            f"See the GitHub repo for training code and real evaluation results."
        ),
        "processing_ms": processing_ms,
        "model_version": "demo-1.0",
    }
