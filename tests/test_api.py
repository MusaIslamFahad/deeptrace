"""
DeepTrace API Tests
Run with: pytest tests/ -v --cov=api
"""

import io
import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Detect whether real torch is available (not a mock)
# ---------------------------------------------------------------------------

def _real_torch_available() -> bool:
    """
    Returns True only when the *real* torch is installed.
    conftest.py injects a MagicMock into sys.modules["torch"] when torch is
    absent, so hasattr(torch, "Tensor") is always True on a MagicMock.
    We check isinstance(torch.Tensor, type) instead — a MagicMock attribute
    is never a real Python type, but torch.Tensor always is.
    """
    try:
        import torch
        return isinstance(torch.Tensor, type)
    except Exception:
        return False

HAS_TORCH = _real_torch_available()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Test client with mocked inference service."""
    with patch("api.services.inference.InferenceService.get_instance") as mock_factory:
        mock_service = MagicMock()
        mock_service.is_healthy.return_value = True
        mock_service.model_version = "test-1.0"
        mock_service.predict.return_value = {
            "image_id": "img_test123",
            "filename": "test.jpg",
            "predicted_source": "stable_diffusion",
            "confidence": 0.92,
            "is_ai_generated": True,
            "per_class_probs": {
                "stable_diffusion": 0.92,
                "midjourney": 0.04,
                "dalle3": 0.02,
                "flux": 0.01,
                "real": 0.01,
            },
            "gradcam_url": None,
            "explanation_text": None,
            "processing_ms": 143,
            "model_version": "test-1.0",
        }
        mock_factory.return_value = mock_service

        from api.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def test_image_bytes() -> bytes:
    """Generate a dummy 224x224 RGB image."""
    img = Image.new("RGB", (224, 224), color=(128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "dev-key-123"}


# ---------------------------------------------------------------------------
# Health tests
# ---------------------------------------------------------------------------

class TestHealth:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()

    def test_health_endpoint(self, client):
        with patch("redis.from_url") as mock_redis:
            mock_redis.return_value.ping.return_value = True
            r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_requires_auth(self, client, test_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 401

    def test_predict_valid_image(self, client, test_image_bytes, auth_headers):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            headers=auth_headers,
        )
        assert r.status_code == 200
        data = r.json()

        assert "image_id" in data
        assert "predicted_source" in data
        assert "confidence" in data
        assert "per_class_probs" in data
        assert "is_ai_generated" in data
        assert "processing_ms" in data

        assert data["predicted_source"] in [
            "stable_diffusion", "midjourney", "dalle3", "flux", "real"
        ]
        assert 0.0 <= data["confidence"] <= 1.0

        probs = data["per_class_probs"]
        assert set(probs.keys()) == {
            "stable_diffusion", "midjourney", "dalle3", "flux", "real"
        }
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_rejects_non_image(self, client, auth_headers):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("doc.pdf", b"fake pdf content", "application/pdf")},
            headers=auth_headers,
        )
        assert r.status_code == 415

    def test_predict_png(self, client, auth_headers):
        img = Image.new("RGB", (224, 224), color=(0, 128, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        r = client.post(
            "/api/v1/predict",
            files={"file": ("test.png", buf.getvalue(), "image/png")},
            headers=auth_headers,
        )
        assert r.status_code == 200

    def test_response_has_model_version(self, client, test_image_bytes, auth_headers):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            headers=auth_headers,
        )
        assert r.status_code == 200
        assert r.json()["model_version"] == "test-1.0"


# ---------------------------------------------------------------------------
# Batch prediction tests
# ---------------------------------------------------------------------------

class TestBatchPredict:
    def test_batch_predict_queues_job(self, client, test_image_bytes, auth_headers):
        files = [
            ("files", ("img1.jpg", test_image_bytes, "image/jpeg")),
            ("files", ("img2.jpg", test_image_bytes, "image/jpeg")),
        ]
        # process_batch_job is imported inside the function body (try/except),
        # so it cannot be patched at module level. In CI, Celery has no broker
        # so the endpoint uses the synchronous fallback automatically.
        # We just verify the response contract is correct either way.
        r = client.post(
            "/api/v1/predict/batch",
            files=files,
            headers=auth_headers,
        )

        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data
        assert "status" in data
        assert data["image_count"] == 2

    def test_batch_rejects_too_many_files(self, client, test_image_bytes, auth_headers):
        files = [
            ("files", (f"img{i}.jpg", test_image_bytes, "image/jpeg"))
            for i in range(60)
        ]
        r = client.post(
            "/api/v1/predict/batch",
            files=files,
            headers=auth_headers,
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Model architecture tests — skipped in CI (require real torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="requires real torch install")
class TestModelArchitecture:
    def test_model_builds(self):
        import torch
        from model.architecture import build_model, NUM_CLASSES
        model = build_model(pretrained=False)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, NUM_CLASSES)

    def test_fft_extractor(self):
        import torch
        from model.architecture import FFTFeatureExtractor
        extractor = FFTFeatureExtractor(out_dim=64)
        dummy = torch.randn(2, 3, 224, 224)
        out = extractor(dummy)
        assert out.shape == (2, 64)

    def test_freeze_unfreeze(self):
        from model.architecture import build_model
        model = build_model(pretrained=False)
        model.freeze_backbones()
        assert all(not p.requires_grad for p in model.effnet.parameters())
        assert any(p.requires_grad for p in model.classifier.parameters())
        model.unfreeze_all()
        assert all(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# Dataset tests — skipped in CI (require real torchvision)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="requires real torch install")
class TestDataset:
    def test_class_mapping_complete(self):
        from data.dataset import CLASS_TO_IDX, IDX_TO_CLASS
        assert len(CLASS_TO_IDX) == 5
        assert len(IDX_TO_CLASS) == 5
        assert "real" in CLASS_TO_IDX
        assert "stable_diffusion" in CLASS_TO_IDX

    def test_transforms_produce_correct_shape(self):
        import torch
        from PIL import Image
        from data.dataset import get_train_transforms, get_val_transforms
        img = Image.new("RGB", (300, 300))
        t1 = get_train_transforms(224)(img)
        t2 = get_val_transforms(224)(img)
        assert t1.shape == torch.Size([3, 224, 224])
        assert t2.shape == torch.Size([3, 224, 224])


# ---------------------------------------------------------------------------
# Schema / config tests — always run
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_class_names_constant(self):
        from model.architecture import CLASS_NAMES, NUM_CLASSES
        assert len(CLASS_NAMES) == NUM_CLASSES
        assert "real" in CLASS_NAMES
        assert "stable_diffusion" in CLASS_NAMES

    def test_prediction_response_schema(self):
        from api.models.response import PredictionResponse
        from datetime import datetime
        p = PredictionResponse(
            image_id="img_abc",
            filename="test.jpg",
            predicted_source="real",
            confidence=0.95,
            is_ai_generated=False,
            per_class_probs={
                "stable_diffusion": 0.01,
                "midjourney": 0.01,
                "dalle3": 0.01,
                "flux": 0.02,
                "real": 0.95,
            },
            processing_ms=120,
            model_version="1.0",
        )
        assert p.predicted_source == "real"
        assert p.confidence == 0.95
        assert not p.is_ai_generated

    def test_settings_load(self):
        from api.config import get_settings
        s = get_settings()
        assert s.app_version is not None
        assert "dev-key-123" in s.api_key_set


# ---------------------------------------------------------------------------
# Demo inference tests — always run (pure Python, no torch needed)
# ---------------------------------------------------------------------------

class TestDemoInference:
    def test_demo_predict_returns_valid_structure(self):
        from api.services.demo_inference import demo_predict
        result = demo_predict(b"fake image bytes", "test.jpg")

        assert "image_id" in result
        assert "filename" in result
        assert "predicted_source" in result
        assert "confidence" in result
        assert "is_ai_generated" in result
        assert "per_class_probs" in result
        assert "processing_ms" in result
        assert "model_version" in result

    def test_demo_predict_valid_source_class(self):
        from api.services.demo_inference import demo_predict
        result = demo_predict(b"some bytes", "img.jpg")
        assert result["predicted_source"] in [
            "stable_diffusion", "midjourney", "dalle3", "flux", "real"
        ]

    def test_demo_predict_probabilities_sum_to_one(self):
        from api.services.demo_inference import demo_predict
        result = demo_predict(b"test", "img.jpg")
        total = sum(result["per_class_probs"].values())
        assert abs(total - 1.0) < 0.01

    def test_demo_predict_confidence_in_range(self):
        from api.services.demo_inference import demo_predict
        result = demo_predict(b"test", "img.jpg")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_demo_predict_is_ai_flag_consistent(self):
        from api.services.demo_inference import demo_predict
        result = demo_predict(b"test", "img.jpg")
        if result["predicted_source"] == "real":
            assert result["is_ai_generated"] is False
        else:
            assert result["is_ai_generated"] is True

    def test_demo_predict_deterministic(self):
        """Same bytes always produce same prediction."""
        from api.services.demo_inference import demo_predict
        r1 = demo_predict(b"identical bytes", "img.jpg")
        r2 = demo_predict(b"identical bytes", "img.jpg")
        assert r1["predicted_source"] == r2["predicted_source"]
        assert r1["confidence"] == r2["confidence"]

    def test_demo_predict_different_images_can_differ(self):
        """Different byte content should produce different image_ids."""
        from api.services.demo_inference import demo_predict
        r1 = demo_predict(b"image_one", "a.jpg")
        r2 = demo_predict(b"image_two", "b.jpg")
        assert r1["image_id"] != r2["image_id"]

    def test_demo_predict_model_version_is_demo(self):
        from api.services.demo_inference import demo_predict
        result = demo_predict(b"x", "x.jpg")
        assert "demo" in result["model_version"]

    def test_demo_predict_per_class_has_all_sources(self):
        from api.services.demo_inference import demo_predict
        result = demo_predict(b"y", "y.jpg")
        expected_keys = {"stable_diffusion", "midjourney", "dalle3", "flux", "real"}
        assert set(result["per_class_probs"].keys()) == expected_keys
