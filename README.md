# DeepTrace — AI Image Provenance & Authenticity Platform

> Multi-generator AI image detection with production-grade MLOps, async serving, and real-time monitoring.

[![CI](https://github.com/yourusername/deeptrace/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/deeptrace/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

DeepTrace classifies images by their source — not just "AI or real", but **which AI system generated it**:

| Source | Description |
|---|---|
| `stable_diffusion` | Stable Diffusion (1.5, XL, 2.1) |
| `midjourney` | Midjourney v5/v6 |
| `dalle3` | OpenAI DALL·E 3 |
| `flux` | Flux.1 dev / schnell |
| `real` | Real photographs |

Each prediction returns calibrated probabilities, a Grad-CAM attention map, and an optional natural-language explanation of the model's decision.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Data ingestion & versioning (DVC + S3)                 │
│  SD · MJ · DALL-E · Flux · Real  →  manifest.csv       │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Training pipeline (MLflow + Optuna)                    │
│  EfficientNet-B0 + ViT-Small + FFT head → ensemble      │
│  Temperature calibration → Model registry               │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Production serving (FastAPI + Celery + Redis)          │
│  /predict  →  single  |  /predict/batch  →  async      │
│  Prometheus metrics   |  Evidently drift detection      │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Clients                                                │
│  React dashboard  ·  REST API + SDK  ·  Browser ext.   │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/deeptrace.git
cd deeptrace

python -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt
cp .env.example .env        # edit API keys etc.
```

### 2. Create a dummy dataset (no downloads needed for testing)

```bash
python data/download.py --dummy
python -c "from data.dataset import build_manifest; build_manifest()"
```

### 3. Start all services with Docker Compose

```bash
docker compose up --build
```

Services available:
| Service | URL |
|---|---|
| API (FastAPI) | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| React Dashboard | http://localhost:3000 |
| MLflow UI | http://localhost:5000 |
| Grafana | http://localhost:3001 (admin / deeptrace) |
| Prometheus | http://localhost:9090 |

### 4. Make a prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: dev-key-123" \
  -F "file=@your_image.jpg"
```

Response:
```json
{
  "image_id": "img_abc12345ef01",
  "predicted_source": "stable_diffusion",
  "confidence": 0.924,
  "is_ai_generated": true,
  "per_class_probs": {
    "stable_diffusion": 0.924,
    "midjourney": 0.041,
    "dalle3": 0.019,
    "flux": 0.009,
    "real": 0.007
  },
  "processing_ms": 143,
  "model_version": "1.0.0"
}
```

---

## Training

### Full pipeline (DVC)

```bash
# Pull data
dvc pull

# Run full training pipeline (builds manifest → trains → calibrates)
dvc repro

# Experiments are tracked in MLflow
# Open http://localhost:5000 to compare runs
```

### Manual training with HPO

```bash
# Hyperparameter search (30 Optuna trials)
python -m model.train --hpo --n-trials 30

# Full training with best params
python -m model.train
```

### Three-phase training schedule

| Phase | Layers trained | Epochs | LR |
|---|---|---|---|
| 1 | Classifier head only | 10 | 1e-3 |
| 2 | + top 2 backbone blocks | 10 | 2e-4 |
| 3 | All layers | 20 | 5e-5 |

---

## Model Architecture

```
Image (3×224×224)
  ├── EfficientNet-B0  → 1280-dim features  (texture / local patterns)
  ├── ViT-Small/16     → 384-dim features   (global structure)
  └── FFT Head         → 64-dim features    (frequency domain artifacts)
           │
           └─ concat [1728-dim]
                   │
              LayerNorm → Linear(512) → GELU → Dropout
                   │
              Linear(256) → GELU → Linear(5)
                   │
           Temperature scaling (calibration)
                   │
           Calibrated probabilities (5 classes)
```

**Why the FFT head?** Exploratory analysis (see `notebooks/eda.ipynb`) revealed that AI-generated images have unnatural high-frequency patterns in the FFT magnitude spectrum — SD images show characteristic ring artifacts, while Midjourney shows smooth frequency roll-off. The FFT head converts this EDA observation into a learned feature.

---

## API Reference

Full OpenAPI docs at `http://localhost:8000/docs`.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Single image, synchronous |
| `POST` | `/api/v1/predict/batch` | Multiple images, async queue |
| `GET` | `/api/v1/jobs/{job_id}` | Poll batch job status |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

### Query parameters for `/predict`

| Param | Type | Default | Description |
|---|---|---|---|
| `gradcam` | bool | false | Include Grad-CAM XAI overlay |
| `lime` | bool | false | Include LIME explanation (slower ~2s) |
| `explain` | bool | false | Include NL explanation via Claude API |

### Authentication

All prediction endpoints require an `X-API-Key` header:
```
X-API-Key: your-api-key
```

---

## Python SDK

```python
import requests

BASE = "http://localhost:8000/api/v1"
HEADERS = {"X-API-Key": "dev-key-123"}

# Single prediction
with open("image.jpg", "rb") as f:
    r = requests.post(f"{BASE}/predict", headers=HEADERS,
                      files={"file": f},
                      params={"gradcam": True, "explain": True})
result = r.json()
print(result["predicted_source"], result["confidence"])

# Batch prediction
files = [("files", open(f"img{i}.jpg", "rb")) for i in range(5)]
r = requests.post(f"{BASE}/predict/batch", headers=HEADERS, files=files)
job_id = r.json()["job_id"]

# Poll until complete
import time
while True:
    status = requests.get(f"{BASE}/jobs/{job_id}", headers=HEADERS).json()
    if status["status"] == "completed":
        for pred in status["results"]:
            print(pred["filename"], pred["predicted_source"], pred["confidence"])
        break
    time.sleep(1)
```

---

## Browser Extension

Install manually in Chrome/Firefox:
1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" → select the `extension/` folder
4. Right-click any image on any webpage → **"Check with DeepTrace"**

A badge appears on the image showing the predicted source and confidence.

---

## Monitoring

### Drift detection

Runs weekly via cron (or trigger manually):
```bash
python -m monitoring.drift_report --threshold 0.25
```

Sends a Slack alert if the KL divergence between current and reference prediction distributions exceeds the threshold. This catches:
- New generators producing images outside the training distribution
- Shifts in the types of images being submitted to the API

### Grafana dashboards

Import `monitoring/grafana/dashboard.json` into Grafana (auto-provisioned in Docker Compose).

Panels include:
- Predictions per second by source class
- Inference latency (p50 / p95 / p99)
- Error rate
- Confidence distribution over time

---

## Project Structure

```
deeptrace/
├── api/                    # FastAPI application
│   ├── main.py             # App factory, middleware, startup
│   ├── config.py           # Pydantic settings
│   ├── routers/            # predict.py, health.py
│   ├── models/             # Pydantic request/response schemas
│   ├── services/           # inference.py (model loading + prediction)
│   └── middleware/         # auth.py (API key validation)
├── model/                  # ML code
│   ├── architecture.py     # EfficientNet + ViT + FFT ensemble
│   ├── train.py            # Three-phase training + Optuna HPO
│   ├── xai_pipeline.py     # Grad-CAM, LIME, FFT visualization
│   └── calibrate.py        # Temperature scaling
├── data/                   # Data pipeline
│   ├── dataset.py          # DeepTraceDataset, DataLoaders, manifest builder
│   ├── download.py         # HuggingFace dataset downloader
│   └── transforms.py       # Train/val/inference augmentations
├── workers/                # Celery async workers
│   ├── celery_app.py
│   └── tasks.py            # process_batch_job
├── monitoring/             # Observability
│   ├── drift_report.py     # Evidently + KL divergence + Slack alert
│   ├── prometheus.yml
│   └── grafana/
├── frontend/               # React + TypeScript + Tailwind
│   └── src/
│       ├── pages/          # Upload.tsx, Analytics.tsx, History.tsx
│       └── api/client.ts   # Typed API wrapper
├── extension/              # Chrome/Firefox browser extension (MV3)
├── scripts/                # evaluate_model.py, check_metrics.py, setup_mlflow.py
├── tests/                  # pytest suite
├── dvc.yaml                # DVC pipeline stages
├── params.yaml             # Training hyperparameters (DVC-tracked)
├── docker-compose.yml      # Full stack: api + worker + redis + mlflow + grafana
├── Dockerfile.api
├── Dockerfile.worker
└── .github/workflows/ci.yml
```

---

## CI/CD Pipeline

```
push to main
    │
    ├── lint (ruff + black)
    ├── test (pytest + coverage)
    ├── model-eval gate (F1 ≥ 0.82, AUC ≥ 0.88)
    └── docker build & push → ghcr.io
```

Weekly cron: drift report → Slack alert if threshold exceeded.

---

## Extending DeepTrace

### Adding a new generator class

1. Collect images → `data/raw/<new_source>/<category>/`
2. Add to `CLASS_TO_IDX` in `data/dataset.py`
3. Update `NUM_CLASSES` in `model/architecture.py`
4. Run `dvc repro` to retrain
5. Bump model version in MLflow registry

### Swapping the backbone

```python
# In model/architecture.py, change:
self.effnet = timm.create_model("efficientnet_b2", ...)  # or convnext_small, etc.
```

Any `timm` model that returns a flat feature vector works as a drop-in.

---

## License

MIT © 2024. See [LICENSE](LICENSE).

---

*Built on top of: [AI vs Real Image Classification](https://www.kaggle.com/datasets/rhythmghai/ai-vs-real-images-dataset) — extended to multi-generator provenance detection with full MLOps deployment.*
