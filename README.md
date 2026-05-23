<div align="center">
  
# DeepTrace

<p align="center">

<a href="https://github.com/MusaIslamFahad/deeptrace/actions/workflows/ci.yml">
  <img src="https://img.shields.io/github/actions/workflow/status/MusaIslamFahad/deeptrace/ci.yml?style=for-the-badge&label=CI%20Pipeline&logo=githubactions&logoColor=white&color=22C55E" />
</a>

<img src="https://img.shields.io/badge/Python-3.11+-0f172a?style=for-the-badge&logo=python&logoColor=FFD43B" />
<img src="https://img.shields.io/badge/FastAPI-0.111+-0f172a?style=for-the-badge&logo=fastapi&logoColor=00C7B7" />
<img src="https://img.shields.io/badge/API-REST-0f172a?style=for-the-badge&logo=swagger&logoColor=85EA2D" />
<img src="https://img.shields.io/badge/License-MIT-0f172a?style=for-the-badge&logo=opensourceinitiative&logoColor=FACC15" />
<img src="https://img.shields.io/github/stars/MusaIslamFahad/deeptrace?style=for-the-badge&logo=github&label=GitHub%20Stars&color=0f172a" />

</p>

DeepTrace goes beyond "AI or real" - it identifies **which AI system generated an image**: Stable Diffusion, Midjourney, DALL·E 3, Flux, or a real photograph. Each prediction returns calibrated probabilities, a Grad-CAM attention map, and an optional natural-language explanation powered by Claude.

</div>

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
- [Model Architecture](#model-architecture)
- [API Reference](#api-reference)
- [Python SDK](#python-sdk)
- [Browser Extension](#browser-extension)
- [Monitoring](#monitoring)
- [CI/CD](#cicd)
- [Environment Variables](#environment-variables)
- [Extending DeepTrace](#extending-deeptrace)
- [License](#license)

---

## What it does

DeepTrace classifies images across five source classes:

| Source | Description |
|---|---|
| `stable_diffusion` | Stable Diffusion 1.5, 2.1, XL |
| `midjourney` | Midjourney v5 / v6 |
| `dalle3` | OpenAI DALL·E 3 |
| `flux` | Flux.1 dev / schnell |
| `real` | Real photographs |

Every prediction includes:

- **Calibrated class probabilities** - temperature-scaled so confidence actually reflects accuracy
- **Grad-CAM overlay** - heatmap showing which image regions drove the decision
- **LIME explanation** - superpixel attribution map (optional, ~2s)
- **Natural-language explanation** - Claude-generated description of why the model decided what it did (optional)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Data ingestion & versioning  (DVC + S3)                     │
│  SD · MJ · DALL·E · Flux · Real  →  manifest.csv             │
└─────────────────────────┬────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Training pipeline  (MLflow + Optuna)                        │
│  EfficientNet-B0 + ViT-Small + FFT head  →  ensemble         │
│  Temperature calibration  →  Model registry                  │
└─────────────────────────┬────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Production serving  (FastAPI + Celery + Redis)              │
│  /predict  →  single sync   |   /predict/batch  →  async     │
│  Prometheus metrics         |   Evidently drift detection    │
└─────────────────────────┬────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Clients                                                     │
│  React dashboard  ·  REST API  ·  Browser extension (MV3)    │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
deeptrace/
├── api/                         # FastAPI application
│   ├── main.py                  # App factory, middleware, lifespan
│   ├── config.py                # Pydantic settings (reads from .env)
│   ├── routers/
│   │   ├── predict.py           # /predict and /predict/batch endpoints
│   │   └── health.py            # /health endpoint
│   ├── models/                  # Pydantic request / response schemas
│   ├── services/
│   │   └── inference.py         # Singleton model loader + prediction logic
│   └── middleware/
│       └── auth.py              # X-API-Key validation
│
├── model/                       # ML code
│   ├── architecture.py          # EfficientNet-B0 + ViT-Small + FFT ensemble
│   ├── train.py                 # Three-phase training loop + Optuna HPO
│   ├── calibrate.py             # Temperature scaling post-training
│   └── xai_pipeline.py          # Grad-CAM, LIME, FFT visualisation
│
├── data/                        # Data pipeline
│   ├── dataset.py               # DeepTraceDataset, DataLoaders, manifest builder
│   ├── download.py              # HuggingFace dataset downloader + dummy mode
│   └── transforms.py            # Train / val / inference augmentations
│
├── workers/                     # Celery async processing
│   ├── celery_app.py
│   └── tasks.py                 # process_batch_job task
│
├── monitoring/                  # Observability
│   ├── drift_report.py          # Evidently + KL divergence + Slack alert
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboard.json       # Pre-built Grafana dashboard
│
├── frontend/                    # React + TypeScript + Tailwind
│   └── src/
│       ├── pages/
│       │   ├── Upload.tsx
│       │   ├── Analytics.tsx
│       │   └── History.tsx
│       └── api/client.ts        # Typed API wrapper
│
├── extension/                   # Chrome / Firefox browser extension (MV3)
│
├── scripts/
│   ├── evaluate_model.py        # Full evaluation on test set
│   ├── check_metrics.py         # CI gate: assert F1 ≥ threshold
│   └── setup_mlflow.py          # Bootstrap MLflow experiment
│
├── tests/                       # pytest suite (no torch required in CI)
│
├── dvc.yaml                     # DVC pipeline stages
├── params.yaml                  # Training hyperparameters (DVC-tracked)
├── docker-compose.yml           # Full local stack
├── Dockerfile.api               # API production image
├── Dockerfile.worker            # Worker production image
├── Dockerfile.render            # CPU-only image for Render free tier
├── render.yaml                  # Render deployment config
└── .github/workflows/ci.yml     # GitHub Actions CI/CD
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Git

### 1. Clone and install

```bash
git clone https://github.com/MusaIslamFahad/deeptrace.git
cd deeptrace

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements/dev.txt

cp .env.example .env   # edit with your keys
```

### 2. Create a dummy dataset (no downloads needed)

```bash
python data/download.py --dummy
python -c "from data.dataset import build_manifest; build_manifest()"
```

### 3. Start all services

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| React Dashboard | http://localhost:3000 |
| MLflow UI | http://localhost:5000 |
| Grafana | http://localhost:3001 (admin / deeptrace) |
| Prometheus | http://localhost:9090 |

### 4. Make your first prediction

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

### Full pipeline via DVC

```bash
# Pull data from remote (S3)
dvc pull

# Run all stages: download → manifest → train → calibrate → evaluate
dvc repro

# Compare runs in MLflow
# Open http://localhost:5000
```

### Manual training

```bash
# Hyperparameter search (30 Optuna trials)
python -m model.train --hpo --n-trials 30

# Full training with best params
python -m model.train

# Temperature calibration
python -m model.calibrate
```

### Training schedule

The model trains in three phases to avoid over-fitting the backbone to the small head:

| Phase | Layers unfrozen | Epochs | Learning rate |
|---|---|---|---|
| 1 | Classifier head only | 10 | 1e-3 |
| 2 | Head + top 2 backbone blocks | 10 | 2e-4 |
| 3 | All layers | 20 | 5e-5 |

### Tracking hyperparameters

All hyperparameters are stored in `params.yaml` and tracked by DVC. Changing a param and running `dvc repro` will only re-run affected stages.

---

## Model Architecture

```
Image (3 × 224 × 224)
  ├── EfficientNet-B0   →  1280-dim   (texture, local patterns)
  ├── ViT-Small/16      →   384-dim   (global structure)
  └── FFT Head          →    64-dim   (frequency-domain artifacts)
           │
           └── concat  [1728-dim]
                    │
               LayerNorm → Linear(512) → GELU → Dropout(0.3)
                    │
               Linear(256) → GELU → Linear(5)
                    │
            Temperature scaling  (calibration)
                    │
            Calibrated probabilities  (5 classes)
```

**Why an FFT head?** Exploratory analysis revealed that AI-generated images carry unnatural high-frequency patterns in the FFT magnitude spectrum. Stable Diffusion images show characteristic ring artifacts; Midjourney produces unusually smooth frequency roll-off. The FFT head converts this observation into a learned 64-dimensional feature that the ensemble head uses alongside spatial features.

---

## API Reference

Full interactive docs at `http://localhost:8000/docs`.

### Authentication

All prediction endpoints require an API key header:

```
X-API-Key: your-api-key
```

API keys are set via the `API_KEYS` environment variable (comma-separated list).

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Single image, synchronous result |
| `POST` | `/api/v1/predict/batch` | Multiple images, returns `job_id` |
| `GET` | `/api/v1/jobs/{job_id}` | Poll batch job status + results |
| `GET` | `/health` | Liveness + readiness check |
| `GET` | `/metrics` | Prometheus metrics |

### Query parameters for `/predict`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gradcam` | bool | `false` | Include Grad-CAM heatmap (base64 PNG) |
| `lime` | bool | `false` | Include LIME superpixel map (~2s extra) |
| `explain` | bool | `false` | Include natural-language explanation via Claude |

### Batch prediction flow

```
POST /api/v1/predict/batch  (upload N images)
  → { "job_id": "job_xyz", "status": "queued", "total": 5 }

GET /api/v1/jobs/job_xyz
  → { "status": "processing", "completed": 3, "total": 5 }

GET /api/v1/jobs/job_xyz   (when done)
  → { "status": "completed", "results": [ ... ] }
```

---

## Python SDK

```python
import time
import requests

BASE = "http://localhost:8000/api/v1"
HEADERS = {"X-API-Key": "dev-key-123"}

# ── Single prediction ─────────────────────────────────────────
with open("image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE}/predict",
        headers=HEADERS,
        files={"file": f},
        params={"gradcam": True, "explain": True},
    )

result = response.json()
print(result["predicted_source"], result["confidence"])
# stable_diffusion 0.924

# ── Batch prediction ──────────────────────────────────────────
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
files = [("files", open(p, "rb")) for p in image_paths]

response = requests.post(f"{BASE}/predict/batch", headers=HEADERS, files=files)
job_id = response.json()["job_id"]

# Poll until complete
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

The `extension/` directory contains a Manifest V3 extension compatible with Chrome and Firefox.

**Install (developer mode):**

1. Open `chrome://extensions/`
2. Enable **Developer mode** (top right toggle)
3. Click **Load unpacked** → select the `extension/` folder
4. Right-click any image on any webpage → **"Check with DeepTrace"**

A badge overlays the image showing the predicted source and confidence score. The extension calls your locally running API by default; change `API_URL` in `extension/config.js` to point at a deployed instance.

---

## Monitoring

### Prometheus + Grafana

Metrics are exposed at `/metrics` in Prometheus format. Docker Compose auto-provisions:

- **Prometheus** at `http://localhost:9090` - scrapes `/metrics` every 15s
- **Grafana** at `http://localhost:3001` - pre-built dashboard (`monitoring/grafana/dashboard.json`)

Dashboard panels:

- Predictions per second by source class
- Inference latency - p50 / p95 / p99
- API error rate
- Confidence score distribution over time

### Drift detection

Runs weekly via GitHub Actions cron, or trigger manually:

```bash
python -m monitoring.drift_report --threshold 0.25
```

Compares the current week's prediction distribution against the reference baseline using KL divergence via Evidently. Sends a Slack alert if the threshold is exceeded, which typically means a new AI generator is producing images outside the training distribution.

Configure the webhook in `.env`:

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

---

## CI/CD

```
push → main
    │
    ├── lint        ruff + black
    ├── test        pytest + coverage (torch mocked, no GPU needed)
    ├── model-eval  F1 ≥ 0.82 gate
    └── docker      parallel build & push to ghcr.io
                    ├── deeptrace-api:main / :latest / :sha-<hash>
                    └── deeptrace-worker:main / :latest / :sha-<hash>

schedule (Monday 08:00 UTC)
    └── drift-check  →  Slack alert if KL divergence > 0.25
```

Images are published to the GitHub Container Registry:

```
ghcr.io/musaislamfahad/deeptrace-api:latest
ghcr.io/musaislamfahad/deeptrace-worker:latest
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. Key variables:

| Variable | Description | Default |
|---|---|---|
| `API_KEYS` | Comma-separated valid API keys | `dev-key-123` |
| `MODEL_URI` | Path to checkpoint or MLflow model URI | `checkpoints/calibrated_model.pt` |
| `MODEL_DEVICE` | `cpu` or `cuda` | `cpu` |
| `MODEL_DEMO_MODE` | Skip model load, return mock predictions | `false` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `ANTHROPIC_API_KEY` | Required for `?explain=true` | — |
| `SLACK_WEBHOOK_URL` | Drift alert destination | — |
| `S3_BUCKET` | S3 bucket for XAI artifacts | — |

---

## Extending DeepTrace

### Adding a new generator class

1. Collect images into `data/raw/<new_source>/<category>/`
2. Add the class to `CLASS_TO_IDX` in `data/dataset.py`
3. Update `NUM_CLASSES = 6` in `model/architecture.py`
4. Run `dvc repro` to retrain from scratch
5. Bump the model version in the MLflow registry

### Swapping the backbone

Any `timm` model that returns a flat feature vector works as a drop-in:

```python
# model/architecture.py
self.effnet = timm.create_model("convnext_small", pretrained=True, num_classes=0)
```

Check the output feature dimension with `model.feature_info[-1]["num_chs"]` and update the `concat_dim` accordingly.

### Deploying to Render (free tier)

The repo includes a CPU-optimised image (`Dockerfile.render`) and a `render.yaml` blueprint. The CPU PyTorch wheel is ~600 MB smaller than the default CUDA build.

```bash
# Build the Render image locally to test
docker build -f Dockerfile.render -t deeptrace-render .
docker run -p 10000:10000 --env-file .env deeptrace-render
```

---
 
## Contributing
 
Contributions are welcome!
 
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request
Please make sure your code passes the linter and tests before opening a PR:
 
```bash
ruff check api/ model/ data/ workers/ tests/
black --check api/ model/ data/ workers/ tests/
pytest tests/ -x
```
 
---
 
## License
 
MIT © 2024 Md. Musa Islam Fahad. Free to use, adapt, and share with attribution. See [LICENSE](LICENSE) for details.
 
---
 
## Acknowledgements
 
- [PyTorch](https://pytorch.org/) - deep learning framework powering the ensemble model
- [timm](https://github.com/huggingface/pytorch-image-models) - EfficientNet and ViT backbones
- [FastAPI](https://fastapi.tiangolo.com/) - async API framework
- [Celery](https://docs.celeryq.dev/) - distributed task queue for batch inference
- [MLflow](https://mlflow.org/) - experiment tracking and model registry
- [DVC](https://dvc.org/) - data and pipeline versioning
- [Evidently](https://www.evidentlyai.com/) - model monitoring and drift detection
- [Anthropic Claude](https://www.anthropic.com/) - natural-language XAI explanations
- [AI vs Real Image Classification](https://www.kaggle.com/datasets/rhythmghai/ai-vs-real-images-dataset) - base dataset extended to multi-generator provenance detection
---
 
## Author
 
**Md. Musa Islam Fahad**
 
- GitHub: [@MusaIslamFahad](https://github.com/MusaIslamFahad)
---
 
> ⭐ If DeepTrace helped you understand AI image detection or MLOps deployment, a star goes a long way. Thank you!
