"""
DeepTrace Health & Metrics Router
GET /health       — liveness + readiness
GET /metrics      — Prometheus metrics
"""

import time
import os
from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from api.models.response import HealthResponse
from api.config import get_settings

router = APIRouter(tags=["observability"])

_startup_time = time.time()

# ---------------------------------------------------------------------------
# Prometheus counters (lazy init)
# ---------------------------------------------------------------------------

def _get_prometheus_metrics() -> str:
    try:
        from prometheus_client import (
            Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
            REGISTRY
        )
        return generate_latest(REGISTRY).decode("utf-8")
    except ImportError:
        return "# prometheus_client not installed\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    settings = get_settings()

    # Check model
    model_ok = False
    try:
        from api.services.inference import InferenceService
        model_ok = InferenceService.get_instance().is_healthy()
    except Exception:
        pass

    # Check Redis
    redis_ok = False
    try:
        import redis
        r = redis.from_url(settings.redis_url, socket_connect_timeout=1)
        r.ping()
        redis_ok = True
    except Exception:
        pass

    overall = "healthy" if (model_ok and redis_ok) else \
              "degraded" if model_ok else "unhealthy"

    return HealthResponse(
        status=overall,
        model_loaded=model_ok,
        redis_connected=redis_ok,
        version=settings.app_version,
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@router.get("/metrics", response_class=PlainTextResponse,
            summary="Prometheus metrics endpoint")
async def prometheus_metrics():
    return _get_prometheus_metrics()
