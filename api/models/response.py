"""
DeepTrace API — Request and Response Schemas
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


SourceType = Literal["stable_diffusion", "midjourney", "dalle3", "flux", "real"]
StatusType = Literal["queued", "processing", "completed", "failed"]


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    image_id: str
    filename: str
    predicted_source: SourceType
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Calibrated probability of top class")
    is_ai_generated: bool
    per_class_probs: Dict[str, float] = Field(
        ..., description="Probability for each of the 5 source classes"
    )
    gradcam_url: Optional[str] = Field(None, description="S3 URL of XAI composite image")
    explanation_text: Optional[str] = Field(None, description="NL explanation of model decision")
    processing_ms: int
    model_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_schema_extra": {
        "example": {
            "image_id": "img_abc123",
            "filename": "photo.jpg",
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
            "explanation_text": "The upper-left region shows the soft, over-smoothed textures characteristic of Stable Diffusion's VAE decoder.",
            "processing_ms": 143,
            "model_version": "1.2.0",
        }
    }}


class BatchJobResponse(BaseModel):
    job_id: str
    status: StatusType
    image_count: int
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    poll_url: str


class BatchJobStatus(BaseModel):
    job_id: str
    status: StatusType
    progress: int = Field(..., description="Number of images completed")
    total: int
    results: Optional[List[PredictionResponse]] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    redis_connected: bool
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
