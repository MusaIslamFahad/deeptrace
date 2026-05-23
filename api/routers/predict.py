"""
DeepTrace Prediction Router
POST /predict         — single image, synchronous
POST /predict/batch   — multiple images, async via Celery
GET  /jobs/{job_id}   — poll batch job status
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from api.config import get_settings
from api.middleware.auth import verify_api_key
from api.models.response import (
    BatchJobResponse, BatchJobStatus, PredictionResponse
)
from api.services.inference import InferenceService

router = APIRouter(prefix="/api/v1", tags=["prediction"])

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_FILE_SIZE_MB = 10


def _validate_image(file: UploadFile):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Accepted: {', '.join(ALLOWED_MIME_TYPES)}",
        )


# ---------------------------------------------------------------------------
# Single image prediction
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a single image",
    description=(
        "Upload one image and receive the predicted AI generator source, "
        "confidence score, per-class probabilities, and optional XAI outputs."
    ),
)
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    gradcam: bool = Query(False, description="Include Grad-CAM XAI overlay"),
    lime: bool = Query(False, description="Include LIME explanation (slower)"),
    explain: bool = Query(False, description="Include natural-language explanation"),
    _api_key: str = Depends(verify_api_key),
):
    _validate_image(file)

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )

    service = InferenceService.get_instance()

    try:
        result = service.predict(
            image_bytes=image_bytes,
            filename=file.filename or "upload.jpg",
            include_xai=gradcam or lime or explain,
            include_lime=lime,
            include_nl=explain,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )

    return PredictionResponse(**result)


# ---------------------------------------------------------------------------
# Batch prediction (async)
# ---------------------------------------------------------------------------

@router.post(
    "/predict/batch",
    response_model=BatchJobResponse,
    summary="Classify multiple images asynchronously",
    description=(
        "Submit a batch of images for classification. Returns a job_id "
        "that you can poll via GET /jobs/{job_id}."
    ),
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Image files (max 50)"),
    _api_key: str = Depends(verify_api_key),
):
    settings = get_settings()

    if len(files) > settings.batch_size_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum batch size is {settings.batch_size_limit}.",
        )

    for f in files:
        _validate_image(f)

    # Read all files into memory
    image_data = []
    for f in files:
        img_bytes = await f.read()
        image_data.append({"bytes": img_bytes, "filename": f.filename or "image.jpg"})

    job_id = str(uuid.uuid4())

    # Dispatch to Celery worker
    try:
        from workers.tasks import process_batch_job
        process_batch_job.apply_async(
            args=[job_id, image_data],
            task_id=job_id,
        )
    except Exception as e:
        # Fallback: process synchronously if Celery unavailable
        print(f"[predict_batch] Celery unavailable ({e}), running synchronously")
        service = InferenceService.get_instance()
        results = service.predict_batch(
            [d["bytes"] for d in image_data],
            filenames=[d["filename"] for d in image_data],
        )
        return BatchJobResponse(
            job_id=job_id,
            status="completed",
            image_count=len(files),
            poll_url=f"/api/v1/jobs/{job_id}",
        )

    return BatchJobResponse(
        job_id=job_id,
        status="queued",
        image_count=len(files),
        poll_url=f"/api/v1/jobs/{job_id}",
    )


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}",
    response_model=BatchJobStatus,
    summary="Poll batch job status",
)
async def get_job_status(
    job_id: str,
    _api_key: str = Depends(verify_api_key),
):
    try:
        from workers.celery_app import celery_app
        task = celery_app.AsyncResult(job_id)

        if task.state == "PENDING":
            return BatchJobStatus(job_id=job_id, status="queued", progress=0, total=0)
        elif task.state == "PROGRESS":
            meta = task.info or {}
            return BatchJobStatus(
                job_id=job_id, status="processing",
                progress=meta.get("current", 0),
                total=meta.get("total", 0),
            )
        elif task.state == "SUCCESS":
            result_data = task.result or {}
            return BatchJobStatus(
                job_id=job_id, status="completed",
                progress=result_data.get("total", 0),
                total=result_data.get("total", 0),
                results=[PredictionResponse(**r) for r in result_data.get("results", [])
                         if "error" not in r],
            )
        else:
            return BatchJobStatus(
                job_id=job_id, status="failed",
                progress=0, total=0,
                error=str(task.info),
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found: {str(e)}",
        )
