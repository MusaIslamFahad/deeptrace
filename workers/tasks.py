"""
DeepTrace Celery Tasks
Handles async batch image processing.
"""

from celery import current_task
from workers.celery_app import celery_app


@celery_app.task(bind=True, name="process_batch_job", max_retries=2)
def process_batch_job(self, job_id: str, image_data: list) -> dict:
    """
    Process a batch of images asynchronously.

    Args:
        job_id: Unique identifier for this batch job
        image_data: List of dicts with keys "bytes" (base64-encoded) and "filename"

    Returns:
        dict with "results" list and "total" count
    """
    import base64
    from api.services.inference import InferenceService

    service = InferenceService.get_instance()
    results = []
    total = len(image_data)

    for i, item in enumerate(image_data):
        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={"current": i, "total": total, "job_id": job_id},
        )

        try:
            # bytes may be base64-encoded (JSON-serializable)
            if isinstance(item["bytes"], str):
                img_bytes = base64.b64decode(item["bytes"])
            else:
                img_bytes = bytes(item["bytes"])

            result = service.predict(img_bytes, filename=item.get("filename", f"image_{i}.jpg"))
            # Convert datetime to string for JSON serialization
            if "created_at" in result:
                result["created_at"] = str(result["created_at"])
            results.append(result)

        except Exception as e:
            results.append({
                "error": str(e),
                "filename": item.get("filename", f"image_{i}.jpg"),
                "image_id": f"err_{i}",
            })

    return {"results": results, "total": total, "job_id": job_id}
