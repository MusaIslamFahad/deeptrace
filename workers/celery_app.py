"""
DeepTrace Celery Application
"""

from celery import Celery
from api.config import get_settings

settings = get_settings()

celery_app = Celery(
    "deeptrace",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_track_started=True,
    worker_max_tasks_per_child=100,  # restart worker after 100 tasks to free GPU memory
)
