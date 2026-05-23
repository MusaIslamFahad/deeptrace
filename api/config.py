"""
DeepTrace API Configuration
All settings loaded from environment variables / .env file
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "DeepTrace API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model
    model_uri: str = "checkpoints/calibrated_model.pt"
    model_device: str = "cpu"          # "cuda" in production
    image_size: int = 224

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Auth
    api_keys: str = "dev-key-123,test-key-456"  # comma-separated; use proper secrets in prod
    rate_limit_per_minute: int = 60
    batch_size_limit: int = 50

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # S3 (for XAI artifact storage)
    s3_bucket: str = "deeptrace-artifacts"
    aws_region: str = "us-east-1"

    # Anthropic (for NL explanations)
    anthropic_api_key: str = ""

    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @property
    def api_key_set(self) -> set:
        return set(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def cors_origin_list(self) -> list:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
