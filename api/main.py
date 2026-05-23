"""
DeepTrace FastAPI Application
Entry point: uvicorn api.main:app --reload
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.routers import predict, health

# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[DeepTrace] Starting up — loading model...")
    try:
        from api.services.inference import InferenceService
        InferenceService.get_instance()
        print("[DeepTrace] Model loaded successfully.")
    except Exception as e:
        print(f"[DeepTrace] WARNING: Model load failed: {e}")
    yield
    print("[DeepTrace] Shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI Image Provenance & Authenticity Platform. "
            "Detects which AI generator (Stable Diffusion, Midjourney, DALL-E 3, Flux) "
            "produced an image, or identifies it as a real photograph."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ---------------------------------------------------------------------------
    # Middleware
    # ---------------------------------------------------------------------------

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        t0 = time.time()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = str(int((time.time() - t0) * 1000))
        return response

    # Prometheus request counter middleware
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        response = await call_next(request)
        try:
            from prometheus_client import Counter
            REQUEST_COUNT = Counter(
                "deeptrace_http_requests_total",
                "Total HTTP requests",
                ["method", "endpoint", "status"],
            )
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).inc()
        except Exception:
            pass
        return response

    # ---------------------------------------------------------------------------
    # Exception handlers
    # ---------------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ---------------------------------------------------------------------------
    # Routers
    # ---------------------------------------------------------------------------

    app.include_router(predict.router)
    app.include_router(health.router)

    # ---------------------------------------------------------------------------
    # Root
    # ---------------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
