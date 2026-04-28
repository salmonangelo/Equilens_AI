"""
EquiLens AI — Backend Application Factory

Creates and configures the production-ready FastAPI application
for local deployment and Cloud Run.

Features:
  - Comprehensive request/response logging
  - Global error handling
  - CORS middleware
  - Health check endpoints
  - Graceful startup/shutdown
"""

import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from config.settings import settings
from backend.routes.health import router as health_router
from backend.routes.upload import router as upload_router
from backend.routes.analysis import router as analysis_router


# =====================================================================
# Logging Setup
# =====================================================================

logger = logging.getLogger(__name__)


# =====================================================================
# Application Lifecycle
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle: startup and shutdown."""
    # --- Startup ---
    logger.info("=" * 80)
    logger.info(f"✓ Starting {settings.APP_NAME} Backend")
    logger.info(f"  Environment: {settings.APP_ENV}")
    logger.info(f"  Debug: {settings.DEBUG}")
    logger.info(f"  CORS Origins: {', '.join(settings.cors_origin_list)}")
    logger.info("=" * 80)

    yield

    # --- Shutdown ---
    logger.info("=" * 80)
    logger.info(f"✓ Shutting down {settings.APP_NAME} Backend")
    logger.info("=" * 80)


# =====================================================================
# FastAPI Application
# =====================================================================

app = FastAPI(
    title=f"{settings.APP_NAME} — Backend API",
    description=(
        "RESTful API for AI-powered fairness and bias auditing. "
        "Upload CSV datasets, run anonymization + fairness metrics + "
        "risk scoring, and retrieve structured JSON reports."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)


# =====================================================================
# Middleware: CORS
# =====================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =====================================================================
# Middleware: Request/Response Logging
# =====================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and outgoing responses."""
    # Skip logging for health checks (to reduce noise)
    skip_logging = request.url.path in ["/health", "/ready", "/api/health", "/api/ready"]

    if not skip_logging:
        logger.info(f"→ {request.method} {request.url.path}")

    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        if not skip_logging:
            logger.info(
                f"← {response.status_code} {request.method} {request.url.path} "
                f"({process_time:.3f}s)"
            )

        return response

    except Exception as exc:
        process_time = time.time() - start_time
        logger.error(
            f"✗ {request.method} {request.url.path} failed after {process_time:.3f}s: {exc}"
        )
        raise


# =====================================================================
# Global Error Handler
# =====================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "path": request.url.path,
        },
    )


# =====================================================================
# Routes
# =====================================================================

# Health routes under /api
app.include_router(health_router, prefix="/api", tags=["Health"])

# Upload + Analysis under /api/v1
app.include_router(upload_router, prefix="/api/v1", tags=["Upload"])
app.include_router(analysis_router, prefix="/api/v1", tags=["Analysis"])


# =====================================================================
# Root Endpoint
# =====================================================================

@app.get("/health", tags=["Root"])
async def health_fallback():
    """Fallback health check at root level."""
    return {"status": "healthy"}


@app.get("/ready", tags=["Root"])
async def ready_fallback():
    """Fallback readiness check at root level."""
    return {"status": "ready"}


@app.get("/", tags=["Root"])
async def root(request: Request):
    """Root endpoint with API information."""
    docs_url = "/docs" if not settings.is_production else None
    endpoints = {
        "health": "GET /api/health",
        "ready": "GET /api/ready",
        "upload": "POST /api/v1/upload",
        "analyze": "POST /api/v1/analyze",
    }

    if "text/html" in request.headers.get("accept", ""):
        docs_link = f"<li><a href='{docs_url}'>API docs</a></li>" if docs_url else ""
        endpoint_items = "".join(
            f"<li><strong>{name}:</strong> {path}</li>" for name, path in endpoints.items()
        )
        return HTMLResponse(
            content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{settings.APP_NAME}</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 40px; }}
    .container {{ max-width: 720px; margin: auto; }}
    a {{ color: #38bdf8; text-decoration: none; }}
    .card {{ background: #1e293b; border-radius: 16px; padding: 24px; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.35); }}
    h1 {{ margin-top: 0; }}
    ul {{ line-height: 1.7; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>{settings.APP_NAME}</h1>
      <p>Version <strong>0.1.0</strong> — Environment: <strong>{settings.APP_ENV}</strong></p>
      <p>Fairness & bias auditing backend API.</p>
      <h2>Endpoints</h2>
      <ul>
        {endpoint_items}
        {docs_link}
      </ul>
      <p>Use the API routes above to upload datasets and analyze fairness metrics.</p>
    </div>
  </div>
</body>
</html>
""",
            status_code=200,
        )

    return {
        "name": settings.APP_NAME,
        "version": "0.1.0",
        "environment": settings.APP_ENV,
        "docs_url": docs_url,
        "endpoints": endpoints,
    }
