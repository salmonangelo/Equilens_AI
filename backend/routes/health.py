"""
EquiLens AI — Health & Readiness Endpoints

Standard health check endpoints for Cloud Run / Kubernetes probes.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from config.settings import settings

router = APIRouter()


@router.get(
    "/health",
    summary="Service health check",
    response_description="Service liveness status",
)
async def health_check():
    """
    Liveness probe — is the service running?

    Returns service name, status, environment, and server timestamp.
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/ready",
    summary="Service readiness check",
    response_description="Service readiness status",
)
async def readiness_check():
    """
    Readiness probe — is the service ready to accept traffic?

    Checks:
        - Database connectivity (future)
        - Model availability (future)
        - External service reachability (future)
    """
    # TODO: Add actual readiness checks
    return {"status": "ready", "checks": {}}
