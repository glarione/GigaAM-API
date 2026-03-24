"""Health check endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from ....config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    """Readiness check response."""

    ready: bool
    models_loaded: list[str] = []


@router.get("/health")
async def health_check():
    """Basic health check."""
    return HealthResponse(status="healthy")


@router.get("/ready")
async def readiness_check():
    """Check if server is ready to accept requests."""
    from ....main import get_app

    app = get_app()
    model_manager = app.state.model_manager

    loaded = model_manager.get_loaded_models()

    return ReadyResponse(
        ready=len(loaded) > 0,
        models_loaded=list(loaded.keys()),
    )
