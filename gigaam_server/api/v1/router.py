"""API v1 router - combines all v1 endpoints."""

from fastapi import APIRouter

from .endpoints.health import router as health_router
from .endpoints.models import router as models_router
from .endpoints.streaming import router as streaming_router
from .endpoints.transcriptions import router as transcriptions_router

api_router = APIRouter()

api_router.include_router(transcriptions_router)
api_router.include_router(streaming_router)
api_router.include_router(models_router)
api_router.include_router(health_router)
