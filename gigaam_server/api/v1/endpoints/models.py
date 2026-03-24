"""Models endpoint - list available models."""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from ....config import get_settings
from ....schemas.transcription import ModelInfo

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models():
    """List available ASR models."""
    settings = get_settings()

    models = [
        ModelInfo(
            id=model_name,
            created=int(datetime.now().timestamp()),
            owned_by="gigaam",
            status="available",
            description=_get_model_description(model_name),
        )
        for model_name in settings.available_models
    ]

    return {"object": "list", "data": models}


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get details for a specific model."""
    settings = get_settings()

    if model_id not in settings.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_id}",
        )

    return ModelInfo(
        id=model_id,
        created=int(datetime.now().timestamp()),
        owned_by="gigaam",
        status="available",
        description=_get_model_description(model_id),
    )


def _get_model_description(model_name: str) -> str:
    """Get human-readable model description."""
    descriptions = {
        "v1_ctc": "GigaAM v1 CTC-based ASR model (2000h training)",
        "v1_rnnt": "GigaAM v1 RNNT-based ASR model (2000h training)",
        "v1_ssl": "GigaAM v1 self-supervised model for embeddings",
        "v2_ctc": "GigaAM v2 CTC-based ASR model (2000h training)",
        "v2_rnnt": "GigaAM v2 RNNT-based ASR model (2000h training)",
        "v2_ssl": "GigaAM v2 self-supervised model for embeddings",
        "v3_ctc": "GigaAM v3 CTC-based ASR model (4000h training)",
        "v3_rnnt": "GigaAM v3 RNNT-based ASR model (4000h training)",
        "v3_ssl": "GigaAM v3 self-supervised model for embeddings",
        "v3_e2e_ctc": "GigaAM v3 end-to-end CTC with punctuation and normalization",
        "v3_e2e_rnnt": "GigaAM v3 end-to-end RNNT with punctuation and normalization",
    }
    return descriptions.get(model_name, "GigaAM speech recognition model")
