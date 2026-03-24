"""Model lifecycle management."""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict

import torch
from loguru import logger

from gigaam import load_model
from gigaam.model import GigaAM, GigaAMASR

from ..config import ServerSettings


class ModelManager:
    """
    Manages ASR model lifecycle.

    Uses gigaam.load_model() for loading models.
    Supports lazy loading and model caching.
    Thread-safe with async locking.
    """

    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self._models: Dict[str, GigaAMASR | GigaAM] = {}
        self._loading: Dict[str, asyncio.Lock] = {}
        self._device = torch.device(
            settings.device if torch.cuda.is_available() else "cpu"
        )

    async def get_model(self, model_name: str) -> GigaAMASR | GigaAM:
        """Get or load model with async locking to prevent duplicate loads."""
        if model_name in self._models:
            return self._models[model_name]

        if model_name not in self._loading:
            self._loading[model_name] = asyncio.Lock()

        async with self._loading[model_name]:
            if model_name in self._models:
                return self._models[model_name]

            logger.info(f"Loading model: {model_name}")
            try:
                model = load_model(
                    model_name=model_name,
                    fp16_encoder=self.settings.fp16_encoder,
                    device=self._device,
                )
                self._models[model_name] = model
                logger.info(f"Model loaded: {model_name}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

    def list_models(self) -> list[str]:
        """Return list of available model names."""
        return self.settings.available_models

    def get_loaded_models(self) -> Dict[str, str]:
        """Return dict of loaded models with their status."""
        return {name: "loaded" for name in self._models.keys()}

    async def preload_models(self, model_names: list[str] | None = None) -> None:
        """Preload specified models on startup."""
        if model_names is None:
            model_names = [self.settings.default_model]

        tasks = [
            self.get_model(name)
            for name in model_names
            if name in self.settings.available_models
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model from memory."""
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Model unloaded: {model_name}")
            return True
        return False

    async def unload_all_models(self) -> None:
        """Unload all models from memory."""
        self._models.clear()
        logger.info("All models unloaded")


@asynccontextmanager
async def model_lifespan(app):
    """Lifespan context manager for model lifecycle."""
    from ..config import get_settings

    settings = get_settings()
    model_manager = ModelManager(settings)

    # Preload default model on startup
    await model_manager.preload_models()

    app.state.model_manager = model_manager

    yield

    # Cleanup on shutdown
    await model_manager.unload_all_models()
