"""GigaAM FastAPI Server - Main application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .api.v1.router import api_router
from .config import ServerSettings, get_settings

# Global app reference for dependency injection
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get the FastAPI app instance."""
    if _app is None:
        raise RuntimeError("App not initialized. Use lifespan context manager.")
    return _app


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global _app
    _app = app

    settings = get_settings()
    logger.info(f"Starting GigaAM Server v0.1.0")
    logger.info(f"Default model: {settings.default_model}")
    logger.info(f"Device: {settings.device}")

    # Initialize model manager
    from .services.model_manager import ModelManager

    model_manager = ModelManager(settings)
    await model_manager.preload_models([settings.default_model])

    app.state.model_manager = model_manager
    app.state.settings = settings

    yield

    # Cleanup
    logger.info("Shutting down GigaAM Server")
    await model_manager.unload_all_models()
    _app = None


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="GigaAM Speech-to-Text API",
        description="OpenAI-compatible STT API powered by GigaAM acoustic models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store settings in app state
    app.state.settings = settings

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "GigaAM Server",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "gigaam_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
