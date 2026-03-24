"""Configuration management for GigaAM server."""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Server configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="GIGAAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Model settings
    default_model: str = "v3_e2e_rnnt"
    available_models: List[str] = [
        "v3_ctc",
        "v3_rnnt",
        "v3_e2e_ctc",
        "v3_e2e_rnnt",
    ]
    device: str = "cuda"  # cuda or cpu
    fp16_encoder: bool = True

    # pyannote/HF settings
    hf_token: str | None = None
    segmentation_model: str = "pyannote/speaker-diarization-community-1"

    # Performance settings
    max_audio_duration: float = 3600.0  # 1 hour max
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    stream_buffer_size: int = 16000  # 1 second at 16kHz

    # CORS settings
    cors_origins: List[str] = ["*"]

    @field_validator("available_models")
    @classmethod
    def validate_models(cls, v: List[str]) -> List[str]:
        valid_models = [
            "v1_ctc",
            "v1_rnnt",
            "v1_ssl",
            "v2_ctc",
            "v2_rnnt",
            "v2_ssl",
            "v3_ctc",
            "v3_rnnt",
            "v3_ssl",
            "v3_e2e_ctc",
            "v3_e2e_rnnt",
        ]
        for model in v:
            if model not in valid_models:
                raise ValueError(f"Invalid model: {model}. Valid: {valid_models}")
        return v


@lru_cache
def get_settings() -> ServerSettings:
    """Get cached settings instance."""
    return ServerSettings()
