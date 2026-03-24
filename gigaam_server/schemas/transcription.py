"""Transcription request/response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """Single transcription segment with timing."""

    id: int
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    tokens: list[int] = Field(default_factory=list, description="Token IDs")
    speaker: str | None = Field(
        default=None, description="Speaker label if diarization enabled"
    )


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str = Field(..., description="Full transcription text")
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    duration: float = Field(..., description="Audio duration in seconds")
    model: str = Field(..., description="Model used for transcription")
    language: str = Field(default="ru", description="Detected or specified language")


class TranscriptionVerboseResponse(TranscriptionResponse):
    """Verbose transcription response with additional metadata."""

    words: list[dict] = Field(default_factory=list, description="Word-level timestamps")
    task: str = Field(default="transcribe", description="Task type")


class ModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(default=0, description="Creation timestamp")
    owned_by: str = Field(default="gigaam", description="Model owner")
    status: str = Field(
        default="available", description="Model status: available, loading, error"
    )
    description: str | None = None
