"""Streaming message schemas."""

from typing import Literal, Optional, List, Dict

from pydantic import BaseModel, Field


class StreamingAudioMessage(BaseModel):
    """Client -> Server: Audio chunk message."""

    type: Literal["audio"] = "audio"
    data: str = Field(..., description="Base64 encoded audio data")
    is_final: bool = Field(default=False, description="Indicates end of audio")


class StreamingPartialMessage(BaseModel):
    """Server -> Client: Partial transcription."""

    type: Literal["partial"] = "partial"
    text: str = Field(..., description="Partial transcription text")
    is_final: bool = Field(default=False)

    # Diarization fields (optional, only present when diarization is enabled)
    speakers: Optional[List[str]] = Field(
        default=None, description="Active speaker labels at this moment"
    )
    speaker_confidence: Optional[float] = Field(
        default=None, description="Confidence score for speaker clustering (0.0-1.0)"
    )
    active_segments: Optional[List[Dict]] = Field(
        default=None, description="Speaker segments with boundaries"
    )


class StreamingFinalMessage(BaseModel):
    """Server -> Client: Final transcription."""

    type: Literal["final"] = "final"
    text: str = Field(..., description="Final transcription text")
    segments: list[dict] = Field(default_factory=list)
    is_final: bool = Field(default=True)


class StreamingErrorMessage(BaseModel):
    """Server -> Client: Error message."""

    type: Literal["error"] = "error"
    message: str = Field(..., description="Error description")
    is_final: bool = Field(default=True)


StreamingResponseMessage = (
    StreamingPartialMessage | StreamingFinalMessage | StreamingErrorMessage
)
