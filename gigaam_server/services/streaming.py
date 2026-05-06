"""Streaming transcription service for real-time audio."""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator

import numpy as np
import torch
from loguru import logger

from ..schemas.streaming import (
    StreamingErrorMessage,
    StreamingFinalMessage,
    StreamingPartialMessage,
)
from .segment_processor import SegmentProcessor


@dataclass
class AudioBuffer:
    """Buffer for streaming audio chunks."""

    samples: list[np.ndarray] = field(default_factory=list)
    max_size: int = 16000 * 60  # 60 seconds max buffer

    def add(self, chunk: np.ndarray) -> None:
        """Add chunk to buffer."""
        self.samples.append(chunk)
        total = sum(len(s) for s in self.samples)
        if total > self.max_size:
            while self.samples and sum(len(s) for s in self.samples) > self.max_size:
                self.samples.pop(0)

    def get_audio(self) -> np.ndarray:
        """Get concatenated audio from buffer."""
        if not self.samples:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.samples)

    def clear(self) -> None:
        """Clear buffer."""
        self.samples.clear()


class StreamingService:
    """
    Real-time streaming transcription service.
    Provides partial results during processing.
    """

    def __init__(self, model_manager, settings, diarization_service=None):
        self.model_manager = model_manager
        self.settings = settings
        self.diarization_service = diarization_service
        self._chunk_size = 32000  # 1 second chunks
        self._overlap_size = 0

    async def stream_transcribe(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        model_name: str,
        enable_diarization: bool = False,
    ) -> AsyncGenerator[
        StreamingPartialMessage | StreamingFinalMessage | StreamingErrorMessage, None
    ]:
        """
        Stream transcription from audio chunks.

        Two modes:
        - Diarization enabled: Use StreamingDiarizationService for speaker-aware transcription
        - Diarization disabled: Use SegmentProcessor for continuous transcription
        """
        if enable_diarization and self.diarization_service:
            # Use DIART-based diarization service
            async for result in self._stream_with_diarization(
                audio_generator, model_name
            ):
                yield result
        else:
            # Use SegmentProcessor for continuous mode
            async for result in self._stream_continuous(audio_generator, model_name):
                yield result

    async def _stream_with_diarization(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        model_name: str,
    ) -> AsyncGenerator[StreamingPartialMessage | StreamingFinalMessage, None]:
        """Stream transcription with DIART diarization."""
        # Placeholder: For now, yield empty results
        # TODO: Implement DIART-based streaming using self.diarization_service.stream_diarize()
        async for _ in audio_generator:
            yield StreamingPartialMessage(
                text="",
                is_final=False,
            )

        yield StreamingFinalMessage(
            text="",
            segments=[],
            is_final=True,
        )

    async def _stream_continuous(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        model_name: str,
    ) -> AsyncGenerator[StreamingPartialMessage | StreamingFinalMessage, None]:
        """Stream transcription without diarization (continuous mode)."""
        processor = SegmentProcessor(self.model_manager, self.settings, None)

        accumulated_text = ""

        async for result in processor.process_stream(
            audio_generator, model_name, enable_diarization=False
        ):
            text = result.get("text", "")
            if text:
                accumulated_text += text + " "

            message = StreamingPartialMessage(
                text=text,
                is_final=result.get("is_final", False),
            )
            yield message

        yield StreamingFinalMessage(
            text=accumulated_text.strip(),
            segments=[],
            is_final=True,
        )
