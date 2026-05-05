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
        # Use SegmentProcessor for modern segment-based processing
        processor = SegmentProcessor(
            self.model_manager, self.settings, self.diarization_service
        )

        # Accumulate all text for final message
        accumulated_text = ""
        all_segments = []

        async for result in processor.process_stream(
            audio_generator, model_name, enable_diarization
        ):
            # Convert result dict to StreamingPartialMessage
            message = StreamingPartialMessage(
                text=result.get("text", ""),
                is_final=result.get("is_final", False),
            )

            # Add diarization info if available
            if enable_diarization:
                message.speakers = [result.get("speaker", "")]
                segment_info = {
                    "speaker": result.get("speaker"),
                    "start": result.get("start", 0.0),
                    "end": result.get("end", 0.0),
                }
                message.active_segments = [segment_info]
                all_segments.append(segment_info)

            # Accumulate text (concatenate for continuous mode, replace for partial updates)
            text = result.get("text", "")
            if text:
                if not enable_diarization:
                    # Continuous mode: concatenate all transcriptions
                    accumulated_text += text + " "
                else:
                    # Diarization mode: keep latest (segments are independent)
                    accumulated_text = text

            yield message

        # Send final message with accumulated text
        yield StreamingFinalMessage(
            text=accumulated_text,
            segments=all_segments if enable_diarization else [],
            is_final=True,
        )
