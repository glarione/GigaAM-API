"""Streaming transcription service for real-time audio."""

import base64
from dataclasses import dataclass, field
from typing import AsyncGenerator, List

import numpy as np
import torch
from loguru import logger

from ..schemas.streaming import (
    StreamingErrorMessage,
    StreamingFinalMessage,
    StreamingPartialMessage,
)


@dataclass
class AudioBuffer:
    """Buffer for streaming audio chunks."""

    samples: List[np.ndarray] = field(default_factory=list)
    max_size: int = 16000 * 60  # 60 seconds max buffer

    def add(self, chunk: np.ndarray) -> None:
        """Add chunk to buffer."""
        self.samples.append(chunk)
        # Trim if too large
        total = sum(len(s) for s in self.samples)
        if total > self.max_size:
            # Remove oldest chunks
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

    Uses CTC model with streaming-friendly decoding.
    Provides partial results during processing.
    """

    def __init__(self, model_manager, settings):
        self.model_manager = model_manager
        self.settings = settings
        self._chunk_size = 16000  # 1 second chunks
        self._overlap_size = 8000  # 0.5 second overlap for context

    async def stream_transcribe(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        model_name: str,
    ) -> AsyncGenerator[
        StreamingPartialMessage | StreamingFinalMessage | StreamingErrorMessage, None
    ]:
        """
        Stream transcription from audio chunks.

        Args:
            audio_generator: Async generator yielding base64-encoded audio chunks
            model_name: Model to use for transcription

        Yields:
            Streaming messages with partial/final results
        """
        buffer = AudioBuffer()
        last_text = ""
        is_final = False

        try:
            model = await self.model_manager.get_model(model_name)

            # Get decoding for CTC models
            if "ctc" in model_name:
                decoding = model.decoding
            else:
                # For RNNT, use the decoding directly
                decoding = model.decoding

            async for chunk in audio_generator:
                try:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(chunk)
                    audio_np = (
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )

                    buffer.add(audio_np)

                    # Process when we have enough audio
                    if len(buffer.get_audio()) >= self._chunk_size:
                        # Get current audio with overlap
                        audio = buffer.get_audio()

                        # Take last chunk_size samples with overlap
                        if len(audio) > self._chunk_size + self._overlap_size:
                            audio = audio[-(self._chunk_size + self._overlap_size) :]

                        # Transcribe
                        audio_tensor = (
                            torch.tensor(audio).unsqueeze(0).to(model._device)
                        )
                        length = torch.tensor([audio_tensor.shape[-1]]).to(
                            model._device
                        )

                        # Forward pass
                        encoded, encoded_len = model.forward(audio_tensor, length)
                        text = decoding.decode(model.head, encoded, encoded_len)[0]

                        # Yield partial result if changed
                        if text and text != last_text:
                            last_text = text
                            yield StreamingPartialMessage(
                                text=text,
                                is_final=False,
                            )

                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue

            # Final transcription
            final_audio = buffer.get_audio()
            if len(final_audio) > 0:
                audio_tensor = torch.tensor(final_audio).unsqueeze(0).to(model._device)
                length = torch.tensor([audio_tensor.shape[-1]]).to(model._device)

                encoded, encoded_len = model.forward(audio_tensor, length)
                final_text = decoding.decode(model.head, encoded, encoded_len)[0]

                yield StreamingFinalMessage(
                    text=final_text or last_text,
                    segments=[],
                    is_final=True,
                )

        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")
            yield StreamingErrorMessage(
                message=str(e),
                is_final=True,
            )

    def _process_audio_chunk(self, chunk: bytes, model) -> str:
        """Process a single audio chunk and return transcription."""
        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.tensor(audio_np).unsqueeze(0).to(model._device)
        length = torch.tensor([audio_tensor.shape[-1]]).to(model._device)

        encoded, encoded_len = model.forward(audio_tensor, length)
        text = model.decoding.decode(model.head, encoded, encoded_len)[0]

        return text
