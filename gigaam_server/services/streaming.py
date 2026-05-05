"""Streaming transcription service for real-time audio."""

import base64
import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional

import numpy as np
import torch
from loguru import logger

from ..schemas.streaming import (
    StreamingErrorMessage,
    StreamingFinalMessage,
    StreamingPartialMessage,
)


async def tee_async_generator(
    gen: AsyncGenerator, num_copies: int
) -> List[AsyncGenerator]:
    """Split an async generator into multiple independent generators."""
    queues: List[asyncio.Queue] = [asyncio.Queue() for _ in range(num_copies)]
    done = asyncio.Event()

    async def producer():
        try:
            async for item in gen:
                for queue in queues:
                    await queue.put(item)
            done.set()
        except Exception:
            done.set()
            raise

    async def consumer(queue_idx: int):
        queue = queues[queue_idx]
        while not done.is_set() or not queue.empty():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield item
            except asyncio.TimeoutError:
                continue

    asyncio.create_task(producer())
    return [consumer(i) for i in range(num_copies)]


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

        Args:
            audio_generator: Async generator yielding base64-encoded audio chunks
            model_name: Model to use for transcription
            enable_diarization: Enable streaming speaker diarization (requires DIART)

        Yields:
            Streaming messages with partial/final results
            Includes speaker information if diarization is enabled
        """
        buffer = AudioBuffer()
        last_text = ""

        # Initialize diarization if enabled
        diarization_result_gen = None
        if enable_diarization and self.diarization_service:
            try:
                # Split audio generator so both transcription and diarization get independent streams
                transcribe_audio_gen, diart_audio_gen = await tee_async_generator(
                    audio_generator, 2
                )
                audio_generator = transcribe_audio_gen
                # Create diarization generator from the split audio stream
                diarization_result_gen = self.diarization_service.stream_diarize(
                    diart_audio_gen
                )
                logger.info("Streaming diarization enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize diarization: {e}")
                diarization_result_gen = None

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
                    # Chunk is already decoded audio bytes from audio_stream_generator
                    audio_bytes = chunk

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
                        with torch.no_grad():
                            audio_tensor = (
                                torch.tensor(audio).unsqueeze(0).to(model._device)
                            )
                            length = torch.tensor([audio_tensor.shape[-1]]).to(
                                model._device
                            )

                            # Forward pass
                            encoded, encoded_len = model.forward(audio_tensor, length)
                            text = decoding.decode(model.head, encoded, encoded_len)[0]

                        # Get diarization result if enabled
                        diarization_result = None
                        if diarization_result_gen:
                            try:
                                diarization_result = (
                                    await diarization_result_gen.__anext__()
                                )
                            except StopAsyncIteration:
                                diarization_result_gen = None

                        # Yield partial result if changed
                        if text and text != last_text:
                            last_text = text
                            partial_msg = StreamingPartialMessage(
                                text=text,
                                is_final=False,
                            )

                            # Add diarization info if available
                            if diarization_result:
                                partial_msg.speakers = diarization_result.get(
                                    "speakers"
                                )
                                partial_msg.speaker_confidence = diarization_result.get(
                                    "confidence"
                                )
                                partial_msg.active_segments = diarization_result.get(
                                    "segments"
                                )

                            yield partial_msg

                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue

            # Final transcription
            final_audio = buffer.get_audio()
            if len(final_audio) > 0:
                with torch.no_grad():
                    audio_tensor = (
                        torch.tensor(final_audio).unsqueeze(0).to(model._device)
                    )
                    length = torch.tensor([audio_tensor.shape[-1]]).to(model._device)

                    encoded, encoded_len = model.forward(audio_tensor, length)
                    final_text = decoding.decode(model.head, encoded, encoded_len)[0]

                # Get final diarization result if enabled
                diarization_result = None
                if diarization_result_gen:
                    try:
                        # Try to get final diarization state
                        diarization_result = await diarization_result_gen.__anext__()
                    except StopAsyncIteration:
                        pass

                message = StreamingFinalMessage(
                    text=final_text or last_text,
                    segments=[],
                    is_final=True,
                )

                # Add diarization info if available
                if diarization_result:
                    # Convert segments to the expected format
                    for seg in diarization_result.get("segments", []):
                        message.segments.append(
                            {
                                "speaker": seg.get("speaker"),
                                "start": seg.get("start"),
                                "end": seg.get("end"),
                            }
                        )

                yield message

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
