"""Segment-based audio processing with VAD and diarization."""

from collections import deque
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Set

import numpy as np
import torch
from loguru import logger

from gigaam.preprocess import SAMPLE_RATE


@dataclass
class Segment:
    """Represents a speech segment with speaker attribution."""

    speaker: str
    start: float  # seconds
    end: float  # seconds
    text: Optional[str] = None
    status: str = "pending"  # pending, processing, complete, skipped


class SegmentProcessor:
    """
    Process audio stream with VAD filtering and segment-based transcription.

    Instead of transcribing every 1-second chunk of mixed audio, this processor:
    1. Detects speech segments using DIART's segmentation model (VAD)
    2. Accumulates audio segments with speaker labels
    3. Transcribes each segment when complete (single-speaker audio)
    4. Streams results with native speaker attribution
    """

    def __init__(self, model_manager, settings, diarization_service=None):
        """
        Initialize segment processor.

        Args:
            model_manager: Model manager for loading ASR models
            settings: Application settings
            diarization_service: Optional diarization service for speaker detection
        """
        self.model_manager = model_manager
        self.settings = settings
        self.diarization_service = diarization_service

        # Audio buffer: deque of (timestamp, audio_chunk)
        # Max size: 120 seconds at 16kHz = 1,920,000 samples
        self.audio_buffer: deque = deque(maxlen=1920000)
        self.current_timestamp = 0.0

        # Segment tracking
        self.segment_queue: List[Segment] = []
        self.processing_segments: Set[str] = set()

        # VAD threshold (energy-based)
        self.vad_threshold = 0.3

        # Diarization configuration
        self.diarization_interval = 3.0  # Run every 3 seconds
        self.last_diarization_time = 0.0

    async def process_stream(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        model_name: str,
        enable_diarization: bool = False,
    ) -> AsyncGenerator[dict, None]:
        """
        Process single audio stream with VAD + segment transcription.

        Args:
            audio_generator: Async generator yielding raw audio bytes (int16)
            model_name: ASR model to use for transcription
            enable_diarization: Enable speaker diarization

        Yields:
            Dict with transcription results:
                - speaker: Speaker label (if diarization enabled)
                - text: Transcribed text
                - start: Segment start time (seconds)
                - end: Segment end time (seconds)
                - is_final: Whether segment is complete
        """
        # Initialize diarization pipeline if enabled
        pipeline = None
        if enable_diarization and self.diarization_service:
            try:
                pipeline = await self.diarization_service.get_pipeline()
                logger.info("Segment processor: diarization enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize diarization: {e}")
                pipeline = None

        # Load ASR model
        model = await self.model_manager.get_model(model_name)
        if "ctc" in model_name:
            decoding = model.decoding
        else:
            decoding = model.decoding

        # Track audio for continuous transcription (when diarization disabled)
        continuous_audio = []
        continuous_start = 0.0

        # Process audio chunks
        async for audio_bytes in audio_generator:
            if len(audio_bytes) == 0:
                continue

            try:
                # Decode and buffer audio
                audio_chunk = self._decode_audio(audio_bytes)
                self._update_buffer(audio_chunk)

                # If no diarization, accumulate audio for continuous transcription
                if not pipeline:
                    continuous_audio.append(audio_chunk)

                    # Transcribe every 3 seconds of accumulated audio
                    accumulated_duration = (
                        len(np.concatenate(continuous_audio)) / SAMPLE_RATE
                    )
                    if accumulated_duration >= 3.0:
                        segment_audio = np.concatenate(continuous_audio)

                        # VAD check
                        if self._is_speech(segment_audio):
                            logger.debug(
                                f"Transcribing continuous audio: {accumulated_duration:.1f}s"
                            )
                            text = await self._transcribe_segment(
                                segment_audio, model, decoding
                            )

                            if text:
                                yield {
                                    "speaker": None,
                                    "text": text,
                                    "start": continuous_start,
                                    "end": continuous_start + accumulated_duration,
                                    "is_final": False,
                                }

                            # Reset buffer (keep last 0.5s for overlap)
                            overlap_samples = int(0.5 * SAMPLE_RATE)
                            if len(segment_audio) > overlap_samples:
                                continuous_audio = [segment_audio[-overlap_samples:]]
                                continuous_start = (
                                    continuous_start + accumulated_duration - 0.5
                                )
                            else:
                                continuous_audio = []
                                continuous_start = (
                                    continuous_start + accumulated_duration
                                )
                        else:
                            logger.debug(
                                f"Skipping silent continuous audio: {accumulated_duration:.1f}s"
                            )
                            continuous_audio = []
                            continuous_start = continuous_start + accumulated_duration

                # Run diarization every 3s if enabled
                if pipeline and self._should_run_diarization():
                    segments = await self._run_diarization(pipeline)
                    self._update_segments(segments)

                # Transcribe completed segments
                completed = self._get_completed_segments()
                for segment in completed:
                    if segment.status == "pending":
                        # Extract segment audio
                        segment_audio = self._extract_audio(segment)

                        # VAD check: skip if no speech
                        if not self._is_speech(segment_audio):
                            logger.debug(
                                f"Skipping silent segment: {segment.start}-{segment.end}"
                            )
                            segment.status = "skipped"
                            continue

                        # Transcribe segment
                        logger.debug(
                            f"Transcribing segment: {segment.speaker} {segment.start:.2f}-{segment.end:.2f}s"
                        )
                        segment.text = await self._transcribe_segment(
                            segment_audio, model, decoding
                        )
                        segment.status = "complete"

                        # Yield result
                        yield {
                            "speaker": segment.speaker,
                            "text": segment.text,
                            "start": segment.start,
                            "end": segment.end,
                            "is_final": True,
                        }

                # Send periodic updates for ongoing segments
                if pipeline and len(self.segment_queue) > 0:
                    latest_segment = self.segment_queue[-1]
                    if (
                        latest_segment.status == "pending"
                        and latest_segment.end <= self.current_timestamp
                    ):
                        # Segment is complete but not yet transcribed
                        # This can happen if diarization lagged behind
                        pass

            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                continue

        # Final transcription for continuous mode (no diarization)
        if not pipeline and continuous_audio:
            segment_audio = np.concatenate(continuous_audio)
            if self._is_speech(segment_audio) and len(segment_audio) > 0:
                logger.debug(
                    f"Final transcription: {len(segment_audio) / SAMPLE_RATE:.1f}s"
                )
                text = await self._transcribe_segment(segment_audio, model, decoding)
                if text:
                    yield {
                        "speaker": None,
                        "text": text,
                        "start": continuous_start,
                        "end": continuous_start + len(segment_audio) / SAMPLE_RATE,
                        "is_final": True,
                    }

        # Transcribe any remaining segments at end of stream
        if pipeline:
            remaining = self._get_completed_segments()
            for segment in remaining:
                if segment.status == "pending":
                    segment_audio = self._extract_audio(segment)
                    if self._is_speech(segment_audio):
                        segment.text = await self._transcribe_segment(
                            segment_audio, model, decoding
                        )
                        segment.status = "complete"
                        yield {
                            "speaker": segment.speaker,
                            "text": segment.text,
                            "start": segment.start,
                            "end": segment.end,
                            "is_final": True,
                        }

    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode raw audio bytes to float32 numpy array."""
        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return audio_np.copy()

    def _update_buffer(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer with timestamp."""
        self.audio_buffer.append((self.current_timestamp, audio_chunk))
        self.current_timestamp += len(audio_chunk) / SAMPLE_RATE

    def _should_run_diarization(self) -> bool:
        """Check if enough audio accumulated for diarization."""
        current_time = self.current_timestamp
        if current_time - self.last_diarization_time >= self.diarization_interval:
            self.last_diarization_time = current_time
            return True
        return False

    async def _run_diarization(self, pipeline):
        """Run DIART segmentation and clustering."""
        # Extract last 3s of audio from buffer
        audio_3s = self._extract_audio_window(self.diarization_interval)

        if len(audio_3s) < 48000:  # Less than 3s
            return []

        # Wrap in SlidingWindowFeature
        from pyannote.core import SlidingWindow, SlidingWindowFeature

        window = SlidingWindow(
            start=self.current_timestamp - self.diarization_interval,
            duration=1.0 / SAMPLE_RATE,
            step=1.0 / SAMPLE_RATE,
        )
        audio_2d = audio_3s.reshape(-1, 1)
        waveform = SlidingWindowFeature(audio_2d, window)

        # Run DIART pipeline
        try:
            # Use manual pipeline to avoid chunk_buffer initialization issues
            batch = torch.stack([torch.from_numpy(w.data) for w in [waveform]])
            seg_output = pipeline.segmentation(batch)

            if seg_output.shape[0] > 0 and seg_output.shape[-1] > 0:
                # Extract segments from segmentation output
                segments = []
                seg_numpy = seg_output.squeeze(0).cpu().numpy()

                # Find active frames (speech detected)
                for frame_idx in range(seg_numpy.shape[0]):
                    if seg_numpy[frame_idx].max() > 0.5:  # Speech threshold
                        # Calculate timestamp
                        frame_time = frame_idx * (
                            self.diarization_interval / seg_numpy.shape[0]
                        )
                        segments.append(
                            Segment(
                                speaker="speaker0",  # Will be updated by clustering
                                start=frame_time,
                                end=frame_time
                                + (self.diarization_interval / seg_numpy.shape[0]),
                                status="pending",
                            )
                        )

                return segments
            else:
                logger.debug("No speech detected in 3s window")
                return []

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            return []

    def _update_segments(self, new_segments: List[Segment]):
        """Update segment queue with new segments."""
        for segment in new_segments:
            # Check if segment already exists (within 0.1s tolerance)
            existing = next(
                (
                    s
                    for s in self.segment_queue
                    if s.speaker == segment.speaker
                    and abs(s.start - segment.start) < 0.1
                ),
                None,
            )

            if existing:
                # Update end time if extended
                if segment.end > existing.end:
                    existing.end = segment.end
            else:
                self.segment_queue.append(segment)

    def _get_completed_segments(self) -> List[Segment]:
        """Get segments that are complete (end_time <= current_time)."""
        return [
            s
            for s in self.segment_queue
            if s.end <= self.current_timestamp and s.status == "pending"
        ]

    def _extract_audio(self, segment: Segment) -> np.ndarray:
        """Extract audio from buffer by timestamp."""
        start_samples = int(segment.start * SAMPLE_RATE)
        end_samples = int(segment.end * SAMPLE_RATE)

        # Concatenate all audio from buffer
        audio_parts = []
        for timestamp, chunk in self.audio_buffer:
            audio_parts.append(chunk)

        if not audio_parts:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(audio_parts)

        # Extract segment range
        start_idx = max(0, start_samples)
        end_idx = min(len(audio), end_samples)

        if end_idx <= start_idx:
            return np.array([], dtype=np.float32)

        return audio[start_idx:end_idx]

    def _extract_audio_window(self, duration: float) -> np.ndarray:
        """Extract last N seconds of audio from buffer."""
        start_time = self.current_timestamp - duration
        start_samples = int(start_time * SAMPLE_RATE)

        # Concatenate all audio from buffer
        audio_parts = []
        for timestamp, chunk in self.audio_buffer:
            audio_parts.append(chunk)

        if not audio_parts:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(audio_parts)

        # Extract window range
        start_idx = max(0, start_samples)
        end_idx = len(audio)

        if end_idx <= start_idx:
            return np.array([], dtype=np.float32)

        return audio[start_idx:end_idx]

    def _is_speech(self, audio: np.ndarray) -> bool:
        """VAD: Check if audio contains speech using energy threshold."""
        if len(audio) == 0:
            return False

        # Simple energy-based VAD
        energy = np.mean(np.abs(audio))
        return energy > self.vad_threshold

    async def _transcribe_segment(self, audio: np.ndarray, model, decoding) -> str:
        """Transcribe single audio segment."""
        if len(audio) == 0:
            return ""

        with torch.no_grad():
            audio_tensor = torch.tensor(audio).unsqueeze(0).to(model._device)
            length = torch.tensor([audio_tensor.shape[-1]]).to(model._device)

            encoded, encoded_len = model.forward(audio_tensor, length)
            text = decoding.decode(model.head, encoded, encoded_len)[0]

        return text
