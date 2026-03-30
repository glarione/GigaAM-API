"""Transcription service wrapping GigaAM models."""

import os
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger

from gigaam.preprocess import SAMPLE_RATE, load_audio, load_audio_bytes

from ..schemas.transcription import TranscriptionSegment


@dataclass
class TranscriptionResult:
    """Internal transcription result."""

    text: str
    segments: List[TranscriptionSegment]
    duration: float
    model: str
    language: str = "ru"
    processing_time: float = 0.0


class TranscriptionService:
    """
    Core transcription service.

    Reuses gigaam package methods:
    - GigaAMASR.transcribe() for short-form (<25s)
    - GigaAMASR.transcribe_longform() for long-form with pyannote VAD
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def _audio_to_segments(
        self,
        utterances: List[dict],
        model_name: str,
        audio_duration: float,
    ) -> Tuple[str, List[TranscriptionSegment]]:
        """Convert utterances to segments and full text."""
        full_text = ""
        segments = []

        for i, utterance in enumerate(utterances):
            text = utterance.get("transcription", "")
            boundaries = utterance.get("boundaries", (0.0, audio_duration))

            segment = TranscriptionSegment(
                id=i,
                start=boundaries[0],
                end=boundaries[1],
                text=text,
                tokens=[],
            )
            segments.append(segment)

            if full_text:
                full_text += " " + text
            else:
                full_text = text

        return full_text, segments

    async def _get_diarization_segments(self, audio_path: str) -> List[Dict]:
        """
        Get speaker-separated segments from diarization pipeline.

        Returns list of dicts with: start, end, speaker, audio_path
        Each segment contains audio for a single speaker.
        """
        from ..config import get_settings
        from .diarization import DiarizationService

        settings = get_settings()
        diarization_service = DiarizationService(settings)

        # Get diarization output with speaker boundaries
        diarization_output = await diarization_service._get_diarization_output(audio_path)

        if diarization_output is None:
            return []

        # Get exclusive diarization (non-overlapping speech)
        diarization = diarization_output.exclusive_speaker_diarization

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        return segments

    async def transcribe_with_diarization(
        self,
        audio_path: str,
        model_name: str,
    ) -> TranscriptionResult:
        """
        Transcribe audio with speaker diarization.

        Splits audio by speaker boundaries and transcribes each speaker separately.
        """
        start_time = time.perf_counter()

        # Get audio duration
        audio = load_audio(audio_path)
        audio_duration = len(audio) / SAMPLE_RATE

        # Get speaker segments from diarization
        speaker_segments = await self._get_diarization_segments(audio_path)

        if not speaker_segments:
            # Fallback to regular transcription if diarization fails
            logger.warning("Diarization failed, falling back to regular transcription")
            return await self.transcribe_from_file(audio_path, model_name, diarization=False)

        model = await self.model_manager.get_model(model_name)

        segments: List[TranscriptionSegment] = []
        full_text = ""
        temp_files = []

        try:
            # Convert audio tensor to numpy for processing
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio

            for i, seg in enumerate(speaker_segments):
                # Extract audio for this speaker segment
                start_sample = int(seg["start"] * SAMPLE_RATE)
                end_sample = int(seg["end"] * SAMPLE_RATE)
                segment_audio = audio_np[start_sample:end_sample]

                # Save to temp WAV
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    temp_files.append(temp_path)

                with wave.open(temp_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(SAMPLE_RATE)
                    wav_file.writeframes((segment_audio * 32767).astype(np.int16).tobytes())

                # Transcribe this speaker's segment
                text = model.transcribe(temp_path)

                segment = TranscriptionSegment(
                    id=i,
                    start=seg["start"],
                    end=seg["end"],
                    text=text,
                    tokens=[],
                    speaker=seg["speaker"],
                )
                segments.append(segment)

                if full_text:
                    full_text += " " + text
                else:
                    full_text = text

            processing_time = time.perf_counter() - start_time
            rtf = processing_time / audio_duration if audio_duration > 0 else 0

            logger.info(
                f"Transcription with diarization complete: "
                f"{len(segments)} segments, RTF={rtf:.3f}x"
            )

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                duration=audio_duration,
                model=model_name,
                processing_time=processing_time,
            )
        finally:
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    async def transcribe_from_bytes(
        self,
        audio_data: bytes,
        model_name: str,
        vad_filter: bool = True,
        diarization: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio from raw bytes without temp file creation.

        This is the optimized path for CPU - avoids subprocess and file I/O overhead.
        Uses load_audio_bytes() for direct tensor conversion.

        For long-form audio that needs pyannote VAD, a temp WAV file is still required
        since pyannote pipeline reads from file path.
        """
        start_time = time.perf_counter()

        # Calculate audio duration from raw data (int16 PCM)
        audio_duration = len(audio_data) / (SAMPLE_RATE * 2)  # 2 bytes per sample

        model = await self.model_manager.get_model(model_name)

        # For short-form: process directly from bytes (no temp file)
        if audio_duration <= 25 and not (vad_filter and audio_duration > 20):
            logger.info(
                f"Using short-form transcription for {audio_duration:.2f}s audio"
            )

            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]
            # Convert bytes to tensor directly
            audio_tensor = load_audio_bytes(audio_data)

            # Run inference directly on tensor
            with torch.inference_mode():
                wav = audio_tensor.to(model._device).to(model._dtype).unsqueeze(0)
                length = torch.full([1], wav.shape[-1], device=model._device)
                encoded, encoded_len = model.forward(wav, length)
                text = model.decoding.decode(model.head, encoded, encoded_len)[0]

            segments = [
                TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=audio_duration,
                    text=text,
                    tokens=[],
                )
            ]
            full_text = text

            processing_time = time.perf_counter() - start_time
            rtf = processing_time / audio_duration if audio_duration > 0 else 0

            logger.info(f"Transcription complete: RTF={rtf:.3f}x")

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                duration=audio_duration,
                model=model_name,
                processing_time=processing_time,
            )

        # For long-form: still need temp file for pyannote VAD
        logger.info(
            f"Using long-form transcription for {audio_duration:.2f}s audio"
        )

        # Save to temp WAV for pyannote compatibility
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        try:
            # Write raw PCM as WAV
            # Ensure buffer size is a multiple of 2 bytes (int16)
            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            with wave.open(wav_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes((audio_np * 32767).astype(np.int16).tobytes())

            logger.info(f"Running VAD on {wav_path}")
            utterances = model.transcribe_longform(wav_path)
            logger.info(
                f"Long-form returned {len(utterances)} segments from VAD"
            )

            # Handle empty utterances - VAD might have failed, use fallback
            if not utterances:
                logger.warning(
                    "VAD returned no segments, falling back to empty result"
                )
                # Return empty result instead of crashing
                utterances = [{"transcription": "", "boundaries": (0.0, audio_duration)}]

            full_text, segments = self._audio_to_segments(
                utterances, model_name, audio_duration
            )

            processing_time = time.perf_counter() - start_time
            rtf = processing_time / audio_duration if audio_duration > 0 else 0

            logger.info(
                f"Transcription complete: {len(segments)} segments, "
                f"RTF={rtf:.3f}x"
            )

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                duration=audio_duration,
                model=model_name,
                processing_time=processing_time,
            )
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    async def transcribe(
        self,
        audio_data: bytes,
        model_name: str,
        vad_filter: bool = True,
        diarization: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio data.

        Auto-routes to short-form or long-form based on duration.
        Deprecated: Use transcribe_from_bytes() for optimized path.
        """
        return await self.transcribe_from_bytes(
            audio_data, model_name, vad_filter, diarization
        )

    async def transcribe_from_file(
        self,
        audio_path: str,
        model_name: str,
        vad_filter: bool = True,
        diarization: bool = False,
    ) -> TranscriptionResult:
        """Transcribe from file path."""
        start_time = time.perf_counter()

        model = await self.model_manager.get_model(model_name)

        # Check duration
        try:
            audio = load_audio(audio_path)
            audio_duration = len(audio) / SAMPLE_RATE
        except Exception:
            audio_duration = 0.0

        if audio_duration > 25 or (vad_filter and audio_duration > 20):
            logger.info(
                f"Using long-form transcription for {audio_duration:.2f}s audio"
            )
            utterances = model.transcribe_longform(audio_path)

            # Handle empty utterances - create single empty segment
            if not utterances:
                logger.warning("Long-form returned empty, creating single segment")
                utterances = [{"transcription": "", "boundaries": (0.0, audio_duration)}]

            full_text, segments = self._audio_to_segments(
                utterances, model_name, audio_duration
            )
        else:
            logger.info(
                f"Using short-form transcription for {audio_duration:.2f}s audio"
            )
            text = model.transcribe(audio_path)
            segments = [
                TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=audio_duration,
                    text=text,
                    tokens=[],
                )
            ]
            full_text = text

        processing_time = time.perf_counter() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0

        logger.info(f"Transcription complete: RTF={rtf:.3f}x")

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            duration=audio_duration,
            model=model_name,
            processing_time=processing_time,
        )
