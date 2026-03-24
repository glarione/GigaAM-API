"""Transcription service wrapping GigaAM models."""

import os
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger

from gigaam.model import GigaAMASR
from gigaam.preprocess import SAMPLE_RATE, load_audio
from gigaam.vad_utils import segment_audio_file

from ..schemas.transcription import TranscriptionResponse, TranscriptionSegment


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
        from .diarization import DiarizationService
        from ..config import get_settings

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
        """
        import tempfile
        import os
        import subprocess

        start_time = time.perf_counter()

        # Calculate audio duration from raw data
        audio_duration = len(audio_data) / (SAMPLE_RATE * 2)  # 2 bytes per sample

        # Save raw audio to temp file, then convert with ffmpeg for pyannote compatibility
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as raw_file:
            raw_file.write(audio_data)
            raw_path = raw_file.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        try:
            # Convert raw PCM to WAV using ffmpeg for proper format
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "s16le",
                    "-ac", "1",
                    "-ar", str(SAMPLE_RATE),
                    "-i", raw_path,
                    "-acodec", "pcm_s16le",
                    wav_path
                ],
                capture_output=True,
                check=True
            )

            model = await self.model_manager.get_model(model_name)

            if audio_duration > 25 or (vad_filter and audio_duration > 20):
                # Use long-form transcription
                logger.info(
                    f"Using long-form transcription for {audio_duration:.2f}s audio"
                )
                utterances = model.transcribe_longform(wav_path)
                full_text, segments = self._audio_to_segments(
                    utterances, model_name, audio_duration
                )
            else:
                # Use short-form transcription
                logger.info(
                    f"Using short-form transcription for {audio_duration:.2f}s audio"
                )
                text = model.transcribe(wav_path)
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

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                duration=audio_duration,
                model=model_name,
                processing_time=processing_time,
            )
        finally:
            if os.path.exists(raw_path):
                os.unlink(raw_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)

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

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            duration=audio_duration,
            model=model_name,
            processing_time=processing_time,
        )
