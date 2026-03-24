"""Speaker diarization service using pyannote."""

import os
import subprocess
from typing import List

import torch
from torch.torch_version import TorchVersion

# Add safe globals for PyTorch 2.6+ compatibility BEFORE pyannote imports
torch.serialization.add_safe_globals([TorchVersion])

import pyannote.audio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.core.task import Problem, Resolution, Specifications

# Add more safe globals after pyannote import
torch.serialization.add_safe_globals([Problem, Resolution, Specifications])

from loguru import logger
from gigaam.preprocess import SAMPLE_RATE


class DiarizationService:
    """
    Speaker diarization using pyannote.audio.

    Uses gigaam.vad_utils for safe model loading with PyTorch 2.6+.
    """

    def __init__(self, settings):
        self.settings = settings
        self._pipeline = None
        self._device = torch.device(
            settings.device if torch.cuda.is_available() else "cpu"
        )

    async def get_diarization_pipeline(self):
        """Lazy load pyannote diarization pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            # Set HF token if provided
            if self.settings.hf_token:
                os.environ["HF_TOKEN"] = self.settings.hf_token

            # Load diarization pipeline using safe loading
            # SpeakerDiarization combines VAD + speaker embedding + clustering
            with torch.serialization.safe_globals([TorchVersion, Problem, Resolution, Specifications]):
                self._pipeline = SpeakerDiarization(
                    segmentation=self.settings.segmentation_model,
                )
                self._pipeline.instantiate({})
            self._pipeline.to(self._device)
            logger.info("Diarization pipeline loaded")
            return self._pipeline
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise

    def _prepare_audio_for_pyannote(self, audio_path: str) -> str:
        """
        Prepare audio file for pyannote by ensuring proper format.

        Pyannote expects 16kHz mono WAV with exact sample counts.
        """
        import tempfile

        # Create temp file with proper format
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            prepared_path = f.name

        try:
            # Convert to proper format: 16kHz, mono, PCM
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", audio_path,
                    "-ar", str(SAMPLE_RATE),
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    "-sample_fmt", "s16",
                    prepared_path
                ],
                capture_output=True,
                check=True,
                timeout=60
            )
            return prepared_path
        except Exception as e:
            logger.warning(f"Failed to prepare audio: {e}")
            # Return original path if conversion fails
            if os.path.exists(prepared_path):
                os.unlink(prepared_path)
            return audio_path

    async def _get_diarization_output(self, audio_path: str):
        """
        Get diarization output for an audio file.

        Returns DiarizeOutput with speaker boundaries, or None on failure.
        This is used by the transcription service to split audio by speaker.
        """
        try:
            pipeline = await self.get_diarization_pipeline()

            # Prepare audio for pyannote
            prepared_path = self._prepare_audio_for_pyannote(audio_path)

            # Run diarization - returns DiarizeOutput
            diarization_output = pipeline(prepared_path)

            # Clean up prepared file if different from original
            if prepared_path != audio_path and os.path.exists(prepared_path):
                os.unlink(prepared_path)

            return diarization_output
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
            return None

    async def diarize(
        self,
        audio_path: str,
        segments: List,
    ) -> List:
        """
        Assign speaker labels to transcription segments.

        Args:
            audio_path: Path to audio file
            segments: List of transcription segments with boundaries

        Returns:
            Segments with speaker labels added
        """
        try:
            # Get diarization output
            diarization_output = self._get_diarization_output(audio_path)

            if diarization_output is None:
                return segments

            # Get the Annotation object from DiarizeOutput
            # Use exclusive_speaker_diarization for non-overlapping speech
            diarization = diarization_output.exclusive_speaker_diarization

            # Map segments to speakers using itertracks on Annotation
            for segment in segments:
                start, end = segment.start, segment.end
                # Find overlapping speaker
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= start < turn.end or turn.start < end <= turn.end:
                        segment.speaker = speaker
                        break

            return segments
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
            # Return segments without speaker labels
            return segments
