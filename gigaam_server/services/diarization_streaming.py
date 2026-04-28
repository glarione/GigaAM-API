"""Streaming speaker diarization using DIART."""

from typing import Any, AsyncGenerator, Dict, List, Optional

import os
import numpy as np
import torch
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from pyannote.core import SlidingWindowFeature, SlidingWindow
from huggingface_hub import login
from loguru import logger

from gigaam.preprocess import SAMPLE_RATE


class StreamingDiarizationService:
    """
    DIART-based streaming speaker diarization.

    Provides real-time speaker identification during audio streaming.
    Uses incremental clustering to update speaker assignments as
    the conversation progresses.

    Example:
        service = StreamingDiarizationService(settings)
        async for result in service.stream_diarize(audio_generator):
            print(f"Active speakers: {result['speakers']}")
    """

    def __init__(self, settings):
        """
        Initialize streaming diarization service.

        Args:
            settings: Application settings with device configuration
        """
        self.settings = settings
        self._pipeline = None
        self._device = torch.device(
            settings.device if torch.cuda.is_available() else "cpu"
        )

        # Default configuration (DIHARD III optimized)
        self._config = {
            "step": 0.5,  # 500ms chunk shift
            "latency": 0.5,  # Minimum latency
            "tau_active": 0.555,
            "rho_update": 0.422,
            "delta_new": 1.517,
        }

    async def get_pipeline(self):
        """
        Lazy load DIART SpeakerDiarization pipeline.

        Returns:
            SpeakerDiarization pipeline instance
        """
        if self._pipeline is not None:
            return self._pipeline

        try:
            # Login to HuggingFace using HF_TOKEN environment variable (consistent with batch diarization)
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
                logger.info("Logged in to HuggingFace using HF_TOKEN")
            else:
                logger.warning(
                    "HF_TOKEN not found. DIART requires pyannote models which need authentication. "
                    "Set HF_TOKEN environment variable or add to .env file."
                )

            # Configure pipeline
            config = SpeakerDiarizationConfig(
                step=self._config["step"],
                latency=self._config["latency"],
                tau_active=self._config["tau_active"],
                rho_update=self._config["rho_update"],
                delta_new=self._config["delta_new"],
            )

            self._pipeline = SpeakerDiarization(config)
            # DIART handles device placement internally, no need to call .to()
            logger.info(
                f"DIART streaming diarization pipeline loaded "
                f"(device={self._device}, latency={self._config['latency']}s)"
            )
            return self._pipeline

        except ImportError as e:
            logger.error(
                "DIART not installed. Install with: "
                "pip install gigaam[streaming-diarization]"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load DIART pipeline: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load DIART pipeline: {e}")
            raise

    async def stream_diarize(
        self,
        audio_generator: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream diarization results alongside audio chunks.

        Args:
            audio_generator: Async generator yielding raw audio bytes (int16)

        Yields:
            Dict with diarization results:
                - timestamp: Current audio timestamp (seconds)
                - speakers: List of active speaker labels
                - segments: List of speaker segments with boundaries
                - confidence: Clustering confidence score
        """
        try:
            pipeline = await self.get_pipeline()
        except Exception as e:
            logger.error(f"Failed to initialize diarization: {e}")
            # Yield empty results for all chunks if diarization fails
            async for _ in audio_generator:
                yield {
                    "timestamp": 0.0,
                    "speakers": [],
                    "segments": [],
                    "confidence": 0.0,
                }
            return

        # Buffer for accumulating audio chunks
        audio_buffer = []
        timestamp = 0.0
        chunk_size = 16000  # 1 second at 16kHz
        diarization_chunk_size = 80000  # 5 seconds for DIART

        try:
            async for audio_bytes in audio_generator:
                if len(audio_bytes) == 0:
                    continue

                # Convert int16 bytes to float32 waveform - ensure it's a real numpy array
                audio_np = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                # Force a copy to ensure it's not a memoryview
                audio_np = audio_np.copy()

                audio_buffer.append(audio_np)
                timestamp += len(audio_np) / SAMPLE_RATE

                # Process when we have enough audio (DIART expects ~5s chunks)
                total_samples = sum(len(chunk) for chunk in audio_buffer)
                if total_samples >= diarization_chunk_size:
                    # Concatenate buffered audio - all chunks are already real arrays
                    audio = np.concatenate(audio_buffer)

                    # Trim to exact 5 seconds if needed
                    if len(audio) > diarization_chunk_size:
                        audio = audio[:diarization_chunk_size]

                    # Final conversion to ensure proper dtype and array type
                    audio_input = np.ascontiguousarray(audio, dtype=np.float32)

                    # Debug: log audio input type and shape
                    logger.debug(
                        f"Diarization input: type={type(audio_input)}, "
                        f"shape={audio_input.shape}, "
                        f"dtype={audio_input.dtype}, "
                        f"flags={audio_input.flags if hasattr(audio_input, 'flags') else 'N/A'}"
                    )

                    # Wrap audio in SlidingWindowFeature (DIART expects this format)
                    # Create a sliding window with proper timing
                    # Based on DIART source code analysis:
                    # - segmentation() expects (samples, channels) or (batch, samples, channels)
                    # - torch.stack of (samples, channels) creates (batch, samples, channels)
                    # - Then assertion checks batch.shape[1] == samples (correct!)
                    window = SlidingWindow(
                        start=timestamp - (diarization_chunk_size / SAMPLE_RATE),
                        duration=diarization_chunk_size / SAMPLE_RATE,
                        step=diarization_chunk_size / SAMPLE_RATE,
                    )
                    # Reshape to (samples, channels) = (80000, 1) for mono audio
                    audio_2d = audio_input.reshape(-1, 1)  # (80000,) -> (80000, 1)
                    waveform = SlidingWindowFeature(audio_2d, window)

                    # Debug: check waveform data shape
                    logger.debug(
                        f"Waveform data: type={type(waveform.data)}, "
                        f"shape={waveform.data.shape if hasattr(waveform.data, 'shape') else 'N/A'}"
                    )

                    # Try calling pipeline and catch the actual error with more context
                    try:
                        # Run diarization inference - pipeline expects Sequence[SlidingWindowFeature]
                        result = pipeline([waveform])

                        # Extract result from the list (we only sent one chunk)
                        if isinstance(result, list) and len(result) > 0:
                            result = result[
                                0
                            ][
                                0
                            ]  # Get Annotation from (Annotation, SlidingWindowFeature) tuple
                        else:
                            result = None
                    except AssertionError as e:
                        # Pipeline assertion failed - might be a bug in DIART's assertion for 2D input
                        logger.error(f"Pipeline assertion failed: {e}")
                        logger.error(
                            "This might be a DIART bug with 2D input. The models expect (batch, features, samples)"
                        )
                        raise
                    except Exception as e:
                        logger.error(f"Pipeline call failed: {e}")
                        logger.error(f"Waveform shape: {waveform.data.shape}")
                        raise

                    if result is None:
                        raise ValueError("Diarization returned no result")

                    try:
                        # Extract active speakers at current timestamp
                        speakers = []
                        segments = []

                        if hasattr(result, "itertracks"):
                            for turn, _, speaker in result.itertracks(yield_label=True):
                                # Convert turn boundaries to seconds
                                turn_start = turn.start
                                turn_end = turn.end

                                segments.append(
                                    {
                                        "speaker": speaker,
                                        "start": float(turn_start),
                                        "end": float(turn_end),
                                    }
                                )

                                # Check if segment is active at current timestamp
                                if turn_start <= timestamp < turn_end:
                                    speakers.append(speaker)

                        # Calculate confidence (simplified: based on segment count)
                        confidence = len(segments) / 5.0 if segments else 0.0
                        confidence = min(confidence, 1.0)

                        yield {
                            "timestamp": timestamp,
                            "speakers": speakers,
                            "segments": segments,
                            "confidence": confidence,
                        }

                    except Exception as e:
                        logger.warning(f"Diarization inference failed: {e}")
                        # Yield empty result on error
                        yield {
                            "timestamp": timestamp,
                            "speakers": [],
                            "segments": [],
                            "confidence": 0.0,
                        }

                    # Keep last 1 second for continuity
                    remaining_samples = max(
                        0, total_samples - diarization_chunk_size + 16000
                    )
                    if remaining_samples > 0 and len(audio_buffer) > 0:
                        # Rebuild buffer with remaining audio
                        new_buffer: List[np.ndarray] = []
                        samples_left = remaining_samples
                        for chunk in reversed(audio_buffer):
                            if samples_left <= 0:
                                break
                            take = min(len(chunk), samples_left)
                            new_buffer.insert(0, chunk[-take:])
                            samples_left -= take
                        audio_buffer = new_buffer
                    else:
                        audio_buffer = []

        except Exception as e:
            logger.error(f"Streaming diarization error: {e}")
            raise

    def configure(self, **kwargs):
        """
        Configure diarization parameters.

        Args:
            **kwargs: Configuration parameters (latency, tau_active, etc.)
        """
        self._config.update(kwargs)
        logger.info(f"Diarization configuration updated: {kwargs}")

        # Reset pipeline to apply new config
        self._pipeline = None

    @classmethod
    def fast_config(cls) -> Dict[str, float]:
        """Fast preset: lower latency, slightly less accurate."""
        return {
            "latency": 0.5,
            "tau_active": 0.507,
            "rho_update": 0.3,
            "delta_new": 1.2,
        }

    @classmethod
    def quality_config(cls) -> Dict[str, float]:
        """Quality preset: higher latency, better accuracy."""
        return {
            "latency": 2.0,
            "tau_active": 0.555,
            "rho_update": 0.422,
            "delta_new": 1.517,
        }
