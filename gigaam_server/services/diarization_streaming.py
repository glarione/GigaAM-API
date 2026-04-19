"""Streaming speaker diarization using DIART."""

from typing import AsyncGenerator, Dict, Any, Optional, List
import torch
import numpy as np
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
            from diart import SpeakerDiarization, SpeakerDiarizationConfig

            # Configure pipeline
            config = SpeakerDiarizationConfig(
                step=self._config["step"],
                latency=self._config["latency"],
                tau_active=self._config["tau_active"],
                rho_update=self._config["rho_update"],
                delta_new=self._config["delta_new"],
            )

            self._pipeline = SpeakerDiarization(config)
            self._pipeline.to(self._device)

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
        pipeline = await self.get_pipeline()

        # Buffer for accumulating audio chunks
        audio_buffer = []
        timestamp = 0.0
        chunk_duration = 1.0  # 1 second chunks (matching streaming service)

        try:
            async for audio_bytes in audio_generator:
                # Convert int16 bytes to float32 waveform
                audio_np = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                audio_buffer.append(audio_np)
                timestamp += chunk_duration

                # Process when we have enough audio (DIART expects ~5s chunks)
                if len(audio_buffer) >= 5:
                    # Concatenate buffered audio
                    audio = np.concatenate(audio_buffer)

                    # Convert to tensor for DIART
                    audio_tensor = torch.tensor(audio).unsqueeze(0).to(self._device)

                    # Run diarization inference
                    # DIART returns annotation with speaker segments
                    try:
                        result = pipeline(audio_tensor)

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

                    # Clear buffer (keep last 1s for continuity)
                    if len(audio_buffer) > 1:
                        audio_buffer = audio_buffer[-1:]

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
