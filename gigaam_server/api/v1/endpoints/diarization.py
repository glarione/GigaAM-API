"""Diarization-only streaming endpoint using DIART's StreamingInference."""

import asyncio
import threading
from typing import AsyncGenerator

import numpy as np
import rx
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sinks import PredictionAccumulator
from diart.sources import AudioSource
from fastapi import APIRouter, WebSocket
from loguru import logger
from rx import operators as ops

from gigaam_server.api.v1.endpoints.streaming import audio_stream_generator

router = APIRouter(prefix="/v1/diarization", tags=["diarization"])


class StreamingAudioSource(AudioSource):
    """Custom AudioSource that receives audio from async generator."""

    def __init__(self, sample_rate: int = 16000, block_duration: float = 0.5):
        super().__init__(uri="stream", sample_rate=sample_rate)
        self._stream = rx.subject.Subject()
        self._chunk_size = int(sample_rate * block_duration)
        self._audio_buffer: list[np.ndarray] = []
        self._is_running = False

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    @property
    def duration(self):
        return None

    def start_feeding(self, audio_generator: AsyncGenerator[bytes, None]):
        """Feed audio chunks from async generator in background."""
        self._is_running = True

        def feed_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def process():
                try:
                    async for audio_bytes in audio_generator:
                        if len(audio_bytes) == 0:
                            continue

                        # Decode audio
                        audio_np = (
                            np.frombuffer(audio_bytes, dtype=np.int16).astype(
                                np.float32
                            )
                            / 32768.0
                        )
                        self._audio_buffer.append(audio_np)

                        # Emit complete chunks
                        while len(self._audio_buffer) >= self._chunk_size:
                            chunk = np.concatenate(
                                self._audio_buffer[: self._chunk_size]
                            )
                            self._audio_buffer = self._audio_buffer[self._chunk_size :]
                            self._stream.on_next(chunk)

                    # Emit remaining
                    if self._audio_buffer:
                        remaining = np.concatenate(self._audio_buffer)
                        if len(remaining) < self._chunk_size:
                            remaining = np.pad(
                                remaining, (0, self._chunk_size - len(remaining))
                            )
                        self._stream.on_next(remaining)

                    self._stream.on_completed()
                except Exception as e:
                    logger.error(f"Feed error: {e}")
                    self._stream.on_error(e)
                finally:
                    self._is_running = False
                    loop.stop()

            loop.create_task(process())
            loop.run_forever()
            loop.close()

    def read(self):
        """Blocking wait (called by StreamingInference)."""
        pass

    def close(self):
        if not self._stream.is_stopped:
            self._stream.on_completed()


@router.websocket("/ws")
async def websocket_diarization(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speaker diarization.

    Uses DIART's StreamingInference for robust speaker segmentation.

    Query parameters:
    - latency: Algorithmic latency (0.5-5.0s, default: 0.5)
    """
    logger.debug(f"Diarization WS connection from {websocket.client}")
    await websocket.accept()
    logger.info("Diarization WS accepted")

    # Get parameters
    latency = float(websocket.query_params.get("latency", 0.5))
    logger.debug(f"Latency: {latency}s")

    try:
        # Create pipeline
        config = SpeakerDiarizationConfig(
            step=0.5,
            latency=latency,
            tau_active=0.555,
            rho_update=0.422,
            delta_new=1.517,
            sample_rate=16000,
        )
        pipeline = SpeakerDiarization(config)

        # Create audio source
        source = StreamingAudioSource(sample_rate=16000, block_duration=0.5)

        # Create accumulator
        accumulator = PredictionAccumulator(uri="stream")

        # Start feeding audio
        feed_thread = threading.Thread(
            target=source.start_feeding,
            args=(audio_stream_generator(websocket),),
            daemon=True,
        )
        feed_thread.start()

        # Create inference
        inference = StreamingInference(
            pipeline,
            source,
            batch_size=1,
            do_plot=False,
            show_progress=False,
        )
        inference.attach_observers(accumulator)

        # Run inference in background
        def run_inference():
            try:
                # Important: call source.read() to start streaming
                # This blocks until the source is closed
                source.read()
                inference()
            except Exception as e:
                logger.error(f"Inference error: {e}")
                import traceback

                traceback.print_exc()

        inf_thread = threading.Thread(target=run_inference, daemon=True)
        inf_thread.start()

        # Stream results
        last_ts: float = 0.0
        update_count = 0
        while inf_thread.is_alive():
            await asyncio.sleep(0.5)
            update_count += 1

            try:
                annotation = accumulator.get_prediction()

                # Skip if no annotation yet
                if annotation is None:
                    if update_count < 5:  # Wait up to 2.5s for first prediction
                        continue
                    else:
                        logger.warning("No predictions received after 2.5s")
                        break

                speakers = []
                segments = []

                for turn, speaker, _ in annotation.itertracks(yield_label=True):
                    speakers.append(speaker)
                    segments.append(
                        {
                            "speaker": speaker,
                            "start": float(turn.start),
                            "end": float(turn.end),
                        }
                    )

                # Calculate timestamp
                current_ts = float(max((s["end"] for s in segments), default=last_ts))

                # Send update
                result = {
                    "timestamp": current_ts,
                    "speakers": list(set(speakers)),
                    "segments": segments,
                    "active_segments": segments[-5:] if segments else [],
                    "confidence": min(len(segments) / 10.0, 1.0),
                    "is_final": False,
                }
                await websocket.send_json(result)
                last_ts = current_ts
                logger.debug(
                    f"Update {update_count}: {len(segments)} segments, {len(speakers)} speakers"
                )

            except Exception as e:
                logger.error(f"Get prediction error: {e}")
                import traceback

                traceback.print_exc()

        # Wait for completion
        inf_thread.join(timeout=5.0)

        # Send final
        try:
            final = accumulator.get_prediction()
            final.patch()

            segments = []
            for turn, speaker, _ in final.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": speaker,
                        "start": float(turn.start),
                        "end": float(turn.end),
                    }
                )

            await websocket.send_json(
                {
                    "timestamp": max(s["end"] for s in segments) if segments else 0.0,
                    "speakers": list(set(s["speaker"] for s in segments)),
                    "segments": segments,
                    "active_segments": segments,
                    "confidence": min(len(segments) / 10.0, 1.0),
                    "is_final": True,
                }
            )
        except Exception as e:
            logger.error(f"Final error: {e}")
            await websocket.send_json({"error": str(e), "is_final": True})

        source.close()

    except Exception as e:
        logger.error(f"WS error: {e}")
        try:
            await websocket.send_json({"error": str(e), "is_final": True})
        except RuntimeError:
            pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
