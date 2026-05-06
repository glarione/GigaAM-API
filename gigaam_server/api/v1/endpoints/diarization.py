"""Diarization-only streaming endpoint using DIART's StreamingInference."""

import asyncio
import json
import threading
import time
from typing import AsyncGenerator

import numpy as np
import rx
from fastapi import APIRouter, WebSocket
from loguru import logger
from rx import operators as ops

from gigaam_server.api.v1.endpoints.streaming import audio_stream_generator

# Import DIART
try:
    from diart import SpeakerDiarization, SpeakerDiarizationConfig
    from diart.sources import AudioSource
    from diart.inference import StreamingInference
    from diart.sinks import PredictionAccumulator

    DIART_AVAILABLE = True
except ImportError:
    DIART_AVAILABLE = False
    logger.warning("DIART not installed")


router = APIRouter(prefix="/v1/diarization", tags=["diarization"])


class DiarizationWebSocketSource(AudioSource):
    """
    AudioSource that receives audio from WebSocket and feeds DIART's StreamingInference.

    This class bridges async WebSocket audio chunks to DIART's RxPY stream system.
    """

    def __init__(self, sample_rate: int = 16000, block_duration: float = 0.5):
        self._stream = rx.subject.Subject()
        self._chunk_size = int(sample_rate * block_duration)
        self._audio_buffer: list[np.ndarray] = []
        self._is_running = False
        self._feed_complete = threading.Event()

        # Initialize base class
        super().__init__(uri="websocket", sample_rate=sample_rate)

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    @property
    def duration(self):
        return None  # Unknown for live streams

    def feed_audio(self, audio_generator: AsyncGenerator[bytes, None]):
        """
        Feed audio chunks from async generator to the stream.

        This runs in a separate thread and converts async audio chunks to
        DIART's synchronous RxPY stream format.
        """
        self._is_running = True

        async def process_audio():
            try:
                async for audio_bytes in audio_generator:
                    if len(audio_bytes) == 0:
                        continue

                    # Decode int16 bytes to float32 waveform
                    audio_np = (
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    self._audio_buffer.append(audio_np)

                    # Emit complete chunks to the stream
                    while len(self._audio_buffer) >= self._chunk_size:
                        chunk = np.concatenate(self._audio_buffer[: self._chunk_size])
                        self._audio_buffer = self._audio_buffer[self._chunk_size :]
                        self._stream.on_next(chunk)

                # Emit remaining audio (padded if necessary)
                if self._audio_buffer:
                    remaining = np.concatenate(self._audio_buffer)
                    if len(remaining) < self._chunk_size:
                        remaining = np.pad(
                            remaining, (0, self._chunk_size - len(remaining))
                        )
                    self._stream.on_next(remaining)

                # Signal stream completion
                self._stream.on_completed()

            except Exception as e:
                logger.error(f"Audio feed error: {e}")
                self._stream.on_error(e)
            finally:
                self._is_running = False
                self._feed_complete.set()

        # Run async generator in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.create_task(process_audio())
            loop.run_until_complete(self._feed_complete.wait())
        finally:
            loop.close()

    def read(self):
        """
        Blocking method called by StreamingInference.

        This method should block until all audio has been read.
        In our case, we wait for the feed to complete.
        """
        # Wait for audio feeding to complete
        self._feed_complete.wait(timeout=120.0)  # Max 2 minutes

    def close(self):
        """Close the audio source."""
        if not self._stream.is_stopped and not self._feed_complete.is_set():
            self._stream.on_completed()


@router.websocket("/ws")
async def websocket_diarization(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speaker diarization using DIART's StreamingInference.

    Client sends base64-encoded audio chunks via WebSocket.
    Server returns speaker segments with timestamps (no transcription).

    Query parameters:
    - latency: Algorithmic latency in seconds (0.5-5.0, default: 0.5)
      - 0.5s: Fastest response, lower accuracy
      - 1.0s: Good balance
      - 2.0s+: Better accuracy, higher delay
    """
    logger.debug(f"Diarization WebSocket connection from {websocket.client}")
    await websocket.accept()
    logger.info("Diarization WebSocket connection accepted")

    if not DIART_AVAILABLE:
        logger.error("DIART not installed on server")
        await websocket.send_json({"error": "DIART not installed", "is_final": True})
        await websocket.close()
        return

    # Get query parameters
    latency = float(websocket.query_params.get("latency", 0.5))
    logger.info(f"Diarization configured with latency={latency}s")

    try:
        # Create DIART pipeline with optimal parameters
        config = SpeakerDiarizationConfig(
            step=0.5,
            latency=latency,
            tau_active=0.555,
            rho_update=0.422,
            delta_new=1.517,
            sample_rate=16000,
        )
        pipeline = SpeakerDiarization(config)
        logger.info("DIART pipeline created successfully")

        # Create audio source
        source = DiarizationWebSocketSource(sample_rate=16000, block_duration=0.5)

        # Create accumulator to collect predictions
        accumulator = PredictionAccumulator(uri="websocket")

        # Start feeding audio in background thread
        feed_thread = threading.Thread(
            target=source.feed_audio,
            args=(audio_stream_generator(websocket),),
            daemon=True,
        )
        feed_thread.start()
        logger.info("Audio feed thread started")

        # Create StreamingInference
        inference = StreamingInference(
            pipeline,
            source,
            batch_size=1,
            do_plot=False,
            show_progress=False,
            do_profile=False,
        )

        # Attach accumulator to collect predictions
        inference.attach_observers(accumulator)
        logger.info("StreamingInference initialized")

        # Run inference in background thread
        def run_inference():
            try:
                logger.info("Starting DIART inference...")
                inference()  # This blocks until source.read() completes
                logger.info("DIART inference completed")
            except Exception as e:
                logger.error(f"Inference error: {e}")
                import traceback

                traceback.print_exc()

        inf_thread = threading.Thread(target=run_inference, daemon=True)
        inf_thread.start()

        # Stream results to WebSocket every 0.5s while inference runs
        last_ts: float = 0.0
        update_count = 0
        last_send_time = 0.0

        while inf_thread.is_alive():
            current_time = time.time()

            # Send updates every 0.5s
            if current_time - last_send_time >= 0.5:
                last_send_time = current_time
                update_count += 1

                try:
                    annotation = accumulator.get_prediction()

                    # Skip if no annotation yet (wait for first prediction)
                    if annotation is None:
                        if update_count < 10:  # Wait up to 5s
                            await asyncio.sleep(0.5)
                            continue
                        else:
                            logger.warning("No predictions received after 5s")
                            break

                    # Extract speaker segments
                    speakers = []
                    segments = []

                    for turn, speaker, _ in annotation.itertracks(yield_label=True):
                        speakers.append(speaker)
                        segments.append(
                            {
                                "speaker": speaker,
                                "start": float(turn.start),
                                "end": float(turn.end),
                                "duration": float(turn.duration),
                            }
                        )

                    # Calculate current timestamp
                    current_ts = float(
                        max((s["end"] for s in segments), default=last_ts)
                    )

                    # Build result
                    result = {
                        "timestamp": current_ts,
                        "speakers": list(set(speakers)),
                        "segments": segments,
                        "active_segments": segments[-5:] if segments else [],
                        "confidence": min(len(segments) / 10.0, 1.0)
                        if segments
                        else 0.0,
                        "is_final": False,
                    }

                    await websocket.send_json(result)
                    last_ts = current_ts

                    if segments:
                        logger.debug(
                            f"Update {update_count}: {len(segments)} segments, "
                            f"{len(set(speakers))} speakers, timestamp={current_ts:.1f}s"
                        )

                except Exception as e:
                    logger.error(f"Get prediction error: {e}")
                    import traceback

                    traceback.print_exc()

            await asyncio.sleep(0.1)

        # Wait for inference to complete
        inf_thread.join(timeout=10.0)
        logger.info("Inference thread completed")

        # Send final result
        try:
            final_annotation = accumulator.get_prediction()

            if final_annotation is not None:
                # Patch to merge nearby same-speaker turns
                final_annotation.patch()

                segments = []
                for turn, speaker, _ in final_annotation.itertracks(yield_label=True):
                    segments.append(
                        {
                            "speaker": speaker,
                            "start": float(turn.start),
                            "end": float(turn.end),
                            "duration": float(turn.duration),
                        }
                    )

                final_result = {
                    "timestamp": float(max((s["end"] for s in segments), default=0.0)),
                    "speakers": list(set(s["speaker"] for s in segments)),
                    "segments": segments,
                    "active_segments": segments,
                    "confidence": min(len(segments) / 10.0, 1.0) if segments else 0.0,
                    "is_final": True,
                }

                await websocket.send_json(final_result)
                logger.info(
                    f"Final result: {len(segments)} segments, {len(set(s['speaker'] for s in segments))} speakers"
                )
            else:
                logger.warning("No final prediction available")
                await websocket.send_json(
                    {
                        "timestamp": 0.0,
                        "speakers": [],
                        "segments": [],
                        "active_segments": [],
                        "confidence": 0.0,
                        "is_final": True,
                    }
                )

        except Exception as e:
            logger.error(f"Final result error: {e}")
            import traceback

            traceback.print_exc()
            await websocket.send_json({"error": str(e), "is_final": True})

        # Cleanup
        source.close()
        logger.info("Diarization session completed")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback

        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e), "is_final": True})
        except RuntimeError:
            logger.error("Could not send error: connection already closed")

    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
