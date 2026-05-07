"""Diarization-only streaming endpoint using DIART's StreamingInference."""
import asyncio
import base64
import json
import time

import numpy as np
import rx
from fastapi import APIRouter, WebSocket
from loguru import logger

try:
    from diart import SpeakerDiarization, SpeakerDiarizationConfig
    from diart.inference import StreamingInference
    from diart.sinks import PredictionAccumulator
    from diart.sources import AudioSource

    DIART_AVAILABLE = True
except ImportError:
    DIART_AVAILABLE = False
    logger.warning("DIART not installed")


router = APIRouter(prefix="/v1/diarization", tags=["diarization"])


class DiarizationQueueSource(AudioSource):
    """
    AudioSource that receives audio from a queue and feeds DIART's StreamingInference.

    This class bridges audio chunks from an asyncio.Queue to DIART's RxPY stream system.
    All operations happen in the same event loop as the WebSocket.
    """

    def __init__(self, sample_rate: int = 16000, block_duration: float = 0.5):
        self._stream = rx.subject.Subject()
        self._chunk_size = int(sample_rate * block_duration)
        self._audio_buffer: list[np.ndarray] = []
        self._chunk_queue = asyncio.Queue()
        self._is_running = False
        self._feed_task = None

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
        return None

    async def start_feeding(self):
        """
        Start consuming audio chunks from queue and emit to the stream.

        This runs in the same event loop as the WebSocket.
        """
        self._is_running = True
        chunks_sent = 0

        logger.info("feed_audio: Starting audio processing loop")

        try:
            while self._is_running:
                try:
                    # Get chunk from queue with timeout
                    audio_bytes = await asyncio.wait_for(
                        self._chunk_queue.get(), timeout=2.0
                    )

                    if audio_bytes is None:  # Sentinel for end of stream
                        logger.info("feed_audio: Received end-of-stream signal")
                        break

                    logger.debug(
                        f"feed_audio: Received chunk: {len(audio_bytes)} bytes"
                    )

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
                        chunks_sent += 1
                        logger.info(
                            f"feed_audio: Emitted chunk {chunks_sent}: {len(chunk)} samples ({len(chunk) / 16000:.2f}s)"
                        )

                except asyncio.TimeoutError:
                    # No chunk received, continue checking
                    continue
                except Exception as e:
                    logger.error(f"feed_audio: Error processing chunk: {e}")
                    import traceback

                    traceback.print_exc()
                    break

            logger.info(
                f"feed_audio: Audio processing completed. Total: {chunks_sent} chunks sent"
            )

            # Emit remaining audio (padded if necessary)
            if self._audio_buffer:
                remaining = np.concatenate(self._audio_buffer)
                if len(remaining) < self._chunk_size:
                    remaining = np.pad(
                        remaining, (0, self._chunk_size - len(remaining))
                    )
                self._stream.on_next(remaining)
                chunks_sent += 1
                logger.info(f"feed_audio: Emitted final chunk {chunks_sent}")

            # Signal stream completion
            logger.info(f"feed_audio: Closing audio stream. Total {chunks_sent} chunks")
            self._stream.on_completed()

        except Exception as e:
            logger.error(f"feed_audio: Fatal error: {e}")
            import traceback

            traceback.print_exc()
            self._stream.on_error(e)
        finally:
            self._is_running = False

    async def add_chunk(self, audio_bytes: bytes):
        """Add an audio chunk to the queue."""
        await self._chunk_queue.put(audio_bytes)

    async def stop(self):
        """Stop the audio source."""
        self._is_running = False
        await self._chunk_queue.put(None)  # Sentinel to stop the feed loop

    def read(self):
        """
        Blocking method called by StreamingInference.

        This method should block until all audio has been read.
        We use an asyncio.Event to signal when the stream is complete.
        """
        # Create an event to wait for stream completion
        # This needs to run in the event loop context
        import threading

        # We need to wait for the stream to complete from this blocking call
        # The stream completion is signaled by start_feeding() calling on_completed()
        # We'll use a simple busy-wait with sleep since we're in a different thread context

        # Actually, the issue is that read() is called from StreamingInference in the main thread
        # but the stream processing happens in the event loop
        # We need to block until the stream completes

        # Simple solution: busy wait until stream is stopped
        while not self._stream.is_stopped:
            time.sleep(0.1)

        logger.info("read(): Stream completed, returning")

    def close(self):
        """Close the audio source."""
        # Stop the stream if it's not already stopped
        if not self._stream.is_stopped:
            self._stream.on_completed()


@router.websocket("/ws")
async def websocket_diarization(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speaker diarization using DIART's StreamingInference.

    Client sends base64-encoded audio chunks via WebSocket.
    Server returns speaker segments with timestamps (no transcription).

    Query parameters:
    - latency: Algorithmic latency in seconds (0.5-5.0, default: 0.5)
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

    # Create DIART pipeline
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
    source = DiarizationQueueSource(sample_rate=16000, block_duration=0.5)

    # Create accumulator
    accumulator = PredictionAccumulator(uri="websocket")

    # Start feeding audio in the same event loop
    feed_task = asyncio.create_task(source.start_feeding())
    logger.info("Audio feed task started")

    # Read audio chunks from WebSocket
    async def read_audio_chunks():
        """Read audio chunks from WebSocket and add to source queue."""
        try:
            chunk_count = 0
            while True:
                message = await websocket.receive_text()
                message_json = json.loads(message)
                msg_type = message_json.get("type")

                if msg_type == "audio":
                    chunk_count += 1
                    data = message_json
                    audio_bytes = base64.b64decode(data["data"])

                    logger.debug(
                        f"WebSocket: Received chunk {chunk_count}: {len(audio_bytes)} bytes, is_final={data.get('is_final', False)}"
                    )

                    await source.add_chunk(audio_bytes)

                    if data.get("is_final"):
                        logger.info(
                            f"WebSocket: Received is_final signal. Total chunks: {chunk_count}. Stopping source..."
                        )
                        await source.stop()
                        logger.info("WebSocket: Source stopped, breaking from loop")
                        break
                elif msg_type == "close":
                    break

            logger.info(f"read_audio: Completed. Total {chunk_count} chunks read")

        except Exception as e:
            logger.error(f"read_audio: Error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            logger.info(
                "read_audio: Finally block - stopping source if not already stopped"
            )
            # Stop the source
            await source.stop()
            logger.info("read_audio: Source stop called")

    # Read audio and run inference concurrently
    read_task = asyncio.create_task(read_audio_chunks())
    logger.info("read_audio_chunks task created")

    # Run inference (this will process the stream)
    logger.info("Starting DIART inference...")
    logger.debug(
        f"Pipeline config: step={pipeline.config.step}s, latency={pipeline.config.latency}s"
    )
    logger.debug(f"Source stream state: {source._stream}")

    try:
        # StreamingInference will process the stream until it's completed
        inference = StreamingInference(
            pipeline,
            source,
            batch_size=1,
            do_plot=False,
            show_progress=False,
            do_profile=False,
        )
        logger.info("StreamingInference object created")

        inference.attach_observers(accumulator)
        logger.info("Accumulator attached to inference")

        # Run inference (blocks until stream completes)
        logger.info("Calling inference() - this will block until stream completes...")
        inference()
        logger.info("DIART inference() returned successfully")

    except Exception as e:
        logger.error(f"Inference error: {e}")
        import traceback

        traceback.print_exc()

    # Wait for read task to complete
    logger.info("Waiting for read_audio_chunks task to complete...")
    await read_task
    logger.info("read_audio_chunks task completed")

    # Stream results to WebSocket
    logger.info("Collecting results from accumulator...")
    last_ts: float = 0.0
    update_count = 0
    last_send_time = 0.0

    # Give inference time to process
    logger.info("Waiting 1s for inference to finalize...")
    await asyncio.sleep(1.0)

    # Collect and send results
    try:
        final_annotation = accumulator.get_prediction()
        logger.info(f"Final prediction: {final_annotation}")

        if final_annotation is not None:
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

            logger.info(f"Final result: {len(segments)} segments")

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
                f"Sent final: {len(segments)} segments, {len(set(s['speaker'] for s in segments))} speakers"
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
    await websocket.close()
    logger.info("Diarization session completed")
