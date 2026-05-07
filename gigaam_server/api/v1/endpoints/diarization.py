"""Diarization-only streaming endpoint using DIART's StreamingInference."""

import asyncio
import base64
import json
import threading
import time
from typing import Optional

import numpy as np
import rx
from fastapi import APIRouter, WebSocket
from loguru import logger

# Import DIART
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


class WebSocketAudioSource(AudioSource):
    """
    Custom AudioSource that bridges WebSocket audio to DIART's RxPY stream.

    This class allows StreamingInference to receive audio chunks from an async WebSocket.
    """

    def __init__(self, sample_rate: int = 16000):
        super().__init__(uri="websocket", sample_rate=sample_rate)
        self._done_event = threading.Event()
        self._chunk_queue = asyncio.Queue()
        self._push_task: Optional[asyncio.Task] = None

    @property
    def duration(self):
        return None  # Unknown for live streams

    def read(self):
        """
        Blocking method called by StreamingInference.

        This waits until close() is called to signal completion.
        """
        logger.info("read(): Blocking until stream completes...")
        self._done_event.wait()
        logger.info("read(): Stream completed, returning")

    async def push_audio(self, audio_chunk: np.ndarray):
        """
        Push an audio chunk to the RxPY stream.

        This is called from the async WebSocket loop.
        """
        logger.debug(f"push_audio: Pushing chunk of {len(audio_chunk)} samples")
        self.stream.on_next(audio_chunk)

    async def push_from_bytes(self, audio_bytes: bytes):
        """
        Push audio from bytes (int16) to the stream.

        Converts bytes to float32 waveform first.
        """
        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        await self.push_audio(audio_np)

    def close(self):
        """
        Signal stream completion.

        Called when WebSocket sends is_final or disconnects.
        """
        logger.info("close(): Signaling stream completion")
        self.stream.on_completed()
        self._done_event.set()

    async def stop_pushing(self):
        """Stop the push task if running."""
        if self._push_task and not self._push_task.done():
            self._push_task.cancel()
            try:
                await self._push_task
            except asyncio.CancelledError:
                pass


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
        logger.error("DIART not installed")
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
    source = WebSocketAudioSource(sample_rate=16000)

    # Create accumulator to collect predictions
    accumulator = PredictionAccumulator(uri="websocket")

    # Create inference
    inference = StreamingInference(
        pipeline,
        source,
        batch_size=1,
        do_plot=False,
        show_progress=False,
        do_profile=False,
    )
    inference.attach_observers(accumulator)
    logger.info("StreamingInference created and accumulator attached")

    # Track results
    results_queue: asyncio.Queue = asyncio.Queue()
    last_send_time = 0.0
    update_count = 0

    # Hook to send results to WebSocket
    def inference_hook(ann_wav):
        """
        Called by StreamingInference when new annotation is available.

        Runs in background thread, so we queue results for async sending.
        """
        nonlocal update_count
        annotation, waveform = ann_wav

        try:
            # Extract segments
            segments = []
            speakers = set()

            for turn, speaker, _ in annotation.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": speaker,
                        "start": float(turn.start),
                        "end": float(turn.end),
                    }
                )
                speakers.add(speaker)

            update_count += 1

            # Queue result for async sending
            asyncio.run_coroutine_threadsafe(
                results_queue.put(
                    {
                        "timestamp": float(
                            max((s["end"] for s in segments), default=0.0)
                        ),
                        "speakers": list(speakers),
                        "segments": segments,
                        "active_segments": segments[-5:] if segments else [],
                        "confidence": min(len(segments) / 10.0, 1.0)
                        if segments
                        else 0.0,
                        "is_final": False,
                    }
                ),
                loop,
            )

            if segments:
                logger.debug(
                    f"Hook update {update_count}: {len(segments)} segments, "
                    f"{len(speakers)} speakers"
                )

        except Exception as e:
            logger.error(f"Hook error: {e}")
            import traceback

            traceback.print_exc()

    # Attach hook
    inference.attach_hooks(inference_hook)
    logger.info("Inference hook attached")

    # Get the event loop for async operations
    loop = asyncio.get_event_loop()

    # Run inference in background thread
    def run_inference():
        """Run StreamingInference in background thread."""
        try:
            logger.info("Starting DIART inference in background thread...")
            inference()
            logger.info("DIART inference completed")
        except Exception as e:
            logger.error(f"Inference thread error: {e}")
            import traceback

            traceback.print_exc()

    inf_thread = threading.Thread(target=run_inference, daemon=True)
    inf_thread.start()
    logger.info("Inference thread started")

    # Read audio from WebSocket and push to source
    chunk_count = 0
    try:
        while True:
            message = await websocket.receive_text()
            message_json = json.loads(message)
            msg_type = message_json.get("type")

            if msg_type == "audio":
                chunk_count += 1
                data = message_json
                audio_bytes = base64.b64decode(data["data"])

                logger.debug(
                    f"WebSocket: Received chunk {chunk_count}: {len(audio_bytes)} bytes"
                )

                # Push to source
                await source.push_from_bytes(audio_bytes)

                if data.get("is_final"):
                    logger.info(
                        f"WebSocket: Received is_final. Total chunks: {chunk_count}"
                    )
                    source.close()
                    break

            elif msg_type == "close":
                logger.info("WebSocket: Received close message")
                source.close()
                break

        # Wait for inference to complete
        logger.info("Waiting for inference thread to complete...")
        inf_thread.join(timeout=10.0)
        logger.info("Inference thread completed")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback

        traceback.print_exc()
        source.close()

    # Send results from queue
    logger.info("Sending accumulated results...")
    try:
        while not results_queue.empty():
            result = await results_queue.get()
            await websocket.send_json(result)
            logger.debug(f"Sent result: {len(result.get('segments', []))} segments")

    except Exception as e:
        logger.error(f"Error sending results: {e}")

    # Get final prediction
    try:
        final_annotation = accumulator.get_prediction()
        logger.info(f"Final prediction: {final_annotation}")

        if final_annotation is not None:
            final_annotation.patch()

            segments = []
            speakers = set()

            for turn, speaker, _ in final_annotation.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": speaker,
                        "start": float(turn.start),
                        "end": float(turn.end),
                    }
                )
                speakers.add(speaker)

            logger.info(f"Final: {len(segments)} segments, {len(speakers)} speakers")

            final_result = {
                "timestamp": float(max((s["end"] for s in segments), default=0.0)),
                "speakers": list(speakers),
                "segments": segments,
                "active_segments": segments,
                "confidence": min(len(segments) / 10.0, 1.0) if segments else 0.0,
                "is_final": True,
            }

            await websocket.send_json(final_result)
            logger.info("Sent final result")

    except Exception as e:
        logger.error(f"Final result error: {e}")
        import traceback

        traceback.print_exc()
        await websocket.send_json({"error": str(e), "is_final": True})

    await websocket.close()
    logger.info("Diarization session completed")
