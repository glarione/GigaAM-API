"""Streaming endpoints for real-time transcription."""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter(prefix="/v1/stream", tags=["streaming"])


async def audio_stream_generator(
    websocket: WebSocket,
) -> AsyncGenerator[bytes, None]:
    """
    Generator that yields audio chunks from WebSocket.

    Yields raw audio bytes (not base64) for processing.
    """
    while True:
        try:
            message = await websocket.receive_text()
            message_json = json.loads(message)
            msg_type = message_json.get("type")
            data_length = len(message_json.get("data", ""))
            logger.debug(
                f"WebSocket message received: type={msg_type}, data_length={data_length} chars"
            )
            data = message_json

            if data.get("type") == "audio":
                import base64

                audio_bytes = base64.b64decode(data["data"])
                logger.debug(f"Audio chunk received: {len(audio_bytes)} bytes")
                yield audio_bytes

                if data.get("is_final"):
                    logger.debug("Final audio chunk received, ending stream")
                    break
            elif data.get("type") == "close":
                logger.debug("Close message received")
                break

        except WebSocketDisconnect:
            logger.warning("WebSocket client disconnected")
            break
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received")
            continue
        except Exception as e:
            logger.error(f"Error in audio_stream_generator: {e}")
            break


@router.websocket("/ws")
async def websocket_streaming(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription.

    Client sends base64-encoded audio chunks.
    Server returns partial and final transcription results.

    Optional diarization: Add ?diarization=true to enable streaming speaker diarization.
    """
    logger.debug(f"WebSocket connection attempt from {websocket.client}")
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    from ....main import get_app

    app = get_app()
    model_manager = app.state.model_manager

    query_params = dict(websocket.query_params)
    logger.debug(f"Query parameters received: {query_params}")
    model_name = query_params.get("model", "v3_ctc")

    # Check if diarization is enabled
    enable_diarization = query_params.get("diarization", "false").lower() == "true"
    diarization_latency = float(query_params.get("diarization_latency", 0.5))

    connection_closed = False

    try:
        from ....services.streaming import StreamingService

        # Initialize diarization service if enabled
        diarization_service = None
        if enable_diarization:
            from ....services.diarization_streaming import StreamingDiarizationService

            diarization_service = StreamingDiarizationService(app.state.settings)
            # Configure latency if specified
            if diarization_latency != 0.5:
                diarization_service.configure(latency=diarization_latency)

        service = StreamingService(
            model_manager, app.state.settings, diarization_service=diarization_service
        )

        async for message in service.stream_transcribe(
            audio_stream_generator(websocket),
            model_name,
            enable_diarization=enable_diarization,
        ):
            try:
                await websocket.send_json(message.model_dump())
            except RuntimeError:
                connection_closed = True
                break

    except Exception as e:
        if not connection_closed:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_json(
                    {"type": "error", "message": str(e), "is_final": True}
                )
            except RuntimeError:
                logger.error("Could not send error message: connection closed")

    finally:
        if not connection_closed:
            try:
                await websocket.close()
            except RuntimeError:
                pass
