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
            data = json.loads(message)

            if data.get("type") == "audio":
                import base64

                audio_bytes = base64.b64decode(data["data"])
                yield audio_bytes

                if data.get("is_final"):
                    break
            elif data.get("type") == "close":
                break

        except WebSocketDisconnect:
            break
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received")
            continue


@router.websocket("/ws")
async def websocket_streaming(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription.

    Client sends base64-encoded audio chunks.
    Server returns partial and final transcription results.

    Optional diarization: Add ?diarization=true to enable streaming speaker diarization.
    """
    await websocket.accept()

    from ....main import get_app

    app = get_app()
    model_manager = app.state.model_manager

    # Get model name from query params or default
    # websocket.query_params is already a QueryParams object, not a string
    query_params = dict(websocket.query_params)
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
                # Connection already closed
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
                # Connection already closed, just log
                logger.error("Could not send error message: connection closed")

    finally:
        if not connection_closed:
            try:
                await websocket.close()
            except RuntimeError:
                # Already closed
                pass
