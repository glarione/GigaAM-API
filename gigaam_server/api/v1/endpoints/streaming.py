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
    """
    await websocket.accept()

    from ....main import get_app

    app = get_app()
    model_manager = app.state.model_manager

    # Get model name from query params or default
    import urllib.parse

    query_params = dict(urllib.parse.parse_qsl(websocket.query_params.decode()))
    model_name = query_params.get("model", "v3_ctc")

    try:
        from ....services.streaming import StreamingService

        service = StreamingService(model_manager, app.settings)

        async for message in service.stream_transcribe(
            audio_stream_generator(websocket),
            model_name,
        ):
            await websocket.send_json(message.model_dump())

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json(
            {"type": "error", "message": str(e), "is_final": True}
        )

    finally:
        await websocket.close()
