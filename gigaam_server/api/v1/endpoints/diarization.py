"""Diarization-only streaming endpoint."""

import base64
import json
from typing import AsyncGenerator

from fastapi import APIRouter, WebSocket
from loguru import logger

from gigaam_server.main import get_app
from gigaam_server.api.v1.endpoints.streaming import audio_stream_generator

router = APIRouter(prefix="/v1/diarization", tags=["diarization"])


@router.websocket("/ws")
async def websocket_diarization(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speaker diarization only.

    Client sends base64-encoded audio chunks.
    Server returns speaker segments with timestamps (no transcription).

    Query parameters:
    - latency: Algorithmic latency in seconds (0.5-5.0, default: 0.5)
      - 0.5s: Fastest, lowest accuracy
      - 1.0s: Good balance
      - 2.0s+: Better accuracy, higher delay
    """
    logger.debug(f"Diarization WebSocket connection attempt from {websocket.client}")
    await websocket.accept()
    logger.info("Diarization WebSocket connection accepted")

    app = get_app()
    diarization_service = app.state.streaming_diarization_service

    # Get query parameters
    latency = float(websocket.query_params.get("latency", 0.5))
    logger.debug(f"Query parameters received: {{'latency': {latency}}}")

    # Configure service if needed
    if latency != 0.5:
        diarization_service.configure(latency=latency)
        logger.info(f"Diarization latency configured to {latency}s")

    connection_closed = False

    try:
        # Process stream with diarization only
        async for result in diarization_service.stream_diarize(
            audio_stream_generator(websocket)
        ):
            try:
                logger.debug(f"{result=}")
                await websocket.send_json(result)
            except RuntimeError:
                connection_closed = True
                break

    except Exception as e:
        if not connection_closed:
            logger.error(f"Diarization WebSocket error: {e}")
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
