"""Transcription endpoints - OpenAI-compatible API."""

import os
import tempfile
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ....schemas.transcription import (
    TranscriptionResponse,
    TranscriptionSegment,
    TranscriptionVerboseResponse,
)
from ....services.transcription import TranscriptionService

router = APIRouter(prefix="/v1/audio", tags=["transcriptions"])


async def get_transcription_service() -> TranscriptionService:
    """Get transcription service from app state."""
    from ....main import get_app

    app = get_app()
    return TranscriptionService(app.state.model_manager, app.state.diarization_service)


@router.post("/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("v3_e2e_rnnt"),
    language: str = Form("ru"),
    response_format: Literal["json", "verbose_json", "text", "srt", "vtt"] = Form(
        "json"
    ),
    stream: bool = Form(False),
    vad_filter: bool = Form(True),
    diarization: bool = Form(False),
    service: TranscriptionService = Depends(get_transcription_service),
):
    """
    Transcribe audio to text.

    OpenAI-compatible endpoint for speech-to-text.

    Optimized for CPU:
    - Short-form audio (<25s): Processes directly from bytes (no temp file)
    - Long-form audio: Uses temp file only for pyannote VAD compatibility
    """
    # Validate model
    from ....config import get_settings

    settings = get_settings()
    if model not in settings.available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: {settings.available_models}",
        )

    # Read file
    try:
        audio_data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # Validate audio
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Transcribe with diarization if requested
    # When diarization is enabled, audio is split by speaker boundaries
    # and each speaker's segment is transcribed separately
    # Diarization requires file-based processing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        temp.write(audio_data)
        temp_path = temp.name

        try:
            if diarization:
                result = await service.transcribe_with_diarization(
                    audio_path=temp_path,
                    model_name=model,
                )

            else:
                # Optimized path: process bytes directly (no temp file for short-form)
                result = await service.transcribe_from_file(
                    audio_path=temp_path,
                    model_name=model,
                    vad_filter=vad_filter,
                    diarization=False,
                )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    # Format response
    if response_format == "text":
        return JSONResponse(content=result.text)

    if response_format == "srt" or response_format == "vtt":
        # Simple SRT/VTT format
        return JSONResponse(
            content=_segments_to_srt(result.segments, vtt=response_format == "vtt")
        )

    segments_data = [
        {
            "id": s.id,
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "tokens": s.tokens,
            **({"speaker": s.speaker} if s.speaker else {}),
        }
        for s in result.segments
    ]

    if response_format == "verbose_json":
        return JSONResponse(
            content=TranscriptionVerboseResponse(
                text=result.text,
                segments=segments_data,
                duration=result.duration,
                model=result.model,
                language=result.language,
            ).model_dump()
        )

    return JSONResponse(
        content=TranscriptionResponse(
            text=result.text,
            segments=segments_data,
            duration=result.duration,
            model=result.model,
            language=result.language,
        ).model_dump()
    )


def _segments_to_srt(segments: list[TranscriptionSegment], vtt: bool = False) -> str:
    """Convert segments to SRT or VTT format."""
    lines = []
    if vtt:
        lines.append("WEBVTT\n")

    for i, seg in enumerate(segments):
        start_str = _format_srt_time(seg.start)
        end_str = _format_srt_time(seg.end)

        if vtt:
            speaker = f"{seg.speaker} " if seg.speaker else ""
            lines.append(f"{i + 1}\n{start_str} --> {end_str}\n{speaker}{seg.text}\n")
        else:
            lines.append(f"{i + 1}\n{start_str} --> {end_str}\n{seg.text}\n")

    return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)

    if ms == 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
