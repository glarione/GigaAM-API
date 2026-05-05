"""Simplified Gradio test tool for STT server with streaming diarization."""

import asyncio
import base64
import json
import queue
from typing import AsyncGenerator

import gradio as gr
import numpy as np
import sounddevice as sd
import websockets
from loguru import logger

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 100ms chunks at 16kHz


class MicrophoneStream:
    """Simple microphone streaming using sounddevice."""

    def __init__(self):
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_streaming = False

    def start(self):
        """Start microphone streaming."""
        self.is_streaming = True
        self.audio_queue = queue.Queue()

        def callback(indata, frames, time, status):
            if status:
                logger.debug(f"Audio status: {status}")
            if self.is_streaming:
                audio_float = indata[:, 0].astype(np.float32)
                audio_int16 = np.clip(audio_float * 32767, -32768, 32767).astype(
                    np.int16
                )
                try:
                    self.audio_queue.put_nowait(audio_int16.tobytes())
                except queue.Full:
                    pass  # Drop if full

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=callback,
            blocksize=CHUNK_SIZE,
            dtype=np.float32,
        )
        self.stream.start()
        logger.info("Microphone streaming started")

    def stop(self):
        """Stop microphone streaming."""
        self.is_streaming = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("Microphone streaming stopped")

    async def audio_generator(self):
        """Generate audio chunks from queue."""
        while self.is_streaming:
            try:
                audio_bytes = self.audio_queue.get(timeout=0.1)
                yield audio_bytes
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio generator error: {e}")
                break


class STTClient:
    """Client for streaming STT server."""

    def __init__(
        self,
        server_url: str,
        model: str = "v3_e2e_rnnt",
        enable_diarization: bool = False,
    ):
        self.server_url = server_url
        self.model = model
        self.enable_diarization = enable_diarization
        self.websocket = None

    async def connect(self):
        """Connect to STT server WebSocket."""
        ws_url = self.server_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        ws_url = f"{ws_url}/v1/stream/ws?model={self.model}&diarization={'true' if self.enable_diarization else 'false'}"

        try:
            self.websocket = await websockets.connect(ws_url)
            logger.info(f"Connected to STT server: {ws_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self):
        """Disconnect from STT server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def send_audio(self, audio_bytes: bytes):
        """Send audio chunk to server."""
        if not self.websocket:
            return

        base64_data = base64.b64encode(audio_bytes).decode("utf-8")
        message = {"type": "audio", "data": base64_data, "is_final": False}
        await self.websocket.send(json.dumps(message))

    async def send_final(self):
        """Send end-of-stream signal."""
        if not self.websocket:
            return

        message = {"type": "audio", "data": "", "is_final": True}
        await self.websocket.send(json.dumps(message))

    async def receive_results(self):
        """Receive transcription results from server."""
        if not self.websocket:
            return

        try:
            async for message in self.websocket:
                data = json.loads(message)
                yield data
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error receiving results: {e}")


def format_speakers(speakers: list, confidence: float) -> str:
    """Format speaker information for display."""
    if not speakers:
        return ""
    speaker_str = ", ".join(speakers)
    return f"[{speaker_str} ({confidence:.2f})]"


async def stream_transcription(server_url: str, model: str, enable_diarization: bool):
    """Stream transcription from microphone to STT server."""
    if not server_url:
        yield "Error: Please enter STT server URL", ""
        return

    mic_stream = MicrophoneStream()
    stt_client = STTClient(server_url, model, enable_diarization)

    if not await stt_client.connect():
        yield f"Error: Failed to connect to {server_url}", ""
        return

    full_transcript = ""
    stopped = False

    try:
        # Start microphone
        mic_stream.start()

        # Start sending audio in background
        async def send_audio_loop():
            try:
                async for audio_chunk in mic_stream.audio_generator():
                    if stopped:
                        break
                    await stt_client.send_audio(audio_chunk)
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Send loop error: {e}")

        send_task = asyncio.create_task(send_audio_loop())

        # Receive and display results
        try:
            async for result in stt_client.receive_results():
                if stopped:
                    break

                text = result.get("text", "")
                is_final = result.get("is_final", False)
                speakers = result.get("speakers", [])
                confidence = result.get("speaker_confidence", 0.0)

                # Build display text with speaker labels
                display_text = text
                if enable_diarization and speakers:
                    speaker_info = format_speakers(speakers, confidence)
                    display_text = f"{speaker_info}: {text}"

                if text:
                    if full_transcript and text.startswith(full_transcript):
                        full_transcript = text
                    else:
                        full_transcript = text

                status = f"Streaming... | {display_text}"
                yield status, full_transcript

                if is_final:
                    break

        except Exception as e:
            if not stopped:
                yield f"Error: {str(e)}", full_transcript

        # Cleanup
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass

        yield "Stream completed", full_transcript

    except Exception as e:
        yield f"Error: {str(e)}", ""
    finally:
        mic_stream.stop()
        await stt_client.disconnect()


def create_gradio_interface():
    """Create simplified Gradio interface."""

    with gr.Blocks(title="STT Streaming Test", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ STT Streaming Test (Simplified)")
        gr.Markdown("Real-time speech recognition with speaker diarization")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                server_url = gr.Textbox(
                    label="STT Server URL",
                    placeholder="http://localhost:8000",
                    value="http://localhost:8000",
                )

                model = gr.Dropdown(
                    label="Model",
                    choices=["v3_e2e_rnnt", "v3_ctc", "v3_rnnt", "v2_ctc"],
                    value="v3_e2e_rnnt",
                )

                enable_diarization = gr.Checkbox(
                    label="Enable Diarization",
                    value=False,
                    info="Requires DIART installed on server",
                )

                start_button = gr.Button("🎤 Start Streaming", variant="primary")
                stop_button = gr.Button("⏹️ Stop", variant="stop", interactive=False)

            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", interactive=False, lines=2)
                transcript = gr.Textbox(label="Transcript", interactive=False, lines=10)

        # Event handlers
        start_click = start_button.click(
            fn=stream_transcription,
            inputs=[server_url, model, enable_diarization],
            outputs=[status, transcript],
        )

        # Disable start, enable stop when streaming
        start_click.then(
            lambda: (False, True),
            outputs=[start_button, stop_button],
        )

        stop_button.click(
            fn=lambda: ("Stream stopped", ""),
            outputs=[status, transcript],
        ).then(
            lambda: (True, False),
            outputs=[start_button, stop_button],
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
