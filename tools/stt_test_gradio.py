"""Gradio test tool for STT server with streaming diarization using DIART."""

import asyncio
import base64
import json
import queue
from typing import AsyncGenerator, Optional

import gradio as gr
import numpy as np
import websockets
import sounddevice as sd

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION


class AudioStream:
    """Manages microphone audio streaming."""

    def __init__(self):
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_streaming = False
        self.sd_stream = None

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"Audio status: {status}")

        if self.is_streaming:
            # Convert to float32 and normalize
            audio_chunk = indata[:, 0].astype(np.float32) / 32768.0

            # Resample if needed (assuming 44.1kHz input)
            if len(audio_chunk) != CHUNK_SIZE:
                # Simple resampling by truncating/padding
                if len(audio_chunk) > CHUNK_SIZE:
                    audio_chunk = audio_chunk[:CHUNK_SIZE]
                else:
                    audio_chunk = np.pad(
                        audio_chunk, (0, CHUNK_SIZE - len(audio_chunk))
                    )

            # Convert to int16 bytes
            audio_bytes = (audio_chunk * 32768).astype(np.int16).tobytes()
            self.audio_queue.put(audio_bytes)

    def start_streaming(self):
        """Start microphone streaming."""
        self.is_streaming = True
        self.audio_queue = queue.Queue()

        try:
            self.sd_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
                dtype=np.int16,
            )
            self.sd_stream.start()
            print("Microphone streaming started")
        except Exception as e:
            print(f"Failed to start microphone: {e}")
            raise

    def stop_streaming(self):
        """Stop microphone streaming."""
        self.is_streaming = False
        if self.sd_stream:
            self.sd_stream.stop()
            self.sd_stream.close()
            self.sd_stream = None
        print("Microphone streaming stopped")

    def get_audio_generator(self):
        """Generate audio chunks from queue."""

        async def generator():
            while self.is_streaming:
                try:
                    audio_bytes = self.audio_queue.get(timeout=0.1)
                    yield audio_bytes
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Audio generator error: {e}")
                    break

        return generator()


class STTClient:
    """Client for streaming STT server."""

    def __init__(
        self, server_url: str, model: str = "v3_ctc", enable_diarization: bool = False
    ):
        self.server_url = server_url
        self.model = model
        self.enable_diarization = enable_diarization
        self.websocket = None
        self.is_connected = False

    async def connect(self):
        """Connect to STT server WebSocket."""
        # Convert HTTP URL to WebSocket URL
        ws_url = self.server_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        ws_url = f"{ws_url}/v1/stream/ws"

        # Add query parameters
        params = f"?model={self.model}&diarization={'true' if self.enable_diarization else 'false'}"
        ws_url += params

        try:
            self.websocket = await websockets.connect(ws_url)
            self.is_connected = True
            print(f"Connected to STT server: {ws_url}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from STT server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.is_connected = False
            print("Disconnected from STT server")

    async def send_audio(self, audio_bytes: bytes):
        """Send audio chunk to server."""
        if not self.websocket or not self.is_connected:
            return

        # Create message
        message = {
            "type": "audio",
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "is_final": False,
        }

        await self.websocket.send(json.dumps(message))

    async def send_final(self):
        """Send end-of-stream signal."""
        if not self.websocket or not self.is_connected:
            return

        message = {"type": "audio", "data": "", "is_final": True}

        await self.websocket.send(json.dumps(message))

    async def receive_results(self):
        """Receive transcription results from server."""
        if not self.websocket or not self.is_connected:
            return

        try:
            async for message in self.websocket:
                data = json.loads(message)
                yield data
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"Error receiving results: {e}")


def format_speakers(speakers: list, confidence: float) -> str:
    """Format speaker information for display."""
    if not speakers:
        return ""

    speaker_str = ", ".join(speakers)
    return f"[{speaker_str} (confidence: {confidence:.2f})]"


async def stream_transcription(
    server_url: str, model: str, enable_diarization: bool, start_streaming: bool
):
    """Main streaming transcription function."""

    if not start_streaming:
        yield "Click 'Start' to begin microphone streaming...", None
        return

    if not server_url:
        yield "Error: Please enter STT server URL", None
        return

    audio_stream = AudioStream()
    stt_client = STTClient(server_url, model, enable_diarization)

    if not await stt_client.connect():
        yield f"Error: Failed to connect to {server_url}", None
        return

    full_transcript: str = ""
    last_speakers: list = []
    last_confidence = 0.0

    try:
        # Start microphone streaming
        audio_stream.start_streaming()

        # Create tasks for sending and receiving
        async def send_task():
            async for audio_chunk in audio_stream.get_audio_generator():
                await stt_client.send_audio(audio_chunk)
                await asyncio.sleep(0)  # Yield to event loop

        async def receive_task():
            nonlocal full_transcript, last_speakers, last_confidence

            async for result in stt_client.receive_results():
                text = result.get("text", "")
                is_final = result.get("is_final", False)

                # Get diarization info if available
                speakers = result.get("speakers", [])
                confidence = result.get("speaker_confidence", 0.0)

                # Update display
                if speakers and speakers != last_speakers:
                    last_speakers = speakers
                    last_confidence = confidence

                # Build display text
                display_text = text

                if enable_diarization and speakers:
                    speaker_info = format_speakers(speakers, confidence)
                    display_text = f"{speaker_info} {text}"

                # Accumulate full transcript
                if text:
                    full_transcript = text

                # Yield current state
                status = f"Status: Streaming... | {display_text}"
                yield status, full_transcript

                if is_final:
                    break

        # Run both tasks concurrently
        send_future = asyncio.create_task(send_task())
        receive_future = asyncio.create_task(receive_task())

        # Wait for receive to complete or error
        try:
            await receive_future
        except Exception as e:
            yield f"Error: {str(e)}", full_transcript

        # Cancel send task
        send_future.cancel()
        try:
            await send_future
        except asyncio.CancelledError:
            pass

        yield "Stream completed", full_transcript

    except Exception as e:
        yield f"Error: {str(e)}", full_transcript
    finally:
        # Cleanup
        audio_stream.stop_streaming()
        await stt_client.disconnect()


def create_gradio_interface():
    """Create and return Gradio interface."""

    with gr.Blocks(title="STT Streaming Test Tool", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ STT Streaming Test Tool")
        gr.Markdown("Test streaming transcription with speaker diarization")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                server_url = gr.Textbox(
                    label="STT Server URL",
                    placeholder="http://localhost:8000",
                    value="http://localhost:8000",
                    info="Base URL of the STT server",
                )

                model = gr.Dropdown(
                    label="ASR Model",
                    choices=["v3_ctc", "v3_rnnt", "v2_ctc", "v1_ctc", "v3_e2e_rnnt"],
                    value="v3_ctc",
                    info="Model to use for transcription",
                )

                enable_diarization = gr.Checkbox(
                    label="Enable Speaker Diarization",
                    value=False,
                    info="Requires DIART to be installed on server",
                )

                start_button = gr.Button("🎤 Start Streaming", variant="primary")
                gr.Markdown(
                    "<small>⚠️ Requires microphone access and DIART installed on server for diarization</small>"
                )

            with gr.Column(scale=2):
                gr.Markdown("### Live Output")

                status = gr.Textbox(label="Status", interactive=False, lines=2)

                transcript = gr.Textbox(
                    label="Transcript",
                    interactive=False,
                    lines=8,
                    placeholder="Transcription will appear here...",
                )

        # Event handler
        start_button.click(
            fn=stream_transcription,
            inputs=[server_url, model, enable_diarization, gr.State(True)],
            outputs=[status, transcript],
        )

        gr.Markdown("---")
        gr.Markdown(
            """
            ## Usage Instructions
            
            1. **Enter Server URL**: Provide the base URL of your STT server (e.g., `http://localhost:8000`)
            2. **Select Model**: Choose the ASR model to use
            3. **Enable Diarization**: Check this if your server has DIART streaming diarization enabled
            4. **Click Start**: Begin microphone streaming
            
            The tool will:
            - Capture audio from your microphone
            - Stream it to the STT server via WebSocket
            - Display real-time transcription with speaker labels (if enabled)
            
            ### Requirements
            - STT server must be running with streaming endpoint at `/v1/stream/ws`
            - For diarization: Server must have DIART installed (`pip install gigaam[streaming-diarization]`)
            - Microphone access required
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
