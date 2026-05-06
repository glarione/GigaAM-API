"""Gradio test tool for diarization-only streaming."""

import asyncio
import base64
import json
import time
from typing import AsyncGenerator

import gradio as gr
import numpy as np
import sounddevice as sd
import websockets
from loguru import logger

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1  # 100ms chunks
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION


class AudioRecorder:
    """Records audio from microphone."""

    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.is_recording = False
        self.sd_stream = None

    def record_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"Record status: {status}")

        if self.is_recording:
            audio_chunk = indata[:, 0].copy()
            self.chunks.append(audio_chunk)

    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.chunks = []

        try:
            self.sd_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self.record_callback,
                blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
                dtype=np.float32,
            )
            self.sd_stream.start()
            print("Recording started")
        except Exception as e:
            print(f"Failed to start recording: {e}")
            raise

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return concatenated audio."""
        self.is_recording = False
        if self.sd_stream:
            self.sd_stream.stop()
            self.sd_stream.close()
            self.sd_stream = None

        print(f"Recording stopped. Total chunks: {len(self.chunks)}")

        if not self.chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(self.chunks)

    def get_duration(self) -> float:
        """Get recording duration in seconds."""
        total_samples = sum(len(chunk) for chunk in self.chunks)
        return total_samples / SAMPLE_RATE


async def diarization_stream(
    server_url: str, latency: float
) -> AsyncGenerator[dict, None]:
    """Connect to diarization WebSocket and stream audio."""
    # Convert HTTP URL to WebSocket URL
    ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/v1/diarization/ws?latency={latency}"

    print(f"Connecting to: {ws_url}")

    try:
        async with websockets.connect(ws_url) as websocket:
            print("Connected to diarization server")

            # Start recording
            recorder = AudioRecorder()
            recorder.start_recording()

            try:
                # Send audio chunks continuously
                start_time = time.time()
                chunk_count = 0

                while recorder.is_recording:
                    await asyncio.sleep(CHUNK_DURATION)

                    # Send current chunk
                    if recorder.chunks:
                        audio_chunk = recorder.chunks.pop(0)

                        # Convert to int16 bytes
                        audio_int16 = np.clip(
                            audio_chunk * 32767, -32768, 32767
                        ).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()

                        # Send as base64
                        base64_data = base64.b64encode(audio_bytes).decode("utf-8")
                        message = {
                            "type": "audio",
                            "data": base64_data,
                            "is_final": False,
                        }

                        await websocket.send(json.dumps(message))
                        chunk_count += 1

                    # Receive diarization result
                    try:
                        message_str = await asyncio.wait_for(
                            websocket.recv(), timeout=0.1
                        )
                        result = json.loads(message_str)

                        # Yield result
                        yield result

                    except asyncio.TimeoutError:
                        # No new result yet, continue
                        pass

                # Send end of stream
                await websocket.send(
                    json.dumps({"type": "audio", "data": "", "is_final": True})
                )

                # Wait for final result
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    result = json.loads(message_str)
                    yield result

                except asyncio.TimeoutError:
                    pass

                # Send end of stream
                await websocket.send(
                    json.dumps({"type": "audio", "data": "", "is_final": True})
                )

                # Wait for final result
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    result = json.loads(message_str)
                    yield result

                except asyncio.TimeoutError:
                    pass

            finally:
                # Stop recording
                recorder.stop_recording()
                duration = time.time() - start_time
                print(f"Session completed: {duration:.1f}s, {chunk_count} chunks sent")

    except Exception as e:
        print(f"Error: {e}")
        yield {"error": str(e), "is_final": True}


def format_segments(segments: list) -> str:
    """Format segments for display."""
    if not segments:
        return "No segments yet"

    lines = []
    for seg in segments[-10:]:  # Show last 10 segments
        lines.append(
            f"{seg.get('speaker', 'unknown')}: [{seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s]"
        )

    return "\n".join(lines)


def format_active(speakers: list, active_segments: list) -> str:
    """Format active speakers for display."""
    if not speakers:
        return "No active speakers"

    lines = [f"Active speakers: {', '.join(speakers)}"]

    if active_segments:
        lines.append("\nCurrently speaking:")
        for seg in active_segments:
            lines.append(
                f"  {seg.get('speaker', 'unknown')}: {seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s"
            )

    return "\n".join(lines)


async def process_diarization(server_url: str, latency: float, duration: float):
    """Process diarization for specified duration."""
    # Start recording in background
    recorder = AudioRecorder()
    recorder.start_recording()

    # Create async generator
    async def audio_stream():
        try:
            async for result in diarization_stream(server_url, latency):
                yield result
        finally:
            recorder.stop_recording()

    # Stream results
    last_timestamp = 0.0
    all_segments = []

    try:
        async for result in audio_stream():
            if "error" in result:
                yield f"Error: {result['error']}", "", ""
                break

            timestamp = result.get("timestamp", 0.0)
            speakers = result.get("speakers", [])
            segments = result.get("segments", [])
            active_segments = result.get("active_segments", [])
            confidence = result.get("confidence", 0.0)
            is_final = result.get("is_final", False)

            if timestamp > last_timestamp:
                last_timestamp = timestamp
                all_segments = segments

            status = f"Timestamp: {timestamp:.1f}s | Speakers: {len(speakers)} | Confidence: {confidence:.2f}"
            if is_final:
                status += " [FINAL]"

            segments_text = format_segments(all_segments)
            active_text = format_active(speakers, active_segments)

            yield status, segments_text, active_text

            if is_final:
                break
    except Exception as e:
        yield f"Error: {str(e)}", "", ""
    finally:
        recorder.stop_recording()


def create_gradio_interface():
    """Create and return Gradio interface."""

    with gr.Blocks(title="Diarization Only Test", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ Diarization Only Test Tool")
        gr.Markdown(
            "Real-time speaker diarization without transcription. See who's speaking and when!"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                server_url = gr.Textbox(
                    label="Server URL",
                    placeholder="http://localhost:8000",
                    value="http://localhost:8000",
                    info="Base URL of the STT server",
                )

                latency = gr.Slider(
                    label="Latency",
                    minimum=0.5,
                    maximum=5.0,
                    step=0.5,
                    value=0.5,
                    info="Algorithmic latency (lower = faster, higher = more accurate)",
                )

                duration = gr.Slider(
                    label="Recording Duration",
                    minimum=5.0,
                    maximum=60.0,
                    step=5.0,
                    value=30.0,
                    info="How long to record (seconds)",
                )

                gr.Markdown("---")

                start_button = gr.Button("🎤 Start Diarization", variant="primary")

                gr.Markdown(
                    "<small>⚠️ Requires microphone access and DIART installed on server</small>"
                )

            with gr.Column(scale=2):
                gr.Markdown("### Output")

                status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Row():
                    segments_output = gr.Textbox(
                        label="All Segments",
                        interactive=False,
                        lines=10,
                        placeholder="Segments will appear here...",
                    )

                    active_output = gr.Textbox(
                        label="Active Speakers",
                        interactive=False,
                        lines=10,
                        placeholder="Active speakers will appear here...",
                    )

        # Event handler
        start_button.click(
            fn=process_diarization,
            inputs=[server_url, latency, duration],
            outputs=[status, segments_output, active_output],
        )

        gr.Markdown("---")
        gr.Markdown(
            """
            ## Usage Instructions
            
            1. **Configure**: Set server URL and latency (0.5s for fastest, 2.0s+ for better accuracy)
            2. **Start**: Click "Start Diarization" to begin recording
            3. **Speak**: Have speakers talk naturally
            4. **Observe**: Watch segments appear in real-time
            
            ## Output
            
            - **All Segments**: Complete list of speaker segments with timestamps
            - **Active Speakers**: Currently speaking speakers
            
            ## Requirements
            
            - STT server must be running with diarization endpoint at `/v1/diarization/ws`
            - Server must have DIART installed
            - Microphone access required
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861)
