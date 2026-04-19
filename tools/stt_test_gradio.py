"""Gradio test tool for STT server with streaming diarization using DIART."""

import asyncio
import base64
import json
import queue
import time
from typing import AsyncGenerator, Optional, List

import gradio as gr
import numpy as np
import websockets
import sounddevice as sd
from loguru import logger

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION


class AudioRecorder:
    """Records audio from microphone to a list of chunks."""

    def __init__(self):
        self.chunks: List[np.ndarray] = []
        self.is_recording = False
        self.sd_stream = None
        self.sample_rate = SAMPLE_RATE

    def record_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream during recording."""
        if status:
            print(f"Record status: {status}")

        if self.is_recording:
            # Copy the audio data
            audio_chunk = indata[:, 0].copy()
            self.chunks.append(audio_chunk)

    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.chunks = []

        try:
            self.sd_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.record_callback,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
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
        return total_samples / self.sample_rate


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

        if self.is_streaming and indata is not None and len(indata) > 0:
            # indata shape: (frames, channels)
            # Extract mono channel (first column)
            audio_float = indata[:, 0].astype(np.float32)

            # Convert to int16 (scale to -32768 to 32767)
            audio_int16 = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)

            # Convert to bytes
            audio_bytes = audio_int16.tobytes()

            # Put in queue (blocking if queue is full)
            try:
                self.audio_queue.put_nowait(audio_bytes)
            except queue.Full:
                pass  # Drop chunk if queue is full

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
        base64_data = base64.b64encode(audio_bytes).decode("utf-8")
        print(
            f"DEBUG: Sending {len(audio_bytes)} bytes as {len(base64_data)} char base64 string"
        )
        message = {
            "type": "audio",
            "data": base64_data,
            "is_final": False,
        }

        json_message = json.dumps(message)
        print(f"DEBUG: JSON message length: {len(json_message)} chars")

        await self.websocket.send(json_message)

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


async def stream_audio_to_server(
    audio,
    server_url: str,
    model: str,
    enable_diarization: bool,
):
    """Stream recorded audio to STT server."""
    if audio is None or (isinstance(audio, np.ndarray) and len(audio) == 0):
        yield "Error: No audio recorded. Please record some audio first.", ""
        return

    if not server_url:
        yield "Error: Please enter STT server URL", ""
        return

    # Handle different audio formats from gr.Audio
    if isinstance(audio, dict):
        audio = audio.get("array", audio.get("data", None))
    elif not isinstance(audio, np.ndarray):
        yield "Error: Invalid audio format", ""
        return

    if len(audio) == 0:
        yield "Error: No audio recorded. Please record some audio first.", ""
        return

    duration = len(audio) / SAMPLE_RATE
    yield f"Streaming {duration:.1f}s of audio...", ""

    stt_client = STTClient(server_url, model, enable_diarization)

    if not await stt_client.connect():
        yield f"Error: Failed to connect to {server_url}", ""
        return

    full_transcript = ""
    last_speakers = []
    last_confidence = 0.0

    try:
        # Convert audio to chunks and send (1-second chunks aligned to frame size)
        chunk_size = int(SAMPLE_RATE * 1.0)  # 1 second chunks (16000 samples)
        frame_size = 160  # 10ms at 16kHz - model frame size
        overlap_size = int(SAMPLE_RATE * 0.5)  # 50% overlap

        async def audio_generator():
            pos = 0
            while pos < len(audio):
                # Take chunk with overlap if available
                end_pos = min(pos + chunk_size, len(audio))
                chunk = audio[pos:end_pos]

                # Pad to exact chunk size if needed
                if len(chunk) < chunk_size:
                    # Pad to nearest frame boundary
                    padded_size = (
                        (chunk_size + frame_size - 1) // frame_size
                    ) * frame_size
                    chunk = np.pad(chunk, (0, padded_size - len(chunk)))
                else:
                    # Trim to chunk size
                    chunk = chunk[:chunk_size]

                # Ensure chunk is multiple of frame size
                if len(chunk) % frame_size != 0:
                    chunk = np.pad(chunk, (0, frame_size - (len(chunk) % frame_size)))

                # Convert to int16 bytes
                audio_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                yield audio_bytes

                # Move forward with overlap
                pos += (
                    chunk_size - overlap_size
                    if pos + chunk_size < len(audio)
                    else chunk_size
                )

            # Send final marker
            yield b""

        # Send audio chunks
        async for audio_chunk in audio_generator():
            if len(audio_chunk) > 0:
                await stt_client.send_audio(audio_chunk)
            await asyncio.sleep(0.05)  # Small delay between chunks

        # Send final signal
        await stt_client.send_final()

        # Receive results
        async for result in stt_client.receive_results():
            text = result.get("text", "")
            is_final = result.get("is_final", False)

            speakers = result.get("speakers", [])
            confidence = result.get("speaker_confidence", 0.0)

            if speakers and speakers != last_speakers:
                last_speakers = speakers
                last_confidence = confidence

            display_text = text
            if enable_diarization and speakers:
                speaker_info = format_speakers(speakers, confidence)
                display_text = f"{speaker_info} {text}"

            if text:
                full_transcript = text

            status = f"Status: Streaming... | {display_text}"
            yield status, full_transcript

            if is_final:
                break

        yield "Stream completed", full_transcript

    except Exception as e:
        yield f"Error: {str(e)}", full_transcript
    finally:
        await stt_client.disconnect()


async def batch_process_audio(
    audio,
    server_url: str,
    model: str,
    enable_diarization: bool,
):
    """Process recorded audio as a single batch (non-streaming).

    Sends the entire audio recording as one base64-encoded chunk to test
    batch processing vs streaming performance.
    """
    if audio is None or (isinstance(audio, np.ndarray) and len(audio) == 0):
        yield "Error: No audio recorded. Please record some audio first.", ""
        return

    if not server_url:
        yield "Error: Please enter STT server URL", ""
        return

    # Handle different audio formats from gr.Audio
    if isinstance(audio, dict):
        audio = audio.get("array", audio.get("data", None))
    elif not isinstance(audio, np.ndarray):
        yield "Error: Invalid audio format", ""
        return

    if len(audio) == 0:
        yield "Error: No audio recorded. Please record some audio first.", ""
        return

    duration = len(audio) / SAMPLE_RATE
    yield f"Processing {duration:.1f}s of audio as batch...", ""

    stt_client = STTClient(server_url, model, enable_diarization)

    if not await stt_client.connect():
        yield f"Error: Failed to connect to {server_url}", ""
        return

    full_transcript = ""
    last_speakers = []
    last_confidence = 0.0

    try:
        # Convert entire audio to int16 bytes (single chunk)
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        logger.debug(
            f"Batch processing: sending {len(audio_bytes)} bytes as single chunk"
        )

        # Send entire audio as single chunk
        await stt_client.send_audio(audio_bytes)

        # Send final signal
        await stt_client.send_final()

        # Receive results
        async for result in stt_client.receive_results():
            text = result.get("text", "")
            is_final = result.get("is_final", False)

            speakers = result.get("speakers", [])
            confidence = result.get("speaker_confidence", 0.0)

            if speakers and speakers != last_speakers:
                last_speakers = speakers
                last_confidence = confidence

            display_text = text
            if enable_diarization and speakers:
                speaker_info = format_speakers(speakers, confidence)
                display_text = f"{speaker_info} {text}"

            if text:
                full_transcript = text

            status = f"Status: Processing... | {display_text}"
            yield status, full_transcript

            if is_final:
                break

        yield "Batch processing completed", full_transcript

    except Exception as e:
        yield f"Error: {str(e)}", full_transcript
    finally:
        await stt_client.disconnect()


def toggle_recording(is_recording):
    """Toggle recording on/off and return status."""
    if not is_recording:
        # Start recording
        recorder = AudioRecorder()
        try:
            recorder.start_recording()
            # Return new is_recording state, recorder object, and status message
            return True, recorder, "Recording... (click Stop Recording)"
        except Exception as e:
            return False, None, f"Error: {str(e)}"
    return False, None, "Not recording"


def stop_recording_wrapper(is_recording, recorder):
    """Stop recording and return audio data."""
    print(
        f"stop_recording_wrapper called: is_recording={is_recording}, recorder={recorder}"
    )
    if recorder:
        audio = recorder.stop_recording()
        print(
            f"Audio shape: {audio.shape if hasattr(audio, 'shape') else 'no shape'}, len={len(audio) if hasattr(audio, '__len__') else 'no len'}"
        )
        duration = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
        # Gradio Audio expects (sampling_rate, array) tuple or dict
        if len(audio) > 0:
            audio_for_gradio = (SAMPLE_RATE, audio)
            audio_for_state = audio
            print(
                f"Returning audio_for_gradio type: {type(audio_for_gradio)}, audio_for_state type: {type(audio_for_state)}"
            )
        else:
            audio_for_gradio = None
            audio_for_state = None
            print("Audio is empty, returning None")
        # Return new states, status message, audio for gr.Audio component
        return (
            False,  # is_recording
            None,  # recorder
            f"Recorded {duration:.1f}s of audio. Ready to stream.",
            audio_for_gradio,  # For gr.Audio (sampling_rate, array)
            audio_for_state,  # For recorded_audio_state
        )
    print("No recorder, returning defaults")
    return False, None, "Not recording.", None, None


def stop_recording(recording_state):
    """Stop recording and return audio data."""
    if recording_state and recording_state[1]:
        recorder = recording_state[1]
        audio = recorder.stop_recording()
        duration = recorder.get_duration()
        return False, None, f"Recorded {duration:.1f}s of audio. Ready to stream."
    return False, None, "Not recording."


def get_recording_status(recording_state):
    """Get current recording status."""
    if recording_state and recording_state[0]:
        duration = recording_state[1].get_duration() if recording_state[1] else 0
        return f"⏺️ Recording... {duration:.1f}s"
    return "⏹️ Not recording"


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
    stopped = False

    try:
        # Start microphone streaming
        audio_stream.start_streaming()

        # Start sending audio in background
        send_task = asyncio.create_task(_send_audio_loop(audio_stream, stt_client))

        # Receive and yield results continuously
        try:
            async for result in stt_client.receive_results():
                if stopped:
                    break

                text = result.get("text", "")
                is_final = result.get("is_final", False)

                # Get diarization info if available
                speakers = result.get("speakers", [])
                confidence = result.get("speaker_confidence", 0.0)

                # Update state
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

                # Yield immediately to update UI
                status = f"Status: Streaming... | {display_text}"
                yield status, full_transcript

                if is_final:
                    break

        except Exception as e:
            if not stopped:
                yield f"Error receiving: {str(e)}", full_transcript

        # Cancel send task
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass

        if stopped:
            yield "Stream stopped by user", full_transcript
        else:
            yield "Stream completed", full_transcript

    except Exception as e:
        yield f"Error: {str(e)}", full_transcript
    finally:
        # Cleanup
        audio_stream.stop_streaming()
        await stt_client.disconnect()


async def _send_audio_loop(audio_stream: AudioStream, stt_client: STTClient):
    """Background task to send audio chunks."""
    try:
        async for audio_chunk in audio_stream.get_audio_generator():
            await stt_client.send_audio(audio_chunk)
            await asyncio.sleep(0)  # Yield to event loop
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Send loop error: {e}")


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
                    placeholder="http://192.168.31.70:8000",
                    value="http://192.168.31.70:8000",
                    info="Base URL of the STT server",
                )

                model = gr.Dropdown(
                    label="ASR Model",
                    choices=[
                        "v3_ctc",
                        "v3_rnnt",
                        "v2_ctc",
                        "v1_ctc",
                        "v3_e2e_rnnt",
                    ],
                    value="v3_e2e_rnnt",
                    info="Model to use for transcription",
                )

                enable_diarization = gr.Checkbox(
                    label="Enable Speaker Diarization",
                    value=False,
                    info="Requires DIART to be installed on server",
                )

                gr.Markdown("---")
                gr.Markdown("### Mode Selection")

                mode_selector = gr.Radio(
                    label="Processing Mode",
                    choices=[
                        ("Live Streaming (microphone)", "live_stream"),
                        ("Record & Stream", "record_stream"),
                        ("Batch Process", "batch_process"),
                    ],
                    value="record_stream",
                    info="Select how to process audio",
                )

                gr.Markdown("### Live Microphone Streaming")

                start_button = gr.Button("🎤 Start Streaming", variant="primary")

                gr.Markdown("### Record & Stream / Batch Process")

                record_button = gr.Button("⏺️ Start Recording", variant="secondary")
                stop_record_button = gr.Button("⏹️ Stop Recording", variant="stop")
                stream_recorded_button = gr.Button(
                    "📡 Stream Recorded Audio", variant="secondary"
                )
                batch_process_button = gr.Button(
                    "🔵 Batch Process Audio", variant="secondary"
                )

                recording_status = gr.Textbox(
                    label="Recording Status", interactive=False, value="⏹️ Not recording"
                )

                # Audio component for playback (visible after recording)
                audio_output = gr.Audio(
                    label="Recorded Audio (click to play)",
                    type="numpy",
                    interactive=True,
                    visible=True,
                    autoplay=False,
                    show_label=True,
                )

                # Hidden state to store recorded audio
                recorded_audio = gr.State(None)

                gr.Markdown(
                    "<small>⚠️ Requires microphone access and DIART installed on server for diarization</small>"
                )

            with gr.Column(scale=2):
                gr.Markdown("### Output")

                status = gr.Textbox(label="Status", interactive=False, lines=2)

                transcript = gr.Textbox(
                    label="Transcript",
                    interactive=False,
                    lines=8,
                    placeholder="Transcription will appear here...",
                )

        # State for recording management
        is_recording_state = gr.State(False)
        recorder_state = gr.State(None)  # Store the AudioRecorder object
        recorded_audio_state = gr.State(None)  # Store the recorded audio array

        # Live streaming event
        start_button.click(
            fn=stream_transcription,
            inputs=[server_url, model, enable_diarization, gr.State(True)],
            outputs=[status, transcript],
        )

        # Recording events
        record_button.click(
            fn=toggle_recording,
            inputs=[is_recording_state],
            outputs=[is_recording_state, recorder_state, recording_status],
        )

        stop_record_button.click(
            fn=stop_recording_wrapper,
            inputs=[is_recording_state, recorder_state],
            outputs=[
                is_recording_state,
                recorder_state,
                recording_status,
                audio_output,  # 4th output: audio_for_gradio (tuple)
                recorded_audio_state,  # 5th output: audio_for_state (array)
            ],
        )

        stream_recorded_button.click(
            fn=stream_audio_to_server,
            inputs=[recorded_audio_state, server_url, model, enable_diarization],
            outputs=[status, transcript],
        )

        batch_process_button.click(
            fn=batch_process_audio,
            inputs=[recorded_audio_state, server_url, model, enable_diarization],
            outputs=[status, transcript],
        )

        gr.Markdown("---")
        gr.Markdown(
            """
            ## Usage Instructions
            
            ### Mode Selection
            Choose one of three processing modes:
            
            1. **Live Streaming (microphone)**: Real-time transcription directly from microphone
            2. **Record & Stream**: Record audio first, then stream it chunk-by-chunk to the server
            3. **Batch Process**: Record audio first, then send as a single file for batch processing
            
            ### Live Microphone Streaming
            1. Select "Live Streaming (microphone)" mode
            2. Enter Server URL: Provide the base URL of your STT server
            3. Select Model: Choose the ASR model to use
            4. Enable Diarization: Check if server has DIART enabled
            5. Click Start Streaming: Begin real-time microphone transcription
            
            ### Record & Stream
            1. Select "Record & Stream" mode
            2. Click Start Recording: Record audio from microphone
            3. Click Stop Recording: Stop recording
            4. Click Stream Recorded Audio: Send recorded audio to server chunk-by-chunk
            
            ### Batch Process
            1. Select "Batch Process" mode
            2. Click Start Recording: Record audio from microphone
            3. Click Stop Recording: Stop recording
            4. Click Batch Process Audio: Send entire recording as a single file
            
            The tool will:
            - Capture audio from your microphone
            - Process it according to selected mode
            - Display transcription results
            
            ### Requirements
            - STT server must be running with streaming endpoint at `/v1/stream/ws`
            - For diarization: Server must have DIART installed
            - Microphone access required
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
