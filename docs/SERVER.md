# GigaAM Server Documentation

FastAPI-based server providing OpenAI-compatible STT API for GigaAM acoustic models.

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install -e ".[server]"

# Or with pip
pip install -e ".[server]"
```

### Running the Server

```bash
# Using uvicorn directly
uvicorn gigaam_server.main:app --reload --host 0.0.0.0 --port 8000

# With environment variables
GIGAAM_DEFAULT_MODEL=v3_ctc GIGAAM_DEVICE=cuda uvicorn gigaam_server.main:app --reload
```

### Using Docker

```bash
# Build and run with GPU support
docker-compose up --build

# CPU-only
docker build -f Dockerfile.cpu -t gigaam-cpu .
docker run -p 8000:8000 gigaam-cpu
```

## API Endpoints

### Transcription (OpenAI-compatible)

**POST `/v1/audio/transcriptions`**

Transcribe audio to text.

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=v3_e2e_rnnt" \
  -F "response_format=json"
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| file | File | Required | Audio file (wav, mp3, flac, m4a) |
| model | str | v3_e2e_rnnt | Model to use |
| language | str | ru | Language code |
| response_format | str | json | json, verbose_json, text, srt, vtt |
| vad_filter | bool | true | Use pyannote VAD for long audio |
| diarization | bool | false | Enable speaker diarization |

**Response:**
```json
{
  "text": "Распознанный текст",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.5,
      "text": "Распознанный текст",
      "tokens": []
    }
  ],
  "duration": 10.5,
  "model": "v3_e2e_rnnt",
  "language": "ru"
}
```

### Streaming (WebSocket)

**WebSocket `/v1/stream/ws`**

Real-time transcription via WebSocket.

```python
import asyncio
import websockets
import base64
import wave
import io
import numpy as np

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/v1/stream/ws?model=v3_ctc") as ws:
        # Send audio chunks (1 second each)
        for chunk in audio_chunks:
            b64 = base64.b64encode(chunk).decode()
            await ws.send({"type": "audio", "data": b64, "is_final": False})

        # Receive results
        async for message in ws:
            print(message)

        # Signal end
        await ws.send({"type": "close"})
```

### Models

**GET `/v1/models`**

List available models.

```bash
curl http://localhost:8000/v1/models
```

**GET `/v1/models/{model_id}`**

Get model details.

### Health

**GET `/health`** - Health check

**GET `/ready`** - Readiness probe

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| GIGAAM_HOST | 0.0.0.0 | Server host |
| GIGAAM_PORT | 8000 | Server port |
| GIGAAM_DEFAULT_MODEL | v3_e2e_rnnt | Default model |
| GIGAAM_DEVICE | cuda | Device (cuda/cpu) |
| GIGAAM_FP16_ENCODER | true | Use FP16 for encoder |
| GIGAAM_HF_TOKEN | - | Hugging Face token for pyannote |

### pyproject.toml

```toml
[project.optional-dependencies]
server = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "python-multipart>=0.0.6",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "websockets>=12.0",
    "aiofiles>=23.2.1",
    "prometheus-fastapi-instrumentator>=0.7.0",
]
longform = [
    "pyannote.audio==4.0",
    "torchcodec==0.7",
]
benchmarks = [
    "pytest-benchmark>=4.0.0",
]
```

## Available Models

| Model | Description |
|-------|-------------|
| v3_ctc | CTC-based ASR (fast) |
| v3_rnnt | RNNT-based ASR (balanced) |
| v3_e2e_ctc | End-to-end with punctuation |
| v3_e2e_rnnt | End-to-end best quality |

## Benchmarks

```bash
# Install benchmark dependencies
uv pip install -e ".[benchmarks,tests]"

# Run benchmarks
pytest gigaam_server/benchmarks/test_benchmark.py -v --benchmark-only

# Metrics collected:
# - tokens_per_second: Tokens processed per second
# - realtime_factor: audio_duration / processing_time
# - latency_p50/p99: Response time percentiles
```

## Docker Deployment

### GPU Deployment

```yaml
# compose.yml
services:
  gigaam-server:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - GIGAAM_DEVICE=cuda
      - HF_TOKEN=${HF_TOKEN}
```

### CPU Deployment

```bash
docker build -f Dockerfile.cpu -t gigaam-cpu .
docker run -p 8000:8000 gigaam-cpu
```

## Architecture

```
gigaam_server/
├── main.py              # FastAPI app
├── config.py            # Settings
├── api/v1/
│   └── endpoints/
│       ├── transcriptions.py  # POST /v1/audio/transcriptions
│       ├── streaming.py       # WebSocket streaming
│       ├── models.py          # Model listing
│       └── health.py          # Health checks
├── services/
│   ├── model_manager.py       # Model lifecycle
│   ├── transcription.py       # Transcription logic
│   ├── streaming.py           # Real-time streaming
│   └── diarization.py         # Speaker diarization
└── schemas/
    ├── transcription.py       # Request/response models
    └── streaming.py           # WebSocket messages
```
