# GigaAM Usage Guide

## Installation

### Requirements

- Python >= 3.10
- ffmpeg installed and in PATH

### Basic Installation

```bash
git clone https://github.com/salute-developers/GigaAM.git
cd GigaAM
pip install -e .
```

### Optional Dependencies

```bash
# For long-form transcription (pyannote)
pip install -e ".[longform]"

# For testing
pip install -e ".[tests]"

# Full installation
pip install -e ".[longform,tests]"
```

### Long-form Setup

For long-form transcription, you need a Hugging Face token:

```bash
# 1. Generate HF token at https://huggingface.co/settings/tokens
# 2. Accept conditions for pyannote/segmentation-3.0 at https://huggingface.co/pyannote/segmentation-3.0
# 3. Set the token
export HF_TOKEN="your_token_here"
```

## Quick Start

### Speech Recognition

```python
import gigaam

# Load model
model = gigaam.load_model("v3_e2e_rnnt")

# Transcribe audio
transcription = model.transcribe("audio.wav")
print(transcription)
```

### Long-form Transcription

```python
import os
import gigaam

os.environ["HF_TOKEN"] = "your_token_here"

model = gigaam.load_model("v3_e2e_rnnt")
utterances = model.transcribe_longform("long_audio.wav")

for utterance in utterances:
    text = utterance["transcription"]
    start, end = utterance["boundaries"]
    print(f"[{gigaam.format_time(start)} - {gigaam.format_time(end)}]: {text}")
```

### Audio Embeddings

```python
import gigaam

model = gigaam.load_model("v3_ssl")
embedding, lengths = model.embed_audio("audio.wav")
print(f"Embedding shape: {embedding.shape}")
```

### Emotion Recognition

```python
import gigaam

model = gigaam.load_model("emo")
emotion_probs = model.get_probs("audio.wav")

for emotion, prob in emotion_probs.items():
    print(f"{emotion}: {prob:.3f}")
```

### ONNX Export and Inference

```python
import gigaam
from gigaam.onnx_utils import load_onnx, infer_onnx

# Export
model = gigaam.load_model("v3_ctc")
model.to_onnx("onnx_model/")

# Inference
sessions, config = load_onnx("onnx_model/", "v3_ctc")
result = infer_onnx("audio.wav", config, sessions)
print(result)
```

## Loading from Hugging Face

```python
from transformers import AutoModel

# Load from Hugging Face Hub
model = AutoModel.from_pretrained(
    "ai-sage/GigaAM-v3",
    revision="e2e_rnnt",
    trust_remote_code=True
)
```

## Model Selection Guide

### For Best Accuracy
- **v3_e2e_rnnt**: Best overall quality with punctuation and normalization

### For Fast Inference
- **v3_ctc**: Simplest architecture, fastest decoding

### For Long-form Audio
- Any ASR model with `transcribe_longform()` method
- Requires pyannote.audio installation

### For Audio Embeddings
- **v3_ssl**: Self-supervised model for feature extraction

### For Emotion Recognition
- **emo**: Specialized emotion classification model

## Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| v3_ctc | CTC-based ASR | Fast transcription |
| v3_rnnt | RNNT-based ASR | Balanced quality/speed |
| v3_e2e_ctc | End-to-end CTC | Transcription with punctuation |
| v3_e2e_rnnt | End-to-end RNNT | Best quality transcription |
| v3_ssl | Self-supervised | Audio embeddings |
| emo | Emotion recognition | Emotion classification |

## Performance Tips

1. **GPU Acceleration**: Models automatically use GPU when available
2. **Flash Attention**: Install `flash-attn` for better performance on long sequences with large batch sizes
3. **ONNX Runtime**: Use ONNX for deployment, supports CPU/GPU inference
4. **Batch Processing**: Use batched inference for multiple audio files

## Testing

```bash
# Run model loading tests
pytest -v tests/test_loading.py

# Run ONNX tests
pytest -v tests/test_onnx.py

# Run long-form tests (requires HF_TOKEN)
HF_TOKEN=your_token pytest -v tests/test_longform.py

# Full test suite with coverage
HF_TOKEN=your_token pytest --cov=gigaam --cov-report=term-missing -v tests/
```
