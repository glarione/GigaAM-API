# GigaAM API Reference

## Package Structure

```
gigaam/
├── __init__.py      # Package exports, load_model function
├── model.py         # Main model classes (GigaAM, GigaAMASR, GigaAMEmo)
├── encoder.py       # Conformer encoder implementation
├── decoder.py       # Head modules (CTC, RNNT)
├── decoding.py      # Decoding algorithms (CTC/RNNT greedy)
├── preprocess.py    # Audio loading, feature extraction
├── onnx_utils.py    # ONNX export and inference
└── vad_utils.py     # Long-form audio segmentation
```

## Main API

### Loading Models

```python
import gigaam

# Load an ASR model
model = gigaam.load_model("v3_ctc")  # or "v3_rnnt", "v3_e2e_ctc", "v3_e2e_rnnt"

# Load SSL model for embeddings
model = gigaam.load_model("v3_ssl")

# Load emotion recognition model
model = gigaam.load_model("emo")
```

### Model Interface

#### GigaAMASR (Speech Recognition)

```python
# Short-form transcription (< 25 seconds)
transcription = model.transcribe("audio.wav")
print(transcription)  # str

# Long-form transcription (any duration)
utterances = model.transcribe_longform("long_audio.wav")
for utterance in utterances:
    text = utterance["transcription"]
    start, end = utterance["boundaries"]
    print(f"[{start:.2f}-{end:.2f}]: {text}")
```

#### GigaAM (Audio Embeddings)

```python
# Extract audio embeddings
embedding, lengths = model.embed_audio("audio.wav")
# embedding: Tensor [batch, seq_len, hidden_dim]
# lengths: Tensor [batch]
```

#### GigaAMEmo (Emotion Recognition)

```python
# Get emotion probabilities
emotion_probs = model.get_probs("audio.wav")
# {emotion_name: probability, ...}
```

### ONNX Inference

```python
from gigaam.onnx_utils import load_onnx, infer_onnx

# Export model to ONNX
model.to_onnx("onnx_output/")

# Load and run ONNX inference
sessions, config = load_onnx("onnx_output/", "v3_ctc")
result = infer_onnx("audio.wav", config, sessions)
```

## Configuration

Models use Hydra/OmegaConf configuration. Example config structure:

```yaml
model_name: v3_ctc

preprocessor:
  _target_: gigaam.preprocess.FeatureExtractor
  sample_rate: 16000
  features: 64

encoder:
  _target_: gigaam.encoder.ConformerEncoder
  feat_in: 64
  n_layers: 16
  d_model: 768
  n_heads: 16
  subsampling: conv2d
  subsampling_factor: 4

head:
  _target_: gigaam.decoder.CTCHead
  feat_in: 768
  num_classes: 128

decoding:
  _target_: gigaam.decoding.CTCGreedyDecoding
  vocabulary: [...]
```

## Constants

- `SAMPLE_RATE = 16000` - Audio sample rate
- `LONGFORM_THRESHOLD = 25 * SAMPLE_RATE` - Max duration for short-form transcribe

## Error Handling

```python
from gigaam.model import GigaAMASR

try:
    model.transcribe("audio.wav")
except ValueError as e:
    # Raised for audio > 25 seconds
    print(f"Too long, use transcribe_longform: {e}")
except RuntimeError as e:
    # Raised for audio loading failures
    print(f"Failed to load audio: {e}")
```
