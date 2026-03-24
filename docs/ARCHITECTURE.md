# GigaAM Architecture

## Overview

GigaAM (Giga Acoustic Model) is a family of open-source Conformer-based acoustic models for Russian speech processing. The project provides state-of-the-art speech recognition (ASR) and emotion recognition capabilities.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GigaAM Model                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Preprocessor │→│   Encoder    │→│      Head/Decoder      │  │
│  │              │  │  (Conformer) │  │  (CTC/RNNT/Emo/SSL)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Preprocessor (`gigaam/preprocess.py`)

Handles audio loading and feature extraction:

- **FeatureExtractor**: Extracts Log-mel spectrogram features
  - Sample rate: 16kHz
  - Mel bins: 64 features
  - Uses torchaudio's MelSpectrogram
  - Log scaling applied to spectrogram values

- **Audio Loading**: Uses ffmpeg for robust audio decoding
  - Converts to 16-bit PCM
  - Mono channel
  - Resampling to 16kHz

### 2. Encoder (`gigaam/encoder.py`)

Conformer-based encoder with the following architecture:

- **StridingSubsampling**: Reduces sequence length by factor of 4 via conv2d
- **Positional Encoding**: Rotary Positional Embeddings (RoPE)
- **Conformer Layers**: Stack of 16 layers with:
  - Feed-forward modules (FFN with SiLU activation)
  - Multi-head self-attention (with RoPE + flash attention support)
  - Conformer convolution module (depthwise conv + GLU)
  - Layer normalization throughout

**Model sizes**: 220-240M parameters

### 3. Heads/Decoders (`gigaam/decoder.py`)

#### CTC Head
- Simple 1D convolution layer
- Outputs log-probabilities over vocabulary + blank token
- Greedy decoding with blank skipping and repeat collapsing

#### RNNT Head
- **RNNTDecoder**: LSTM-based prediction network
- **RNNTJoint**: Combines encoder and decoder outputs
  - Linear projections for both streams
  - ReLU + linear for final joint representation
  - Log-softmax output

### 4. Decoding (`gigaam/decoding.py`)

- **Tokenizer**: Supports both character-wise and SentencePiece tokenization
- **CTCGreedyDecoding**: Standard CTC greedy search
- **RNNTGreedyDecoding**: RNNT greedy decoding with max symbols per step limit

### 5. VAD Utils (`gigaam/vad_utils.py`)

Long-form audio processing using pyannote:

- **VoiceActivityDetection**: Segments audio using pyannote/segmentation-3.0
- **Segment Merging**: Combines VAD segments into optimal chunks (15-22s)
- **Boundary Tracking**: Maintains timestamps for each transcription segment

### 6. ONNX Utils (`gigaam/onnx_utils.py`)

ONNX export and inference support:

- **load_onnx**: Loads ONNX sessions with CPU/GPU provider selection
- **infer_onnx**: Runs inference through ONNX runtime
- Supports batched processing
- Optimized session options (16 threads, sequential execution)

## Model Variants

| Version | Pretrain | Pretrain Hours | ASR Hours | Models |
|---------|----------|----------------|-----------|--------|
| v1 | Wav2vec 2.0 | 50,000 | 2,000 | v1_ssl, emo, v1_ctc, v1_rnnt |
| v2 | HuBERT-CTC | 50,000 | 2,000 | v2_ssl, v2_ctc, v2_rnnt |
| v3 | HuBERT-CTC | 700,000 | 4,000 | v3_ssl, v3_ctc, v3_rnnt, v3_e2e_ctc, v3_e2e_rnnt |

### Model Types

- **SSL**: Self-supervised learning (audio embeddings)
- **CTC**: Connectionist Temporal Classification ASR
- **RNNT**: Recurrent Neural Network Transducer ASR
- **Emo**: Emotion recognition
- **e2e**: End-to-end with punctuation and text normalization

## Data Flow

### Short-form ASR
```
Audio File → ffmpeg → Raw Audio → Mel Spectrogram → Conformer Encoder →
Head (CTC/RNNT) → Decoding → Text Output
```

### Long-form ASR
```
Audio File → pyannote VAD → Segments → [Parallel Processing] →
Each Segment → Short-form ASR → Merged Transcriptions with Timestamps
```

### ONNX Export
```
PyTorch Model → Export Encoder (and Decoder/Joint for RNNT) →
ONNX Files + Config YAML → ONNX Runtime Inference
```

## Performance Benchmarks

### ASR WER (%) - GigaAM-v3

| Domain | CTC | RNNT | E2E CTC | E2E RNNT |
|--------|-----|------|---------|----------|
| Golos Farfield | 4.5 | 3.9 | 6.1 | 5.5 |
| Mozilla Common Voice | 1.3 | 0.9 | 3.2 | 3.0 |
| Disordered Speech | 20.6 | 19.2 | 22.8 | 23.1 |
| **Average** | **9.1** | **8.3** | **12.0** | **11.2** |

### Inference Speed (CUDA)

| Batch, Duration | Custom Attn | SDPA | Flash |
|-----------------|-------------|------|-------|
| 1, 10s | 10.14ms | 10.06ms | 11.57ms |
| 128, 30s | 324.53ms | 324.48ms | 293.80ms |

## Dependencies

- Python >= 3.10
- torch >= 2.5, < 2.9
- torchaudio >= 2.5, < 2.9
- onnxruntime >= 1.23
- sentencepiece
- hydra-core
- Optional: pyannote.audio (longform), flash-attn (GPU acceleration)
