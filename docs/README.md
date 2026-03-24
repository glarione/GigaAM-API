# GigaAM Documentation

Welcome to the GigaAM documentation. This documentation covers the architecture, API, usage, and performance of the GigaAM family of speech models.

## Documentation Index

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System architecture and component design
- **[API.md](./API.md)** - API reference and code examples
- **[USAGE.md](./USAGE.md)** - Usage guide with installation and quick start
- **[PERFORMANCE.md](./PERFORMANCE.md)** - Performance benchmarks and evaluation

## Quick Links

- [GitHub Repository](https://github.com/salute-developers/GigaAM)
- [Hugging Face Models](https://huggingface.co/ai-sage/GigaAM-v3)
- [Research Paper](https://arxiv.org/abs/2506.01192)
- [Colab Example](https://colab.research.google.com/github/salute-developers/GigaAM/blob/main/colab_example.ipynb)

## Overview

GigaAM is a family of Conformer-based acoustic models for Russian speech processing, providing:

- **Speech Recognition (ASR)**: CTC, RNNT, and end-to-end models
- **Audio Embeddings**: Self-supervised models for feature extraction
- **Emotion Recognition**: Specialized model for emotion classification

### Key Features

- State-of-the-art performance on Russian speech benchmarks
- Support for short-form and long-form audio
- ONNX export for production deployment
- Pyannote integration for long-form transcription
- GPU acceleration with flash attention support
