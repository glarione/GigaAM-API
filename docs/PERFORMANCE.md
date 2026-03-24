# GigaAM Performance Evaluation

## ASR Performance

### Word Error Rate (%) Across Validation Sets

| Set Name | V3 CTC | V3 RNNT | E2E CTC* | E2E RNNT* | V2 CTC | V2 RNNT | V1 CTC | V1 RNNT | Whisper* |
|----------|-------:|--------:|---------:|----------:|-------:|--------:|-------:|--------:|---------:|
| Golos Farfield | 4.5 | 3.9 | 6.1 | 5.5 | 4.3 | 4.0 | 5.8 | 4.8 | 16.4 |
| Golos Crowd | 2.8 | 2.4 | 9.7 | 9.1 | 2.5 | 2.3 | 3.1 | 2.3 | 19.0 |
| Russian LibriSpeech | 4.7 | 4.4 | 6.4 | 6.4 | 5.2 | 5.2 | 7.5 | 7.7 | 9.4 |
| Mozilla Common Voice 19 | 1.3 | 0.9 | 3.2 | 3.0 | 1.5 | 0.9 | 8.4 | 8.0 | 5.5 |
| Natural Speech | 7.8 | 6.9 | 9.6 | 8.5 | 10.8 | 10.3 | 12.6 | 11.4 | 13.4 |
| Disordered Speech | 20.6 | 19.2 | 22.8 | 23.1 | 28.0 | 27.5 | 37.5 | 40.8 | 58.6 |
| Callcenter | 10.3 | 9.5 | 13.3 | 12.6 | 13.6 | 12.9 | 15.5 | 15.0 | 23.1 |
| OpenSTT Phone Calls | 18.6 | 17.4 | 20.0 | 19.1 | 20.7 | 19.8 | 23.0 | 21.1 | 27.4 |
| OpenSTT Youtube | 11.6 | 10.6 | 12.7 | 11.8 | 13.9 | 13.0 | 16.0 | 14.7 | 17.8 |
| OpenSTT Audiobooks | 8.7 | 8.2 | 10.3 | 9.3 | 10.8 | 10.3 | 12.7 | 11.7 | 14.3 |
| **Average** | **9.1** | **8.3** | **12.0** | **11.2** | **11.1** | **10.6** | **14.2** | **13.8** | **21.0** |

*\*With post-processing applied (removing punctuation and capitalization)*

### Key Improvements

- **GigaAM-v3 vs v2**: ~18% WER reduction on average
- **GigaAM-v3 vs v1**: ~36% WER reduction on average
- **vs Whisper**: ~57% WER reduction on average

## End-to-End ASR Performance

### Side-by-Side vs Whisper (LLM-as-a-Judge)

GigaAM e2e models win against Whisper by **70:30** margin across domains.

| Model | Normalization | F1(,) | F1(.) | F1(?) | WER | CER |
|-------|---------------|-------|-------|-------|-----|-----|
| GigaChat Max Audio | Full | 84.2 | 85.6 | 74.9 | 18.4 | 10.9 |
| Whisper Punctuator | punctuation only | 62.2 | 85.0 | 77.7 | 0.0 | 0.0 |
| GigaAM from Whisper labels | punctuation only | 50.3 | 84.1 | 77.7 | 12.0 | 7.8 |
| **GigaAM-e2e-ctc** | **Full** | **83.7** | **86.7** | **78.6** | **16.0** | **8.7** |
| **GigaAM-e2e-rnnt** | **Full** | **84.5** | **86.7** | **79.8** | **14.2** | **8.8** |

## Emotion Recognition Performance

### DUSHA Dataset Results

| Model | Crowd UAcc | Crowd WAcc | Crowd Macro F1 | Podcast UAcc | Podcast WAcc | Podcast Macro F1 |
|-------|-----------:|-----------:|---------------:|-------------:|-------------:|-----------------:|
| DUSHA baseline (MobileNetV2 + Self-Attention) | 0.83 | 0.76 | 0.77 | 0.89 | 0.53 | 0.54 |
| ABC (TIM-Net) | 0.84 | 0.77 | 0.78 | 0.90 | 0.50 | 0.55 |
| **GigaAM-Emo** | **0.90** | **0.87** | **0.84** | **0.90** | **0.76** | **0.67** |

## Inference Speed Benchmarks

### Attention Mechanism Performance (CUDA, Time in ms ± std)

| Batch, Duration | Custom | SDPA | Flash |
|-----------------|--------|------|-------|
| 1, 10s | 0.03 ± 0.00 | 0.03 ± 0.00 | 0.05 ± 0.03 |
| 8, 20s | 0.15 ± 0.01 | 0.14 ± 0.01 | 0.66 ± 0.14 |
| 128, 30s | 3.60 ± 0.10 | 3.59 ± 0.04 | 1.40 ± 0.06 |

### Full Encoder Inference (CUDA, Time in ms ± std)

| Batch, Duration | Custom | SDPA | Flash |
|-----------------|--------|------|-------|
| 1, 10s | 10.14 ± 0.17 | 10.06 ± 0.12 | 11.57 ± 0.25 |
| 8, 20s | 15.84 ± 0.07 | 15.90 ± 0.02 | 25.26 ± 0.26 |
| 128, 30s | 324.53 ± 0.17 | 324.48 ± 0.09 | 293.80 ± 0.89 |

### Recommendations

- **Single/batch inference**: SDPA is recommended (best balance)
- **Large batch + long sequences**: Flash attention provides ~10% speedup
- **CPU inference**: SDPA provides best compatibility

## Test Coverage

- **Coverage**: 91%
- **Excluded**: flash-attn (requires GPU execution)

```bash
HF_TOKEN=<token> pytest --cov=gigaam --cov-report=term-missing -v tests/
```

## Benchmarking Your Setup

To benchmark performance on your hardware:

```python
import time
import gigaam
import torch

# Load model
model = gigaam.load_model("v3_ctc")
model.cuda()  # Move to GPU

# Warmup
_ = model.transcribe("warmup.wav")

# Benchmark
times = []
for _ in range(10):
    start = time.perf_counter()
    _ = model.transcribe("test.wav")
    times.append(time.perf_counter() - start)

print(f"Average inference time: {sum(times)/len(times):.3f}s")
print(f"Std deviation: {(max(times)-min(times))/2:.3f}s")
```
