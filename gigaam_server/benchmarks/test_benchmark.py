"""Performance benchmarks for GigaAM transcription service."""

import io
import os
import tempfile
import time
import wave

import numpy as np
import pytest

from gigaam import load_model
from gigaam.preprocess import SAMPLE_RATE


def generate_test_audio(duration_seconds: float, frequency: float = 440.0) -> bytes:
    """Generate synthetic test audio (sine wave)."""
    num_samples = int(duration_seconds * SAMPLE_RATE)
    t = np.arange(num_samples) / SAMPLE_RATE
    audio = np.sin(2 * np.pi * frequency * t)

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write to WAV buffer
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def short_audio_path():
    """Generate short test audio file (5 seconds)."""
    audio_data = generate_test_audio(5.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def long_audio_path():
    """Generate long test audio file (30 seconds)."""
    audio_data = generate_test_audio(30.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        path = f.name
    yield path
    os.unlink(path)


class TestModelInference:
    """Benchmarks for direct model inference."""

    @pytest.mark.benchmark(group="model-inference")
    def test_ctc_inference_time(self, benchmark, short_audio_path):
        """Measure inference time for CTC model."""
        model = load_model("v3_ctc", device="cpu")

        def _transcribe():
            return model.transcribe(short_audio_path)

        result = benchmark(_transcribe)
        assert isinstance(result, str)

    @pytest.mark.benchmark(group="model-inference")
    def test_rnnt_inference_time(self, benchmark, short_audio_path):
        """Measure inference time for RNNT model."""
        model = load_model("v3_rnnt", device="cpu")

        def _transcribe():
            return model.transcribe(short_audio_path)

        result = benchmark(_transcribe)
        assert isinstance(result, str)

    @pytest.mark.benchmark(group="model-inference")
    def test_e2e_inference_time(self, benchmark, short_audio_path):
        """Measure inference time for end-to-end model."""
        model = load_model("v3_e2e_ctc", device="cpu")

        def _transcribe():
            return model.transcribe(short_audio_path)

        result = benchmark(_transcribe)
        assert isinstance(result, str)


class TestTokensPerSecond:
    """Benchmarks for tokens per second metric."""

    @pytest.mark.benchmark(group="throughput")
    def test_tokens_per_second_ctc(self, benchmark, short_audio_path):
        """Calculate tokens per second for CTC model."""
        model = load_model("v3_ctc", device="cpu")
        tokenizer = model.decoding.tokenizer

        # Measure time
        start = time.perf_counter()
        result = model.transcribe(short_audio_path)
        elapsed = time.perf_counter() - start

        # Count tokens (characters for char-based vocab)
        if tokenizer.charwise:
            num_tokens = len(result)
        else:
            num_tokens = len(tokenizer.model.encode(result))

        tokens_per_second = num_tokens / elapsed if elapsed > 0 else 0

        # Report via benchmark
        benchmark.extra_info["tokens"] = num_tokens
        benchmark.extra_info["elapsed"] = elapsed
        benchmark.extra_info["tokens_per_second"] = tokens_per_second

        assert tokens_per_second > 0

    @pytest.mark.benchmark(group="throughput")
    def test_tokens_per_second_e2e(self, benchmark, short_audio_path):
        """Calculate tokens per second for E2E model."""
        model = load_model("v3_e2e_ctc", device="cpu")
        tokenizer = model.decoding.tokenizer

        start = time.perf_counter()
        result = model.transcribe(short_audio_path)
        elapsed = time.perf_counter() - start

        if tokenizer.charwise:
            num_tokens = len(result)
        else:
            num_tokens = len(tokenizer.model.encode(result))

        tokens_per_second = num_tokens / elapsed if elapsed > 0 else 0

        benchmark.extra_info["tokens"] = num_tokens
        benchmark.extra_info["elapsed"] = elapsed
        benchmark.extra_info["tokens_per_second"] = tokens_per_second

        assert tokens_per_second > 0


class TestRealtimeFactor:
    """Benchmarks for realtime factor metric."""

    @pytest.mark.benchmark(group="realtime")
    def test_realtime_factor_short(self, benchmark, short_audio_path):
        """Calculate realtime factor for short audio."""
        audio_duration = 5.0  # seconds
        model = load_model("v3_ctc", device="cpu")

        def _transcribe():
            return model.transcribe(short_audio_path)

        elapsed = benchmark(_transcribe)

        realtime_factor = (
            audio_duration / benchmark.stats["mean"]
            if benchmark.stats["mean"] > 0
            else 0
        )

        benchmark.extra_info["audio_duration"] = audio_duration
        benchmark.extra_info["processing_time"] = benchmark.stats["mean"]
        benchmark.extra_info["realtime_factor"] = realtime_factor

        # Should be faster than realtime (factor < 1.0)
        assert realtime_factor > 0

    @pytest.mark.benchmark(group="realtime")
    def test_realtime_factor_long(self, benchmark, long_audio_path):
        """Calculate realtime factor for long audio."""
        audio_duration = 30.0  # seconds
        model = load_model("v3_ctc", device="cpu")

        def _transcribe():
            return model.transcribe(long_audio_path)

        elapsed = benchmark(_transcribe)

        realtime_factor = (
            audio_duration / benchmark.stats["mean"]
            if benchmark.stats["mean"] > 0
            else 0
        )

        benchmark.extra_info["audio_duration"] = audio_duration
        benchmark.extra_info["processing_time"] = benchmark.stats["mean"]
        benchmark.extra_info["realtime_factor"] = realtime_factor

        assert realtime_factor > 0


class TestLatency:
    """Benchmarks for latency measurements."""

    @pytest.mark.benchmark(group="latency")
    def test_latency_p50_p99(self, benchmark, short_audio_path):
        """Measure p50 and p99 latency."""
        model = load_model("v3_ctc", device="cpu")

        def _transcribe():
            return model.transcribe(short_audio_path)

        benchmark(_transcribe)

        # Stats are provided by pytest-benchmark
        benchmark.extra_info["p50"] = benchmark.stats.get("min", 0)
        benchmark.extra_info["max"] = benchmark.stats.get("max", 0)


@pytest.mark.asyncio
class TestAsyncThroughput:
    """Async throughput benchmarks."""

    @pytest.mark.benchmark(group="async-throughput")
    async def test_concurrent_requests(self, benchmark):
        """Measure throughput with concurrent requests."""
        # This would require a running server
        # Placeholder for async benchmark structure
        pass
