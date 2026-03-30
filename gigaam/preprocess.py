import warnings
from subprocess import CalledProcessError, run
from typing import Literal, Tuple

import torch
import torchaudio
from torch import Tensor, nn

SAMPLE_RATE = 16000


def load_audio(
    audio_path: str,
    sample_rate: int = SAMPLE_RATE,
    backend: Literal["ffmpeg-subprocess", "torchaudio"] = "ffmpeg-subprocess",
) -> Tensor:
    """
    Load an audio file and resample it to the specified sample rate.

    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate (default: 16000)
        backend: Loading backend to use
            - "ffmpeg-subprocess": Uses ffmpeg subprocess (original, robust)
            - "torchaudio": Uses torchaudio native loading (faster on CPU, no subprocess)

    Returns:
        Tensor of audio samples normalized to [-1, 1]
    """
    if backend == "torchaudio":
        return _load_audio_torchaudio(audio_path, sample_rate)

    # Original ffmpeg subprocess method
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        audio = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        raise RuntimeError("Failed to load audio") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return torch.frombuffer(audio, dtype=torch.int16).float() / 32768.0


def _load_audio_torchaudio(audio_path: str, sample_rate: int = SAMPLE_RATE) -> Tensor:
    """
    Load audio using torchaudio native backend (no subprocess overhead).

    This is faster on CPU as it avoids subprocess spawning overhead.
    Uses torchaudio's built-in ffmpeg/soundfile backend directly.

    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate (default: 16000)

    Returns:
        Tensor of audio samples normalized to [-1, 1]
    """
    # Load audio with torchaudio (uses ffmpeg or soundfile backend)
    waveform, src_sr = torchaudio.load(audio_path)

    # Resample if needed
    if src_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(src_sr, sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize to [-1, 1] (torchaudio returns float already for most formats)
    if waveform.dtype == torch.int16:
        waveform = waveform.float() / 32768.0
    else:
        # Already float, just ensure mono
        waveform = waveform.squeeze(0)

    return waveform.squeeze(0)


def load_audio_bytes(
    audio_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
    dtype: Literal["int16", "float32"] = "int16",
) -> Tensor:
    """
    Load audio from raw bytes (no file I/O).

    This is the fastest method as it avoids both subprocess and file I/O.
    Ideal for server-side processing of uploaded audio.

    Args:
        audio_bytes: Raw audio bytes (PCM format)
        sample_rate: Sample rate of the audio
        channels: Number of channels (default: 1 for mono)
        dtype: Data type of input bytes ("int16" or "float32")

    Returns:
        Tensor of audio samples normalized to [-1, 1]
    """
    if dtype == "int16":
        audio = torch.frombuffer(audio_bytes, dtype=torch.int16).float() / 32768.0
    else:
        audio = torch.frombuffer(audio_bytes, dtype=torch.float32)

    # Reshape if multi-channel
    if channels > 1:
        audio = audio.view(-1, channels).mean(dim=1)

    return audio


class SpecScaler(nn.Module):
    """
    Module that applies logarithmic scaling to spectrogram values.
    This module clamps the input values within a certain range and then applies a natural logarithm.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class FeatureExtractor(nn.Module):
    """
    Module for extracting Log-mel spectrogram features from raw audio signals.
    This module uses Torchaudio's MelSpectrogram transform to extract features
    and applies logarithmic scaling.
    """

    def __init__(self, sample_rate: int, features: int, **kwargs):
        super().__init__()
        self.hop_length = kwargs.get("hop_length", sample_rate // 100)
        self.win_length = kwargs.get("win_length", sample_rate // 40)
        self.n_fft = kwargs.get("n_fft", sample_rate // 40)
        self.center = kwargs.get("center", True)
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=features,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                center=self.center,
            ),
            SpecScaler(),
        )

    def out_len(self, input_lengths: Tensor) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        if self.center:
            return (
                input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()
            )
        else:
            return (
                (input_lengths - self.win_length)
                .div(self.hop_length, rounding_mode="floor")
                .add(1)
                .long()
            )

    def forward(self, input_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.
        """
        return self.featurizer(input_signal), self.out_len(length)
