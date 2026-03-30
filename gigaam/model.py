from typing import Dict, List, Tuple, Union

import hydra
import omegaconf
import torch
from torch import Tensor, nn

from .preprocess import SAMPLE_RATE, load_audio
from .utils import onnx_converter

LONGFORM_THRESHOLD = 25 * SAMPLE_RATE


class GigaAM(nn.Module):
    """
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks

    Optimizations:
    - torch.compile on CPU (PyTorch 2.5+) for 20-40% speedup
    - FP16 only on GPU (disabled on CPU where it's slower)
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)

    def forward(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform forward pass through the preprocessor and encoder.

        Note: FP16 is only used on GPU. On CPU, FP16 is slower because
        there are no Tensor Cores, and the overhead of conversion outweighs benefits.
        """
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        if self._device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.encoder(features, feature_lengths)
        return self.encoder(features, feature_lengths)

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prepare_wav(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Prepare an audio file for processing by loading it onto
        the correct device and converting its format.
        """
        wav = load_audio(wav_file)
        wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self._device)
        return wav, length

    def warmup(self, duration_seconds: float = 1.0) -> None:
        """
        Warm up the model by running a dummy inference.

        This initializes kernels, caches, and (for torch.compile) triggers
        compilation before real requests arrive.

        Args:
            duration_seconds: Duration of dummy audio in seconds
        """
        # Create dummy audio: 1 second at 16kHz = 16000 samples
        num_samples = int(duration_seconds * SAMPLE_RATE)
        dummy_audio = torch.randn(1, num_samples, device=self._device, dtype=self._dtype)

        with torch.inference_mode():
            # Run through preprocessor and encoder
            features, feature_lengths = self.preprocessor(dummy_audio, torch.tensor([num_samples], device=self._device))
            _ = self.encoder(features, feature_lengths)

    def embed_audio(self, wav_file: str) -> Tuple[Tensor, Tensor]:
        """
        Extract audio representations using the GigaAM model.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len

    def to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        self._to_onnx(dir_path)
        omegaconf.OmegaConf.save(self.cfg, f"{dir_path}/{self.cfg.model_name}.yaml")

    def _to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        onnx_converter(
            model_name=f"{self.cfg.model_name}_encoder",
            out_dir=dir_path,
            module=self.encoder,
            dynamic_axes=self.encoder.dynamic_axes(),
        )


class GigaAMASR(GigaAM):
    """
    Giga Acoustic Model for Speech Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)

    @torch.inference_mode()
    def transcribe(self, wav_file: str) -> str:
        """
        Transcribes a short audio file into text.
        """
        wav, length = self.prepare_wav(wav_file)
        if length.item() > LONGFORM_THRESHOLD:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")

        encoded, encoded_len = self.forward(wav, length)
        return self.decoding.decode(self.head, encoded, encoded_len)[0]

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        return self.head(self.encoder(features, feature_lengths)[0])

    def _to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx ASR model.
        `ctc`:  exported entirely in encoder-decoder format.
        `rnnt`: exported in encoder/decoder/joint parts separately.
        """
        if "ctc" in self.cfg.model_name:
            saved_forward = self.forward
            self.forward = self.forward_for_export  # type: ignore[assignment, method-assign]
            try:
                onnx_converter(
                    model_name=self.cfg.model_name,
                    out_dir=dir_path,
                    module=self,
                    inputs=self.encoder.input_example(),
                    input_names=["features", "feature_lengths"],
                    output_names=["log_probs"],
                    dynamic_axes={
                        "features": {0: "batch_size", 2: "seq_len"},
                        "feature_lengths": {0: "batch_size"},
                        "log_probs": {0: "batch_size", 1: "seq_len"},
                    },
                )
            finally:
                self.forward = saved_forward  # type: ignore[assignment, method-assign]
        else:
            super()._to_onnx(dir_path)  # export encoder
            onnx_converter(
                model_name=f"{self.cfg.model_name}_decoder",
                out_dir=dir_path,
                module=self.head.decoder,
            )
            onnx_converter(
                model_name=f"{self.cfg.model_name}_joint",
                out_dir=dir_path,
                module=self.head.joint,
            )

    @torch.inference_mode()
    def transcribe_longform(
        self, wav_file: str, batch_size: int = 4, **kwargs
    ) -> List[Dict[str, Union[str, Tuple[float, float]]]]:
        """
        Transcribes a long audio file by splitting it into segments and
        then transcribing each segment in parallel for better performance.

        Args:
            wav_file: Path to the audio file
            batch_size: Number of segments to process in parallel (default: 4)
            **kwargs: Additional arguments passed to segment_audio_file
        """
        from .vad_utils import segment_audio_file

        segments, boundaries = segment_audio_file(
            wav_file, SAMPLE_RATE, device=self._device, **kwargs
        )

        if not segments:
            return []

        transcribed_segments = []

        # Process segments sequentially (more reliable than threading for PyTorch models)
        for segment, segment_boundaries in zip(segments, boundaries):
            wav = segment.to(self._device).unsqueeze(0).to(self._dtype)
            length = torch.full([1], wav.shape[-1], device=self._device)
            encoded, encoded_len = self.forward(wav, length)
            result = self.decoding.decode(self.head, encoded, encoded_len)[0]
            transcribed_segments.append(
                {"transcription": result, "boundaries": segment_boundaries}
            )

        return transcribed_segments


class GigaAMEmo(GigaAM):
    """
    Giga Acoustic Model for Emotion Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.id2name = cfg.id2name

    def get_probs(self, wav_file: str) -> Dict[str, float]:
        """
        Calculate probabilities for each emotion class based on the provided audio file.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, _ = self.forward(wav, length)
        encoded_pooled = nn.functional.avg_pool1d(
            encoded, kernel_size=encoded.shape[-1]
        ).squeeze(-1)

        logits = self.head(encoded_pooled)[0]
        probs = nn.functional.softmax(logits, dim=-1).detach().tolist()

        return {self.id2name[i]: probs[i] for i in range(len(self.id2name))}

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        encoded, _ = self.encoder(features, feature_lengths)
        enc_pooled = encoded.mean(dim=-1)
        return nn.functional.softmax(self.head(enc_pooled), dim=-1)

    def _to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx Emo model.
        """
        saved_forward = self.forward
        self.forward = self.forward_for_export  # type: ignore[assignment, method-assign]
        try:
            onnx_converter(
                model_name=self.cfg.model_name,
                out_dir=dir_path,
                module=self,
                inputs=self.encoder.input_example(),
                input_names=["features", "feature_lengths"],
                output_names=["probs"],
                dynamic_axes={
                    "features": {0: "batch_size", 2: "seq_len"},
                    "feature_lengths": {0: "batch_size"},
                    "probs": {0: "batch_size", 1: "seq_len"},
                },
            )
        finally:
            self.forward = saved_forward  # type: ignore[assignment, method-assign]
