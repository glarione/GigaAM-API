import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import omegaconf
import onnxruntime as rt

warnings.simplefilter("ignore", category=UserWarning)


DTYPE = np.float32
MAX_LETTERS_PER_FRAME = 3


def _get_optimal_thread_count() -> int:
    """
    Get optimal thread count for ONNX Runtime on CPU.

    Uses physical core count for better performance.
    Falls back to 4 if unable to detect.
    """
    try:
        import psutil

        # Use physical cores (not logical/hyperthreaded)
        return psutil.cpu_count(logical=False) or 4
    except ImportError:
        # Fallback: try os.sched_getaffinity or default to 4
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            # Fallback to 4 threads
            return 4


def load_onnx(
    onnx_dir: str,
    model_version: str,
    provider: Optional[str] = None,
) -> Tuple[
    List[rt.InferenceSession], Union[omegaconf.DictConfig, omegaconf.ListConfig]
]:
    """Load ONNX sessions for the given versions and cpu / cuda provider

    Optimized for CPU:
    - Auto-detects physical core count for thread configuration
    - Uses parallel execution mode for better throughput
    - Enables CPU memory pattern optimization
    """
    # Auto-detect provider
    if provider is None and "CUDAExecutionProvider" in rt.get_available_providers():
        provider = "CUDAExecutionProvider"
    elif provider is None:
        provider = "CPUExecutionProvider"

    # Get optimal thread count based on physical cores
    num_cores = _get_optimal_thread_count()

    # Configure session options for optimal CPU performance
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = num_cores
    opts.inter_op_num_threads = max(1, num_cores // 4)  # Small inter-op parallelism
    opts.execution_mode = (
        rt.ExecutionMode.ORT_PARALLEL
    )  # Parallel for better throughput
    opts.log_severity_level = 3
    opts.enable_cpu_mem_pattern = True  # Memory pattern optimization

    model_cfg = omegaconf.OmegaConf.load(f"{onnx_dir}/{model_version}.yaml")

    if "rnnt" not in model_version and "ssl" not in model_version:
        model_path = f"{onnx_dir}/{model_version}.onnx"
        sessions = [
            rt.InferenceSession(model_path, providers=[provider], sess_options=opts)
        ]
    elif "ssl" in model_version:
        pth = f"{onnx_dir}/{model_version}"
        enc_sess = rt.InferenceSession(
            f"{pth}_encoder.onnx", providers=[provider], sess_options=opts
        )
        sessions = [enc_sess]
    else:
        pth = f"{onnx_dir}/{model_version}"
        enc_sess = rt.InferenceSession(
            f"{pth}_encoder.onnx", providers=[provider], sess_options=opts
        )
        pred_sess = rt.InferenceSession(
            f"{pth}_decoder.onnx", providers=[provider], sess_options=opts
        )
        joint_sess = rt.InferenceSession(
            f"{pth}_joint.onnx", providers=[provider], sess_options=opts
        )
        sessions = [enc_sess, pred_sess, joint_sess]

    return sessions, model_cfg
