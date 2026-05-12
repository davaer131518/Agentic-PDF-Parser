"""PaddleOCR-VL device resolution.

PaddleOCR device strings differ from torch device strings:
  - "cpu"        → CPU inference
  - "gpu:0"      → GPU 0 (NVIDIA CUDA)
  - "gpu:0,1"    → Multi-GPU (not used for single-page inference)

``paddle`` (the PaddlePaddle framework) is imported lazily at call-time so
this module can be loaded without a full Paddle installation.

``resolve_n_gpu_layers`` is used separately to configure the llama-server
subprocess that handles VLM recognition.
"""
from __future__ import annotations

from ...config import DeviceChoice


def resolve_device(choice: DeviceChoice) -> str:
    """Translate a :class:`DeviceChoice` to a PaddleOCR device string.

    When ``gpu`` is requested but no CUDA device is found via the Paddle
    framework, a ``RuntimeError`` is raised rather than silently falling
    back to CPU.

    Raises
    ------
    RuntimeError
        If ``gpu`` is explicitly requested but Paddle reports no CUDA device.
    """
    match choice:
        case DeviceChoice.AUTO:
            return _auto_detect_device()
        case DeviceChoice.GPU:
            if not _is_gpu_available():
                raise RuntimeError(
                    "device=gpu requested but no CUDA device is available to Paddle. "
                    "Use device=auto to fall back to CPU."
                )
            return "gpu:0"
        case DeviceChoice.CPU:
            return "cpu"
        case _:
            return "cpu"


def resolve_n_gpu_layers(choice: DeviceChoice) -> int:
    """Return the ``-ngl`` value to pass to ``llama-server.exe``.

    ``-1`` (all layers on GPU) is used for ``gpu`` and ``auto`` when a CUDA
    device is detected.  ``0`` forces CPU-only inference.
    """
    match choice:
        case DeviceChoice.GPU:
            return -1
        case DeviceChoice.CPU:
            return 0
        case DeviceChoice.AUTO:
            return -1 if _is_gpu_available() else 0
        case _:
            return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_gpu_available() -> bool:
    """Return True if PaddlePaddle reports at least one CUDA device."""
    try:
        import paddle  # noqa: PLC0415

        return (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:
        return False


def _auto_detect_device() -> str:
    """Return the best available device string for Paddle."""
    return "gpu:0" if _is_gpu_available() else "cpu"
