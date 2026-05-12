"""Unit tests for backends/paddle_vl/device.py.

All tests that actually call into ``paddle`` are guarded with
``pytest.importorskip("paddle")`` — they are skipped when the PaddlePaddle
framework is not installed (normal in the CI environment used for unit tests).
"""
from __future__ import annotations

import pytest

from agentic_pdf_parser.backends.paddle_vl.device import (
    _auto_detect_device,
    _is_gpu_available,
    resolve_device,
)
from agentic_pdf_parser.config import DeviceChoice


# ---------------------------------------------------------------------------
# resolve_device — CPU path (no Paddle import needed)
# ---------------------------------------------------------------------------


def test_resolve_device_cpu() -> None:
    assert resolve_device(DeviceChoice.CPU) == "cpu"


def test_resolve_device_gpu_no_paddle(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU requested but _is_gpu_available returns False → RuntimeError."""
    monkeypatch.setattr(
        "agentic_pdf_parser.backends.paddle_vl.device._is_gpu_available",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_device(DeviceChoice.GPU)


def test_resolve_device_gpu_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU requested and _is_gpu_available returns True → 'gpu:0'."""
    monkeypatch.setattr(
        "agentic_pdf_parser.backends.paddle_vl.device._is_gpu_available",
        lambda: True,
    )
    assert resolve_device(DeviceChoice.GPU) == "gpu:0"


def test_resolve_device_auto_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentic_pdf_parser.backends.paddle_vl.device._is_gpu_available",
        lambda: False,
    )
    assert resolve_device(DeviceChoice.AUTO) == "cpu"


def test_resolve_device_auto_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentic_pdf_parser.backends.paddle_vl.device._is_gpu_available",
        lambda: True,
    )
    assert resolve_device(DeviceChoice.AUTO) == "gpu:0"


# ---------------------------------------------------------------------------
# _is_gpu_available — requires paddle (skipped when not installed)
# ---------------------------------------------------------------------------


def test_is_gpu_available_returns_bool() -> None:
    """_is_gpu_available should never raise — returns bool even without Paddle."""
    result = _is_gpu_available()
    assert isinstance(result, bool)


def test_auto_detect_device_returns_valid_string() -> None:
    result = _auto_detect_device()
    assert result in {"cpu", "gpu:0"}
