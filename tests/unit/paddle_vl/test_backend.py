"""Unit tests for backends/paddle_vl/backend.py.

The PaddleOCR / Paddle framework and llama-server are NOT needed to run
these tests.  We inject a fake ``paddleocr`` module into ``sys.modules``
and mock ``subprocess.Popen`` + ``urlopen`` so that ``backend.load()``
completes without any real processes or network calls.
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch
import tempfile

import pytest

from agentic_pdf_parser.backends.paddle_vl.backend import PaddleVLBackend, _result_to_raw
from agentic_pdf_parser.backends.base import BackendPageResult
from agentic_pdf_parser.config import DeviceChoice, ParseConfig

from .conftest import CANNED_FULL_PAGE


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _FakeBlock:
    """Minimal stand-in for PaddleOCRVLBlock."""

    def __init__(self, label: str, content: str, bbox: list[int]) -> None:
        self.label = label
        self.content = content
        self.bbox = list(bbox)


class _FakeResult(dict):
    """Minimal stand-in for PaddleOCRVLResult (dict subclass)."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(data)


def _build_fake_result_from_canned(canned: dict) -> _FakeResult:
    blocks = [
        _FakeBlock(
            label=b["block_label"],
            content=b["block_content"],
            bbox=b["block_bbox"],
        )
        for b in canned["parsing_res_list"]
    ]
    return _FakeResult(
        {
            "page_index": canned["page_index"],
            "width": canned["width"],
            "height": canned["height"],
            "parsing_res_list": blocks,
        }
    )


def _make_fake_paddleocr_module() -> ModuleType:
    """Return a fake ``paddleocr`` module whose PaddleOCRVL accepts any kwargs."""

    class FakePipelineInst:
        def predict(self, *args, **kwargs):
            return []

        def close(self):
            pass

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            self._inst = FakePipelineInst()

        def predict(self, *args, **kwargs):
            return self._inst.predict(*args, **kwargs)

        def close(self):
            self._inst.close()

    fake_mod = ModuleType("paddleocr")
    fake_mod.PaddleOCRVL = FakePaddleOCRVL  # type: ignore[attr-defined]
    return fake_mod


def _make_config(device: DeviceChoice = DeviceChoice.CPU) -> ParseConfig:
    out = Path(tempfile.mkdtemp()) / "out"
    return ParseConfig(backend="paddle_vl", device=device, output_dir=out)


def _make_fake_popen() -> MagicMock:
    """Return a mock Popen instance that appears healthy (poll() returns None)."""
    proc = MagicMock()
    proc.poll.return_value = None   # process is still running
    proc.returncode = None
    return proc


def _make_healthy_response() -> MagicMock:
    """Return a mock urlopen context manager that returns HTTP 200."""
    resp = MagicMock()
    resp.status = 200
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Context manager: load backend with all external calls mocked
# ---------------------------------------------------------------------------


def _load_backend(
    backend: PaddleVLBackend,
    fake_pipeline_predict=None,
) -> tuple[MagicMock, ModuleType]:
    """Load the backend with subprocess.Popen and urlopen mocked.

    Returns (fake_popen_instance, fake_paddleocr_module).
    """
    fake_proc = _make_fake_popen()
    fake_resp = _make_healthy_response()
    fake_mod = _make_fake_paddleocr_module()

    prev = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_mod

    with (
        patch("agentic_pdf_parser.backends.paddle_vl.backend.subprocess.Popen",
              return_value=fake_proc) as mock_popen,
        patch("agentic_pdf_parser.backends.paddle_vl.backend.urlopen",
              return_value=fake_resp),
        patch.object(
            backend._cfg.paddle_vl.__class__,
            "__get__",
            return_value=backend._cfg.paddle_vl,
        ) if False else patch("builtins.open", MagicMock()) if False else _noop(),
    ):
        # Patch path existence checks so FileNotFoundError isn't raised
        with patch("pathlib.Path.exists", return_value=True):
            backend.load()

    if prev is None:
        sys.modules.pop("paddleocr", None)
    else:
        sys.modules["paddleocr"] = prev

    if fake_pipeline_predict is not None:
        backend._pipeline.predict = fake_pipeline_predict

    return fake_proc, fake_mod


class _noop:
    """No-op context manager for use in with-chains."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# Tests: __init__ and protocol attributes
# ---------------------------------------------------------------------------


def test_backend_name() -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)
    assert backend.name == "paddle_vl"


def test_backend_implements_protocol() -> None:
    from agentic_pdf_parser.backends.base import ParserBackend
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)
    assert callable(getattr(backend, "load", None))
    assert callable(getattr(backend, "unload", None))
    assert callable(getattr(backend, "parse_page", None))


def test_resolved_device_cpu() -> None:
    cfg = _make_config(DeviceChoice.CPU)
    backend = PaddleVLBackend(cfg)
    assert backend.resolved_device == "cpu"


# ---------------------------------------------------------------------------
# Tests: load / unload lifecycle
# ---------------------------------------------------------------------------


def test_load_starts_llama_server(page_input) -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)

    fake_proc = _make_fake_popen()
    fake_resp = _make_healthy_response()
    fake_mod = _make_fake_paddleocr_module()

    prev = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_mod
    try:
        with (
            patch("agentic_pdf_parser.backends.paddle_vl.backend.subprocess.Popen",
                  return_value=fake_proc) as mock_popen,
            patch("agentic_pdf_parser.backends.paddle_vl.backend.urlopen",
                  return_value=fake_resp),
            patch("pathlib.Path.exists", return_value=True),
        ):
            backend.load()
            # llama-server must have been launched
            mock_popen.assert_called_once()
            call_cmd = mock_popen.call_args[0][0]
            expected_binary = "llama-server.exe" if sys.platform == "win32" else "llama-server"
            assert expected_binary in str(call_cmd[0])
    finally:
        if prev is None:
            sys.modules.pop("paddleocr", None)
        else:
            sys.modules["paddleocr"] = prev


def test_load_sets_pipeline(page_input) -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)

    fake_mod = _make_fake_paddleocr_module()
    prev = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_mod
    try:
        with (
            patch("agentic_pdf_parser.backends.paddle_vl.backend.subprocess.Popen",
                  return_value=_make_fake_popen()),
            patch("agentic_pdf_parser.backends.paddle_vl.backend.urlopen",
                  return_value=_make_healthy_response()),
            patch("pathlib.Path.exists", return_value=True),
        ):
            backend.load()
        assert backend._pipeline is not None
        assert backend.version == "PaddleOCR-VL-1.5"
    finally:
        if prev is None:
            sys.modules.pop("paddleocr", None)
        else:
            sys.modules["paddleocr"] = prev


def test_unload_clears_pipeline_and_terminates_server(page_input) -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)

    fake_proc = _make_fake_popen()
    fake_mod = _make_fake_paddleocr_module()
    prev = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_mod
    try:
        with (
            patch("agentic_pdf_parser.backends.paddle_vl.backend.subprocess.Popen",
                  return_value=fake_proc),
            patch("agentic_pdf_parser.backends.paddle_vl.backend.urlopen",
                  return_value=_make_healthy_response()),
            patch("pathlib.Path.exists", return_value=True),
        ):
            backend.load()
        backend.unload()
        assert backend._pipeline is None
        assert backend._llama_proc is None
        fake_proc.kill.assert_called_once()
    finally:
        if prev is None:
            sys.modules.pop("paddleocr", None)
        else:
            sys.modules["paddleocr"] = prev


def test_unload_before_load_is_safe() -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)
    backend.unload()  # should not raise


def test_parse_page_before_load_raises(page_input) -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)
    with pytest.raises(RuntimeError, match="before load"):
        backend.parse_page(page_input)


# ---------------------------------------------------------------------------
# Tests: parse_page contract
# ---------------------------------------------------------------------------


def _load_with_fake_predict(backend: PaddleVLBackend, fake_result: _FakeResult) -> None:
    fake_mod = _make_fake_paddleocr_module()
    prev = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_mod
    try:
        with (
            patch("agentic_pdf_parser.backends.paddle_vl.backend.subprocess.Popen",
                  return_value=_make_fake_popen()),
            patch("agentic_pdf_parser.backends.paddle_vl.backend.urlopen",
                  return_value=_make_healthy_response()),
            patch("pathlib.Path.exists", return_value=True),
        ):
            backend.load()
    finally:
        if prev is None:
            sys.modules.pop("paddleocr", None)
        else:
            sys.modules["paddleocr"] = prev
    backend._pipeline.predict = lambda *a, **kw: [fake_result]


def test_parse_page_returns_backend_result(page_input) -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)
    fake_result = _build_fake_result_from_canned(CANNED_FULL_PAGE)
    _load_with_fake_predict(backend, fake_result)

    result = backend.parse_page(page_input)
    assert isinstance(result, BackendPageResult)


def test_parse_page_page_is_normalized(page_input) -> None:
    from agentic_pdf_parser.schema import Page
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)
    fake_result = _build_fake_result_from_canned(CANNED_FULL_PAGE)
    _load_with_fake_predict(backend, fake_result)

    result = backend.parse_page(page_input)
    assert isinstance(result.page, Page)
    assert result.page.number == 1
    assert len(result.page.blocks) > 0


def test_parse_page_raw_is_dict(page_input) -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)
    fake_result = _build_fake_result_from_canned(CANNED_FULL_PAGE)
    _load_with_fake_predict(backend, fake_result)

    result = backend.parse_page(page_input)
    assert isinstance(result.raw, dict)
    assert "parsing_res_list" in result.raw


def test_parse_page_raw_is_json_serializable(page_input) -> None:
    import json
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)
    fake_result = _build_fake_result_from_canned(CANNED_FULL_PAGE)
    _load_with_fake_predict(backend, fake_result)

    result = backend.parse_page(page_input)
    json.dumps(result.raw)  # should not raise


# ---------------------------------------------------------------------------
# Tests: _result_to_raw serialization helper
# ---------------------------------------------------------------------------


def test_result_to_raw_structure() -> None:
    fake_result = _build_fake_result_from_canned(CANNED_FULL_PAGE)
    raw = _result_to_raw(fake_result)
    assert "page_index" in raw
    assert "width" in raw
    assert "height" in raw
    assert "parsing_res_list" in raw
    assert isinstance(raw["parsing_res_list"], list)


def test_result_to_raw_preserves_blocks() -> None:
    fake_result = _build_fake_result_from_canned(CANNED_FULL_PAGE)
    raw = _result_to_raw(fake_result)
    assert len(raw["parsing_res_list"]) == len(CANNED_FULL_PAGE["parsing_res_list"])


def test_result_to_raw_block_fields() -> None:
    fake_result = _build_fake_result_from_canned(CANNED_FULL_PAGE)
    raw = _result_to_raw(fake_result)
    for block in raw["parsing_res_list"]:
        assert "block_label" in block
        assert "block_content" in block
        assert "block_bbox" in block
        assert "block_id" in block


def test_result_to_raw_empty_list() -> None:
    fake = _FakeResult({"page_index": 0, "width": 100, "height": 100, "parsing_res_list": []})
    raw = _result_to_raw(fake)
    assert raw["parsing_res_list"] == []


# ---------------------------------------------------------------------------
# Tests: load fails gracefully without paddleocr
# ---------------------------------------------------------------------------


def test_load_raises_if_paddleocr_missing() -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)

    prev = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = None  # type: ignore[assignment]
    try:
        with (
            patch("agentic_pdf_parser.backends.paddle_vl.backend.subprocess.Popen",
                  return_value=_make_fake_popen()),
            patch("agentic_pdf_parser.backends.paddle_vl.backend.urlopen",
                  return_value=_make_healthy_response()),
            patch("pathlib.Path.exists", return_value=True),
        ):
            with pytest.raises((RuntimeError, ImportError)):
                backend.load()
    finally:
        if prev is None:
            sys.modules.pop("paddleocr", None)
        else:
            sys.modules["paddleocr"] = prev


# ---------------------------------------------------------------------------
# Tests: load fails if required paths missing
# ---------------------------------------------------------------------------


def test_load_raises_if_binary_missing() -> None:
    cfg = _make_config()
    backend = PaddleVLBackend(cfg)
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            backend.load()
