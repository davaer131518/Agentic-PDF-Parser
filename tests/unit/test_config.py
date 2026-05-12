"""Tests for config.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentic_pdf_parser.config import (
    BackendName,
    DeviceChoice,
    PaddleVLConfig,
    ParseConfig,
    RasterConfig,
)


def test_backend_name_values() -> None:
    assert BackendName.PADDLE_VL == "paddle_vl"


def test_device_choice_values() -> None:
    assert DeviceChoice.CPU == "cpu"
    assert DeviceChoice.GPU == "gpu"
    assert DeviceChoice.AUTO == "auto"


def test_parse_config_defaults(tmp_path: Path) -> None:
    cfg = ParseConfig(backend=BackendName.PADDLE_VL, output_dir=tmp_path)
    assert cfg.device == DeviceChoice.AUTO
    assert cfg.keep_raw is False
    assert cfg.raster.dpi == 200


def test_raster_config_default_dpi() -> None:
    assert RasterConfig().dpi == 200


def test_paddle_vl_config_defaults() -> None:
    p = PaddleVLConfig()
    assert p.use_doc_orientation_classify is False
    assert p.use_doc_unwarping is False


def test_parse_config_backend_enum_validation(tmp_path: Path) -> None:
    cfg = ParseConfig(backend="paddle_vl", output_dir=tmp_path)  # type: ignore[arg-type]
    assert cfg.backend == BackendName.PADDLE_VL


def test_parse_config_invalid_backend(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        ParseConfig(backend="nonexistent", output_dir=tmp_path)  # type: ignore[arg-type]


def test_parse_config_keep_raw(tmp_path: Path) -> None:
    cfg = ParseConfig(backend=BackendName.PADDLE_VL, output_dir=tmp_path, keep_raw=True)
    assert cfg.keep_raw is True


def test_parse_config_output_dir_as_string(tmp_path: Path) -> None:
    cfg = ParseConfig(backend=BackendName.PADDLE_VL, output_dir=str(tmp_path))
    assert isinstance(cfg.output_dir, Path)


def test_parse_config_nested_sub_config_override(tmp_path: Path) -> None:
    cfg = ParseConfig(
        backend=BackendName.PADDLE_VL,
        output_dir=tmp_path,
        paddle_vl=PaddleVLConfig(
            gguf_model_path=tmp_path / "custom.gguf",
            server_port=9090,
        ),
    )
    assert cfg.paddle_vl.gguf_model_path == tmp_path / "custom.gguf"
    assert cfg.paddle_vl.server_port == 9090
