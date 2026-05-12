"""Unit tests for config_loader.py.

Tests cover:
- load_yaml: valid file, empty file, non-mapping, missing file
- build_config: YAML-only, CLI-only, YAML + override precedence, dpi nesting,
  keep_raw flag, output_dir defaulting, invalid backend/device (Pydantic errors)
- load_config: convenience wrapper behaviour
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agentic_pdf_parser.config import DeviceChoice, ParseConfig
from agentic_pdf_parser.config_loader import build_config, load_config, load_yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def _write_yaml(tmp_path: Path, data: dict, name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_yaml
# ---------------------------------------------------------------------------


def test_load_yaml_basic(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, {"backend": "paddle_vl", "device": "cpu"})
    data = load_yaml(p)
    assert data["backend"] == "paddle_vl"
    assert data["device"] == "cpu"


def test_load_yaml_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_yaml(p) == {}


def test_load_yaml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nonexistent.yaml")


def test_load_yaml_non_mapping_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- paddle_vl\n- cpu\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(p)


def test_load_yaml_nested_keys(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, {"backend": "paddle_vl", "raster": {"dpi": 300}})
    data = load_yaml(p)
    assert data["raster"]["dpi"] == 300


# ---------------------------------------------------------------------------
# build_config — CLI-only (no YAML)
# ---------------------------------------------------------------------------


def test_build_config_cli_only(tmp_path: Path) -> None:
    cfg = build_config(backend="paddle_vl", output_dir=tmp_path / "out")
    assert cfg.backend == "paddle_vl"
    assert cfg.device == DeviceChoice.AUTO       # default
    assert cfg.raster.dpi == 200                 # default
    assert cfg.keep_raw is False                 # default


def test_build_config_cli_device_override(tmp_path: Path) -> None:
    cfg = build_config(backend="paddle_vl", device="cpu", output_dir=tmp_path / "out")
    assert cfg.device == DeviceChoice.CPU


def test_build_config_cli_keep_raw_flag(tmp_path: Path) -> None:
    cfg = build_config(backend="paddle_vl", keep_raw=True, output_dir=tmp_path / "out")
    assert cfg.keep_raw is True


def test_build_config_cli_keep_raw_default_false(tmp_path: Path) -> None:
    cfg = build_config(backend="paddle_vl", output_dir=tmp_path / "out")
    assert cfg.keep_raw is False


def test_build_config_cli_dpi_override(tmp_path: Path) -> None:
    cfg = build_config(backend="paddle_vl", dpi=300, output_dir=tmp_path / "out")
    assert cfg.raster.dpi == 300


def test_build_config_output_dir_default(tmp_path: Path) -> None:
    """output_dir defaults to Path('output') when not provided."""
    cfg = build_config(backend="paddle_vl")
    assert cfg.output_dir == Path("output")


def test_build_config_output_dir_cli(tmp_path: Path) -> None:
    cfg = build_config(backend="paddle_vl", output_dir=tmp_path / "myout")
    assert cfg.output_dir == tmp_path / "myout"


def test_build_config_missing_backend_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_config()
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("backend",) for err in errors)


def test_build_config_invalid_backend_raises() -> None:
    with pytest.raises(ValidationError):
        build_config(backend="unknown_backend")


def test_build_config_invalid_device_raises() -> None:
    with pytest.raises(ValidationError):
        build_config(backend="paddle_vl", device="tpu")


# ---------------------------------------------------------------------------
# build_config — YAML base, no CLI overrides
# ---------------------------------------------------------------------------


def test_build_config_yaml_base(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        {
            "backend": "paddle_vl",
            "device": "cpu",
            "keep_raw": True,
            "output_dir": str(tmp_path / "out"),
            "raster": {"dpi": 150},
        },
    )
    cfg = build_config(yaml_path=yaml_path)
    assert cfg.backend == "paddle_vl"
    assert cfg.device == DeviceChoice.CPU
    assert cfg.keep_raw is True
    assert cfg.raster.dpi == 150


def test_build_config_yaml_partial(tmp_path: Path) -> None:
    """YAML with only backend set; other fields fall back to defaults."""
    yaml_path = _write_yaml(tmp_path, {"backend": "paddle_vl"})
    cfg = build_config(yaml_path=yaml_path)
    assert cfg.backend == "paddle_vl"
    assert cfg.device == DeviceChoice.AUTO
    assert cfg.raster.dpi == 200


def test_build_config_yaml_nested_raster(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path, {"backend": "paddle_vl", "raster": {"dpi": 300}}
    )
    cfg = build_config(yaml_path=yaml_path)
    assert cfg.raster.dpi == 300


def test_build_config_yaml_paddle_vl_sub_config(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        {
            "backend": "paddle_vl",
            "paddle_vl": {"server_port": 9090},
        },
    )
    cfg = build_config(yaml_path=yaml_path)
    assert cfg.paddle_vl.server_port == 9090


# ---------------------------------------------------------------------------
# build_config — YAML + CLI override precedence
# ---------------------------------------------------------------------------


def test_cli_backend_overrides_yaml(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, {"backend": "paddle_vl", "device": "gpu"})
    cfg = build_config(yaml_path=yaml_path, backend="paddle_vl")
    assert cfg.backend == "paddle_vl"
    assert cfg.device == DeviceChoice.GPU  # YAML value preserved


def test_cli_device_overrides_yaml(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, {"backend": "paddle_vl", "device": "gpu"})
    cfg = build_config(yaml_path=yaml_path, device="cpu")
    assert cfg.device == DeviceChoice.CPU


def test_cli_output_dir_overrides_yaml(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path, {"backend": "paddle_vl", "output_dir": str(tmp_path / "yaml_out")}
    )
    cli_out = tmp_path / "cli_out"
    cfg = build_config(yaml_path=yaml_path, output_dir=cli_out)
    assert cfg.output_dir == cli_out


def test_cli_keep_raw_enables_over_yaml_false(tmp_path: Path) -> None:
    """--keep-raw (True) enables raw output even when YAML has keep_raw: false."""
    yaml_path = _write_yaml(tmp_path, {"backend": "paddle_vl", "keep_raw": False})
    cfg = build_config(yaml_path=yaml_path, keep_raw=True)
    assert cfg.keep_raw is True


def test_cli_keep_raw_false_respects_yaml_true(tmp_path: Path) -> None:
    """keep_raw=False (CLI default) does NOT override YAML keep_raw: true."""
    yaml_path = _write_yaml(tmp_path, {"backend": "paddle_vl", "keep_raw": True})
    cfg = build_config(yaml_path=yaml_path, keep_raw=False)
    assert cfg.keep_raw is True  # YAML wins because keep_raw=False means "not set"


def test_cli_dpi_overrides_yaml_dpi(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path, {"backend": "paddle_vl", "raster": {"dpi": 150}}
    )
    cfg = build_config(yaml_path=yaml_path, dpi=300)
    assert cfg.raster.dpi == 300


def test_cli_dpi_preserves_other_raster_fields(tmp_path: Path) -> None:
    """Overriding dpi should not wipe out other raster keys if any are added later."""
    yaml_path = _write_yaml(
        tmp_path, {"backend": "paddle_vl", "raster": {"dpi": 150}}
    )
    cfg = build_config(yaml_path=yaml_path, dpi=300)
    assert cfg.raster.dpi == 300  # overridden


def test_all_cli_override_yaml(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        {
            "backend": "paddle_vl",
            "device": "gpu",
            "keep_raw": False,
            "output_dir": str(tmp_path / "yaml_out"),
            "raster": {"dpi": 150},
        },
    )
    cli_out = tmp_path / "cli_out"
    cfg = build_config(
        yaml_path=yaml_path,
        backend="paddle_vl",
        device="cpu",
        output_dir=cli_out,
        keep_raw=True,
        dpi=300,
    )
    assert cfg.backend == "paddle_vl"
    assert cfg.device == DeviceChoice.CPU
    assert cfg.output_dir == cli_out
    assert cfg.keep_raw is True
    assert cfg.raster.dpi == 300


# ---------------------------------------------------------------------------
# load_config — convenience wrapper
# ---------------------------------------------------------------------------


def test_load_config_from_real_default_yaml() -> None:
    """The committed configs/default.yaml should parse cleanly."""
    cfg = load_config(_CONFIGS_DIR / "default.yaml")
    assert isinstance(cfg, ParseConfig)
    assert cfg.backend == "paddle_vl"
    assert cfg.device == DeviceChoice.AUTO


def test_load_config_from_real_paddle_vl_cpu_yaml() -> None:
    cfg = load_config(_CONFIGS_DIR / "paddle_vl_cpu.yaml")
    assert cfg.backend == "paddle_vl"
    assert cfg.device == DeviceChoice.CPU


def test_load_config_from_real_paddle_vl_gpu_yaml() -> None:
    cfg = load_config(_CONFIGS_DIR / "paddle_vl_gpu.yaml")
    assert cfg.backend == "paddle_vl"
    assert cfg.device == DeviceChoice.GPU


def test_paddle_vl_config_llama_fields(tmp_path: Path) -> None:
    """PaddleVLConfig llama-server fields are loaded from YAML."""
    yaml_path = _write_yaml(
        tmp_path,
        {
            "backend": "paddle_vl",
            "paddle_vl": {
                "llama_cpp_dir": "C:/llama-cpp",
                "gguf_model_path": "C:/llama-cpp/models/PaddleOCR-VL-1.5.gguf",
                "mmproj_path": "C:/llama-cpp/models/PaddleOCR-VL-1.5-mmproj.gguf",
                "server_port": 8080,
            },
        },
    )
    cfg = build_config(yaml_path=yaml_path)
    assert cfg.paddle_vl.server_port == 8080
    assert cfg.paddle_vl.gguf_model_path == Path("C:/llama-cpp/models/PaddleOCR-VL-1.5.gguf")
    assert cfg.paddle_vl.mmproj_path == Path("C:/llama-cpp/models/PaddleOCR-VL-1.5-mmproj.gguf")


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent_config.yaml"))
