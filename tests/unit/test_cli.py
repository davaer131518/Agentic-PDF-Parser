"""Unit tests for cli.py (the parse-pdf Typer command).

Strategy
--------
- ``parse_pdf`` is patched in ``agentic_pdf_parser.cli`` so no real backend or
  PDF processing occurs.
- A minimal real-looking PDF-like file is created with ``tmp_path`` for tests
  that exercise file-existence checks.
- Tests cover: argument parsing, YAML config loading, CLI override precedence,
  validation errors, and missing-file error paths.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from agentic_pdf_parser.cli import app
from agentic_pdf_parser.config import DeviceChoice, ParseConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

runner = CliRunner()


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    """A file with a .pdf extension that exists on disk (content irrelevant)."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


@pytest.fixture
def fake_result(tmp_path: Path) -> MagicMock:
    """Minimal ParseResult mock returned by the patched parse_pdf."""
    result = MagicMock()
    result.json_path = tmp_path / "output" / "document.json"
    result.markdown_path = tmp_path / "output" / "document.md"
    result.raw_dir = None
    return result


def _write_yaml(tmp_path: Path, data: dict, name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Input file validation
# ---------------------------------------------------------------------------


def test_missing_input_pdf_gives_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["nonexistent.pdf", "--backend", "paddle_vl"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_help_flag_exits_cleanly() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "parse" in result.output.lower()


# ---------------------------------------------------------------------------
# Config building from CLI flags
# ---------------------------------------------------------------------------


def test_cli_backend_and_device(fake_pdf: Path, fake_result: MagicMock, tmp_path: Path) -> None:
    captured: list[ParseConfig] = []

    def _mock_parse_pdf(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock_parse_pdf):
        result = runner.invoke(
            app,
            [
                str(fake_pdf),
                "--backend", "paddle_vl",
                "--device", "cpu",
                "--out", str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.backend == "paddle_vl"
    assert cfg.device == DeviceChoice.CPU


def test_cli_paddle_vl_backend(fake_pdf: Path, fake_result: MagicMock, tmp_path: Path) -> None:
    captured: list[ParseConfig] = []

    def _mock(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock):
        result = runner.invoke(
            app,
            [str(fake_pdf), "--backend", "paddle_vl", "--out", str(tmp_path / "out")],
        )

    assert result.exit_code == 0, result.output
    assert captured[0].backend == "paddle_vl"


def test_cli_dpi_override(fake_pdf: Path, fake_result: MagicMock, tmp_path: Path) -> None:
    captured: list[ParseConfig] = []

    def _mock(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock):
        result = runner.invoke(
            app,
            [str(fake_pdf), "--backend", "paddle_vl", "--dpi", "300",
             "--out", str(tmp_path / "out")],
        )

    assert result.exit_code == 0
    assert captured[0].raster.dpi == 300


def test_cli_keep_raw_flag(fake_pdf: Path, fake_result: MagicMock, tmp_path: Path) -> None:
    captured: list[ParseConfig] = []

    def _mock(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock):
        result = runner.invoke(
            app,
            [str(fake_pdf), "--backend", "paddle_vl", "--keep-raw",
             "--out", str(tmp_path / "out")],
        )

    assert result.exit_code == 0
    assert captured[0].keep_raw is True


def test_cli_default_device_is_auto(fake_pdf: Path, fake_result: MagicMock, tmp_path: Path) -> None:
    captured: list[ParseConfig] = []

    def _mock(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock):
        runner.invoke(
            app,
            [str(fake_pdf), "--backend", "paddle_vl", "--out", str(tmp_path / "out")],
        )

    assert captured[0].device == DeviceChoice.AUTO


def test_cli_default_output_dir(fake_pdf: Path, fake_result: MagicMock) -> None:
    """When --out is omitted, output_dir defaults to Path('output')."""
    captured: list[ParseConfig] = []

    def _mock(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock):
        runner.invoke(app, [str(fake_pdf), "--backend", "paddle_vl"])

    assert captured[0].output_dir == Path("output")


# ---------------------------------------------------------------------------
# YAML config loading via --config
# ---------------------------------------------------------------------------


def test_cli_config_file_sets_backend(
    fake_pdf: Path, fake_result: MagicMock, tmp_path: Path
) -> None:
    yaml_path = _write_yaml(tmp_path, {"backend": "paddle_vl", "device": "cpu"})
    captured: list[ParseConfig] = []

    def _mock(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock):
        result = runner.invoke(
            app,
            [str(fake_pdf), "--config", str(yaml_path), "--out", str(tmp_path / "out")],
        )

    assert result.exit_code == 0, result.output
    assert captured[0].backend == "paddle_vl"
    assert captured[0].device == DeviceChoice.CPU


def test_cli_flag_overrides_yaml_backend(
    fake_pdf: Path, fake_result: MagicMock, tmp_path: Path
) -> None:
    yaml_path = _write_yaml(tmp_path, {"backend": "paddle_vl", "device": "gpu"})
    captured: list[ParseConfig] = []

    def _mock(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock):
        result = runner.invoke(
            app,
            [
                str(fake_pdf),
                "--config", str(yaml_path),
                "--backend", "paddle_vl",
                "--out", str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0
    assert captured[0].backend == "paddle_vl"
    assert captured[0].device == DeviceChoice.GPU  # from YAML, not overridden


def test_cli_flag_overrides_yaml_device(
    fake_pdf: Path, fake_result: MagicMock, tmp_path: Path
) -> None:
    yaml_path = _write_yaml(tmp_path, {"backend": "paddle_vl", "device": "gpu"})
    captured: list[ParseConfig] = []

    def _mock(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock):
        result = runner.invoke(
            app,
            [
                str(fake_pdf),
                "--config", str(yaml_path),
                "--device", "cpu",
                "--out", str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0
    assert captured[0].device == DeviceChoice.CPU


def test_cli_flag_overrides_yaml_dpi(
    fake_pdf: Path, fake_result: MagicMock, tmp_path: Path
) -> None:
    yaml_path = _write_yaml(
        tmp_path, {"backend": "paddle_vl", "raster": {"dpi": 150}}
    )
    captured: list[ParseConfig] = []

    def _mock(input_path, cfg, **kw):
        captured.append(cfg)
        return fake_result

    with patch("agentic_pdf_parser.api.parse_pdf", side_effect=_mock):
        result = runner.invoke(
            app,
            [
                str(fake_pdf),
                "--config", str(yaml_path),
                "--dpi", "300",
                "--out", str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0
    assert captured[0].raster.dpi == 300


def test_cli_missing_config_file_gives_error(fake_pdf: Path) -> None:
    result = runner.invoke(
        app,
        [str(fake_pdf), "--config", "no_such_file.yaml", "--backend", "paddle_vl"],
    )
    assert result.exit_code == 1
    combined = result.output
    assert "not found" in combined.lower()


# ---------------------------------------------------------------------------
# Validation error paths
# ---------------------------------------------------------------------------


def test_cli_missing_backend_gives_error(fake_pdf: Path, tmp_path: Path) -> None:
    """No --backend and no --config â†’ ValidationError with helpful hint."""
    result = runner.invoke(
        app, [str(fake_pdf), "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == 1
    combined = result.output
    # Should mention the problem and give a hint
    assert "backend" in combined.lower() or "configuration" in combined.lower()


def test_cli_invalid_backend_gives_error(fake_pdf: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [str(fake_pdf), "--backend", "unknown_model", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 1


def test_cli_invalid_device_gives_error(fake_pdf: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            str(fake_pdf),
            "--backend", "paddle_vl",
            "--device", "tpu",
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Output reporting
# ---------------------------------------------------------------------------


def test_cli_output_paths_printed(
    fake_pdf: Path, fake_result: MagicMock, tmp_path: Path
) -> None:
    """parse-pdf should print the JSON and MD paths on success."""
    fake_result.json_path = tmp_path / "out" / "document.json"
    fake_result.markdown_path = tmp_path / "out" / "document.md"
    fake_result.raw_dir = None

    with patch("agentic_pdf_parser.api.parse_pdf", return_value=fake_result):
        result = runner.invoke(
            app,
            [str(fake_pdf), "--backend", "paddle_vl", "--out", str(tmp_path / "out")],
        )

    assert result.exit_code == 0
    assert "document.json" in result.output
    assert "document.md" in result.output


def test_cli_raw_dir_printed_when_keep_raw(
    fake_pdf: Path, fake_result: MagicMock, tmp_path: Path
) -> None:
    fake_result.raw_dir = tmp_path / "out" / "raw"

    with patch("agentic_pdf_parser.api.parse_pdf", return_value=fake_result):
        result = runner.invoke(
            app,
            [
                str(fake_pdf),
                "--backend", "paddle_vl",
                "--keep-raw",
                "--out", str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0
    assert "raw" in result.output.lower()


# ---------------------------------------------------------------------------
# Runtime error from backend (missing dependencies)
# ---------------------------------------------------------------------------


def test_cli_runtime_error_shows_hint(fake_pdf: Path, tmp_path: Path) -> None:
    """If parse_pdf raises RuntimeError, the CLI should show it and exit 1."""
    with patch(
        "agentic_pdf_parser.api.parse_pdf",
        side_effect=RuntimeError("torch not found"),
    ):
        result = runner.invoke(
            app,
            [str(fake_pdf), "--backend", "paddle_vl", "--out", str(tmp_path / "out")],
        )

    assert result.exit_code == 1
    combined = result.output
    assert "torch not found" in combined or "parse failed" in combined.lower()
