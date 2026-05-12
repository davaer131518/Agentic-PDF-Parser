"""YAML-based configuration loading with CLI override support.

Precedence (highest → lowest):
  1. Explicit CLI flag values
  2. YAML file values (when ``--config`` is provided)
  3. Pydantic defaults defined in ``ParseConfig``

The only field that has no Pydantic default and is therefore **required** is
``backend``.  Every other field falls back to its model default if absent from
both YAML and CLI.

``output_dir`` is special: it has no Pydantic default but defaults to
``Path("output")`` here so callers do not have to provide it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .config import ParseConfig


# ---------------------------------------------------------------------------
# Low-level YAML reader
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a plain dict.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is valid YAML but not a mapping at the top level.
    """
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML config must be a mapping at the top level, "
            f"got {type(data).__name__!r} in {path}"
        )
    return data


# ---------------------------------------------------------------------------
# Config builder — YAML base + CLI overrides
# ---------------------------------------------------------------------------


def build_config(
    *,
    yaml_path: Path | None = None,
    backend: str | None = None,
    device: str | None = None,
    output_dir: Path | None = None,
    keep_raw: bool = False,
    debug: bool = False,
    dpi: int | None = None,
) -> ParseConfig:
    """Build a :class:`~agentic_pdf_parser.config.ParseConfig`.

    Parameters
    ----------
    yaml_path:
        Optional path to a YAML config file.  Its contents form the base
        configuration layer.
    backend:
        CLI override for ``config.backend`` (e.g. ``"paddle_vl"``).
    device:
        CLI override for ``config.device`` (e.g. ``"cpu"``).
    output_dir:
        CLI override for ``config.output_dir``.
    keep_raw:
        If ``True``, override config to enable raw output saving.
        ``False`` means "not specified by CLI" — the YAML value is used.
    debug:
        If ``True``, override config to enable bbox debug images.
        ``False`` means "not specified by CLI" — the YAML value is used.
    dpi:
        CLI override for ``config.raster.dpi``.

    Returns
    -------
    ParseConfig
        Validated configuration ready for use by the pipeline.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* is provided but the file does not exist.
    ValueError
        If the YAML content is not a top-level mapping.
    pydantic.ValidationError
        If the merged config is invalid (e.g. unknown backend, bad device).
    """
    raw: dict[str, Any] = {}

    # --- Layer 1: YAML base ---
    if yaml_path is not None:
        raw = load_yaml(yaml_path)

    # --- Layer 2: CLI overrides (only when explicitly provided) ---
    if backend is not None:
        raw["backend"] = backend
    if device is not None:
        raw["device"] = device
    if output_dir is not None:
        raw["output_dir"] = output_dir
    if keep_raw:
        # Only set to True; never force to False (YAML can still enable it).
        raw["keep_raw"] = True
    if debug:
        raw["debug"] = True
    if dpi is not None:
        # dpi is nested under raster; preserve any other raster fields from YAML.
        raster = raw.get("raster")
        if not isinstance(raster, dict):
            raster = {}
        raster["dpi"] = dpi
        raw["raster"] = raster

    # --- Layer 3: Pydantic defaults (applied implicitly via model_validate) ---
    # output_dir has no Pydantic default; supply one here so it is not required.
    raw.setdefault("output_dir", Path("output"))

    return ParseConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Convenience: load directly from a YAML path (no CLI overrides)
# ---------------------------------------------------------------------------


def load_config(yaml_path: Path) -> ParseConfig:
    """Load a :class:`ParseConfig` directly from a YAML file.

    Equivalent to ``build_config(yaml_path=yaml_path)`` with no CLI overrides.
    """
    return build_config(yaml_path=yaml_path)
