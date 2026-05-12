"""Example: parse a PDF using the Python API.

Two usage patterns are shown:
  A) Programmatic config — build ParseConfig directly in code.
  B) YAML-based config   — load from a config file, optionally override fields.

Run from the project root::

    python examples/parse_example.py                         # uses default.yaml
    python examples/parse_example.py --programmatic          # builds config in code
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the src layout is importable when run directly (no install).
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from agentic_pdf_parser import parse_pdf
from agentic_pdf_parser.config import BackendName, DeviceChoice, ParseConfig
from agentic_pdf_parser.config_loader import build_config, load_config
from agentic_pdf_parser.schema import NormalizedDocument


def run_programmatic(input_pdf: Path) -> None:
    """Parse using a config built entirely in Python."""
    config = ParseConfig(
        backend=BackendName.PADDLE_VL,
        device=DeviceChoice.AUTO,
        output_dir=Path("output"),
        keep_raw=False,
    )
    _run(input_pdf, config, label="programmatic")


def run_from_yaml(input_pdf: Path, yaml_path: Path) -> None:
    """Parse using a YAML config file (no CLI overrides)."""
    config = load_config(yaml_path)
    _run(input_pdf, config, label=f"yaml:{yaml_path.name}")


def run_with_overrides(input_pdf: Path, yaml_path: Path) -> None:
    """Parse using a YAML base config with selective CLI-style overrides.

    This mirrors what the ``parse-pdf`` CLI does internally.
    """
    config = build_config(
        yaml_path=yaml_path,
        # Override only what you need:
        device="cpu",
        keep_raw=True,
        dpi=150,
        output_dir=Path("output_override"),
    )
    _run(input_pdf, config, label="yaml+overrides")


def _run(input_pdf: Path, config: ParseConfig, label: str) -> None:
    print(
        f"\n[{label}] Parsing {input_pdf.name} "
        f"(backend={config.backend}, device={config.device}, "
        f"dpi={config.raster.dpi}, out={config.output_dir})"
    )

    result = parse_pdf(input_pdf, config)

    print(f"  JSON  → {result.json_path}")
    print(f"  MD    → {result.markdown_path}")
    if result.raw_dir is not None:
        print(f"  Raw   → {result.raw_dir}")

    doc: NormalizedDocument = result.document
    print(f"  Pages : {len(doc.pages)}")
    for page in doc.pages:
        print(f"    Page {page.number}: {len(page.blocks)} blocks")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    input_pdf = Path("sample.pdf")
    if not input_pdf.exists():
        print(f"Place a PDF at '{input_pdf}' and re-run.")
        print("  e.g.  cp /path/to/document.pdf sample.pdf")
        return

    yaml_path = _ROOT / "configs" / "default.yaml"

    if "--programmatic" in sys.argv:
        run_programmatic(input_pdf)
    else:
        run_from_yaml(input_pdf, yaml_path)


if __name__ == "__main__":
    main()
