"""CLI entrypoint for the agentic-pdf-parser pipeline.

Usage
-----
::

    # Minimal — backend required:
    parse-pdf input.pdf --backend paddle_vl --out output/

    # With YAML config (backend, device, dpi already set):
    parse-pdf input.pdf --config configs/default.yaml --out output/

    # CLI flags always override YAML values:
    parse-pdf input.pdf --config configs/default.yaml --backend paddle_vl --device cpu

    # Save raw per-page backend output:
    parse-pdf input.pdf --backend paddle_vl --out output/ --keep-raw

Design
------
The CLI is intentionally thin:

1. Parse flags.
2. Build a :class:`~agentic_pdf_parser.config.ParseConfig` via
   :func:`~agentic_pdf_parser.config_loader.build_config`.
3. Call :func:`~agentic_pdf_parser.api.parse_pdf` (the existing Python API).
4. Print output paths.

No orchestration logic lives here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich.console import Console

from .config_loader import build_config

app = typer.Typer(
    name="parse-pdf",
    help="Parse a PDF into canonical JSON and Markdown.",
    add_completion=False,
    no_args_is_help=True,
)

_err = Console(stderr=True, highlight=False)


@app.command()
def main(
    input_pdf: Path = typer.Argument(
        ...,
        metavar="INPUT_PDF",
        help="Path to the PDF file to parse.",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        metavar="NAME",
        help="Backend to use: paddle_vl.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        metavar="DEVICE",
        help="Compute device: cpu | gpu | auto.  Default: auto.",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        metavar="DIR",
        help="Output directory.  Default: ./output (or value from --config).",
    ),
    keep_raw: bool = typer.Option(
        False,
        "--keep-raw/--no-keep-raw",
        help="Save raw per-page backend output under <out>/raw/.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Save bbox-annotated page images to <out>/debug/ for visual inspection.",
    ),
    dpi: Optional[int] = typer.Option(
        None,
        "--dpi",
        metavar="INT",
        help="Rasterization DPI.  Default: 200 (or value from --config).",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        metavar="YAML",
        help="Path to a YAML config file.  CLI flags override YAML values.",
    ),
) -> None:
    """Parse INPUT_PDF and write canonical JSON + Markdown to the output dir."""
    # --- Validate input file ------------------------------------------------
    if not input_pdf.exists():
        _err.print(f"[red]Error:[/red] input file not found: {input_pdf}")
        raise typer.Exit(1)
    if not input_pdf.is_file():
        _err.print(f"[red]Error:[/red] not a file: {input_pdf}")
        raise typer.Exit(1)

    # --- Validate config file (if provided) ---------------------------------
    if config_file is not None and not config_file.exists():
        _err.print(f"[red]Error:[/red] config file not found: {config_file}")
        raise typer.Exit(1)

    # --- Build config (YAML base + CLI overrides) ---------------------------
    try:
        cfg = build_config(
            yaml_path=config_file,
            backend=backend,
            device=device,
            output_dir=out,
            keep_raw=keep_raw,
            debug=debug,
            dpi=dpi,
        )
    except ValidationError as exc:
        _err.print("[red]Configuration error:[/red]")
        for err in exc.errors():
            field = " → ".join(str(loc) for loc in err["loc"])
            msg = err["msg"]
            _err.print(f"  {field}: {msg}")
        _err.print(
            "\nHint: provide [bold]--backend[/bold] (paddle_vl) "
            "or set it in a [bold]--config[/bold] YAML file."
        )
        raise typer.Exit(1)
    except (ValueError, OSError) as exc:
        _err.print(f"[red]Config load error:[/red] {exc}")
        raise typer.Exit(1)

    # --- Run pipeline -------------------------------------------------------
    from .api import parse_pdf  # deferred to keep startup fast  # noqa: PLC0415

    typer.echo(
        f"Parsing {input_pdf.name} "
        f"[backend={cfg.backend}, device={cfg.device}, dpi={cfg.raster.dpi}"
        + (", debug=on" if cfg.debug else "")
        + "]"
    )

    try:
        result = parse_pdf(input_pdf, cfg)
    except RuntimeError as exc:
        import traceback  # noqa: PLC0415
        _err.print(f"[red]Parse failed:[/red] {exc}")
        if exc.__cause__:
            _err.print(traceback.format_exc())
        _hint = _backend_hint(cfg.backend)
        if _hint:
            _err.print(_hint)
        raise typer.Exit(1)
    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        _err.print(f"[red]Unexpected error:[/red] {exc}")
        _err.print(traceback.format_exc())
        raise typer.Exit(1)

    # --- Report results -----------------------------------------------------
    typer.echo(f"JSON  -> {result.json_path}")
    typer.echo(f"MD    -> {result.markdown_path}")
    if result.raw_dir is not None:
        typer.echo(f"Raw   -> {result.raw_dir}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _backend_hint(backend: str) -> str:
    """Return an install hint for missing backend dependencies."""
    hints = {
        "paddle_vl": (
            "Hint: install PaddleVL dependencies with\n"
            "  pip install paddlepaddle  # CPU\n"
            "  # or: pip install paddlepaddle-gpu  # GPU"
        ),
    }
    return hints.get(str(backend), "")
