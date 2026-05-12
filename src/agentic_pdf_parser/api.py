"""Public Python API entrypoint."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ParseConfig
from .schema import NormalizedDocument


@dataclass
class ParseResult:
    """Returned by :func:`parse_pdf`."""

    document: NormalizedDocument
    output_dir: Path
    json_path: Path
    markdown_path: Path
    raw_dir: Path | None  # set only when ParseConfig.keep_raw is True


def parse_pdf(
    input_path: str | Path,
    config: ParseConfig,
    *,
    _backend: object | None = None,
) -> ParseResult:
    """Parse a PDF and return a :class:`ParseResult`.

    Parameters
    ----------
    input_path:
        Path to the source PDF.
    config:
        Full pipeline configuration including backend, device, and output dir.
    _backend:
        For testing only: inject a pre-built backend instance, bypassing
        :func:`~agentic_pdf_parser.backends.registry.build_backend`.
    """
    from .backends.base import ParserBackend
    from .backends.registry import build_backend
    from .orchestrator import run

    pdf_path = Path(input_path)
    backend: ParserBackend = (
        _backend  # type: ignore[assignment]
        if isinstance(_backend, ParserBackend)
        else build_backend(config)
    )
    return run(pdf_path=pdf_path, config=config, backend=backend)
