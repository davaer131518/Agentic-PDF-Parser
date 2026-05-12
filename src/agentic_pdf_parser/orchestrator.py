"""Pipeline orchestrator — coordinates rasterization, backend calls, and export.

Orchestrator responsibilities (in order):
1. Hash source file and read PDF metadata.
2. For each page: rasterize, call backend.parse_page(), optionally save raw output,
   extract/crop figure assets, aggregate normalized Page.
3. Build NormalizedDocument from collected Pages.
4. Export canonical JSON and Markdown.

The orchestrator never normalizes. It only consumes the already-normalized
``Page`` inside ``BackendPageResult``.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .assets import extract_figures
from .backends.base import BackendPageResult, PageInput, ParserBackend
from .config import ParseConfig
from .export import json_export, markdown_export
from .rasterize import iter_pages, read_metadata
from .schema import (
    BackendMetadata,
    DocumentMetadata,
    NormalizedDocument,
    Page,
    parse_pdf_date,
)
from .utils.hashing import sha256_file
from .utils.logging import get_logger

logger = get_logger(__name__)


def run(
    pdf_path: Path,
    config: ParseConfig,
    backend: ParserBackend,
) -> "ParseResult":
    """Execute the full parsing pipeline and return a :class:`ParseResult`.

    Parameters
    ----------
    pdf_path:
        Path to the input PDF file.
    config:
        Pipeline configuration (device, DPI, keep_raw, output directory …).
    backend:
        An already-constructed backend instance.  *Not* loaded yet; this
        function calls ``backend.load()`` and ``backend.unload()``.
    """
    from .api import ParseResult  # local import avoids circular dependency

    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    raw_dir: Path | None = (out_dir / "raw") if config.keep_raw else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
    debug_dir: Path | None = (out_dir / "debug") if config.debug else None

    # --- Document-level metadata -------------------------------------------
    source_hash = sha256_file(pdf_path)
    fitz_meta, num_pages = read_metadata(pdf_path)
    doc_meta = DocumentMetadata(
        source_filename=pdf_path.name,
        source_sha256=source_hash,
        num_pages=num_pages,
        title=fitz_meta.get("title") or None,
        author=fitz_meta.get("author") or None,
        created_at=parse_pdf_date(fitz_meta.get("creationDate")),
    )

    # --- Per-page loop -------------------------------------------------------
    backend.load()
    pages: list[Page] = []

    try:
        with tempfile.TemporaryDirectory() as _tmpdir:
            tmp_path = Path(_tmpdir)
            for page_image, dims, tmp_img_path in iter_pages(
                pdf_path, config.raster.dpi, tmp_path
            ):
                page_index = len(pages)
                logger.debug("Processing page %d / %d", page_index + 1, num_pages)

                page_input = PageInput(
                    page_index=page_index,
                    page_number=page_index + 1,
                    dimensions=dims,
                    image=page_image,
                    temp_image_path=tmp_img_path,
                    pdf_path=pdf_path,
                )

                # backend owns inference AND normalization
                result: BackendPageResult = backend.parse_page(page_input)

                if raw_dir is not None:
                    raw_file = raw_dir / f"page_{page_index + 1:04}.json"
                    raw_file.write_text(
                        json.dumps(result.raw, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                extract_figures(page_image, result.page, assets_dir)

                if debug_dir is not None:
                    from .debug_viz import save_debug_page  # noqa: PLC0415
                    save_debug_page(page_image, result.page, debug_dir)
                    logger.debug("Debug bbox image saved for page %d", page_index + 1)

                pages.append(result.page)
    finally:
        backend.unload()

    # --- Assemble NormalizedDocument ----------------------------------------
    normalized_doc = NormalizedDocument(
        document=doc_meta,
        backend=BackendMetadata(
            name=backend.name,
            version=backend.version,
            model_id=backend.model_id,
            device=backend.resolved_device,
        ),
        pages=pages,
    )

    # --- Export canonical outputs -------------------------------------------
    json_path = out_dir / "document.json"
    md_path = out_dir / "document.md"
    json_export.write(normalized_doc, json_path)
    markdown_export.write(normalized_doc, md_path)

    logger.info(
        "Parsed %d page(s) → %s | %s",
        num_pages,
        json_path,
        md_path,
    )

    return ParseResult(
        document=normalized_doc,
        output_dir=out_dir,
        json_path=json_path,
        markdown_path=md_path,
        raw_dir=raw_dir,
    )
