"""PDF page rasterization using PyMuPDF."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterator

import fitz  # PyMuPDF
from PIL import Image

from .schema import PageDimensions


def read_metadata(pdf_path: Path) -> tuple[dict[str, Any], int]:
    """Open the PDF, extract metadata dict and page count, then close.

    Returns:
        (metadata_dict, page_count) where metadata_dict has keys like
        ``title``, ``author``, ``creationDate`` (raw PDF date strings).
    """
    doc = fitz.open(str(pdf_path))
    meta: dict[str, Any] = dict(doc.metadata)
    count = len(doc)
    doc.close()
    return meta, count


def iter_pages(
    pdf_path: Path,
    dpi: int,
    work_dir: Path,
) -> Iterator[tuple[Image.Image, PageDimensions, Path]]:
    """Yield ``(PIL image, PageDimensions, temp PNG path)`` for every page.

    PNG files are written into *work_dir*; the caller is responsible for
    cleaning up the directory when processing is complete.
    """
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(len(doc)):
            page = doc[i]
            rect = page.rect
            pix = page.get_pixmap(matrix=matrix)

            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            dims = PageDimensions(
                width_pt=float(rect.width),
                height_pt=float(rect.height),
                width_px=pix.width,
                height_px=pix.height,
                dpi=dpi,
            )

            tmp_path = work_dir / f"page_{i:04}.png"
            image.save(tmp_path, format="PNG")

            yield image, dims, tmp_path
    finally:
        doc.close()
