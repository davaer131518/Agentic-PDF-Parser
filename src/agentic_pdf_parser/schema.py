"""Canonical document schema — the source of truth for all pipeline outputs."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Coordinates in PDF points (72 pt = 1 inch), top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float


# ---------------------------------------------------------------------------
# Provenance — links a block back to its backend source
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    backend: str
    backend_version: str | None = None
    bbox: BoundingBox | None = None
    polygon: list[tuple[float, float]] | None = None
    confidence: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Block-level content types
# ---------------------------------------------------------------------------

BlockType = Literal[
    "heading",
    "paragraph",
    "list",
    "list_item",
    "caption",
    "figure",
    "table",
    "formula",
    "code",
    "footnote",
    "page_header",
    "page_footer",
    "section",
]


class TableCell(BaseModel):
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str
    is_header: bool = False
    bbox: BoundingBox | None = None


class Table(BaseModel):
    rows: int
    cols: int
    cells: list[TableCell]
    html: str | None = None   # always set; exporter fallback for complex tables
    otsl: str | None = None   # reserved for OTSL structured table representation
    caption_id: str | None = None


class Figure(BaseModel):
    asset_path: str            # relative to output_dir, e.g. "assets/p0001_fig01.png"
    caption_id: str | None = None
    alt_text: str | None = None


class Formula(BaseModel):
    latex: str | None = None
    mathml: str | None = None
    inline: bool = False


class Block(BaseModel):
    id: str                    # e.g. "p0001_b0003"
    type: BlockType
    text: str | None = None
    level: int | None = None   # heading depth (1–6) or list nesting level
    reading_order: int
    provenance: Provenance
    table: Table | None = None
    figure: Figure | None = None
    formula: Formula | None = None
    children: list[Block] | None = None  # for list/section trees


# Required for self-referential model in Pydantic v2
Block.model_rebuild()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


class PageDimensions(BaseModel):
    width_pt: float
    height_pt: float
    width_px: int
    height_px: int
    dpi: int


class Page(BaseModel):
    index: int                         # 0-based
    number: int                        # 1-based
    dimensions: PageDimensions
    blocks: list[Block]                # ordered by reading_order
    backend_metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Document-level metadata
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    source_filename: str
    source_sha256: str
    num_pages: int
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None


class BackendMetadata(BaseModel):
    name: str
    version: str | None = None
    model_id: str | None = None
    device: str                        # resolved string, e.g. "cuda:0"
    options: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level canonical document
# ---------------------------------------------------------------------------


class NormalizedDocument(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    document: DocumentMetadata
    backend: BackendMetadata
    pages: list[Page]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")


def parse_pdf_date(date_str: str | None) -> datetime | None:
    """Parse a PDF date string (``D:YYYYMMDDHHmmSS...``) into a UTC datetime."""
    if not date_str:
        return None
    m = _PDF_DATE_RE.match(date_str)
    if not m:
        return None
    year, month, day, hour, minute, second = (int(x) for x in m.groups())
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None
