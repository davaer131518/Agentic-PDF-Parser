"""PaddleVL normalizer: raw Paddle result dict → canonical Page.

This module is an **internal implementation detail** of the PaddleVL backend.
Nothing outside ``backends/paddle_vl/`` should import from here.

Input contract (raw dict)
-------------------------
The normalizer consumes the dict produced by ``_result_to_raw()`` in
``backend.py``.  Its top-level structure is::

    {
        "page_index": int,
        "width": int,          # page pixel width as seen by the model
        "height": int,         # page pixel height as seen by the model
        "parsing_res_list": [
            {
                "block_label": str,
                "block_content": str,
                "block_bbox": [x1, y1, x2, y2],  # pixels, TOPLEFT origin
                "block_id": int,
            },
            ...
        ],
    }

The ``parsing_res_list`` is already in reading order as determined by
PP-DocLayoutV3 layout detection inside PaddleOCR-VL 1.5.

Coordinate system
-----------------
``block_bbox`` contains pixel coordinates from the PaddleOCR-VL pipeline's
layout detector.  The origin is top-left.  We convert to PDF points using
the page dimensions stored in ``PageInput``.  The result width/height
reported by the Paddle result *may* differ from ``page_input.dimensions``
because Paddle can internally rescale images; we therefore use
``page_input.dimensions`` as the ground truth and compute a scaling factor.

Table HTML
----------
Table blocks carry their full HTML as ``block_content``.  We parse it with
``BeautifulSoup`` + ``lxml`` to extract cells, preserving rowspan/colspan
and header semantics.

Formula LaTeX
-------------
Formula blocks carry LaTeX in ``block_content``, typically wrapped in
``$$...$$`` or ``$...$``.  Delimiters are stripped before storing in
:class:`~agentic_pdf_parser.schema.Formula`.
"""
from __future__ import annotations

from typing import Any

from ...backends.base import PageInput
from ...schema import (
    Block,
    BoundingBox,
    Figure,
    Formula,
    Page,
    Provenance,
    Table,
    TableCell as SchemaTableCell,
)

# ---------------------------------------------------------------------------
# Block-label vocabulary (verified from paddleocr 3.4.1 result.py)
# ---------------------------------------------------------------------------

# Maps Paddle label → (our BlockType, heading_level | None)
# Heading level None means the type carries no level.
_LABEL_MAP: dict[str, tuple[str, int | None]] = {
    # --- Headings ---
    "doc_title": ("heading", 1),
    "paragraph_title": ("heading", 2),
    "abstract_title": ("heading", 2),
    "reference_title": ("heading", 2),
    "content_title": ("heading", 2),
    # --- Captions (titles of floating elements) ---
    "table_title": ("caption", None),
    "figure_title": ("caption", None),
    "chart_title": ("caption", None),
    # --- Body text ---
    "text": ("paragraph", None),
    "ocr": ("paragraph", None),
    "vertical_text": ("paragraph", None),
    "reference_content": ("paragraph", None),
    "abstract": ("paragraph", None),
    "content": ("paragraph", None),
    "reference": ("paragraph", None),
    "algorithm": ("paragraph", None),
    "spotting": ("paragraph", None),
    "number": ("paragraph", None),
    "vision_footnote": ("paragraph", None),
    "aside_text": ("paragraph", None),
    # --- Structural ---
    "footnote": ("footnote", None),
    "header": ("page_header", None),
    "header_image": ("page_header", None),
    "footer": ("page_footer", None),
    "footer_image": ("page_footer", None),
    # --- Rich content ---
    "table": ("table", None),
    "formula": ("formula", None),
    "display_formula": ("formula", None),
    "inline_formula": ("formula", None),
    "image": ("figure", None),
    "chart": ("figure", None),
    "seal": ("figure", None),
}

# Labels whose content is always treated as figure (image asset)
_FIGURE_LABELS: frozenset[str] = frozenset(
    {"image", "chart", "seal", "header_image", "footer_image"}
)

# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def _bbox_to_schema(
    raw_bbox: list[int],
    page_input: PageInput,
    model_width: int,
    model_height: int,
) -> BoundingBox | None:
    """Convert a Paddle pixel bbox to PDF-point BoundingBox (TOPLEFT origin).

    Paddle bboxes are ``[x1, y1, x2, y2]`` in pixels at the model's internal
    resolution, which may differ from ``page_input.dimensions`` if PaddleOCR
    rescaled the image internally.  We normalise by the model's reported
    dimensions and scale to PDF points.

    Returns ``None`` if the bbox is malformed.
    """
    if not raw_bbox or len(raw_bbox) < 4:
        return None

    x1, y1, x2, y2 = raw_bbox[:4]

    # Guard against degenerate boxes
    if x1 >= x2 or y1 >= y2:
        return None

    mw = model_width if model_width > 0 else page_input.dimensions.width_px
    mh = model_height if model_height > 0 else page_input.dimensions.height_px

    pt_w = page_input.dimensions.width_pt
    pt_h = page_input.dimensions.height_pt

    return BoundingBox(
        x0=(x1 / mw) * pt_w,
        y0=(y1 / mh) * pt_h,
        x1=(x2 / mw) * pt_w,
        y1=(y2 / mh) * pt_h,
    )


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------


def _strip_latex_delimiters(content: str) -> str:
    """Remove ``$$...$$`` or ``$...$`` wrappers from a LaTeX string."""
    s = content.strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1].strip()
    return s


# ---------------------------------------------------------------------------
# Table HTML parsing
# ---------------------------------------------------------------------------


def _parse_table_html(html: str) -> tuple[list[SchemaTableCell], int, int]:
    """Parse Paddle's table HTML into a flat list of SchemaTableCell objects.

    Returns ``(cells, num_rows, num_cols)``.

    Paddle produces HTML tables directly from its table-structure recognition
    model.  The HTML may use ``<th>`` for headers and ``rowspan``/``colspan``
    attributes.  We parse using BeautifulSoup + lxml.

    If parsing fails (malformed HTML, unexpected structure), returns an empty
    cell list with ``(0, 0)`` dimensions rather than raising.
    """
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415
    except ImportError:
        return [], 0, 0

    try:
        soup = BeautifulSoup(html, "lxml")
        rows = soup.find_all("tr")
        num_rows = len(rows)
        num_cols = 0
        cells: list[SchemaTableCell] = []

        # Two-pass approach: first collect cells, then infer column count.
        for row_idx, row in enumerate(rows):
            col_cursor = 0
            for cell_tag in row.find_all(["th", "td"]):
                rowspan = int(cell_tag.get("rowspan", 1))
                colspan = int(cell_tag.get("colspan", 1))
                is_header = cell_tag.name == "th"
                text = cell_tag.get_text(separator=" ", strip=True)

                cells.append(
                    SchemaTableCell(
                        row=row_idx,
                        col=col_cursor,
                        rowspan=rowspan,
                        colspan=colspan,
                        text=text,
                        is_header=is_header,
                    )
                )
                col_cursor += colspan

            if col_cursor > num_cols:
                num_cols = col_cursor

        return cells, num_rows, num_cols

    except Exception:
        return [], 0, 0


# ---------------------------------------------------------------------------
# Per-block conversion
# ---------------------------------------------------------------------------


def _block_to_schema(
    raw_block: dict[str, Any],
    block_index: int,
    page_input: PageInput,
    model_width: int,
    model_height: int,
    block_prefix: str,
) -> Block | None:
    """Convert one raw block dict to a canonical :class:`Block`.

    Returns ``None`` for unrecognised labels (silently skipped).
    """
    label: str = raw_block.get("block_label", "")
    content: str = raw_block.get("block_content", "")
    raw_bbox: list[int] = raw_block.get("block_bbox", [])

    mapped = _LABEL_MAP.get(label)
    if mapped is None:
        # Unknown label — skip to avoid polluting the canonical output
        return None

    block_type, level = mapped
    bid = f"{block_prefix}_b{block_index:04}"
    prov = Provenance(
        backend="paddle_vl",
        bbox=_bbox_to_schema(raw_bbox, page_input, model_width, model_height),
    )

    # --- Table ---
    if block_type == "table":
        cells, num_rows, num_cols = _parse_table_html(content)
        return Block(
            id=bid,
            type="table",
            reading_order=block_index,
            provenance=prov,
            table=Table(
                rows=num_rows,
                cols=num_cols,
                cells=cells,
                html=content,  # raw HTML from Paddle (already well-formed)
            ),
        )

    # --- Formula ---
    if block_type == "formula":
        latex = _strip_latex_delimiters(content)
        inline = label == "inline_formula"
        return Block(
            id=bid,
            type="formula",
            reading_order=block_index,
            provenance=prov,
            formula=Formula(latex=latex, inline=inline),
        )

    # --- Figure / image ---
    if block_type == "figure" or label in _FIGURE_LABELS:
        return Block(
            id=bid,
            type="figure",
            reading_order=block_index,
            provenance=prov,
            figure=Figure(asset_path=""),  # orchestrator fills this in
        )

    # --- All text-like types (heading, paragraph, caption, footnote, …) ---
    return Block(
        id=bid,
        type=block_type,  # type: ignore[arg-type]
        text=content,
        level=level,
        reading_order=block_index,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def raw_result_to_page(
    raw: dict[str, Any],
    page_input: PageInput,
    backend_name: str,
    backend_version: str | None,
) -> Page:
    """Convert a serialised Paddle result dict to a canonical :class:`Page`.

    This is the single entry point called by :class:`PaddleVLBackend`.

    Parameters
    ----------
    raw:
        The dict produced by ``backend._result_to_raw()``.  Keys:
        ``page_index``, ``width``, ``height``, ``parsing_res_list``.
    page_input:
        The ``PageInput`` for this page; provides canonical dimensions.
    backend_name / backend_version:
        Stored verbatim in each block's ``Provenance``.
    """
    block_prefix = f"p{page_input.page_number:04}"
    model_width: int = raw.get("width", page_input.dimensions.width_px) or 0
    model_height: int = raw.get("height", page_input.dimensions.height_px) or 0

    blocks: list[Block] = []
    block_index = 0

    for raw_block in raw.get("parsing_res_list", []):
        blk = _block_to_schema(
            raw_block,
            block_index=block_index,
            page_input=page_input,
            model_width=model_width,
            model_height=model_height,
            block_prefix=block_prefix,
        )
        if blk is not None:
            blocks.append(blk)
            block_index += 1

    return Page(
        index=page_input.page_index,
        number=page_input.page_number,
        dimensions=page_input.dimensions,
        blocks=blocks,
        backend_metadata={
            "backend": backend_name,
            "backend_version": backend_version,
        },
    )
