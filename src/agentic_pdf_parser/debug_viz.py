"""Debug visualisation — draws bounding boxes on rasterised page images.

Usage
-----
Enable with ``ParseConfig(debug=True, ...)``.  For each page the orchestrator
calls :func:`save_debug_page` **after** ``backend.parse_page()`` returns.
The annotated image is saved to ``<output_dir>/debug/page_XXXX.png``.

Colour key (written into the image legend):
    heading    — red
    paragraph  — blue
    figure     — bright green
    table      — orange
    formula    — purple
    list_item  — cyan
    caption    — gold
    footnote   — pink
    other      — grey
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schema import Page

# ---------------------------------------------------------------------------
# Block-type → RGB colour
# ---------------------------------------------------------------------------

_COLOURS: dict[str, tuple[int, int, int]] = {
    "heading":    (220, 50,  50),   # red
    "paragraph":  (50,  100, 220),  # blue
    "figure":     (0,   200, 80),   # bright green
    "table":      (230, 130, 0),    # orange
    "formula":    (160, 0,   200),  # purple
    "list_item":  (0,   190, 210),  # cyan
    "caption":    (200, 180, 0),    # gold
    "footnote":   (220, 110, 180),  # pink
    "page_header": (140, 140, 140), # grey
    "page_footer": (140, 140, 140), # grey
    "code":       (80,  160, 80),   # muted green
}
_DEFAULT_COLOUR = (160, 160, 160)  # grey for unknown types


def _colour(block_type: str) -> tuple[int, int, int]:
    return _COLOURS.get(block_type, _DEFAULT_COLOUR)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_debug_page(
    page_image: Image.Image,
    page: Page,
    debug_dir: Path,
) -> Path:
    """Draw bounding boxes on *page_image* and save to *debug_dir*.

    Boxes are drawn in the colour corresponding to their block type.
    Each box is labelled with the block type and a short text preview.

    Parameters
    ----------
    page_image:
        The rasterised page image (RGB, in pixels at the pipeline DPI).
    page:
        The normalised page produced by the backend.
    debug_dir:
        Directory to save annotated images.  Created if it does not exist.

    Returns
    -------
    Path
        The saved annotated image path.
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"page_{page.number:04}.png"

    # Work on a copy so the original is not mutated
    img = page_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    dims = page.dimensions
    # Convert PDF-point bbox → pixel bbox
    scale_x = dims.width_px / dims.width_pt
    scale_y = dims.height_px / dims.height_pt

    # Try to load a small font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, int(14 * scale_x / 2.78)))
    except Exception:
        font = ImageFont.load_default()

    for block in page.blocks:
        bbox = block.provenance.bbox if block.provenance else None
        if bbox is None:
            continue

        left   = max(0, int(bbox.x0 * scale_x))
        top    = max(0, int(bbox.y0 * scale_y))
        right  = min(img.width,  int(bbox.x1 * scale_x))
        bottom = min(img.height, int(bbox.y1 * scale_y))

        if right <= left or bottom <= top:
            continue

        r, g, b = _colour(block.type)

        # Semi-transparent fill
        draw.rectangle(
            [left, top, right, bottom],
            fill=(r, g, b, 40),
            outline=(r, g, b, 230),
            width=2,
        )

        # Label: type + short text preview
        label_parts = [block.type]
        if block.text:
            preview = block.text[:40].replace("\n", " ")
            label_parts.append(f'"{preview}"')
        elif block.type == "figure":
            label_parts.append("[figure]")
        elif block.type == "table":
            label_parts.append(f"[{block.table.rows}×{block.table.cols}]" if block.table else "[table]")
        elif block.type == "formula":
            latex = block.formula.latex[:30] if block.formula and block.formula.latex else ""
            label_parts.append(f"[{latex}]")
        label = " ".join(label_parts)

        # White background behind label for readability
        label_x = left + 3
        label_y = max(0, top - 16)
        draw.rectangle(
            [label_x - 1, label_y, label_x + len(label) * 7, label_y + 14],
            fill=(255, 255, 255, 200),
        )
        draw.text((label_x, label_y), label, fill=(r, g, b), font=font)

    # Draw a legend in the top-right corner
    _draw_legend(draw, img.width, font)

    img.save(out_path, format="PNG")
    return out_path


def _draw_legend(draw: ImageDraw.ImageDraw, img_width: int, font: ImageFont.ImageFont) -> None:
    """Draw a colour legend in the top-right corner of the image."""
    items = [k for k in _COLOURS if k not in ("page_header", "page_footer", "code")]
    pad = 6
    row_h = 16
    box_w = 12
    legend_w = 130
    legend_h = pad * 2 + len(items) * row_h

    x0 = img_width - legend_w - pad
    y0 = pad

    draw.rectangle([x0, y0, x0 + legend_w, y0 + legend_h], fill=(255, 255, 255, 220))

    for i, btype in enumerate(items):
        r, g, b = _colour(btype)
        bx = x0 + pad
        by = y0 + pad + i * row_h
        draw.rectangle([bx, by + 2, bx + box_w, by + row_h - 2], fill=(r, g, b))
        draw.text((bx + box_w + 4, by), btype, fill=(40, 40, 40), font=font)
