"""Shared fixtures and canned result data for PaddleVL backend tests.

Canned fixtures are plain Python dicts — they are the *raw dict* format
produced by ``PaddleVLBackend._result_to_raw()``.  The normalizer tests
consume these directly without needing paddleocr or paddle installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from agentic_pdf_parser.backends.base import PageInput
from agentic_pdf_parser.schema import PageDimensions

# ---------------------------------------------------------------------------
# Page dimensions + input
# ---------------------------------------------------------------------------

A4_W_PT = 595.0
A4_H_PT = 842.0
A4_W_PX = 595
A4_H_PX = 842


@pytest.fixture
def a4_dims() -> PageDimensions:
    return PageDimensions(
        width_pt=A4_W_PT,
        height_pt=A4_H_PT,
        width_px=A4_W_PX,
        height_px=A4_H_PX,
        dpi=72,
    )


@pytest.fixture
def page_input(a4_dims: PageDimensions, tmp_path: Path) -> PageInput:
    img = Image.new("RGB", (A4_W_PX, A4_H_PX), color=(255, 255, 255))
    img_path = tmp_path / "page_0001.png"
    img.save(img_path)
    return PageInput(
        page_index=0,
        page_number=1,
        dimensions=a4_dims,
        image=img,
        temp_image_path=img_path,
        pdf_path=tmp_path / "dummy.pdf",
    )


# ---------------------------------------------------------------------------
# Canned raw dicts  (verified structure from PaddleOCR-VL 3.4.1 result.py)
# ---------------------------------------------------------------------------

#: Minimal page with heading + paragraph.
CANNED_HEADING_TEXT: dict = {
    "page_index": 0,
    "width": A4_W_PX,
    "height": A4_H_PX,
    "parsing_res_list": [
        {
            "block_label": "doc_title",
            "block_content": "My Document Title",
            "block_bbox": [10, 20, 500, 50],
            "block_id": 0,
        },
        {
            "block_label": "text",
            "block_content": "This is a paragraph.",
            "block_bbox": [10, 60, 500, 100],
            "block_id": 1,
        },
    ],
}

#: Page with a simple 2×2 HTML table.
CANNED_TABLE: dict = {
    "page_index": 0,
    "width": A4_W_PX,
    "height": A4_H_PX,
    "parsing_res_list": [
        {
            "block_label": "table",
            "block_content": (
                "<table>"
                "<tr><th>Name</th><th>Value</th></tr>"
                "<tr><td>Alice</td><td>100</td></tr>"
                "</table>"
            ),
            "block_bbox": [10, 110, 500, 300],
            "block_id": 0,
        },
    ],
}

#: Page with a table containing rowspan.
CANNED_TABLE_ROWSPAN: dict = {
    "page_index": 0,
    "width": A4_W_PX,
    "height": A4_H_PX,
    "parsing_res_list": [
        {
            "block_label": "table",
            "block_content": (
                "<table>"
                "<tr><th rowspan='2'>Group</th><th>A</th></tr>"
                "<tr><td>B</td></tr>"
                "</table>"
            ),
            "block_bbox": [10, 110, 500, 300],
            "block_id": 0,
        },
    ],
}

#: Page with display formula ($$-wrapped LaTeX).
CANNED_FORMULA_DISPLAY: dict = {
    "page_index": 0,
    "width": A4_W_PX,
    "height": A4_H_PX,
    "parsing_res_list": [
        {
            "block_label": "display_formula",
            "block_content": "$$E = mc^2$$",
            "block_bbox": [10, 310, 500, 350],
            "block_id": 0,
        },
    ],
}

#: Page with inline formula ($-wrapped LaTeX).
CANNED_FORMULA_INLINE: dict = {
    "page_index": 0,
    "width": A4_W_PX,
    "height": A4_H_PX,
    "parsing_res_list": [
        {
            "block_label": "inline_formula",
            "block_content": "$x^2$",
            "block_bbox": [10, 200, 100, 220],
            "block_id": 0,
        },
    ],
}

#: Page with an image block.
CANNED_IMAGE: dict = {
    "page_index": 0,
    "width": A4_W_PX,
    "height": A4_H_PX,
    "parsing_res_list": [
        {
            "block_label": "image",
            "block_content": "",
            "block_bbox": [10, 360, 500, 600],
            "block_id": 0,
        },
    ],
}

#: Page with all main block types together.
CANNED_FULL_PAGE: dict = {
    "page_index": 0,
    "width": A4_W_PX,
    "height": A4_H_PX,
    "parsing_res_list": [
        {
            "block_label": "doc_title",
            "block_content": "Full Page Example",
            "block_bbox": [10, 10, 500, 40],
            "block_id": 0,
        },
        {
            "block_label": "paragraph_title",
            "block_content": "Section 1",
            "block_bbox": [10, 50, 500, 75],
            "block_id": 1,
        },
        {
            "block_label": "text",
            "block_content": "Body text here.",
            "block_bbox": [10, 80, 500, 110],
            "block_id": 2,
        },
        {
            "block_label": "table",
            "block_content": "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>",
            "block_bbox": [10, 120, 500, 300],
            "block_id": 3,
        },
        {
            "block_label": "display_formula",
            "block_content": "$$x = 1$$",
            "block_bbox": [10, 310, 500, 340],
            "block_id": 4,
        },
        {
            "block_label": "image",
            "block_content": "",
            "block_bbox": [10, 350, 500, 600],
            "block_id": 5,
        },
        {
            "block_label": "footnote",
            "block_content": "1 This is a footnote.",
            "block_bbox": [10, 750, 500, 780],
            "block_id": 6,
        },
        {
            "block_label": "header",
            "block_content": "Page Header",
            "block_bbox": [10, 5, 500, 18],
            "block_id": 7,
        },
        {
            "block_label": "footer",
            "block_content": "Page Footer",
            "block_bbox": [10, 820, 500, 835],
            "block_id": 8,
        },
    ],
}
