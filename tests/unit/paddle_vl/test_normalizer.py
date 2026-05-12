"""Unit tests for backends/paddle_vl/normalizer.py.

These tests exercise the normalizer with canned raw-dict fixtures.
No ``paddleocr`` or ``paddle`` imports are required — the normalizer
works purely on plain Python dicts.
"""
from __future__ import annotations

import pytest

from agentic_pdf_parser.backends.paddle_vl.normalizer import (
    _parse_table_html,
    _strip_latex_delimiters,
    raw_result_to_page,
)
from agentic_pdf_parser.schema import Page

from .conftest import (
    A4_H_PT,
    A4_H_PX,
    A4_W_PT,
    A4_W_PX,
    CANNED_FORMULA_DISPLAY,
    CANNED_FORMULA_INLINE,
    CANNED_FULL_PAGE,
    CANNED_HEADING_TEXT,
    CANNED_IMAGE,
    CANNED_TABLE,
    CANNED_TABLE_ROWSPAN,
)


# ---------------------------------------------------------------------------
# Helpers — _strip_latex_delimiters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$$E = mc^2$$", "E = mc^2"),
        ("$x^2$", "x^2"),
        ("x^2", "x^2"),
        ("$$  spaced  $$", "spaced"),
        ("$$$$", ""),          # edge: only delimiters
        ("$$$a$$$", "$a$"),    # edge: triple dollar — outer $$ stripped
    ],
)
def test_strip_latex_delimiters(raw: str, expected: str) -> None:
    assert _strip_latex_delimiters(raw) == expected


# ---------------------------------------------------------------------------
# Helpers — _parse_table_html
# ---------------------------------------------------------------------------


def test_parse_table_html_basic() -> None:
    html = (
        "<table>"
        "<tr><th>Name</th><th>Value</th></tr>"
        "<tr><td>Alice</td><td>100</td></tr>"
        "</table>"
    )
    cells, num_rows, num_cols = _parse_table_html(html)
    assert num_rows == 2
    assert num_cols == 2
    header_cells = [c for c in cells if c.is_header]
    assert len(header_cells) == 2
    assert header_cells[0].text == "Name"
    assert header_cells[1].text == "Value"
    data_cells = [c for c in cells if not c.is_header]
    assert data_cells[0].text == "Alice"
    assert data_cells[1].text == "100"


def test_parse_table_html_rowspan() -> None:
    html = (
        "<table>"
        "<tr><th rowspan='2'>Group</th><th>A</th></tr>"
        "<tr><td>B</td></tr>"
        "</table>"
    )
    cells, num_rows, num_cols = _parse_table_html(html)
    assert num_rows == 2
    assert num_cols == 2
    group_cell = next(c for c in cells if c.text == "Group")
    assert group_cell.rowspan == 2
    assert group_cell.col == 0


def test_parse_table_html_empty() -> None:
    cells, num_rows, num_cols = _parse_table_html("<table></table>")
    assert cells == []
    assert num_rows == 0
    assert num_cols == 0


def test_parse_table_html_malformed_returns_empty() -> None:
    cells, num_rows, num_cols = _parse_table_html("not html at all <<<")
    # Should not raise — returns empty
    assert isinstance(cells, list)


def test_parse_table_html_html_escape() -> None:
    html = "<table><tr><td>a &amp; b</td></tr></table>"
    cells, _, _ = _parse_table_html(html)
    assert cells[0].text == "a & b"


# ---------------------------------------------------------------------------
# raw_result_to_page — core normalisation
# ---------------------------------------------------------------------------


def test_page_metadata(page_input) -> None:
    page = raw_result_to_page(
        CANNED_HEADING_TEXT, page_input, "paddle_vl", "1.5"
    )
    assert isinstance(page, Page)
    assert page.index == 0
    assert page.number == 1
    assert page.dimensions.width_pt == A4_W_PT
    assert page.backend_metadata["backend"] == "paddle_vl"
    assert page.backend_metadata["backend_version"] == "1.5"


def test_heading_and_paragraph(page_input) -> None:
    page = raw_result_to_page(
        CANNED_HEADING_TEXT, page_input, "paddle_vl", "1.5"
    )
    assert len(page.blocks) == 2

    title_block = page.blocks[0]
    assert title_block.type == "heading"
    assert title_block.level == 1
    assert title_block.text == "My Document Title"
    assert title_block.reading_order == 0

    para_block = page.blocks[1]
    assert para_block.type == "paragraph"
    assert para_block.text == "This is a paragraph."
    assert para_block.reading_order == 1


def test_heading_ids_are_unique(page_input) -> None:
    page = raw_result_to_page(
        CANNED_HEADING_TEXT, page_input, "paddle_vl", "1.5"
    )
    ids = [b.id for b in page.blocks]
    assert len(ids) == len(set(ids))


def test_table_block(page_input) -> None:
    page = raw_result_to_page(CANNED_TABLE, page_input, "paddle_vl", "1.5")
    assert len(page.blocks) == 1
    blk = page.blocks[0]
    assert blk.type == "table"
    assert blk.table is not None
    assert blk.table.rows == 2
    assert blk.table.cols == 2
    assert len(blk.table.cells) == 4
    assert blk.table.html is not None
    assert "<table>" in blk.table.html


def test_table_with_rowspan(page_input) -> None:
    page = raw_result_to_page(CANNED_TABLE_ROWSPAN, page_input, "paddle_vl", "1.5")
    blk = page.blocks[0]
    group_cell = next(c for c in blk.table.cells if c.text == "Group")
    assert group_cell.rowspan == 2


def test_formula_display(page_input) -> None:
    page = raw_result_to_page(
        CANNED_FORMULA_DISPLAY, page_input, "paddle_vl", "1.5"
    )
    blk = page.blocks[0]
    assert blk.type == "formula"
    assert blk.formula is not None
    assert blk.formula.latex == "E = mc^2"
    assert blk.formula.inline is False


def test_formula_inline(page_input) -> None:
    page = raw_result_to_page(
        CANNED_FORMULA_INLINE, page_input, "paddle_vl", "1.5"
    )
    blk = page.blocks[0]
    assert blk.formula.latex == "x^2"
    assert blk.formula.inline is True


def test_image_block(page_input) -> None:
    page = raw_result_to_page(CANNED_IMAGE, page_input, "paddle_vl", "1.5")
    blk = page.blocks[0]
    assert blk.type == "figure"
    assert blk.figure is not None
    assert blk.figure.asset_path == ""  # orchestrator fills this later


def test_empty_result(page_input) -> None:
    raw = {
        "page_index": 0,
        "width": A4_W_PX,
        "height": A4_H_PX,
        "parsing_res_list": [],
    }
    page = raw_result_to_page(raw, page_input, "paddle_vl", "1.5")
    assert page.blocks == []


def test_unknown_label_skipped(page_input) -> None:
    """Blocks with labels not in _LABEL_MAP should be silently dropped."""
    raw = {
        "page_index": 0,
        "width": A4_W_PX,
        "height": A4_H_PX,
        "parsing_res_list": [
            {
                "block_label": "unknown_future_label",
                "block_content": "whatever",
                "block_bbox": [0, 0, 100, 20],
                "block_id": 0,
            },
            {
                "block_label": "text",
                "block_content": "kept",
                "block_bbox": [0, 30, 100, 50],
                "block_id": 1,
            },
        ],
    }
    page = raw_result_to_page(raw, page_input, "paddle_vl", "1.5")
    assert len(page.blocks) == 1
    assert page.blocks[0].text == "kept"


def test_bbox_converted_to_pdf_points(page_input) -> None:
    """Bbox should be converted from pixel coords to PDF points."""
    # Title bbox: x1=10, y1=20, x2=500, y2=50 in pixels (595×842)
    page = raw_result_to_page(
        CANNED_HEADING_TEXT, page_input, "paddle_vl", "1.5"
    )
    bbox = page.blocks[0].provenance.bbox
    assert bbox is not None

    # Since model_width == page_input.dimensions.width_px (595),
    # and model_height == height_px (842), scaling is 1:1 (pt == px at 72 DPI).
    assert abs(bbox.x0 - 10.0) < 0.01
    assert abs(bbox.y0 - 20.0) < 0.01
    assert abs(bbox.x1 - 500.0) < 0.01
    assert abs(bbox.y1 - 50.0) < 0.01


def test_bbox_scales_when_model_dimensions_differ(page_input) -> None:
    """If the model reports different width/height, bbox should be rescaled."""
    # Model processes at 1190×1684 (2× of A4 at 72 DPI)
    raw = {
        "page_index": 0,
        "width": 1190,
        "height": 1684,
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "hello",
                "block_bbox": [100, 200, 1000, 400],
                "block_id": 0,
            }
        ],
    }
    page = raw_result_to_page(raw, page_input, "paddle_vl", "1.5")
    bbox = page.blocks[0].provenance.bbox
    assert bbox is not None
    # x0 = (100 / 1190) * 595.0 ≈ 50.0
    assert abs(bbox.x0 - (100 / 1190) * A4_W_PT) < 0.1
    assert abs(bbox.y0 - (200 / 1684) * A4_H_PT) < 0.1


def test_provenance_backend(page_input) -> None:
    page = raw_result_to_page(
        CANNED_HEADING_TEXT, page_input, "paddle_vl", "1.5"
    )
    for blk in page.blocks:
        assert blk.provenance.backend == "paddle_vl"


def test_block_ids_use_page_prefix(page_input) -> None:
    page = raw_result_to_page(
        CANNED_HEADING_TEXT, page_input, "paddle_vl", "1.5"
    )
    for blk in page.blocks:
        assert blk.id.startswith("p0001")


def test_reading_order_preserved(page_input) -> None:
    """reading_order should match the original list order, contiguously."""
    page = raw_result_to_page(CANNED_FULL_PAGE, page_input, "paddle_vl", "1.5")
    for i, blk in enumerate(page.blocks):
        assert blk.reading_order == i


def test_full_page_block_types(page_input) -> None:
    page = raw_result_to_page(CANNED_FULL_PAGE, page_input, "paddle_vl", "1.5")
    types = [b.type for b in page.blocks]
    assert "heading" in types
    assert "paragraph" in types
    assert "table" in types
    assert "formula" in types
    assert "figure" in types
    assert "footnote" in types
    assert "page_header" in types
    assert "page_footer" in types


def test_degenerate_bbox_ignored(page_input) -> None:
    """Bbox where x1>=x2 or y1>=y2 should result in provenance.bbox == None."""
    raw = {
        "page_index": 0,
        "width": A4_W_PX,
        "height": A4_H_PX,
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "degenerate",
                "block_bbox": [100, 100, 100, 100],  # zero-area
                "block_id": 0,
            }
        ],
    }
    page = raw_result_to_page(raw, page_input, "paddle_vl", "1.5")
    assert page.blocks[0].provenance.bbox is None


def test_missing_bbox_ignored(page_input) -> None:
    raw = {
        "page_index": 0,
        "width": A4_W_PX,
        "height": A4_H_PX,
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "no bbox",
                "block_bbox": [],
                "block_id": 0,
            }
        ],
    }
    page = raw_result_to_page(raw, page_input, "paddle_vl", "1.5")
    assert page.blocks[0].provenance.bbox is None


def test_section_heading_level(page_input) -> None:
    """paragraph_title should map to heading level 2."""
    raw = {
        "page_index": 0,
        "width": A4_W_PX,
        "height": A4_H_PX,
        "parsing_res_list": [
            {
                "block_label": "paragraph_title",
                "block_content": "Section",
                "block_bbox": [10, 50, 500, 70],
                "block_id": 0,
            }
        ],
    }
    page = raw_result_to_page(raw, page_input, "paddle_vl", "1.5")
    assert page.blocks[0].level == 2


def test_caption_block(page_input) -> None:
    raw = {
        "page_index": 0,
        "width": A4_W_PX,
        "height": A4_H_PX,
        "parsing_res_list": [
            {
                "block_label": "figure_title",
                "block_content": "Figure 1: My figure",
                "block_bbox": [10, 610, 500, 630],
                "block_id": 0,
            }
        ],
    }
    page = raw_result_to_page(raw, page_input, "paddle_vl", "1.5")
    blk = page.blocks[0]
    assert blk.type == "caption"
    assert blk.text == "Figure 1: My figure"


def test_chart_treated_as_figure(page_input) -> None:
    raw = {
        "page_index": 0,
        "width": A4_W_PX,
        "height": A4_H_PX,
        "parsing_res_list": [
            {
                "block_label": "chart",
                "block_content": "",
                "block_bbox": [10, 100, 500, 400],
                "block_id": 0,
            }
        ],
    }
    page = raw_result_to_page(raw, page_input, "paddle_vl", "1.5")
    assert page.blocks[0].type == "figure"
