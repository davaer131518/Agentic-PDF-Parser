"""Tests for the canonical schema (schema.py)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agentic_pdf_parser.schema import (
    BackendMetadata,
    Block,
    BoundingBox,
    DocumentMetadata,
    Figure,
    Formula,
    NormalizedDocument,
    Page,
    PageDimensions,
    Provenance,
    Table,
    TableCell,
    parse_pdf_date,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provenance(**kwargs: object) -> Provenance:
    return Provenance(backend="test", **kwargs)  # type: ignore[arg-type]


def _make_dims() -> PageDimensions:
    return PageDimensions(
        width_pt=595.0, height_pt=842.0, width_px=1654, height_px=2339, dpi=200
    )


def _make_doc_meta(num_pages: int = 1) -> DocumentMetadata:
    return DocumentMetadata(
        source_filename="test.pdf",
        source_sha256="abc123",
        num_pages=num_pages,
    )


def _make_backend_meta() -> BackendMetadata:
    return BackendMetadata(name="fake", device="cpu")


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------


def test_bounding_box_fields() -> None:
    bbox = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0)
    assert bbox.x0 == 10.0
    assert bbox.y1 == 200.0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_defaults() -> None:
    p = Provenance(backend="paddle_vl")
    assert p.backend_version is None
    assert p.bbox is None
    assert p.confidence is None
    assert p.extra == {}


def test_provenance_with_bbox() -> None:
    bbox = BoundingBox(x0=0, y0=0, x1=100, y1=100)
    p = Provenance(backend="paddle_vl", bbox=bbox, confidence=0.95)
    assert p.bbox is not None
    assert p.confidence == 0.95


# ---------------------------------------------------------------------------
# Block types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block_type",
    [
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
    ],
)
def test_block_type_valid(block_type: str) -> None:
    blk = Block(
        id="p0001_b0001",
        type=block_type,  # type: ignore[arg-type]
        reading_order=0,
        provenance=_make_provenance(),
    )
    assert blk.type == block_type


def test_block_optional_fields_default() -> None:
    blk = Block(id="b1", type="paragraph", reading_order=0, provenance=_make_provenance())
    assert blk.text is None
    assert blk.level is None
    assert blk.table is None
    assert blk.figure is None
    assert blk.formula is None
    assert blk.children is None


# ---------------------------------------------------------------------------
# Recursive Block (children)
# ---------------------------------------------------------------------------


def test_block_with_children() -> None:
    child = Block(
        id="p0001_b0002",
        type="list_item",
        text="item",
        reading_order=0,
        provenance=_make_provenance(),
    )
    parent = Block(
        id="p0001_b0001",
        type="list",
        reading_order=0,
        provenance=_make_provenance(),
        children=[child],
    )
    assert parent.children is not None
    assert len(parent.children) == 1
    assert parent.children[0].text == "item"


def test_block_deeply_nested() -> None:
    """Three levels of nesting round-trips through JSON."""
    leaf = Block(
        id="leaf",
        type="list_item",
        text="deep",
        reading_order=0,
        provenance=_make_provenance(),
    )
    mid = Block(
        id="mid",
        type="list",
        reading_order=0,
        provenance=_make_provenance(),
        children=[leaf],
    )
    root = Block(
        id="root",
        type="section",
        reading_order=0,
        provenance=_make_provenance(),
        children=[mid],
    )
    data = root.model_dump(mode="json")
    restored = Block.model_validate(data)
    assert restored.children is not None
    assert restored.children[0].children is not None
    assert restored.children[0].children[0].text == "deep"


# ---------------------------------------------------------------------------
# Table, Figure, Formula
# ---------------------------------------------------------------------------


def test_table_defaults() -> None:
    tbl = Table(rows=2, cols=2, cells=[])
    assert tbl.html is None
    assert tbl.caption_id is None


def test_table_cell() -> None:
    cell = TableCell(row=0, col=1, text="hello", is_header=True, rowspan=2)
    assert cell.rowspan == 2
    assert cell.colspan == 1


def test_figure_defaults() -> None:
    fig = Figure(asset_path="assets/p0001_fig01.png")
    assert fig.caption_id is None
    assert fig.alt_text is None


def test_formula_inline() -> None:
    f = Formula(latex=r"E=mc^2", inline=True)
    assert f.inline is True
    assert f.mathml is None


# ---------------------------------------------------------------------------
# Page and PageDimensions
# ---------------------------------------------------------------------------


def test_page_defaults() -> None:
    page = Page(
        index=0, number=1, dimensions=_make_dims(), blocks=[]
    )
    assert page.backend_metadata == {}


def test_page_dimensions() -> None:
    dims = _make_dims()
    assert dims.width_pt == 595.0
    assert dims.height_px == 2339


# ---------------------------------------------------------------------------
# NormalizedDocument round-trip
# ---------------------------------------------------------------------------


def _build_full_document() -> NormalizedDocument:
    """Build a NormalizedDocument with every block type."""
    prov = _make_provenance(
        bbox=BoundingBox(x0=10, y0=10, x1=200, y1=50),
        confidence=0.9,
    )
    blocks: list[Block] = [
        Block(id="b_h1", type="heading", text="Title", level=1, reading_order=0, provenance=prov),
        Block(id="b_p", type="paragraph", text="Body", reading_order=1, provenance=prov),
        Block(
            id="b_table",
            type="table",
            reading_order=2,
            provenance=prov,
            table=Table(
                rows=2,
                cols=2,
                cells=[
                    TableCell(row=0, col=0, text="A", is_header=True),
                    TableCell(row=0, col=1, text="B", is_header=True),
                    TableCell(row=1, col=0, text="1"),
                    TableCell(row=1, col=1, text="2"),
                ],
                html="<table></table>",
            ),
        ),
        Block(
            id="b_fig",
            type="figure",
            reading_order=3,
            provenance=prov,
            figure=Figure(asset_path="assets/p0001_fig01.png", caption_id="b_cap"),
        ),
        Block(id="b_cap", type="caption", text="Figure 1", reading_order=4, provenance=prov),
        Block(
            id="b_formula",
            type="formula",
            reading_order=5,
            provenance=prov,
            formula=Formula(latex=r"x^2 + y^2 = z^2"),
        ),
        Block(id="b_code", type="code", text="print('hi')", reading_order=6, provenance=prov),
        Block(id="b_footnote", type="footnote", text="note", reading_order=7, provenance=prov),
        Block(
            id="b_list",
            type="list",
            reading_order=8,
            provenance=prov,
            children=[
                Block(
                    id="b_li",
                    type="list_item",
                    text="item",
                    level=1,
                    reading_order=0,
                    provenance=prov,
                )
            ],
        ),
    ]
    page = Page(index=0, number=1, dimensions=_make_dims(), blocks=blocks)
    return NormalizedDocument(
        document=_make_doc_meta(),
        backend=_make_backend_meta(),
        pages=[page],
    )


def test_normalized_document_round_trip() -> None:
    doc = _build_full_document()
    data = doc.model_dump(mode="json")
    restored = NormalizedDocument.model_validate(data)

    assert restored.schema_version == "1.0"
    assert restored.document.source_filename == "test.pdf"
    assert restored.backend.name == "fake"
    assert len(restored.pages) == 1
    assert len(restored.pages[0].blocks) == len(doc.pages[0].blocks)


def test_normalized_document_json_serializable() -> None:
    doc = _build_full_document()
    data = doc.model_dump(mode="json")
    # Must be JSON-serializable without errors
    raw_json = json.dumps(data)
    assert "heading" in raw_json


def test_normalized_document_preserves_all_block_types() -> None:
    doc = _build_full_document()
    block_types = {b.type for b in doc.pages[0].blocks}
    for expected in ("heading", "paragraph", "table", "figure", "caption", "formula", "code", "footnote", "list"):
        assert expected in block_types, f"Block type {expected!r} missing"


# ---------------------------------------------------------------------------
# parse_pdf_date
# ---------------------------------------------------------------------------


def test_parse_pdf_date_valid() -> None:
    dt = parse_pdf_date("D:20231015120000")
    assert dt is not None
    assert dt.year == 2023
    assert dt.month == 10
    assert dt.day == 15
    assert dt.tzinfo == timezone.utc


def test_parse_pdf_date_none() -> None:
    assert parse_pdf_date(None) is None


def test_parse_pdf_date_empty() -> None:
    assert parse_pdf_date("") is None


def test_parse_pdf_date_invalid() -> None:
    assert parse_pdf_date("not-a-date") is None
