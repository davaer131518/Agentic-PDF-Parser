"""Tests for export/markdown_export.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentic_pdf_parser.export.markdown_export import (
    _is_simple_table,
    _render_block,
    _render_page,
    _render_table,
    write,
)
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
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(backend="test")


def _dims() -> PageDimensions:
    return PageDimensions(
        width_pt=595.0, height_pt=842.0, width_px=1654, height_px=2339, dpi=200
    )


def _render(block: Block, captions: dict[str, str] | None = None, referenced: set[str] | None = None) -> str:
    return _render_block(block, captions or {}, referenced or set())


# ---------------------------------------------------------------------------
# _is_simple_table
# ---------------------------------------------------------------------------


def test_simple_table_all_1x1() -> None:
    cells = [TableCell(row=0, col=0, text="A"), TableCell(row=0, col=1, text="B")]
    tbl = Table(rows=1, cols=2, cells=cells)
    assert _is_simple_table(tbl) is True


def test_complex_table_with_rowspan() -> None:
    cells = [TableCell(row=0, col=0, text="A", rowspan=2)]
    tbl = Table(rows=2, cols=1, cells=cells)
    assert _is_simple_table(tbl) is False


def test_complex_table_with_pipe_in_text() -> None:
    cells = [TableCell(row=0, col=0, text="A|B")]
    tbl = Table(rows=1, cols=1, cells=cells)
    assert _is_simple_table(tbl) is False


def test_complex_table_with_newline_in_text() -> None:
    cells = [TableCell(row=0, col=0, text="line1\nline2")]
    tbl = Table(rows=1, cols=1, cells=cells)
    assert _is_simple_table(tbl) is False


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _simple_2x2_table() -> Table:
    return Table(
        rows=2,
        cols=2,
        cells=[
            TableCell(row=0, col=0, text="Name", is_header=True),
            TableCell(row=0, col=1, text="Value", is_header=True),
            TableCell(row=1, col=0, text="foo"),
            TableCell(row=1, col=1, text="42"),
        ],
        html="<table><tr><td>Name</td><td>Value</td></tr><tr><td>foo</td><td>42</td></tr></table>",
    )


def test_render_simple_table_gfm() -> None:
    md = _render_table(_simple_2x2_table())
    assert "| Name | Value |" in md
    assert "| --- | --- |" in md
    assert "| foo | 42 |" in md


def test_render_complex_table_uses_html() -> None:
    cells = [TableCell(row=0, col=0, text="A", rowspan=2)]
    tbl = Table(
        rows=2, cols=1, cells=cells,
        html="<table><tr><td rowspan='2'>A</td></tr></table>",
    )
    md = _render_table(tbl)
    assert "<table>" in md
    assert "rowspan" in md


def test_render_table_gfm_separator_after_first_row() -> None:
    md = _render_table(_simple_2x2_table())
    lines = md.splitlines()
    assert lines[1].startswith("|") and "---" in lines[1]


# ---------------------------------------------------------------------------
# Block renderers
# ---------------------------------------------------------------------------


def test_render_heading_level_1() -> None:
    blk = Block(id="b1", type="heading", text="Hello", level=1, reading_order=0, provenance=_prov())
    assert _render(blk) == "# Hello"


def test_render_heading_level_3() -> None:
    blk = Block(id="b1", type="heading", text="Sub", level=3, reading_order=0, provenance=_prov())
    assert _render(blk) == "### Sub"


def test_render_heading_level_clamped_at_6() -> None:
    blk = Block(id="b1", type="heading", text="Deep", level=9, reading_order=0, provenance=_prov())
    md = _render(blk)
    assert md.startswith("######")


def test_render_paragraph() -> None:
    blk = Block(id="b1", type="paragraph", text="Body text.", reading_order=0, provenance=_prov())
    assert _render(blk) == "Body text."


def test_render_list_item_level_1() -> None:
    blk = Block(id="b1", type="list_item", text="item", level=1, reading_order=0, provenance=_prov())
    assert _render(blk) == "- item"


def test_render_list_item_level_2() -> None:
    blk = Block(id="b1", type="list_item", text="sub", level=2, reading_order=0, provenance=_prov())
    assert _render(blk).startswith("  - sub")


def test_render_list_with_children() -> None:
    item = Block(id="c", type="list_item", text="A", level=1, reading_order=0, provenance=_prov())
    lst = Block(id="p", type="list", reading_order=0, provenance=_prov(), children=[item])
    md = _render(lst)
    assert "- A" in md


def test_render_inline_formula() -> None:
    blk = Block(
        id="b1", type="formula", reading_order=0, provenance=_prov(),
        formula=Formula(latex=r"E=mc^2", inline=True),
    )
    assert _render(blk) == r"$E=mc^2$"


def test_render_block_formula() -> None:
    blk = Block(
        id="b1", type="formula", reading_order=0, provenance=_prov(),
        formula=Formula(latex=r"x^2", inline=False),
    )
    md = _render(blk)
    assert md.startswith("$$")
    assert r"x^2" in md
    assert md.endswith("$$")


def test_render_figure_no_caption() -> None:
    blk = Block(
        id="b1", type="figure", reading_order=0, provenance=_prov(),
        figure=Figure(asset_path="assets/p0001_fig01.png"),
    )
    assert _render(blk) == "![](assets/p0001_fig01.png)"


def test_render_figure_with_alt_text() -> None:
    blk = Block(
        id="b1", type="figure", reading_order=0, provenance=_prov(),
        figure=Figure(asset_path="assets/fig.png", alt_text="A chart"),
    )
    assert _render(blk) == "![A chart](assets/fig.png)"


def test_render_figure_with_caption_inline() -> None:
    blk = Block(
        id="b_fig",
        type="figure",
        reading_order=0,
        provenance=_prov(),
        figure=Figure(asset_path="assets/fig.png", caption_id="b_cap"),
    )
    captions = {"b_cap": "My Caption"}
    md = _render(blk, captions=captions)
    assert "![](assets/fig.png)" in md
    assert "*My Caption*" in md


def test_render_caption_standalone() -> None:
    blk = Block(id="b_cap", type="caption", text="Stand-alone", reading_order=0, provenance=_prov())
    assert _render(blk) == "*Stand-alone*"


def test_render_caption_referenced_is_empty() -> None:
    blk = Block(id="b_cap", type="caption", text="Skip me", reading_order=0, provenance=_prov())
    md = _render(blk, referenced={"b_cap"})
    assert md == ""


def test_render_code_block() -> None:
    blk = Block(id="b1", type="code", text="x = 1", reading_order=0, provenance=_prov())
    md = _render(blk)
    assert "```" in md
    assert "x = 1" in md


def test_render_footnote() -> None:
    blk = Block(id="b1", type="footnote", text="A footnote.", reading_order=0, provenance=_prov())
    md = _render(blk)
    assert md == "*A footnote.*"


def test_render_table_block_simple() -> None:
    tbl = _simple_2x2_table()
    blk = Block(id="b1", type="table", reading_order=0, provenance=_prov(), table=tbl)
    md = _render(blk)
    assert "| Name |" in md


def test_render_table_block_with_caption() -> None:
    tbl = Table(
        rows=1, cols=1,
        cells=[TableCell(row=0, col=0, text="X")],
        html="<table></table>",
        caption_id="cap1",
    )
    blk = Block(id="b1", type="table", reading_order=0, provenance=_prov(), table=tbl)
    md = _render(blk, captions={"cap1": "Table 1"})
    assert "*Table 1*" in md


# ---------------------------------------------------------------------------
# Full page rendering
# ---------------------------------------------------------------------------


def _make_page_with_all_types() -> Page:
    prov = _prov()
    blocks = [
        Block(id="b_h", type="heading", text="Chapter 1", level=2, reading_order=0, provenance=prov),
        Block(id="b_p", type="paragraph", text="Introduction.", reading_order=1, provenance=prov),
        Block(
            id="b_tbl",
            type="table",
            reading_order=2,
            provenance=prov,
            table=_simple_2x2_table(),
        ),
        Block(
            id="b_fig",
            type="figure",
            reading_order=3,
            provenance=prov,
            figure=Figure(asset_path="assets/fig.png", caption_id="b_cap"),
        ),
        Block(id="b_cap", type="caption", text="Fig 1", reading_order=4, provenance=prov),
        Block(
            id="b_formula",
            type="formula",
            reading_order=5,
            provenance=prov,
            formula=Formula(latex=r"\int_0^1 x\,dx"),
        ),
    ]
    return Page(index=0, number=1, dimensions=_dims(), blocks=blocks)


def test_render_page_has_all_content() -> None:
    page = _make_page_with_all_types()
    md = _render_page(page)
    assert "## Chapter 1" in md
    assert "Introduction." in md
    assert "| Name |" in md
    assert "![](assets/fig.png)" in md
    assert "*Fig 1*" in md


def test_caption_not_duplicated() -> None:
    """Caption block should appear at most once (inline after figure)."""
    page = _make_page_with_all_types()
    md = _render_page(page)
    assert md.count("Fig 1") == 1


# ---------------------------------------------------------------------------
# write() — full document
# ---------------------------------------------------------------------------


def _make_doc_for_write() -> NormalizedDocument:
    page = _make_page_with_all_types()
    return NormalizedDocument(
        document=DocumentMetadata(
            source_filename="test.pdf",
            source_sha256="abc",
            num_pages=1,
            title="My Document",
        ),
        backend=BackendMetadata(name="test", device="cpu"),
        pages=[page],
    )


def test_write_creates_file(tmp_path: Path) -> None:
    doc = _make_doc_for_write()
    out = tmp_path / "document.md"
    write(doc, out)
    assert out.exists()


def test_write_has_title(tmp_path: Path) -> None:
    doc = _make_doc_for_write()
    out = tmp_path / "document.md"
    write(doc, out)
    content = out.read_text(encoding="utf-8")
    assert "# My Document" in content


def test_write_has_page_markers(tmp_path: Path) -> None:
    doc = _make_doc_for_write()
    out = tmp_path / "document.md"
    write(doc, out)
    content = out.read_text(encoding="utf-8")
    assert "<!-- page: 1 -->" in content


def test_write_multi_page_markers(tmp_path: Path) -> None:
    page1 = _make_page_with_all_types()
    page2 = Page(
        index=1,
        number=2,
        dimensions=_dims(),
        blocks=[
            Block(id="b_h2", type="heading", text="Page 2", level=1, reading_order=0, provenance=_prov())
        ],
    )
    doc = NormalizedDocument(
        document=DocumentMetadata(
            source_filename="t.pdf", source_sha256="x", num_pages=2
        ),
        backend=BackendMetadata(name="test", device="cpu"),
        pages=[page1, page2],
    )
    out = tmp_path / "doc.md"
    write(doc, out)
    content = out.read_text(encoding="utf-8")
    assert "<!-- page: 1 -->" in content
    assert "<!-- page: 2 -->" in content


def test_write_no_title_when_none(tmp_path: Path) -> None:
    page = Page(index=0, number=1, dimensions=_dims(), blocks=[])
    doc = NormalizedDocument(
        document=DocumentMetadata(source_filename="f.pdf", source_sha256="x", num_pages=1),
        backend=BackendMetadata(name="test", device="cpu"),
        pages=[page],
    )
    out = tmp_path / "doc.md"
    write(doc, out)
    content = out.read_text(encoding="utf-8")
    assert not content.startswith("# ")
