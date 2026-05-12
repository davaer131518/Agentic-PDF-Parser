"""Tests for export/json_export.py."""
from __future__ import annotations

import json
from pathlib import Path

from agentic_pdf_parser.export.json_export import load, write
from agentic_pdf_parser.schema import (
    BackendMetadata,
    Block,
    DocumentMetadata,
    NormalizedDocument,
    Page,
    PageDimensions,
    Provenance,
    Table,
    TableCell,
)


def _make_doc() -> NormalizedDocument:
    dims = PageDimensions(
        width_pt=595.0, height_pt=842.0, width_px=1654, height_px=2339, dpi=200
    )
    prov = Provenance(backend="test")
    blocks = [
        Block(id="b1", type="heading", text="Title", level=1, reading_order=0, provenance=prov),
        Block(id="b2", type="paragraph", text="Body text.", reading_order=1, provenance=prov),
        Block(
            id="b3",
            type="table",
            reading_order=2,
            provenance=prov,
            table=Table(
                rows=2,
                cols=2,
                cells=[
                    TableCell(row=0, col=0, text="H1", is_header=True),
                    TableCell(row=0, col=1, text="H2", is_header=True),
                    TableCell(row=1, col=0, text="v1"),
                    TableCell(row=1, col=1, text="v2"),
                ],
                html="<table></table>",
            ),
        ),
    ]
    page = Page(index=0, number=1, dimensions=dims, blocks=blocks)
    return NormalizedDocument(
        document=DocumentMetadata(
            source_filename="test.pdf", source_sha256="deadbeef", num_pages=1
        ),
        backend=BackendMetadata(name="test", device="cpu"),
        pages=[page],
    )


def test_write_creates_file(tmp_path: Path) -> None:
    doc = _make_doc()
    out = tmp_path / "document.json"
    write(doc, out)
    assert out.exists()


def test_write_produces_valid_json(tmp_path: Path) -> None:
    doc = _make_doc()
    out = tmp_path / "document.json"
    write(doc, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["schema_version"] == "1.0"


def test_round_trip(tmp_path: Path) -> None:
    original = _make_doc()
    out = tmp_path / "document.json"
    write(original, out)
    restored = load(out)

    assert restored.schema_version == original.schema_version
    assert restored.document.source_sha256 == original.document.source_sha256
    assert len(restored.pages) == 1
    assert len(restored.pages[0].blocks) == 3


def test_round_trip_preserves_table_cells(tmp_path: Path) -> None:
    original = _make_doc()
    out = tmp_path / "document.json"
    write(original, out)
    restored = load(out)

    table_block = next(b for b in restored.pages[0].blocks if b.type == "table")
    assert table_block.table is not None
    assert len(table_block.table.cells) == 4
    assert table_block.table.cells[0].is_header is True


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    doc = _make_doc()
    nested = tmp_path / "a" / "b" / "c" / "document.json"
    write(doc, nested)
    assert nested.exists()


def test_write_utf8_content(tmp_path: Path) -> None:
    doc = _make_doc()
    # Add unicode text to a page block
    doc.pages[0].blocks[0].text = "日本語テスト"
    out = tmp_path / "document.json"
    write(doc, out)
    raw = out.read_text(encoding="utf-8")
    assert "日本語テスト" in raw
