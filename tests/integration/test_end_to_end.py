"""End-to-end integration test using FakeBackend.

Verifies the full orchestrator pipeline without invoking any real model:
- PDF rasterization
- per-page backend call (single contract: normalized Page + raw)
- raw file saving
- figure asset extraction
- NormalizedDocument assembly
- JSON export
- Markdown export
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_pdf_parser.config import BackendName, DeviceChoice, ParseConfig
from agentic_pdf_parser.export.json_export import load as load_json
from agentic_pdf_parser.orchestrator import run
from agentic_pdf_parser.schema import Block, BoundingBox, Provenance
from tests.fixtures.fake_backend import (
    FakeBackend,
    make_fake_page,
    make_figure_block,
    make_formula_block,
    make_table_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(out_dir: Path, keep_raw: bool = False) -> ParseConfig:
    return ParseConfig(
        backend=BackendName.PADDLE_VL,  # doesn't matter for FakeBackend
        device=DeviceChoice.AUTO,
        keep_raw=keep_raw,
        output_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Basic pipeline
# ---------------------------------------------------------------------------


def test_end_to_end_files_created(sample_pdf: Path, tmp_path: Path) -> None:
    """All expected output files are present after a successful run."""
    out_dir = tmp_path / "output"
    cfg = _config(out_dir)

    pages = [make_fake_page(0), make_fake_page(1)]
    backend = FakeBackend(pages=pages)

    result = run(pdf_path=sample_pdf, config=cfg, backend=backend)

    assert result.json_path.exists(), "document.json not created"
    assert result.markdown_path.exists(), "document.md not created"
    assert result.output_dir == out_dir


def test_end_to_end_backend_called_once_per_page(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path / "out")
    pages = [make_fake_page(0), make_fake_page(1)]
    backend = FakeBackend(pages=pages)
    run(pdf_path=sample_pdf, config=cfg, backend=backend)
    assert backend.call_count == 2


def test_end_to_end_json_valid(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path / "out")
    pages = [make_fake_page(0), make_fake_page(1)]
    result = run(pdf_path=sample_pdf, config=_config(tmp_path / "out"), backend=FakeBackend(pages=pages))

    doc = load_json(result.json_path)
    assert doc.schema_version == "1.0"
    assert doc.document.num_pages == 2
    assert len(doc.pages) == 2


def test_end_to_end_document_metadata(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path / "out")
    pages = [make_fake_page(0), make_fake_page(1)]
    result = run(pdf_path=sample_pdf, config=cfg, backend=FakeBackend(pages=pages))

    doc = load_json(result.json_path)
    assert doc.document.source_filename == sample_pdf.name
    assert len(doc.document.source_sha256) == 64  # hex SHA-256


def test_end_to_end_backend_metadata(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path / "out")
    pages = [make_fake_page(0), make_fake_page(1)]
    result = run(pdf_path=sample_pdf, config=cfg, backend=FakeBackend(pages=pages))

    doc = load_json(result.json_path)
    assert doc.backend.name == "fake"
    assert doc.backend.device == "cpu"


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def test_end_to_end_markdown_has_page_markers(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path / "out")
    pages = [make_fake_page(0), make_fake_page(1)]
    result = run(pdf_path=sample_pdf, config=cfg, backend=FakeBackend(pages=pages))

    md = result.markdown_path.read_text(encoding="utf-8")
    assert "<!-- page: 1 -->" in md
    assert "<!-- page: 2 -->" in md


def test_end_to_end_markdown_content_from_pages(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path / "out")
    pages = [make_fake_page(0), make_fake_page(1)]
    result = run(pdf_path=sample_pdf, config=cfg, backend=FakeBackend(pages=pages))

    md = result.markdown_path.read_text(encoding="utf-8")
    assert "# Page 1 Heading" in md
    assert "# Page 2 Heading" in md


# ---------------------------------------------------------------------------
# Raw output
# ---------------------------------------------------------------------------


def test_end_to_end_raw_disabled_by_default(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path / "out", keep_raw=False)
    pages = [make_fake_page(0), make_fake_page(1)]
    result = run(pdf_path=sample_pdf, config=cfg, backend=FakeBackend(pages=pages))

    assert result.raw_dir is None
    assert not (tmp_path / "out" / "raw").exists()


def test_end_to_end_raw_files_created_when_enabled(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path / "out", keep_raw=True)
    pages = [make_fake_page(0), make_fake_page(1)]
    raw_data = [{"page": 1, "source": "fake"}, {"page": 2, "source": "fake"}]
    result = run(
        pdf_path=sample_pdf,
        config=cfg,
        backend=FakeBackend(pages=pages, raw=raw_data),
    )

    assert result.raw_dir is not None
    assert (result.raw_dir / "page_0001.json").exists()
    assert (result.raw_dir / "page_0002.json").exists()


def test_end_to_end_raw_json_content(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path / "out", keep_raw=True)
    pages = [make_fake_page(0), make_fake_page(1)]
    raw_data = [{"marker": "page-one"}, {"marker": "page-two"}]
    result = run(
        pdf_path=sample_pdf,
        config=cfg,
        backend=FakeBackend(pages=pages, raw=raw_data),
    )

    assert result.raw_dir is not None
    raw1 = json.loads((result.raw_dir / "page_0001.json").read_text(encoding="utf-8"))
    raw2 = json.loads((result.raw_dir / "page_0002.json").read_text(encoding="utf-8"))
    assert raw1["marker"] == "page-one"
    assert raw2["marker"] == "page-two"


# ---------------------------------------------------------------------------
# Figure asset extraction
# ---------------------------------------------------------------------------


def test_end_to_end_figure_asset_extracted(sample_pdf: Path, tmp_path: Path) -> None:
    """Figure blocks with valid bboxes produce cropped PNG files."""
    cfg = _config(tmp_path / "out")

    fig_block = make_figure_block(
        page_number=1,
        block_index=2,
        bbox=BoundingBox(x0=50, y0=50, x1=200, y1=200),
    )
    page0 = make_fake_page(index=0, extra_blocks=[fig_block])
    page1 = make_fake_page(index=1)

    result = run(pdf_path=sample_pdf, config=cfg, backend=FakeBackend(pages=[page0, page1]))

    assets_dir = result.output_dir / "assets"
    assert assets_dir.exists()
    png_files = list(assets_dir.glob("*.png"))
    assert len(png_files) == 1
    assert png_files[0].name == "p0001_fig01.png"


def test_end_to_end_figure_asset_path_in_json(sample_pdf: Path, tmp_path: Path) -> None:
    """After extraction, figure.asset_path is updated in the persisted JSON."""
    cfg = _config(tmp_path / "out")

    fig_block = make_figure_block(
        page_number=1,
        block_index=2,
        bbox=BoundingBox(x0=50, y0=50, x1=200, y1=200),
    )
    page0 = make_fake_page(index=0, extra_blocks=[fig_block])
    page1 = make_fake_page(index=1)

    result = run(pdf_path=sample_pdf, config=cfg, backend=FakeBackend(pages=[page0, page1]))

    doc = load_json(result.json_path)
    fig_blocks = [b for b in doc.pages[0].blocks if b.type == "figure"]
    assert len(fig_blocks) == 1
    assert fig_blocks[0].figure is not None
    assert fig_blocks[0].figure.asset_path == "assets/p0001_fig01.png"


# ---------------------------------------------------------------------------
# Rich page content round-trip
# ---------------------------------------------------------------------------


def test_end_to_end_table_in_json(sample_pdf: Path, tmp_path: Path) -> None:
    table_blk = make_table_block(page_number=1, block_index=2)
    page0 = make_fake_page(index=0, extra_blocks=[table_blk])
    page1 = make_fake_page(index=1)

    cfg = _config(tmp_path / "out")
    result = run(pdf_path=sample_pdf, config=cfg, backend=FakeBackend(pages=[page0, page1]))

    doc = load_json(result.json_path)
    tbl_blocks = [b for b in doc.pages[0].blocks if b.type == "table"]
    assert len(tbl_blocks) == 1
    assert tbl_blocks[0].table is not None
    assert len(tbl_blocks[0].table.cells) == 4


def test_end_to_end_formula_in_markdown(sample_pdf: Path, tmp_path: Path) -> None:
    formula_blk = make_formula_block(page_number=1, block_index=2, latex=r"E = mc^2")
    page0 = make_fake_page(index=0, extra_blocks=[formula_blk])
    page1 = make_fake_page(index=1)

    cfg = _config(tmp_path / "out")
    result = run(pdf_path=sample_pdf, config=cfg, backend=FakeBackend(pages=[page0, page1]))

    md = result.markdown_path.read_text(encoding="utf-8")
    assert r"E = mc^2" in md


# ---------------------------------------------------------------------------
# parse_pdf() public API
# ---------------------------------------------------------------------------


def test_parse_pdf_api_with_fake_backend(sample_pdf: Path, tmp_path: Path) -> None:
    """parse_pdf() accepts _backend= for testing."""
    from agentic_pdf_parser import parse_pdf, ParseConfig, BackendName, DeviceChoice

    cfg = ParseConfig(
        backend=BackendName.PADDLE_VL,
        device=DeviceChoice.AUTO,
        output_dir=tmp_path / "api_out",
    )
    pages = [make_fake_page(0), make_fake_page(1)]
    backend = FakeBackend(pages=pages)

    result = parse_pdf(sample_pdf, cfg, _backend=backend)

    assert result.json_path.exists()
    assert result.markdown_path.exists()
    assert backend.call_count == 2
