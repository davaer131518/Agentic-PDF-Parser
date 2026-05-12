"""Tests for assets.py."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from agentic_pdf_parser.assets import extract_figures
from agentic_pdf_parser.schema import (
    Block,
    BoundingBox,
    Figure,
    Page,
    PageDimensions,
    Provenance,
)


def _make_white_image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _dims(width_pt: float = 72.0, height_pt: float = 72.0, px: int = 200) -> PageDimensions:
    """1-inch square page at 200 DPI → 200×200 px."""
    return PageDimensions(
        width_pt=width_pt,
        height_pt=height_pt,
        width_px=px,
        height_px=px,
        dpi=200,
    )


def _figure_block(
    bid: str,
    bbox: BoundingBox | None,
    reading_order: int = 0,
) -> Block:
    return Block(
        id=bid,
        type="figure",
        reading_order=reading_order,
        provenance=Provenance(backend="test", bbox=bbox),
        figure=Figure(asset_path=""),
    )


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_extract_saves_cropped_file(tmp_path: Path) -> None:
    img = _make_white_image(200, 200)
    bbox = BoundingBox(x0=0, y0=0, x1=36, y1=36)  # half the 72pt page
    blk = _figure_block("p0001_b0001", bbox)
    page = Page(index=0, number=1, dimensions=_dims(), blocks=[blk])

    extract_figures(img, page, tmp_path / "assets")

    expected = tmp_path / "assets" / "p0001_fig01.png"
    assert expected.exists()


def test_extract_updates_asset_path(tmp_path: Path) -> None:
    img = _make_white_image(200, 200)
    bbox = BoundingBox(x0=0, y0=0, x1=36, y1=36)
    blk = _figure_block("p0001_b0001", bbox)
    page = Page(index=0, number=1, dimensions=_dims(), blocks=[blk])

    extract_figures(img, page, tmp_path / "assets")

    assert blk.figure is not None
    assert blk.figure.asset_path == "assets/p0001_fig01.png"


def test_extract_crop_dimensions(tmp_path: Path) -> None:
    """Bbox (0,0)→(36,36) on a 72pt/200px page gives a 100×100 crop."""
    img = _make_white_image(200, 200)
    bbox = BoundingBox(x0=0, y0=0, x1=36, y1=36)
    blk = _figure_block("p0001_b0001", bbox)
    page = Page(index=0, number=1, dimensions=_dims(), blocks=[blk])

    assets_dir = tmp_path / "assets"
    extract_figures(img, page, assets_dir)

    cropped = Image.open(assets_dir / "p0001_fig01.png")
    assert cropped.size == (100, 100)


# ---------------------------------------------------------------------------
# No bbox → skip
# ---------------------------------------------------------------------------


def test_no_bbox_skips_extraction(tmp_path: Path) -> None:
    img = _make_white_image(200, 200)
    blk = _figure_block("p0001_b0001", bbox=None)
    page = Page(index=0, number=1, dimensions=_dims(), blocks=[blk])

    assets_dir = tmp_path / "assets"
    extract_figures(img, page, assets_dir)

    assert blk.figure is not None
    assert blk.figure.asset_path == ""  # unchanged sentinel
    assert not list(assets_dir.glob("*.png")) if assets_dir.exists() else True


# ---------------------------------------------------------------------------
# Multiple figures → unique filenames
# ---------------------------------------------------------------------------


def test_multiple_figures_unique_names(tmp_path: Path) -> None:
    img = _make_white_image(200, 200)
    bbox = BoundingBox(x0=0, y0=0, x1=20, y1=20)
    blk1 = _figure_block("p0001_b0001", bbox, reading_order=0)
    blk2 = _figure_block("p0001_b0002", bbox, reading_order=1)
    page = Page(index=0, number=1, dimensions=_dims(), blocks=[blk1, blk2])

    assets_dir = tmp_path / "assets"
    extract_figures(img, page, assets_dir)

    assert (assets_dir / "p0001_fig01.png").exists()
    assert (assets_dir / "p0001_fig02.png").exists()
    assert blk1.figure is not None
    assert blk2.figure is not None
    assert blk1.figure.asset_path != blk2.figure.asset_path


# ---------------------------------------------------------------------------
# Non-figure blocks are ignored
# ---------------------------------------------------------------------------


def test_non_figure_blocks_ignored(tmp_path: Path) -> None:
    img = _make_white_image(200, 200)
    para = Block(
        id="p0001_b0001",
        type="paragraph",
        text="text",
        reading_order=0,
        provenance=Provenance(backend="test"),
    )
    page = Page(index=0, number=1, dimensions=_dims(), blocks=[para])
    assets_dir = tmp_path / "assets"
    extract_figures(img, page, assets_dir)

    assert not list(assets_dir.glob("*.png")) if assets_dir.exists() else True


# ---------------------------------------------------------------------------
# assets_dir is created automatically
# ---------------------------------------------------------------------------


def test_assets_dir_created(tmp_path: Path) -> None:
    img = _make_white_image(200, 200)
    blk = _figure_block("b1", BoundingBox(x0=0, y0=0, x1=10, y1=10))
    page = Page(index=0, number=1, dimensions=_dims(), blocks=[blk])
    assets_dir = tmp_path / "deep" / "path" / "assets"

    assert not assets_dir.exists()
    extract_figures(img, page, assets_dir)
    assert assets_dir.exists()


# ---------------------------------------------------------------------------
# Bbox clamped to image bounds
# ---------------------------------------------------------------------------


def test_bbox_clamped_to_image_bounds(tmp_path: Path) -> None:
    """A bbox larger than the image is clamped and still saves a file."""
    img = _make_white_image(200, 200)
    bbox = BoundingBox(x0=0, y0=0, x1=999, y1=999)  # way outside image
    blk = _figure_block("b1", bbox)
    page = Page(index=0, number=1, dimensions=_dims(), blocks=[blk])
    assets_dir = tmp_path / "assets"
    extract_figures(img, page, assets_dir)

    expected = assets_dir / "p0001_fig01.png"
    assert expected.exists()
