"""Tests for rasterize.py."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from agentic_pdf_parser.rasterize import iter_pages, read_metadata


def test_read_metadata_page_count(sample_pdf: Path) -> None:
    meta, count = read_metadata(sample_pdf)
    assert count == 2


def test_read_metadata_returns_dict(sample_pdf: Path) -> None:
    meta, _ = read_metadata(sample_pdf)
    assert isinstance(meta, dict)
    # Keys present in fitz metadata dict (even if empty strings)
    for key in ("title", "author", "format"):
        assert key in meta


def test_iter_pages_count(sample_pdf: Path, tmp_path: Path) -> None:
    pages = list(iter_pages(sample_pdf, dpi=72, work_dir=tmp_path))
    assert len(pages) == 2


def test_iter_pages_yields_pil_images(sample_pdf: Path, tmp_path: Path) -> None:
    for img, dims, tmp_img_path in iter_pages(sample_pdf, dpi=72, work_dir=tmp_path):
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"


def test_iter_pages_dimensions_consistent(sample_pdf: Path, tmp_path: Path) -> None:
    for img, dims, tmp_img_path in iter_pages(sample_pdf, dpi=72, work_dir=tmp_path):
        assert img.width == dims.width_px
        assert img.height == dims.height_px
        assert dims.dpi == 72


def test_iter_pages_dpi_scales_dimensions(sample_pdf: Path, tmp_path: Path) -> None:
    (tmp_path / "lo").mkdir()
    (tmp_path / "hi").mkdir()
    pages_lo = list(iter_pages(sample_pdf, dpi=72, work_dir=tmp_path / "lo"))
    pages_hi = list(iter_pages(sample_pdf, dpi=144, work_dir=tmp_path / "hi"))

    # 144 DPI should give roughly double the pixels
    lo_w = pages_lo[0][1].width_px
    hi_w = pages_hi[0][1].width_px
    assert hi_w > lo_w


def test_iter_pages_writes_png_files(sample_pdf: Path, tmp_path: Path) -> None:
    for _, _, tmp_img_path in iter_pages(sample_pdf, dpi=72, work_dir=tmp_path):
        assert tmp_img_path.exists()
        assert tmp_img_path.suffix == ".png"


def test_iter_pages_sequential_filenames(sample_pdf: Path, tmp_path: Path) -> None:
    paths = [p for _, _, p in iter_pages(sample_pdf, dpi=72, work_dir=tmp_path)]
    assert paths[0].name == "page_0000.png"
    assert paths[1].name == "page_0001.png"
