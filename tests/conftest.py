"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a 2-page A4 PDF created with PyMuPDF."""
    tmp = tmp_path_factory.mktemp("pdfs")
    path = tmp / "sample.pdf"

    doc = fitz.open()

    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 100), "Page 1 — Hello, World!")
    page1.insert_text((50, 150), "Some body text on the first page.")

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 100), "Page 2 — Second Page")
    page2.insert_text((50, 150), "Some body text on the second page.")

    doc.save(str(path))
    doc.close()
    return path
