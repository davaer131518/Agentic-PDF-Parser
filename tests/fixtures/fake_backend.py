"""FakeBackend — a deterministic test double for ParserBackend.

Usage::

    from tests.fixtures.fake_backend import FakeBackend, make_fake_page

    pages = [make_fake_page(index=0), make_fake_page(index=1)]
    backend = FakeBackend(pages=pages)
    result = orchestrator.run(pdf_path=..., config=..., backend=backend)
    assert backend.call_count == 2
"""
from __future__ import annotations

from typing import Any

from agentic_pdf_parser.backends.base import BackendPageResult, PageInput
from agentic_pdf_parser.schema import (
    Block,
    BoundingBox,
    Figure,
    Formula,
    Page,
    PageDimensions,
    Provenance,
    Table,
    TableCell,
)


class FakeBackend:
    """Returns pre-built canonical Pages without running any model."""

    name = "fake"
    version = "0.1.0-test"
    model_id: str | None = None
    resolved_device = "cpu"

    def __init__(
        self,
        pages: list[Page],
        raw: list[Any] | None = None,
    ) -> None:
        self._pages = pages
        self._raw = raw if raw is not None else [{"fake": True}] * len(pages)
        self.call_count = 0
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def parse_page(self, page_input: PageInput) -> BackendPageResult:
        if self.call_count >= len(self._pages):
            raise RuntimeError(
                f"FakeBackend has {len(self._pages)} page(s) but was called "
                f"{self.call_count + 1} time(s). "
                "Make sure the PDF page count matches the number of pre-built pages."
            )
        page = self._pages[self.call_count]
        raw = self._raw[self.call_count]
        self.call_count += 1
        return BackendPageResult(page=page, raw=raw)

    def unload(self) -> None:
        self._loaded = False


# ---------------------------------------------------------------------------
# Page builder helpers
# ---------------------------------------------------------------------------

_A4_DIMS = PageDimensions(
    width_pt=595.0,
    height_pt=842.0,
    width_px=1654,
    height_px=2339,
    dpi=200,
)


def make_fake_page(index: int, extra_blocks: list[Block] | None = None) -> Page:
    """Build a minimal canonical Page for testing.

    Always includes a heading and a paragraph.  Pass *extra_blocks* to add
    tables, formulas, figures, etc.
    """
    blocks: list[Block] = [
        Block(
            id=f"p{index + 1:04}_b0001",
            type="heading",
            text=f"Page {index + 1} Heading",
            level=1,
            reading_order=0,
            provenance=Provenance(backend="fake"),
        ),
        Block(
            id=f"p{index + 1:04}_b0002",
            type="paragraph",
            text=f"This is paragraph content on page {index + 1}.",
            reading_order=1,
            provenance=Provenance(backend="fake"),
        ),
    ]
    if extra_blocks:
        for i, blk in enumerate(extra_blocks, start=2):
            # Preserve provided reading_order; if unset keep position order
            if blk.reading_order == 0 and i > 0:
                object.__setattr__(blk, "reading_order", i)
            blocks.append(blk)

    return Page(
        index=index,
        number=index + 1,
        dimensions=_A4_DIMS,
        blocks=blocks,
    )


def make_figure_block(
    page_number: int,
    block_index: int,
    *,
    bbox: BoundingBox | None = None,
    caption_id: str | None = None,
) -> Block:
    return Block(
        id=f"p{page_number:04}_b{block_index:04}",
        type="figure",
        reading_order=block_index,
        provenance=Provenance(backend="fake", bbox=bbox),
        figure=Figure(asset_path="", caption_id=caption_id),
    )


def make_table_block(
    page_number: int,
    block_index: int,
    *,
    rows: int = 2,
    cols: int = 2,
    caption_id: str | None = None,
) -> Block:
    cells = [
        TableCell(row=r, col=c, text=f"r{r}c{c}")
        for r in range(rows)
        for c in range(cols)
    ]
    html = "<table><tr><td>r0c0</td><td>r0c1</td></tr><tr><td>r1c0</td><td>r1c1</td></tr></table>"
    return Block(
        id=f"p{page_number:04}_b{block_index:04}",
        type="table",
        reading_order=block_index,
        provenance=Provenance(backend="fake"),
        table=Table(
            rows=rows,
            cols=cols,
            cells=cells,
            html=html,
            caption_id=caption_id,
        ),
    )


def make_formula_block(
    page_number: int,
    block_index: int,
    *,
    latex: str = r"E = mc^2",
    inline: bool = False,
) -> Block:
    return Block(
        id=f"p{page_number:04}_b{block_index:04}",
        type="formula",
        reading_order=block_index,
        provenance=Provenance(backend="fake"),
        formula=Formula(latex=latex, inline=inline),
    )
