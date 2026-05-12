"""Backend protocol and data-transfer objects.

Contract
--------
Every backend implementation MUST:

1. Own both inference **and** normalization inside ``parse_page()``.
2. Return ``BackendPageResult(page=<already-normalized Page>, raw=<JSON-serializable>)``.
3. Never expose backend-native types (DoclingDocument, Paddle result objects, etc.)
   outside of its own package.

The orchestrator calls ``backend.parse_page()`` and receives a normalized ``Page``.
It **never** calls a normalizer itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from ..schema import Page, PageDimensions


@dataclass(frozen=True)
class PageInput:
    """Everything a backend needs to process one page."""

    page_index: int          # 0-based
    page_number: int         # 1-based
    dimensions: PageDimensions
    image: Image.Image
    temp_image_path: Path    # persists until the orchestrator's temp-dir is cleaned up
    pdf_path: Path


@dataclass
class BackendPageResult:
    """Single-contract return value from every backend.

    ``page`` is the already-normalized canonical Page.
    ``raw`` is the backend-native output, guaranteed JSON-serializable.
    """

    page: Page
    raw: Any  # dict for PaddleVL; JSON-serializable backend-native output


@runtime_checkable
class ParserBackend(Protocol):
    """Structural protocol satisfied by PaddleVLBackend and any future backend."""

    name: str
    version: str | None
    model_id: str | None
    resolved_device: str

    def load(self) -> None:
        """Load model weights into memory / initialise the pipeline."""
        ...

    def parse_page(self, page_input: PageInput) -> BackendPageResult:
        """Run inference + normalization on a single page.

        Returns a ``BackendPageResult`` whose ``page`` field is a fully
        normalized canonical ``Page``.  The orchestrator never normalizes.
        """
        ...

    def unload(self) -> None:
        """Release model weights and free resources."""
        ...
