"""Explicit backend registry — a plain match, not a plugin system."""
from __future__ import annotations

from ..config import BackendName, ParseConfig
from .base import ParserBackend


def build_backend(cfg: ParseConfig) -> ParserBackend:
    """Instantiate and return the backend specified by *cfg.backend*.

    Imports are lazy so a user without the required packages installed can
    still get a helpful error message rather than a bare ImportError.
    """
    match cfg.backend:
        case BackendName.PADDLE_VL:
            from .paddle_vl.backend import PaddleVLBackend  # noqa: PLC0415

            return PaddleVLBackend(cfg)  # type: ignore[return-value]
        case _:
            raise ValueError(f"Unknown backend: {cfg.backend!r}")
