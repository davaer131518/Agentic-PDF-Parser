"""Logging helpers."""
from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
