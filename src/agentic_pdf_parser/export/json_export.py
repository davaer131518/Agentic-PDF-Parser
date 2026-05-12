"""JSON serialisation of NormalizedDocument."""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import NormalizedDocument


def write(doc: NormalizedDocument, output_path: Path) -> None:
    """Serialize *doc* to a pretty-printed JSON file at *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = doc.model_dump(mode="json")
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load(path: Path) -> NormalizedDocument:
    """Deserialize a JSON file produced by :func:`write`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return NormalizedDocument.model_validate(data)
