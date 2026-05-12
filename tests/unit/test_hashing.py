"""Tests for utils/hashing.py."""
from __future__ import annotations

import hashlib
from pathlib import Path

from agentic_pdf_parser.utils.hashing import sha256_file


def test_sha256_known_content(tmp_path: Path) -> None:
    content = b"hello world"
    f = tmp_path / "test.bin"
    f.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(f) == expected


def test_sha256_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    expected = hashlib.sha256(b"").hexdigest()
    assert sha256_file(f) == expected


def test_sha256_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"some data")
    assert sha256_file(f) == sha256_file(f)


def test_sha256_different_files_differ(tmp_path: Path) -> None:
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"aaaa")
    f2.write_bytes(b"bbbb")
    assert sha256_file(f1) != sha256_file(f2)


def test_sha256_returns_lowercase_hex(tmp_path: Path) -> None:
    f = tmp_path / "test.bin"
    f.write_bytes(b"data")
    result = sha256_file(f)
    assert result == result.lower()
    assert all(c in "0123456789abcdef" for c in result)
    assert len(result) == 64
