"""Pipeline configuration models."""
from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class DeviceChoice(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class BackendName(StrEnum):
    # Public backend names are stable identifiers.
    PADDLE_VL = "paddle_vl"


class RasterConfig(BaseModel):
    dpi: int = 200  # 300 for small-font documents; 200 balances quality vs speed


def _default_llama_cpp_dir() -> Path:
    if sys.platform == "win32":
        return Path("C:/llama-cpp")
    return Path("~/llama-cpp").expanduser()


def _default_gguf_path() -> Path:
    if sys.platform == "win32":
        return Path("C:/llama-cpp/models/PaddleOCR-VL-1.5.gguf")
    return Path("~/models/paddle-ocr-vl/PaddleOCR-VL-1.5.gguf").expanduser()


def _default_mmproj_path() -> Path:
    if sys.platform == "win32":
        return Path("C:/llama-cpp/models/PaddleOCR-VL-1.5-mmproj.gguf")
    return Path("~/models/paddle-ocr-vl/PaddleOCR-VL-1.5-mmproj.gguf").expanduser()


class PaddleVLConfig(BaseModel):
    # llama-server serves the VLM recognition component (OCR, tables, formulas).
    # PP-DocLayoutV3 layout detection still runs via PaddleOCR Python.
    #
    # Defaults are platform-aware (Windows: C:/llama-cpp, others: ~/llama-cpp).
    # Override in your YAML config — all paths support ~ expansion:
    #   Windows:        llama_cpp_dir: "C:/llama-cpp"
    #   macOS Homebrew: llama_cpp_dir: "/opt/homebrew"
    #   manual install: llama_cpp_dir: "~/llama-cpp"
    llama_cpp_dir: Path = Field(default_factory=_default_llama_cpp_dir)
    gguf_model_path: Path = Field(default_factory=_default_gguf_path)
    mmproj_path: Path = Field(default_factory=_default_mmproj_path)
    # Port for llama-server; change if 8080 is already in use.
    server_port: int = 8080
    # PP-DocLayoutV3 pipeline flags (passed to PaddleOCRVL).
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False

    @field_validator("llama_cpp_dir", "gguf_model_path", "mmproj_path", mode="before")
    @classmethod
    def _expand_user(cls, v: object) -> object:
        """Expand leading ~ so YAML values like ~/models/... work on all platforms."""
        if isinstance(v, (str, Path)):
            return Path(str(v)).expanduser()
        return v


class ParseConfig(BaseModel):
    backend: BackendName
    device: DeviceChoice = DeviceChoice.AUTO
    keep_raw: bool = False
    debug: bool = False  # save bbox-annotated page images to <output_dir>/debug/
    output_dir: Path
    raster: RasterConfig = Field(default_factory=RasterConfig)
    paddle_vl: PaddleVLConfig = Field(default_factory=PaddleVLConfig)
