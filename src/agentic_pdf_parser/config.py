"""Pipeline configuration models."""
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from pydantic import BaseModel, Field


class DeviceChoice(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class BackendName(StrEnum):
    # Public backend names are stable identifiers.
    PADDLE_VL = "paddle_vl"


class RasterConfig(BaseModel):
    dpi: int = 200  # 300 for small-font documents; 200 balances quality vs speed


class PaddleVLConfig(BaseModel):
    # llama-server.exe serves the VLM recognition component (OCR, tables, formulas).
    # PP-DocLayoutV3 layout detection still runs via PaddleOCR Python.
    llama_cpp_dir: Path = Path("C:/llama-cpp")
    gguf_model_path: Path = Path("C:/llama-cpp/models/PaddleOCR-VL-1.5.gguf")
    mmproj_path: Path = Path("C:/llama-cpp/models/PaddleOCR-VL-1.5-mmproj.gguf")
    # Port for llama-server; change if 8080 is already in use.
    server_port: int = 8080
    # PP-DocLayoutV3 pipeline flags (passed to PaddleOCRVL).
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False


class ParseConfig(BaseModel):
    backend: BackendName
    device: DeviceChoice = DeviceChoice.AUTO
    keep_raw: bool = False
    debug: bool = False  # save bbox-annotated page images to <output_dir>/debug/
    output_dir: Path
    raster: RasterConfig = Field(default_factory=RasterConfig)
    paddle_vl: PaddleVLConfig = Field(default_factory=PaddleVLConfig)
