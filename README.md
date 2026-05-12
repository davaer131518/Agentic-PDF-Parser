# Agentic PDF Parser

A local-first, page-oriented PDF parsing pipeline that produces a canonical
machine-readable JSON and a Markdown export, preserving tables, formulas,
images, layout, and reading order.

## Supported backend

| Public name | Underlying model | Inference engine | Strengths |
|-------------|-----------------|------------------|-----------|
| `paddle_vl` | [PaddleOCR-VL-1.5](https://github.com/PaddlePaddle/PaddleOCR) | PP-DocLayoutV3 (Python) + `llama-server` (GGUF) | Dense tables, multi-language, scanned documents |

## Prerequisites

**Python ≥ 3.11** is required.

The backend requires **llama.cpp** binaries installed locally:

1. Download the latest [llama.cpp release](https://github.com/ggerganov/llama.cpp/releases)
   and extract to `C:/llama-cpp/` (or any path — update the configs accordingly).
2. Download the GGUF model files into `C:/llama-cpp/models/`:
   - `PaddleOCR-VL-1.5.gguf`
   - `PaddleOCR-VL-1.5-mmproj.gguf`

## Installation

```bash
# Core only (schema, orchestrator, exporters — no inference backend)
pip install -e .

# With PaddleVL backend (paddlepaddle + paddleocr for layout detection)
pip install -e ".[paddle]"

# All extras (paddle + dev)
pip install -e ".[all]"

# Development extras (pytest, ruff, mypy)
pip install -e ".[dev]"
```

For GPU inference, install the GPU-enabled PaddlePaddle build for your
CUDA version before installing paddleocr:

```bash
# Example for CUDA 12.x
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
pip install paddleocr>=3.0
```

The VLM recognition component (OCR, tables, formulas) runs through
`llama-server` — no PyTorch is required.

## Quick start

### CLI — PaddleVL (GPU)

```bash
# Using the pre-built GPU config (recommended)
parse-pdf input.pdf --config configs/paddle_vl_gpu.yaml --out output/

# CPU (slower; useful without NVIDIA GPU)
parse-pdf input.pdf --config configs/paddle_vl_cpu.yaml --out output/

# Custom DPI override on top of config file
parse-pdf input.pdf --config configs/paddle_vl_gpu.yaml --dpi 300 --out output/

# With raw per-page backend output saved alongside canonical JSON
parse-pdf input.pdf --config configs/paddle_vl_gpu.yaml --keep-raw --out output/
```

### CLI — Using a YAML config file with overrides

```bash
# YAML as base; CLI flags override individual fields
parse-pdf input.pdf --config configs/paddle_vl_gpu.yaml --device cpu --out my_output/
```

CLI flags always win over YAML values.  See [Configuration](#configuration) for details.

### Python API

```python
from pathlib import Path
from agentic_pdf_parser import parse_pdf
from agentic_pdf_parser.config_loader import load_config

# Load from a YAML config file
config = load_config(Path("configs/paddle_vl_gpu.yaml"))
result = parse_pdf("input.pdf", config)

print(result.json_path)       # output/document.json
print(result.markdown_path)   # output/document.md

for page in result.document.pages:
    print(f"Page {page.number}: {len(page.blocks)} blocks")
```

### Python API — programmatic config with overrides

```python
from pathlib import Path
from agentic_pdf_parser import parse_pdf
from agentic_pdf_parser.config_loader import build_config

config = build_config(
    yaml_path=Path("configs/paddle_vl_gpu.yaml"),
    keep_raw=True,
    output_dir=Path("my_output"),
)
result = parse_pdf("input.pdf", config)
```

## Output structure

```
output/
├── document.json      # Canonical NormalizedDocument — source of truth
├── document.md        # Markdown derived from document.json
├── assets/            # Cropped figure/image assets
│   ├── p0001_fig00.png
│   └── ...
├── debug/             # Bbox-annotated page images (--debug only)
│   ├── page_0001.png
│   └── ...
└── raw/               # Per-page raw backend output (--keep-raw only)
    ├── page_0001.json  # PaddleVL: {"parsing_res_list": [...]}
    └── ...
```

`document.json` is the canonical source of truth; `document.md` is always
derived from it.

## Configuration

### Supported CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` / `-b` | *(from config)* | `paddle_vl` |
| `--device` / `-d` | `auto` | `cpu`, `gpu`, or `auto` |
| `--out` / `-o` | `./output` | Output directory |
| `--keep-raw` | off | Save raw per-page backend output under `<out>/raw/` |
| `--dpi` | `200` | Page rasterization DPI |
| `--debug` | off | Save bbox-annotated page images to `<out>/debug/` |
| `--config` / `-c` | *(none)* | Path to a YAML config file |

### Device behaviour

| `--device` | Behaviour |
|------------|-----------|
| `auto` | GPU if available, otherwise CPU *(default)* |
| `gpu` | GPU only; raises `RuntimeError` if no CUDA device is found |
| `cpu` | CPU only; significantly slower |

### YAML config files

Pre-built configs live in `configs/`:

| File | Backend | Device |
|------|---------|--------|
| `configs/paddle_vl_cpu.yaml` | `paddle_vl` | `cpu` |
| `configs/paddle_vl_gpu.yaml` | `paddle_vl` | `gpu` |

YAML keys mirror `ParseConfig` fields. Example for PaddleVL GPU:

```yaml
backend: paddle_vl
device: gpu

raster:
  dpi: 200

paddle_vl:
  llama_cpp_dir: "C:/llama-cpp"
  gguf_model_path: "C:/llama-cpp/models/PaddleOCR-VL-1.5.gguf"
  mmproj_path: "C:/llama-cpp/models/PaddleOCR-VL-1.5-mmproj.gguf"
  server_port: 8080
  use_doc_orientation_classify: false
  use_doc_unwarping: false
```

**Precedence (highest → lowest):** CLI flags › YAML file › Pydantic defaults.

**`keep_raw` flag:** passing `--keep-raw` forces `keep_raw=True` regardless of
YAML; `--no-keep-raw` overrides a YAML `keep_raw: true` back to `false`.

### Fixed invariants

The following behaviours are **always on** and not configurable:

- Tables are always extracted and represented as structured cells + HTML.
- Formulas are always extracted as LaTeX (where the model provides it).
- Images and figures are always cropped and saved to `assets/`.
- Both `document.json` and `document.md` are always emitted.
- Page boundaries are always marked in Markdown as `<!-- page: N -->`.

## Architecture overview

See [`docs/architecture.md`](docs/architecture.md) for the full design.

## Smoke tests

Results for representative PDFs are stored in `smoke_tests/`:

| PDF | Notes |
|-----|-------|
| `simple_adobe_demo.pdf` | Simple 4-page document |
| `complex_attention_is_all_you_need.pdf` | Multi-column, figures, formulas |
| `table_heavy_world_bank.pdf` | 84 pages, dense tables |

Run a smoke test yourself:

```bash
# PaddleVL GPU on the World Bank report
python smoke_tests/run_one.py --backend paddle_vl --device gpu --pdf examples/pdfs/table_heavy_world_bank.pdf

# PaddleVL CPU
python smoke_tests/run_one.py --backend paddle_vl --device cpu --pdf examples/pdfs/simple_adobe_demo.pdf
```

Output is written to `smoke_tests/<pdf_name>/<backend_device>/`.

## Related modules

### Knowledge graph — `kg_table_relationships.ipynb`

Consumes a `document.json` produced by this parser and builds a **Neo4j
knowledge graph** that encodes how every table in the document connects to
surrounding content (paragraphs, headings, captions, figures, other tables)
using structural rules, regex, embeddings, and a local LLM.

See [`kg_table_relationships.md`](kg_table_relationships.md) for full
documentation of the graph schema, relationship extraction pipeline, and
configuration.

### Evaluation notebooks

| Notebook | What it evaluates |
|----------|-------------------|
| `evaluate_pdf_parser_doclaynet.ipynb` | Layout localisation (IoU/F1), label accuracy, OCR (CER/WER), reading order (Kendall tau), table TEDS, formula extraction — against DocLayNet v1.2 annotations |
| `evaluate_pdf_parser_omnidocbench_v15.ipynb` | Layout, text, table, formula, reading-order, and Markdown/JSON quality metrics against OmniDocBench v1.5 (English, digital PDF pages) |
