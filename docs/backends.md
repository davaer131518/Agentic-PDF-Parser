# Backends

This document covers the supported backend: its public name, underlying
model, dependency installation, device handling, internal design, and the
raw output format it produces.

## The single-contract rule (enforced by all backends)

Every backend satisfies exactly one contract:

```python
def parse_page(self, page_input: PageInput) -> BackendPageResult:
    ...

# BackendPageResult:
#   page: Page   — already-normalized canonical Page
#   raw:  dict   — JSON-serializable backend-native output
```

**Normalization is entirely internal to each backend's `parse_page()`.**
The orchestrator never calls a normalizer. It calls `backend.parse_page()`
and receives a `Page` that is already expressed in the canonical schema.

Backend-native types (PaddleOCR result dicts, etc.) never
escape their package boundary.

## Backend lifecycle (same for all backends)

```python
backend.load()      # load weights / start server / init pipeline
try:
    result = backend.parse_page(page_input)
finally:
    backend.unload()  # release weights, terminate server, free GPU memory
```

`__init__` only stores config and resolves the device string. All heavy
imports (`paddleocr`) and process launches (`llama-server`) are deferred
to `load()` so that the rest of the package can be used without any
inference framework installed.

---

## `paddle_vl` — PaddleOCR-VL-1.5

### Overview

A two-stage layout-aware document parsing pipeline. **PP-DocLayoutV3** (Python,
runs in the same process) detects regions; **`llama-server.exe`** (llama.cpp
HTTP server, spawned as a subprocess) runs the VLM to recognise content within
each region. PP-DocLayoutV3 is an **internal component** — it is not exposed
as a separate public backend name.

- **Public name:** `paddle_vl`
- **Layout detection:** PP-DocLayoutV3 via `PaddleOCRVL` (PaddleOCR 3.x)
- **VLM recognition:** `PaddleOCR-VL-1.5` GGUF served by `llama-server.exe`
- **Strengths:** Dense tables (each table gets a dedicated VLM call on just
  that region), multi-language, scanned documents

### Prerequisites

1. [llama.cpp](https://github.com/ggerganov/llama.cpp/releases) binaries in
   `<llama_cpp_dir>/`.
2. GGUF model files in `<llama_cpp_dir>/models/`:
   - `PaddleOCR-VL-1.5.gguf`
   - `PaddleOCR-VL-1.5-mmproj.gguf`

### Installation

```bash
# Core Python dependencies
pip install -e ".[paddle]"
# Installs: paddlepaddle (CPU), paddleocr>=3.0, openai>=1.63

# GPU PaddlePaddle (replace cu123 with your CUDA version)
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

The VLM component runs through `llama-server` — no PyTorch required.

### Device handling

| `--device` / `DeviceChoice` | PP-DocLayoutV3 | llama-server flag | Notes |
|-----------------------------|----------------|-------------------|-------|
| `auto` | `"gpu:0"` if CUDA, else `"cpu"` | `-ngl -1` / `-ngl 0` | Detects via `paddle.device.is_compiled_with_cuda()` |
| `gpu` | `"gpu:0"` | `-ngl -1` | Raises `RuntimeError` if no CUDA device |
| `cpu` | `"cpu"` | `-ngl 0` | Noticeably slower for page parsing |

### Sub-config (`config.paddle_vl`)

```yaml
paddle_vl:
  llama_cpp_dir: "C:/llama-cpp"           # directory containing llama-server.exe
  gguf_model_path: "C:/llama-cpp/models/PaddleOCR-VL-1.5.gguf"
  mmproj_path: "C:/llama-cpp/models/PaddleOCR-VL-1.5-mmproj.gguf"
  server_port: 8080           # HTTP port for llama-server; change if 8080 is taken
  use_doc_orientation_classify: false  # auto-rotate skewed scans
  use_doc_unwarping: false             # correct perspective distortion
```

Enable orientation classification and unwarping for photographed or scanned
documents.

### Internal flow

```
PaddleVLBackend.load()
  ├─ subprocess: llama-server.exe
  │    --model <gguf_model_path> --mmproj <mmproj_path>
  │    --port <server_port> -ngl <n_gpu_layers>
  │    (polls GET /health until ready)
  │
  └─ PaddleOCRVL(
         vl_rec_backend="llama-cpp-server",
         vl_rec_server_url="http://localhost:<server_port>",
         device=<paddle_device>,
     )

PaddleVLBackend.parse_page(page_input)
  │
  ├─ pipeline.predict(str(page_input.temp_image_path))
  │    PP-DocLayoutV3: detect layout regions → bboxes + labels
  │    For each region: POST to llama-server → VLM recognition result
  │    → PaddleOCRVLResult (dict-like)
  │
  ├─ _result_to_raw(result)
  │    → JSON-serializable dict
  │
  └─ normalizer.raw_result_to_page(raw, page_input)
       _block_to_schema() for each block in parsing_res_list
         _LABEL_MAP → (BlockType, level)
         _parse_table_html() for table blocks (BeautifulSoup + lxml)
         _strip_latex_delimiters() for formula blocks
         _bbox_to_schema() pixel → PDF points
       → canonical Page

PaddleVLBackend.unload()
  └─ llama-server.exe process is killed (SIGKILL on Windows)
```

### Raw output format

Stored in `raw/page_XXXX.json` when `--keep-raw` is set:

```json
{
  "page_index": 0,
  "width": 595,
  "height": 842,
  "parsing_res_list": [
    {
      "block_label": "doc_title",
      "block_content": "My Document Title",
      "block_bbox": [10, 20, 500, 50],
      "block_id": 0
    },
    {
      "block_label": "table",
      "block_content": "<table><tr><th>Name</th><th>Value</th></tr>…</table>",
      "block_bbox": [10, 110, 500, 300],
      "block_id": 1
    },
    {
      "block_label": "display_formula",
      "block_content": "$$E = mc^2$$",
      "block_bbox": [10, 310, 500, 350],
      "block_id": 2
    }
  ]
}
```

### Block label vocabulary

The normalizer maps Paddle's `block_label` strings to canonical `BlockType`
values. Key mappings:

| Paddle label | Canonical type | Notes |
|--------------|----------------|-------|
| `doc_title` | `heading` (level 1) | |
| `paragraph_title`, `abstract_title`, `content_title`, `reference_title` | `heading` (level 2) | |
| `table_title`, `figure_title`, `chart_title` | `caption` | |
| `text`, `content`, `abstract`, `ocr`, `vertical_text` | `paragraph` | |
| `table` | `table` | HTML in `block_content` |
| `formula`, `display_formula` | `formula` (block) | |
| `inline_formula` | `formula` (inline) | |
| `image`, `chart`, `seal` | `figure` | |
| `footnote` | `footnote` | |
| `header`, `header_image` | `page_header` | |
| `footer`, `footer_image` | `page_footer` | |

Unrecognised labels are silently skipped.

### Coordinate system note

Paddle bboxes are `[x1, y1, x2, y2]` in pixels (top-left origin) at the
pipeline's internal resolution. The normalizer converts to PDF points:

```
x_pt = (x_px / model_width)  × page_width_pt
y_pt = (y_px / model_height) × page_height_pt
```

`model_width`/`model_height` come from `result["width"]`/`result["height"]`
(the resolution the pipeline actually used), which may differ from
`page_input.dimensions` if PaddleOCR internally rescaled the image.

### Assumptions and known uncertainties

- `predict()` receives the PNG file path (not a PIL Image) to guarantee
  compatibility across PaddleX versions.
- `block_bbox` is assumed to always be a 4-element list; inputs with fewer
  elements produce `provenance.bbox = None`.
- Table content (`block_content`) is HTML produced directly by the pipeline's
  table structure recognition model; we re-parse it with BeautifulSoup.
- On Windows, `llama-server.exe` ignores `terminate()` (CTRL_BREAK); `kill()`
  is used in `unload()` to guarantee the process exits.

---

## Adding a new backend

1. Create `backends/<name>/` with `__init__.py`, `device.py`, `normalizer.py`,
   `backend.py`.
2. Implement `backend.py` so that `parse_page()` returns
   `BackendPageResult(page=<normalized Page>, raw=<dict>)`.
3. Add the public name to `BackendName` in `config.py`.
4. Add a branch to `build_backend()` in `backends/registry.py`.
5. Add unit tests with canned fixtures (no real model required).

The orchestrator requires no changes.
