# Architecture

This document describes the as-built design of the Agentic PDF Parser pipeline.

## Guiding principle: the single-contract rule

Every backend must satisfy exactly one contract:

```
parse_page(page_input: PageInput) -> BackendPageResult(
    page: Page,   # already-normalized canonical Page
    raw:  dict,   # JSON-serializable backend-native output
)
```

**Normalization is internal to each backend.** The orchestrator never
normalizes. It calls `backend.parse_page()` and receives a `Page` that
is already expressed in the canonical schema.

This isolates backend-specific types (`PaddleOCRVLResult`, etc.) entirely
inside their own packages. Nothing outside `backends/paddle_vl/` ever imports
a backend-native type.

## Pipeline stages

```
PDF file
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  Orchestrator (orchestrator.py)                         │
│                                                         │
│  1. Hash PDF (SHA-256) + read metadata (PyMuPDF)        │
│  2. For each page:                                      │
│      a. Rasterize page → PNG (PyMuPDF, configurable DPI)│
│      b. backend.parse_page(PageInput)                   │
│           └─► [backend owns inference + normalization]  │
│           └─► returns BackendPageResult                 │
│      c. If keep_raw: write raw/page_XXXX.json           │
│      d. Extract + crop figure assets → assets/          │
│      e. Append normalized Page to list                  │
│  3. Build NormalizedDocument from Pages                 │
│  4. Export document.json (json_export)                  │
│  5. Export document.md  (markdown_export)               │
└─────────────────────────────────────────────────────────┘
```

## Component map

```
src/agentic_pdf_parser/
│
├── __init__.py          Public API: parse_pdf, ParseConfig, load_config, …
├── api.py               parse_pdf() → ParseResult (thin wrapper over orchestrator)
├── cli.py               Typer parse-pdf command (thin: builds config, calls api)
├── config.py            Pydantic models: ParseConfig, PaddleVLConfig, …
├── config_loader.py     YAML loading + CLI override merging → ParseConfig
├── schema.py            Canonical data models (NormalizedDocument and children)
├── orchestrator.py      Pipeline coordination (see stages above)
├── rasterize.py         PDF → per-page PNG via PyMuPDF
├── assets.py            Figure crop + save
├── debug_viz.py         Bbox-annotated page image generation (--debug)
│
├── export/
│   ├── json_export.py   NormalizedDocument ↔ JSON
│   └── markdown_export.py NormalizedDocument → Markdown
│
├── backends/
│   ├── base.py          PageInput, BackendPageResult, ParserBackend (Protocol)
│   ├── registry.py      build_backend(config) factory (explicit match, no plugin system)
│   │
│   └── paddle_vl/
│       ├── device.py    paddle device resolution
│       ├── normalizer.py raw dict → Page  (internal to paddle_vl)
│       └── backend.py   PaddleVLBackend: load / parse_page / unload
│
└── utils/
    ├── hashing.py       SHA-256 file hash
    └── logging.py       Logging helpers
```

## Backend lifecycle

The orchestrator drives every backend through the same three-step lifecycle:

```python
backend.load()        # load weights into memory / init pipeline
try:
    for page in pages:
        result = backend.parse_page(page_input)
finally:
    backend.unload()  # release weights, free GPU memory
```

`load()` is where heavy work (downloading/loading model weights) happens.
`__init__` only stores config and resolves device — it never imports
paddleocr.

## Backend internals

### PaddleVL (`paddle_vl`)

```
PaddleVLBackend.load()
   ├── subprocess: llama-server.exe (persistent, polls /health until ready)
   └── PaddleOCRVL(vl_rec_backend="llama-cpp-server", ...)

PaddleVLBackend.parse_page(page_input)
   │
   ├── pipeline.predict(str(page_input.temp_image_path))
   │     PP-DocLayoutV3: detect regions → bboxes + labels
   │     For each region: POST llama-server → VLM recognition
   │     → PaddleOCRVLResult
   │
   ├── _result_to_raw(result)
   │     → JSON-serializable dict
   │
   └── normalizer.raw_result_to_page(raw, page_input)
         _block_to_schema() for each block in parsing_res_list
         → canonical Page

PaddleVLBackend.unload()
   └── kill() llama-server.exe subprocess
```

Raw output: `{"page_index": 0, "width": ..., "height": ..., "parsing_res_list": [...]}`

## Config and CLI layering

```
CLI flags  ──►┐
              ├──► build_config() ──► ParseConfig ──► orchestrator
YAML file  ──►┘                          ▲
                                         │
                               Pydantic defaults
```

Precedence: **CLI flags › YAML › Pydantic defaults**.

`backend` has no Pydantic default and is therefore required from CLI or YAML.
`output_dir` defaults to `Path("output")` in `build_config` if absent from both.

## Canonical JSON as source of truth

`document.json` is written first. `document.md` is derived from it by the
`markdown_export` module — there is no direct path from backend output to
Markdown. This means:

- Re-generating Markdown from a saved JSON requires only `json_export.read` +
  `markdown_export.write`, with no backend dependency.
- The JSON file is a complete audit record of what was parsed.

## Fixed invariants

These are not configuration options. They are always on:

| Invariant | Where enforced |
|-----------|----------------|
| Tables always extracted | Backend normalizers; `BlockType = "table"` |
| Formulas always extracted | Backend normalizers; `BlockType = "formula"` |
| Images/figures always extracted | `assets.extract_figures()` in orchestrator |
| `document.json` always written | `orchestrator.run()` |
| `document.md` always written | `orchestrator.run()` |
| Page markers (`<!-- page: N -->`) always in Markdown | `markdown_export.write()` |

## Dependency isolation

| Component | Hard deps | Optional deps |
|-----------|-----------|---------------|
| Core schema, orchestrator, exporters | pydantic, pymupdf, pillow, pyyaml, lxml, beautifulsoup4, typer, rich | — |
| PaddleVL backend | *(above)* + paddlepaddle, paddleocr, openai | — (VLM via llama-server.exe binary) |

The backend requires **llama.cpp** binaries and GGUF model files installed
locally — these are not Python packages and are not managed by pip.

All backend-specific Python imports are deferred to `load()`. The core package
can be imported and all non-backend functionality used without any inference
framework installed.
