# Canonical Schema

`document.json` is the single source of truth for every parse run.
`document.md` is derived from it; no backend output reaches the Markdown
exporter directly.

All models are defined in `src/agentic_pdf_parser/schema.py` using Pydantic v2.

## Top-level structure

```json
{
  "schema_version": "1.0",
  "document": { ... },
  "backend":  { ... },
  "pages":    [ ... ]
}
```

### `NormalizedDocument`

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `"1.0"` | Literal version tag |
| `document` | `DocumentMetadata` | Source file metadata |
| `backend` | `BackendMetadata` | Which backend/model/device was used |
| `pages` | `list[Page]` | Pages in document order |

### `DocumentMetadata`

| Field | Type | Description |
|-------|------|-------------|
| `source_filename` | `str` | Original filename |
| `source_sha256` | `str` | SHA-256 hex digest of the PDF |
| `num_pages` | `int` | Total page count |
| `title` | `str \| null` | From PDF metadata |
| `author` | `str \| null` | From PDF metadata |
| `created_at` | `datetime \| null` | From PDF `D:…` date string (UTC) |

### `BackendMetadata`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Public backend name: `paddle_vl` |
| `version` | `str \| null` | Model version string (set after `load()`) |
| `model_id` | `str \| null` | HuggingFace ID or pipeline name |
| `device` | `str` | Resolved device string, e.g. `"cpu"`, `"cuda"`, `"gpu:0"` |
| `options` | `dict` | Reserved for backend-specific metadata |

## Page

```json
{
  "index": 0,
  "number": 1,
  "dimensions": {
    "width_pt": 595.0, "height_pt": 842.0,
    "width_px": 1654,  "height_px": 2339,
    "dpi": 200
  },
  "blocks": [ ... ],
  "backend_metadata": { "backend": "paddle_vl", "backend_version": "..." }
}
```

### `PageDimensions`

| Field | Type | Description |
|-------|------|-------------|
| `width_pt` / `height_pt` | `float` | Page size in PDF points (72 pt = 1 inch) |
| `width_px` / `height_px` | `int` | Rasterized image dimensions |
| `dpi` | `int` | DPI used for rasterization |

## Block

Blocks are the atomic content units within a page. They are ordered by
`reading_order` (0-based integer reflecting the backend's reading-order
judgement).

```json
{
  "id": "p0001_b0003",
  "type": "table",
  "text": null,
  "level": null,
  "reading_order": 3,
  "provenance": { "backend": "paddle_vl", "bbox": { ... } },
  "table": { ... },
  "figure": null,
  "formula": null,
  "children": null
}
```

### `Block` fields

| Field | Type | Notes |
|-------|------|-------|
| `id` | `str` | Format `p{page_number:04}_b{index:04}`, e.g. `p0001_b0003` |
| `type` | `BlockType` | See block types below |
| `text` | `str \| null` | Primary text content (headings, paragraphs, etc.) |
| `level` | `int \| null` | Heading depth (1–6) or list nesting level |
| `reading_order` | `int` | Position in the page's reading sequence (0-based) |
| `provenance` | `Provenance` | Source metadata and bounding box |
| `table` | `Table \| null` | Set when `type == "table"` |
| `figure` | `Figure \| null` | Set when `type == "figure"` |
| `formula` | `Formula \| null` | Set when `type == "formula"` |
| `children` | `list[Block] \| null` | For list and section trees |

### Block types (`BlockType`)

| Type | Description |
|------|-------------|
| `heading` | Section heading; `level` = 1–6 |
| `paragraph` | Body text |
| `list` | List container; children hold `list_item` blocks |
| `list_item` | Single list item; `level` = nesting depth |
| `caption` | Figure or table caption |
| `figure` | Image/chart; `figure.asset_path` points to `assets/` |
| `table` | Structured table; `table.cells` + `table.html` always set |
| `formula` | Mathematical expression; `formula.latex` holds LaTeX |
| `code` | Code block |
| `footnote` | Footnote text |
| `page_header` | Running header text |
| `page_footer` | Running footer text |
| `section` | Logical section container with nested children |

## Provenance

Every block carries a `Provenance` object linking it to its backend source.

| Field | Type | Description |
|-------|------|-------------|
| `backend` | `str` | Backend name (`paddle_vl`) |
| `backend_version` | `str \| null` | Version string from the model |
| `bbox` | `BoundingBox \| null` | Bounding box in PDF points, top-left origin |
| `polygon` | `list[tuple[float,float]] \| null` | Polygon points (reserved) |
| `confidence` | `float \| null` | Model confidence (backend-dependent) |
| `extra` | `dict` | Arbitrary backend-specific passthrough |

### `BoundingBox`

Coordinates in **PDF points** (72 pt = 1 inch), **top-left origin**.

```json
{ "x0": 56.7, "y0": 113.4, "x1": 538.6, "y1": 141.7 }
```

Both backends convert their native pixel/token coordinates to this
representation inside their normalizers before returning a `Page`.

## Rich content types

### `Table`

| Field | Type | Description |
|-------|------|-------------|
| `rows` | `int` | Row count |
| `cols` | `int` | Column count |
| `cells` | `list[TableCell]` | Flat list of cells |
| `html` | `str \| null` | Full HTML `<table>…</table>` string (always set) |
| `otsl` | `str \| null` | OTSL string (reserved; currently `null`) |
| `caption_id` | `str \| null` | Block ID of the associated caption |

#### `TableCell`

| Field | Type | Default |
|-------|------|---------|
| `row` | `int` | — |
| `col` | `int` | — |
| `rowspan` | `int` | `1` |
| `colspan` | `int` | `1` |
| `text` | `str` | — |
| `is_header` | `bool` | `False` |
| `bbox` | `BoundingBox \| null` | `null` |

**Markdown rendering rule:** simple tables (no merged cells, no `|` or
newlines in cell text) are rendered as GFM tables; all others fall back to
the raw HTML string.

### `Figure`

| Field | Type | Description |
|-------|------|-------------|
| `asset_path` | `str` | Path relative to `output_dir`, e.g. `assets/p0001_fig00.png` |
| `caption_id` | `str \| null` | Block ID of the associated caption |
| `alt_text` | `str \| null` | Alt text for Markdown `![alt](path)` |

Figure assets are cropped from the rasterized page by the orchestrator
immediately after `backend.parse_page()` returns. The `asset_path` field is
empty (`""`) in the `Page` returned by the backend; the orchestrator fills it.

### `Formula`

| Field | Type | Description |
|-------|------|-------------|
| `latex` | `str \| null` | LaTeX expression (without delimiters) |
| `mathml` | `str \| null` | MathML (not currently populated) |
| `inline` | `bool` | `True` for inline (`$…$`), `False` for block (`$$…$$`) |

LaTeX delimiters (`$$…$$`, `$…$`) are stripped by the backend normalizers
and re-applied by the Markdown exporter according to `inline`.

## Markdown rendering rules

| Block type | Markdown output |
|-----------|----------------|
| `heading` (level N) | `#{N} text` |
| `paragraph` | plain text |
| `list` | nested via children |
| `list_item` | `- text` (indented by level) |
| `table` (simple) | GFM `\| … \|` table |
| `table` (complex) | raw `<table>…</table>` HTML |
| `formula` (inline) | `$latex$` |
| `formula` (block) | `$$\nlatex\n$$` |
| `figure` | `![alt](asset_path)` |
| `caption` (referenced) | suppressed (merged into figure/table) |
| `caption` (standalone) | `*caption text*` |
| `code` | ` ``` \ntext\n ``` ` |
| `footnote`, `page_header`, `page_footer` | `*text*` |
| Page boundary | `<!-- page: N -->` |

Page markers are always emitted at the start of every page, unconditionally.

## Minimal JSON example

```json
{
  "schema_version": "1.0",
  "document": {
    "source_filename": "report.pdf",
    "source_sha256": "a3f2...",
    "num_pages": 2,
    "title": "Quarterly Report",
    "author": null,
    "created_at": null
  },
  "backend": {
    "name": "paddle_vl",
    "version": "PaddleOCR-VL-1.5",
    "model_id": "PaddleOCR-VL-1.5",
    "device": "cpu",
    "options": {}
  },
  "pages": [
    {
      "index": 0,
      "number": 1,
      "dimensions": {
        "width_pt": 595.0, "height_pt": 842.0,
        "width_px": 1654, "height_px": 2339, "dpi": 200
      },
      "blocks": [
        {
          "id": "p0001_b0000",
          "type": "heading",
          "text": "Quarterly Report",
          "level": 1,
          "reading_order": 0,
          "provenance": {
            "backend": "paddle_vl",
            "bbox": { "x0": 56.7, "y0": 56.7, "x1": 538.6, "y1": 85.0 }
          },
          "table": null, "figure": null, "formula": null, "children": null
        },
        {
          "id": "p0001_b0001",
          "type": "table",
          "text": null,
          "level": null,
          "reading_order": 1,
          "provenance": {
            "backend": "paddle_vl",
            "bbox": { "x0": 56.7, "y0": 100.0, "x1": 538.6, "y1": 300.0 }
          },
          "table": {
            "rows": 2, "cols": 2,
            "cells": [
              { "row": 0, "col": 0, "rowspan": 1, "colspan": 1,
                "text": "Item", "is_header": true },
              { "row": 0, "col": 1, "rowspan": 1, "colspan": 1,
                "text": "Value", "is_header": true },
              { "row": 1, "col": 0, "rowspan": 1, "colspan": 1,
                "text": "Revenue", "is_header": false },
              { "row": 1, "col": 1, "rowspan": 1, "colspan": 1,
                "text": "$1M", "is_header": false }
            ],
            "html": "<table><tr><th>Item</th><th>Value</th></tr><tr><td>Revenue</td><td>$1M</td></tr></table>",
            "otsl": null,
            "caption_id": null
          },
          "figure": null, "formula": null, "children": null
        }
      ],
      "backend_metadata": {
        "backend": "paddle_vl",
        "backend_version": "PaddleOCR-VL-1.5"
      }
    }
  ]
}
```
