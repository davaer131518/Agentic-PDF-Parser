"""Markdown export derived from canonical NormalizedDocument.

Rules:
- Simple tables (all 1×1 cells, no pipes or newlines in text) → GFM table.
- Complex tables (merged cells, embedded newlines) → raw HTML ``<table>``.
- Formulas → LaTeX: inline ``$...$``, block ``$$\\n...\\n$$``.
- Figures → ``![alt](asset_path)`` + italicised caption on the next line.
- Caption blocks referenced by a figure or table are skipped at the top level.
- Page boundaries are marked with HTML comments: ``<!-- page: N -->``.
"""
from __future__ import annotations

from pathlib import Path

from ..schema import Block, NormalizedDocument, Page, Table


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _is_simple_table(table: Table) -> bool:
    """True when every cell is 1×1 and contains no ``|`` or newline."""
    return all(
        cell.rowspan == 1
        and cell.colspan == 1
        and "\n" not in cell.text
        and "|" not in cell.text
        for cell in table.cells
    )


def _render_table_gfm(table: Table) -> str:
    """Render as GitHub-Flavored Markdown table."""
    if not table.cells:
        return ""

    grid: dict[int, dict[int, str]] = {}
    for cell in table.cells:
        grid.setdefault(cell.row, {})[cell.col] = cell.text

    lines: list[str] = []
    for row_idx in sorted(grid):
        row_cells = [grid[row_idx].get(c, "") for c in range(table.cols)]
        lines.append("| " + " | ".join(row_cells) + " |")
        if row_idx == 0:
            lines.append("| " + " | ".join("---" for _ in range(table.cols)) + " |")

    return "\n".join(lines)


def _render_table(table: Table) -> str:
    if not table.cells:
        return ""
    if _is_simple_table(table):
        return _render_table_gfm(table)
    if table.html:
        return table.html
    # Fallback: best-effort GFM even for complex tables
    return _render_table_gfm(table)


# ---------------------------------------------------------------------------
# Block renderer
# ---------------------------------------------------------------------------


def _render_block(
    block: Block,
    captions: dict[str, str],
    referenced_captions: set[str],
) -> str:
    """Return the Markdown representation of a single block."""
    match block.type:
        case "heading":
            hashes = "#" * min(block.level or 1, 6)
            return f"{hashes} {block.text or ''}"

        case "paragraph":
            return block.text or ""

        case "list":
            if not block.children:
                return ""
            return "\n".join(
                _render_block(c, captions, referenced_captions)
                for c in block.children
            )

        case "list_item":
            depth = block.level or 1
            indent = "  " * (depth - 1)
            return f"{indent}- {block.text or ''}"

        case "section":
            if not block.children:
                return ""
            return "\n\n".join(
                _render_block(c, captions, referenced_captions)
                for c in block.children
                if _render_block(c, captions, referenced_captions).strip()
            )

        case "table":
            if not block.table:
                return ""
            table_md = _render_table(block.table)
            if block.table.caption_id and block.table.caption_id in captions:
                caption_text = captions[block.table.caption_id]
                return f"{table_md}\n*{caption_text}*"
            return table_md

        case "figure":
            if not block.figure:
                return ""
            alt = block.figure.alt_text or ""
            path = block.figure.asset_path
            img_line = f"![{alt}]({path})"
            if block.figure.caption_id and block.figure.caption_id in captions:
                caption_text = captions[block.figure.caption_id]
                return f"{img_line}\n*{caption_text}*"
            return img_line

        case "formula":
            if not block.formula:
                return ""
            latex = block.formula.latex or ""
            if block.formula.inline:
                return f"${latex}$"
            return f"$$\n{latex}\n$$"

        case "caption":
            if block.id in referenced_captions:
                return ""
            return f"*{block.text or ''}*"

        case "code":
            return f"```\n{block.text or ''}\n```"

        case "footnote" | "page_header" | "page_footer":
            return f"*{block.text or ''}*"

        case _:
            return block.text or ""


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def _render_page(page: Page) -> str:
    """Render all blocks of *page* as Markdown text."""
    # Pass 1: collect caption text by block id
    captions: dict[str, str] = {
        b.id: (b.text or "")
        for b in page.blocks
        if b.type == "caption"
    }

    # Pass 2: collect caption ids referenced by figures and tables
    referenced_captions: set[str] = set()
    for b in page.blocks:
        if b.type == "figure" and b.figure and b.figure.caption_id:
            referenced_captions.add(b.figure.caption_id)
        if b.type == "table" and b.table and b.table.caption_id:
            referenced_captions.add(b.table.caption_id)

    # Pass 3: render blocks, joining non-empty chunks with a blank line
    chunks: list[str] = []
    for b in page.blocks:
        chunk = _render_block(b, captions, referenced_captions)
        if chunk.strip():
            chunks.append(chunk.strip())

    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Public writer
# ---------------------------------------------------------------------------


def write(doc: NormalizedDocument, output_path: Path) -> None:
    """Write Markdown derived from *doc* to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    if doc.document.title:
        lines.append(f"# {doc.document.title}")
        lines.append("")

    for page in doc.pages:
        lines.append(f"<!-- page: {page.number} -->")
        lines.append("")
        content = _render_page(page)
        if content:
            lines.append(content)
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
