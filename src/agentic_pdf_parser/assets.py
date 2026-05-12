"""Figure asset extraction — crops figure regions from rasterized page images."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .schema import Page


def extract_figures(
    page_image: Image.Image,
    page: Page,
    assets_dir: Path,
) -> None:
    """Crop figure regions from *page_image* and update ``Figure.asset_path`` in-place.

    Only top-level blocks are inspected (nested figures inside section/list
    children are not extracted in v1).

    If a figure block has no bounding-box in its provenance, it is skipped
    and its ``asset_path`` is left unchanged.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_counter = 0

    for block in page.blocks:
        if block.type != "figure" or block.figure is None:
            continue

        fig_counter += 1
        page_label = f"p{page.number:04}"
        fig_name = f"{page_label}_fig{fig_counter:02}.png"
        asset_full_path = assets_dir / fig_name

        bbox = block.provenance.bbox
        if bbox is not None:
            dims = page.dimensions
            scale_x = dims.width_px / dims.width_pt
            scale_y = dims.height_px / dims.height_pt

            left = max(0, int(bbox.x0 * scale_x))
            top = max(0, int(bbox.y0 * scale_y))
            right = min(page_image.width, int(bbox.x1 * scale_x))
            bottom = min(page_image.height, int(bbox.y1 * scale_y))

            if right > left and bottom > top:
                cropped = page_image.crop((left, top, right, bottom))
                cropped.save(asset_full_path)
                block.figure.asset_path = f"assets/{fig_name}"
                continue

        # No valid bbox — skip this figure; asset_path remains as set by normalizer
