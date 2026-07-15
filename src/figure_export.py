"""
Shared figure-export helpers for Journal of Chemometrics (Wiley) submission format
(addendum Task 8 -- format only, no analysis/data changes).

Wiley final-art requirements applied here:
  - TIFF (LZW-compressed, lossless) at 600 dpi for line art, alongside a 300 dpi PNG
    kept for internal preview/working use (Word/Jenni preview PNG better than TIFF).
  - Figure width within 80-180 mm, max reproduction size 140x200 mm.
  - Each figure a single file, named unambiguously with its figure number.
"""

import os

MM_PER_INCH = 25.4
MAX_WIDTH_MM = 180
MAX_HEIGHT_MM = 200


def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH


def in_to_mm(inches: float) -> float:
    return inches * MM_PER_INCH


def _enforce_size_limit(path: str, dpi: int, compression: str = None,
                         max_width_mm: float = MAX_WIDTH_MM, max_height_mm: float = MAX_HEIGHT_MM) -> None:
    """Safety net: matplotlib's bbox_inches='tight' can grow the saved canvas beyond the
    nominal figsize when a legend or a rotated colorbar label overflows the axes (that
    overflow is driven by text length in inches, not by the target figsize, so shrinking
    figsize alone does not reliably fix it). Reads the actual saved image back and, if it
    still exceeds the Wiley limits, downscales the pixel raster in place so the physical
    size (at the stated dpi) is guaranteed compliant."""
    from PIL import Image
    with Image.open(path) as im:
        width_px, height_px = im.size
        width_mm = width_px / dpi * MM_PER_INCH
        height_mm = height_px / dpi * MM_PER_INCH
        scale = min(max_width_mm / width_mm, max_height_mm / height_mm, 1.0)
        if scale >= 1.0:
            return
        new_size = (max(1, int(width_px * scale)), max(1, int(height_px * scale)))
        resized = im.convert('RGB').resize(new_size, Image.LANCZOS)
    save_kwargs = {'dpi': (dpi, dpi)}
    if compression:
        save_kwargs['compression'] = compression
    resized.save(path, **save_kwargs)


def save_figure(fig, figures_dir: str, filename_stem: str) -> tuple:
    """Save a matplotlib Figure as both TIFF (600 dpi, LZW) and PNG (300 dpi) under the
    same base filename (which must already contain the unambiguous fig_N figure number,
    e.g. 'fig_5_response_surfaces'), then enforce the Wiley 180x200 mm size limit by
    downscaling in place if the saved canvas overshot it. Returns (tiff_path, png_path)."""
    os.makedirs(figures_dir, exist_ok=True)
    tiff_path = os.path.join(figures_dir, f"{filename_stem}.tiff")
    png_path = os.path.join(figures_dir, f"{filename_stem}.png")
    fig.savefig(tiff_path, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    _enforce_size_limit(tiff_path, dpi=600, compression='tiff_lzw')
    _enforce_size_limit(png_path, dpi=300)
    return tiff_path, png_path
