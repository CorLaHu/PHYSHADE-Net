"""Otsu-threshold a single-band raster into a binary shadow mask.

Standalone utility: ``python -m physhade.inference.Otsu --input X.tif --output Y.tif``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from physhade.config import OUTPUT_DIR


def otsu_shadow_mask(
    in_path: str | Path,
    out_path: str | Path,
    blur_sigma: float = 1.0,
    percentile_floor: float = 2.0,
) -> Path:
    """Read band 1 of ``in_path``, threshold it, write a uint8 0/1 GeoTIFF."""
    with rasterio.open(in_path) as src:
        band = src.read(1).astype(np.float32)
        profile = src.profile
        nodata = src.nodata

    if nodata is not None:
        band[band == nodata] = np.nan

    band_blur = gaussian_filter(band, sigma=blur_sigma)
    lo, hi = np.nanmin(band_blur), np.nanmax(band_blur)
    band_norm = (band_blur - lo) / (hi - lo) if hi > lo else np.zeros_like(band_blur)

    valid = band_norm[~np.isnan(band_norm)]
    threshold = max(threshold_otsu(valid), np.nanpercentile(band_norm, percentile_floor))

    shadow_mask = (band_norm <= threshold).astype(np.uint8)  # shadow = darker than threshold

    profile.update(dtype=rasterio.uint8, count=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(shadow_mask, 1)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="single-band raster")
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "main_model" / "otsu" / "otsu_mask.tif",
        help="destination GeoTIFF (default: Output/main_model/otsu/otsu_mask.tif)",
    )
    p.add_argument("--blur-sigma", type=float, default=1.0)
    p.add_argument("--percentile-floor", type=float, default=2.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = otsu_shadow_mask(args.input, args.output, args.blur_sigma, args.percentile_floor)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
