"""Split class-encoded PNG masks (0/1/2) into per-class binary GeoTIFFs.

    python -m physhade.data.split_annotations --all-dir Dataset/main_model/all

Reads ``<all-dir>/class_masks/*.png`` (0 = background, 1 = building, 2 = shadow)
and writes ``building_footprint/<stem>.tif`` / ``annotated_shadow/<stem>.tif``,
copying the grid (CRS + transform) from the companion tile in ``<all-dir>/image``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from physhade.config import DATA_DIR


def true_stem(p: Path) -> str:
    """Filename without *any* suffixes: ``foo.tif.png`` -> ``foo``."""
    return p.with_suffix("").stem


def png_to_geotiff_mask(png_path: Path, image_dir: Path, out_dir: Path, label_value: int) -> Path | None:
    """Write ``out_dir/<stem>.tif`` = (class-PNG == label_value), on the image grid."""
    stem = true_stem(png_path)
    candidates = list(image_dir.glob(stem + ".tif*"))
    if not candidates:
        print(f"[WARN] GeoTIFF for {stem} not found; skipped.")
        return None

    with rasterio.open(candidates[0]) as src_img:
        meta = src_img.meta.copy()
        h, w = src_img.height, src_img.width

    mask_png = np.array(Image.open(png_path))
    if mask_png.ndim == 3:
        mask_png = mask_png[..., 0]
    if mask_png.shape != (h, w):
        print(f"[WARN] Size mismatch: {png_path.name}; skipped.")
        return None

    mask = (mask_png == label_value).astype(np.uint8) * 255
    meta.update(driver="GTiff", count=1, dtype="uint8")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask, 1)
    return out_path


def split_imagery(all_dir: Path) -> int:
    """Split every ``<all-dir>/class_masks/*.png`` into building (1) / shadow (2) GeoTIFFs."""
    class_dir, image_dir = all_dir / "class_masks", all_dir / "image"
    bf_dir, shadow_dir = all_dir / "building_footprint", all_dir / "annotated_shadow"
    n = 0
    for png_path in sorted(class_dir.glob("*.png")):
        png_to_geotiff_mask(png_path, image_dir, bf_dir, label_value=1)
        png_to_geotiff_mask(png_path, image_dir, shadow_dir, label_value=2)
        n += 1
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all-dir", type=Path, default=DATA_DIR / "main_model" / "all")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    n = split_imagery(args.all_dir)
    print(f"split {n} class masks under {args.all_dir}")


if __name__ == "__main__":
    main()
