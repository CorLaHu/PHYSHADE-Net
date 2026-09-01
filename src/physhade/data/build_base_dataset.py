"""Build ``Dataset/base_model/{train,val}`` from AISD for RGB U-Net pretraining.

    python -m physhade.data.build_base_dataset          # AISD from data_sources/ or Dataset/aisd/

AISD masks in this dataset encode ``1 = sunlit`` / ``0 = shadow``; the model wants
``shadow = 1``, so masks are flipped by default (``--no-flip`` to keep them as-is).
``Test51`` is never built into ``base_model`` - it stays eval-only under ``Dataset/aisd``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from physhade.config import DATA_DIR, SOURCES_DIR


def _resolve_aisd(cli: Path | None) -> Path:
    for cand in [cli, DATA_DIR / "aisd", SOURCES_DIR / "AISD"]:
        if cand and (cand / "Train412").is_dir():
            return cand
    raise SystemExit("AISD/Train412 not found - run `build_dataset` or pass --aisd")


def _convert_split(src_split: Path, dst_split: Path, flip: bool) -> tuple[int, float]:
    (dst_split / "shadow").mkdir(parents=True, exist_ok=True)
    (dst_split / "mask").mkdir(parents=True, exist_ok=True)
    shadow_frac_num = shadow_frac_den = 0.0
    n = 0
    for img_path in sorted((src_split / "shadow").glob("*.tif")):
        msk_path = src_split / "mask" / img_path.name
        if not msk_path.exists():
            continue
        with rasterio.open(img_path) as s:
            rgb = s.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        with rasterio.open(msk_path) as s:
            m = s.read(1)
        shadow = (m == 0) if flip else (m > 0)
        Image.fromarray(rgb).save((dst_split / "shadow" / img_path.name).with_suffix(".tif"))
        Image.fromarray((shadow * 255).astype(np.uint8), mode="L").save(
            (dst_split / "mask" / img_path.name).with_suffix(".tif")
        )
        shadow_frac_num += float(shadow.sum())
        shadow_frac_den += shadow.size
        n += 1
    return n, (shadow_frac_num / shadow_frac_den if shadow_frac_den else 0.0)


def build(aisd: Path, out: Path, flip: bool, overwrite: bool) -> None:
    base = out / "base_model"
    if overwrite:
        shutil.rmtree(base, ignore_errors=True)
    for src_name, dst_name in [("Train412", "train"), ("Val51", "val")]:
        n, frac = _convert_split(aisd / src_name, base / dst_name, flip)
        print(f"{dst_name}: {n} tiles, shadow pixel fraction {frac:.3f}")
        if flip and frac > 0.5:
            print(f"  WARNING: shadow fraction > 0.5 for {dst_name} - check --no-flip")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aisd", type=Path, default=None, help="dir containing Train412/ Val51/")
    p.add_argument("--out", type=Path, default=DATA_DIR)
    p.add_argument("--no-flip", action="store_true", help="keep AISD mask polarity as-is")
    p.add_argument("--overwrite", action="store_true", help="wipe Dataset/base_model first")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build(_resolve_aisd(args.aisd), args.out, flip=not args.no_flip, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
