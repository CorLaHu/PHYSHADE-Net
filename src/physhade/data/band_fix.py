"""Swap class values 1 <-> 2 in Supervisely ``masks_machine`` PNGs.

One-off annotation-repair utility. Requires GDAL (``osgeo``), which is not a
pip dependency of this package - install it via conda (it ships with the
``physhade`` env) or ``pip install gdal`` matching your libgdal.

    python -m physhade.data.band_fix --input-folder A --output-folder B
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from physhade.config import DATA_DIR

_DEFAULT_IN = DATA_DIR / "annotation_exports" / "dataset 2025-04-23 19-28-48" / "masks_machine"
_DEFAULT_OUT = DATA_DIR / "annotation_exports" / "dataset 2025-04-23 19-28-48" / "mask_machine_fixed"


def remap_classes(in_folder: str | Path, out_folder: str | Path) -> int:
    """Rewrite every ``*.png`` in ``in_folder`` with classes 1 and 2 swapped."""
    from osgeo import gdal  # lazy: GDAL is an optional, conda-only dependency

    in_folder, out_folder = Path(in_folder), Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in os.listdir(in_folder):
        if not name.lower().endswith(".png"):
            continue
        ds = gdal.Open(str(in_folder / name))
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()

        out_arr = np.copy(arr)
        out_arr[arr == 1] = 2
        out_arr[arr == 2] = 1

        mem = gdal.GetDriverByName("MEM").Create("", ds.RasterXSize, ds.RasterYSize, 1, band.DataType)
        mem.SetProjection(ds.GetProjection())
        mem.SetGeoTransform(ds.GetGeoTransform())
        mem_band = mem.GetRasterBand(1)
        mem_band.WriteArray(out_arr)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            mem_band.SetNoDataValue(nodata)

        gdal.GetDriverByName("PNG").CreateCopy(str(out_folder / name), mem, strict=0)
        mem = ds = None
        n += 1
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-folder", type=Path, default=_DEFAULT_IN)
    p.add_argument("--output-folder", type=Path, default=_DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    n = remap_classes(args.input_folder, args.output_folder)
    print(f"remapped {n} masks -> {args.output_folder}")


if __name__ == "__main__":
    main()
