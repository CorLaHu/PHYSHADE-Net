#!/usr/bin/env python
"""Regenerate the figures referenced by the README, into docs/figures/.

Run once the dataset is built:

    python scripts/make_figures.py

Produces:
    docs/figures/smear_showcase.png     - the pseudo-shadow accumulation, step by step
    docs/figures/assign_and_break.gif   - the blob separation, animated
    docs/figures/blob_map.png           - a per-tile height diagnostic (blob id | RGB + match)
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import rasterio  # noqa: E402

from physhade.config import DATA_DIR, SOURCES_DIR  # noqa: E402
from physhade.height.blob_showcase import _pick_tile, visualize_assign_and_break  # noqa: E402
from physhade.height.height_pipeline import (  # noqa: E402
    _estimate_tile,
    build_dissolved_buildings,
    layout,
    load_bag_heights,
    solar_angle_lookup,
)
from physhade.physics.smear_showcase import visualize_shadow_mosaic  # noqa: E402

FIG_DIR = REPO / "docs" / "figures"


def _smear_showcase() -> None:
    footprints = sorted((DATA_DIR / "main_model" / "all" / "building_footprint").glob("*winter*.tif"))
    if not footprints:
        raise SystemExit("no footprint rasters found - run `python -m physhade.data.build_dataset` first")
    out = visualize_shadow_mosaic(
        raster_path=str(footprints[len(footprints) // 2]),
        azimuth_deg=116.0,
        length_mapunits=(2.0, 15.0),
        decay_type="delayed_gradient",
        mask_buildings=True,
        erode_buildings_px=1,
        out_path=FIG_DIR / "smear_showcase.png",
    )
    print(f"wrote {out}")


def _assign_and_break_gif() -> None:
    root = DATA_DIR / "val_set"
    if not (root / "building_footprint").is_dir():
        print("skipping assign_and_break.gif - Dataset/val_set not built")
        return
    tile = _pick_tile(root)
    with rasterio.open(root / "annotated_shadows" / f"{tile}.tif") as s:
        shadow = (s.read(1) > 0).astype("float32")
    with rasterio.open(root / "building_footprint" / f"{tile}.tif") as s:
        building = (s.read(1) > 0).astype("uint8")
    with rasterio.open(root / "georef_image" / f"{tile}.tif") as s:
        rgb = s.read([1, 2, 3]).transpose(1, 2, 0).astype("float32") / 255.0

    shp = DATA_DIR / "main_model" / "shpfiles"
    solar = solar_angle_lookup(
        layout(root)["image"],
        shp / "2023_beeldmiddenoverzicht_lrl.shp",
        shp / "2023_beeldmiddenoverzicht_hrl.shp",
    )
    az = solar.get(tile, (135.0, 0.0))[0]
    out = visualize_assign_and_break(shadow, building, az, rgb=rgb, out_path=FIG_DIR / "assign_and_break.gif")
    print(f"wrote {out}  (tile {tile})")


def _blob_map_figure() -> None:
    root = DATA_DIR / "val_set"
    bag_path = SOURCES_DIR / "heights_validation_area.gpkg"
    if not (root / "building_footprint").is_dir() or not bag_path.exists():
        print("skipping blob_map.png - Dataset/val_set or heights_validation_area.gpkg missing")
        return
    tile = _pick_tile(root)
    fp_path = root / "building_footprint" / f"{tile}.tif"
    with rasterio.open(root / "annotated_shadows" / f"{tile}.tif") as s:
        shadow = (s.read(1) > 0).astype("float32")
    with rasterio.open(fp_path) as s:
        fp_mask = (s.read(1) > 0).astype("float32")
        transform = s.transform
    with rasterio.open(root / "georef_image" / f"{tile}.tif") as s:
        rgb = s.read([1, 2, 3]).astype("float32") / 255.0

    shp = DATA_DIR / "main_model" / "shpfiles"
    solar = solar_angle_lookup(
        layout(root)["image"],
        shp / "2023_beeldmiddenoverzicht_lrl.shp",
        shp / "2023_beeldmiddenoverzicht_hrl.shp",
    )
    az, elev = solar.get(tile, (135.0, 30.0))
    bag, _ = load_bag_heights(bag_path)
    bag = bag.to_crs(rasterio.open(fp_path).crs)
    dissolved = build_dissolved_buildings(bag, fp_path)

    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / tile
        _estimate_tile(shadow, fp_mask, transform, dissolved, az, elev, 0.25, 99, rgb=rgb, diag_stub=stub)
        made = Path(f"{stub}_map.png")
        if made.exists():
            shutil.copy(made, FIG_DIR / "blob_map.png")
            print(f"wrote {FIG_DIR / 'blob_map.png'}  (tile {tile})")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _smear_showcase()
    _assign_and_break_gif()
    _blob_map_figure()


if __name__ == "__main__":
    main()
