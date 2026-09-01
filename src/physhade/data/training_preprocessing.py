"""Stage A2: per-tile solar geometry -> pseudo-shadow priors -> train/ split.

    python -m physhade.data.training_preprocessing          # the training tiles
    python -m physhade.data.training_preprocessing --val    # regen Dataset/val_set/smear_shadow

Reads ``Dataset/main_model/all`` (built by ``build_dataset``) and the two flight
shapefiles under ``main_model/shpfiles``; writes ``all/smear_shadow`` and the
``main_model/train/{image,smear_shadow,annotated_shadow}`` tree + ``pairs.txt`` /
``singles.txt``.

``--val`` instead regenerates ``Dataset/val_set/smear_shadow/`` + ``pairs.txt`` /
``singles.txt`` from the shipped ``val_set/building_footprint/`` rasters (the
val annotations themselves are primary artifacts - see DATA.md).
"""

from __future__ import annotations

import argparse
import glob
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
import rasterio
from rasterio.warp import transform
from sklearn.neighbors import KDTree

from physhade.config import DATA_DIR
from physhade.physics import smear


def normalize_stem(stem):
    return stem.replace("summer", "").replace("winter", "")


def get_image_center(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        x = (bounds.left + bounds.right) / 2
        y = (bounds.top + bounds.bottom) / 2

        if src.crs.to_string() != "EPSG:4326":
            lon, lat = transform(src.crs, rasterio.CRS.from_epsg(code=4326), [x], [y])
            return lat[0], lon[0]
        else:
            return x, y


def true_stem(p: Path) -> str:
    """Return the filename without *any* suffixes (handles .tif.png etc.)."""
    return p.with_suffix("").stem


def preprocess(
    all_dir: Path,
    shapefile_summer_path,
    shapefile_winter_path,
    max_distance_km=5,
    timezone_offset=0,
    building_min=2,
    building_max=42.90,
    accepted_stems: list[str] | None = None,
):
    all_dir = Path(all_dir)
    input_dir = all_dir / "image"
    footprint_dir = all_dir / "building_footprint"
    mask_out_dir = all_dir / "smear_shadow"
    accepted_dir = all_dir / "human" / "accepted"
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("building_footprint", "annotated_shadow", "smear_shadow"):
        (all_dir / sub / "accepted").mkdir(parents=True, exist_ok=True)

    points_summer_gdf = gpd.read_file(shapefile_summer_path)
    points_winter_gdf = gpd.read_file(shapefile_winter_path)

    points_winter_gdf["fotodatum"] = pd.to_datetime(points_winter_gdf["fotodatum"])
    points_winter_gdf["fototyd"] = points_winter_gdf["fototyd"].astype(str)

    # Project points to correct crs only for xy
    if points_summer_gdf.crs != "EPSG:4326":
        points_summer_gdf = points_summer_gdf.to_crs("EPSG:4326")

    if points_winter_gdf.crs != "EPSG:4326":
        points_winter_gdf = points_winter_gdf.to_crs("EPSG:4326")

    points_summer_gdf["x"] = points_summer_gdf.geometry.x
    points_summer_gdf["y"] = points_summer_gdf.geometry.y

    points_winter_gdf["x"] = points_winter_gdf.geometry.x
    points_winter_gdf["y"] = points_winter_gdf.geometry.y

    points_summer_gdf["xy"] = list(zip(points_summer_gdf["x"], points_summer_gdf["y"]))
    points_winter_gdf["xy"] = list(zip(points_winter_gdf["x"], points_winter_gdf["y"]))

    summer_coords = np.array(points_summer_gdf["xy"].tolist())
    winter_coords = np.array(points_winter_gdf["xy"].tolist())
    tree_summer = KDTree(summer_coords)
    tree_winter = KDTree(winter_coords)

    tiles = sorted(glob.glob(str(input_dir / "*.tif")) + glob.glob(str(input_dir / "*.tiff")))
    for tif in tiles:
        lat, lon = get_image_center(tif)
        query_pt = np.array([[lon, lat]])

        name = true_stem(Path(tif)) + ".tif"
        l_min = building_min

        if "winter" in tif:
            distance, index = tree_winter.query(query_pt, k=1)
            nearest_row = points_winter_gdf.iloc[index.item()]

            date = nearest_row["fotodatum"]
            time = nearest_row["fototyd"]

            time_obj = datetime.strptime(time, "%H:%M:%S").time()
            dt = datetime.combine(date.date(), time_obj)
            tz = timezone(timedelta(hours=timezone_offset))
            dt = dt.replace(tzinfo=tz)

            times = pd.DatetimeIndex([dt])
            solar_position = pvlib.solarposition.get_solarposition(latitude=lat, longitude=lon, time=times)

            sp0 = solar_position.iloc[0]
            azimuth = sp0.azimuth
            elevation = sp0.apparent_elevation
            l_max = abs(building_max / np.tan(np.radians(elevation)))

            smear.multi_shift_shadow_raster(
                raster_path=footprint_dir / name,
                azimuth_deg=azimuth,
                length_mapunits=(l_min, l_max),
                decay_type="delayed_gradient",
                out_tiff=mask_out_dir / name,
                out_shape=(512, 512),
                mask_buildings=True,
            )

        elif "summer" in tif:
            distance, index = tree_summer.query(query_pt, k=1)
            nearest_row = points_summer_gdf.iloc[index.item()]

            date = nearest_row["OPNAMEDATU"]
            time = nearest_row["OPNAMETIJD"]

            date_str = str(int(date)).zfill(6)
            time_str = str(int(time)).zfill(6)

            dt = datetime.strptime(date_str + time_str, "%y%m%d%H%M%S")
            tz = timezone(timedelta(hours=timezone_offset))
            dt = dt.replace(tzinfo=tz)

            times = pd.DatetimeIndex([dt])
            solar_position = pvlib.solarposition.get_solarposition(latitude=lat, longitude=lon, time=times)

            sp0 = solar_position.iloc[0]
            azimuth = sp0.azimuth
            elevation = sp0.apparent_elevation
            l_max = abs(building_max / np.tan(np.radians(elevation)))

            smear.multi_shift_shadow_raster(
                raster_path=footprint_dir / name,
                azimuth_deg=azimuth,
                length_mapunits=(l_min, l_max),
                decay_type="delayed_gradient",
                out_tiff=mask_out_dir / name,
                out_shape=(512, 512),
                mask_buildings=True,
            )
    # ---- accepted subset: copy the per-layer GeoTIFFs into <layer>/accepted ----
    if accepted_stems is not None:
        stems = list(accepted_stems)
    else:
        stems = sorted(true_stem(Path(p)) for p in glob.glob(str(accepted_dir / "*")))

    winter_list, summer_list = [], []
    for stem in stems:
        (winter_list if "winter" in stem else summer_list).append(stem)
        for layer in ("building_footprint", "annotated_shadow", "smear_shadow"):
            src = all_dir / layer / f"{stem}.tif"
            if src.exists():
                shutil.copyfile(src, all_dir / layer / "accepted" / f"{stem}.tif")

    # ---- pair winter/summer tiles of the same location ----
    normalized_summer = {normalize_stem(s): s for s in summer_list}
    normalized_winter = {normalize_stem(w): w for w in winter_list}
    pairs, singles = [], []
    for key in sorted(set(normalized_summer) | set(normalized_winter)):
        s, w = normalized_summer.get(key), normalized_winter.get(key)
        if s and w:
            pairs.append((w, s))
        elif s or w:
            singles.append(s or w)

    (all_dir / "singles.txt").write_text("".join(f"{x}\n" for x in singles))
    (all_dir / "pairs.txt").write_text("".join(f"{w} {s}\n" for w, s in pairs))

    # ---- flat train/ tree the training scripts consume ----
    train_dir = all_dir.parent / "train"
    layers = ("smear_shadow", "annotated_shadow", "building_footprint")
    for sub in ("image", *layers):
        (train_dir / sub).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        img = next(iter(input_dir.glob(f"{stem}.tif*")), None)
        if img:
            shutil.copyfile(img, train_dir / "image" / f"{stem}.tif")
        for layer in layers:
            src = all_dir / layer / f"{stem}.tif"
            if src.exists():
                shutil.copyfile(src, train_dir / layer / f"{stem}.tif")
    shutil.copyfile(all_dir / "pairs.txt", train_dir / "pairs.txt")
    shutil.copyfile(all_dir / "singles.txt", train_dir / "singles.txt")
    print(f"preprocess: {len(stems)} accepted tiles, {len(pairs)} pairs + {len(singles)} singles")


def preprocess_val(
    val_root: Path,
    shp_summer: Path,
    shp_winter: Path,
    timezone_offset: int = 0,
    building_min: float = 2.0,
    building_max: float = 42.90,
) -> None:
    """Regenerate ``val_set/smear_shadow/`` + ``pairs.txt`` / ``singles.txt``.

    Uses the same per-tile solar lookup as the height stage
    (``solar_angle_lookup``, incl. the Winter tile override).
    """
    from physhade.height.height_pipeline import solar_angle_lookup

    val_root = Path(val_root)
    fp_dir = val_root / "building_footprint"
    img_dir = val_root / "georef_image"
    smear_dir = val_root / "smear_shadow"
    smear_dir.mkdir(parents=True, exist_ok=True)

    solar = solar_angle_lookup(img_dir, shp_summer, shp_winter, timezone_offset)
    stems = sorted(true_stem(p) for p in fp_dir.glob("*.tif"))
    for stem in stems:
        if stem not in solar:
            print(f"  no solar geometry for {stem} - skipped")
            continue
        azimuth, elevation = solar[stem]
        l_max = abs(building_max / np.tan(np.radians(elevation)))
        smear.multi_shift_shadow_raster(
            raster_path=fp_dir / f"{stem}.tif",
            azimuth_deg=azimuth,
            length_mapunits=(building_min, l_max),
            decay_type="delayed_gradient",
            out_tiff=smear_dir / f"{stem}.tif",
            out_shape=(512, 512),
            mask_buildings=True,
        )

    def _norm(stem: str) -> str:  # season-insensitive key (val stems are capitalised)
        return stem.lower().replace("summer", "").replace("winter", "")

    normalized_summer = {_norm(s): s for s in stems if "summer" in s.lower()}
    normalized_winter = {_norm(w): w for w in stems if "winter" in w.lower()}
    pairs, singles = [], []
    for key in sorted(set(normalized_summer) | set(normalized_winter)):
        s, w = normalized_summer.get(key), normalized_winter.get(key)
        if s and w:
            pairs.append((w, s))
        elif s or w:
            singles.append(s or w)
    (val_root / "singles.txt").write_text("".join(f"{x}\n" for x in singles))
    (val_root / "pairs.txt").write_text("".join(f"{w} {s}\n" for w, s in pairs))
    print(f"preprocess --val: {len(stems)} tiles, {len(pairs)} pairs + {len(singles)} singles -> {smear_dir}")


def build_parser() -> argparse.ArgumentParser:
    shp = DATA_DIR / "main_model" / "shpfiles"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all-dir", type=Path, default=DATA_DIR / "main_model" / "all")
    p.add_argument("--val", action="store_true", help="regenerate Dataset/val_set/smear_shadow instead")
    p.add_argument("--val-dir", type=Path, default=DATA_DIR / "val_set")
    p.add_argument("--shp-summer", type=Path, default=shp / "2023_beeldmiddenoverzicht_lrl.shp")
    p.add_argument("--shp-winter", type=Path, default=shp / "2023_beeldmiddenoverzicht_hrl.shp")
    p.add_argument("--timezone-offset", type=int, default=0)
    p.add_argument("--max-distance-km", type=float, default=5.0)
    p.add_argument("--building-min", type=float, default=2.0)
    p.add_argument("--building-max", type=float, default=42.90)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.val:
        preprocess_val(
            val_root=args.val_dir,
            shp_summer=args.shp_summer,
            shp_winter=args.shp_winter,
            timezone_offset=args.timezone_offset,
            building_min=args.building_min,
            building_max=args.building_max,
        )
        return
    preprocess(
        all_dir=args.all_dir,
        shapefile_summer_path=str(args.shp_summer),
        shapefile_winter_path=str(args.shp_winter),
        max_distance_km=args.max_distance_km,
        timezone_offset=args.timezone_offset,
        building_min=args.building_min,
        building_max=args.building_max,
    )


if __name__ == "__main__":
    main()
