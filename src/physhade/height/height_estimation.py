"""Building-height retrieval from shadows.

Exports ``subpixel_flood_shadow_height`` (the raycasting height algorithm) and
``export_solar_angles``. Run as a script it executes the *synthetic-shadow
control experiment* from the thesis: a solid shadow is generated from footprints
+ per-tile solar geometry, the height algorithm is run on it, and the estimate is
compared against the known synthetic length to isolate the algorithm's own error.

    python -m physhade.height.height_estimation

Height estimation on *annotated* or *model-predicted* shadows (thesis stage 8 /
8b) is not wired here yet - see docs/pipeline.md.
"""

import argparse
import glob
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import rasterio
from rasterio.warp import transform
from scipy.ndimage import label
from scipy.stats import trim_mean
from skimage.measure import profile_line
from sklearn.neighbors import KDTree

from physhade.config import DATA_DIR, OUTPUT_DIR
from physhade.physics import smear

ROOT = DATA_DIR / "main_model" / "all"
IMAGE_DIR = ROOT / "image"
FOOTPRINT_DIR = ROOT / "building_footprint"  # holds the .png exported from Supervisely
MASK_OUT_DIR = ROOT / "smear_shadow"  # smear outputs
SOLAR_TEST_MASK_DIR = ROOT / "solar_test_mask"
ACCEPTED_DIR = ROOT / "human/accepted"


def export_solar_angles(
    input_dir, shapefile_summer_path, shapefile_winter_path, timezone_offset=0, output_csv="solar_angles.csv"
):
    # --- load your two point‐clouds and build KD‐trees exactly as in preprocess() ---
    points_summer = gpd.read_file(shapefile_summer_path).to_crs("EPSG:4326")
    points_winter = gpd.read_file(shapefile_winter_path).to_crs("EPSG:4326")

    # parse winter timestamps
    points_winter["fotodatum"] = pd.to_datetime(points_winter["fotodatum"])
    points_winter["fototyd"] = points_winter["fototyd"].astype(str)

    # extract lon/lat
    for gdf in (points_summer, points_winter):
        gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y
        gdf["xy"] = list(zip(gdf["x"], gdf["y"]))

    summer_coords = np.array(points_summer["xy"].tolist())
    winter_coords = np.array(points_winter["xy"].tolist())
    tree_summer = KDTree(summer_coords)
    tree_winter = KDTree(winter_coords)

    records = []
    for tif_path in glob.glob(os.path.join(input_dir, "*.tif*")):
        lat, lon = get_image_center(tif_path)
        query = np.array([[lon, lat]])
        name = Path(tif_path).name

        if "winter" in tif_path.lower():
            dist, idx = tree_winter.query(query, k=1)
            row = points_winter.iloc[idx.item()]
            date = row["fotodatum"].date()
            time = datetime.strptime(row["fototyd"], "%H:%M:%S").time()
            dt = datetime.combine(date, time, tzinfo=timezone(timedelta(hours=timezone_offset)))
        elif "summer" in tif_path.lower():
            dist, idx = tree_summer.query(query, k=1)
            row = points_summer.iloc[idx.item()]
            # OPNAMEDATU is yymmdd, OPNAMETIJD is hhmmss
            dstr = str(int(row["OPNAMEDATU"])).zfill(6)
            tstr = str(int(row["OPNAMETIJD"])).zfill(6)
            dt = datetime.strptime(dstr + tstr, "%y%m%d%H%M%S")
            dt = dt.replace(tzinfo=timezone(timedelta(hours=timezone_offset)))
        else:
            # skip images that aren't season‐tagged
            continue

        # compute solar position
        sp = pvlib.solarposition.get_solarposition(
            latitude=lat, longitude=lon, time=pd.DatetimeIndex([dt])
        ).iloc[0]

        records.append({"file": name, "azimuth": sp.azimuth, "elevation": sp.apparent_elevation})

    # write out
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"  Saved solar angles to {output_csv}")


def subpixel_flood_shadow_height(
    mask,
    building_mask,
    azimuth_deg,
    solar_elevation,
    pixel_size=0.25,
    min_blob_area=30,
    max_steps=100,
    percentile=80,
    require_connection=True,
    connection_strictness="none",
):
    """
    Estimate height by marching rays along shadow direction from shadow pixels connected to buildings.

    Parameters:
    - mask: shadow mask (binary or float)
    - building_mask: binary mask of building footprints
    - azimuth_deg: sun azimuth (deg, clockwise from north)
    - solar_elevation: sun elevation in degrees
    - pixel_size: size of pixel in map units (e.g., 0.25 m)
    - min_blob_area: minimum blob size to consider
    - max_steps: maximum steps to march per ray
    - percentile: which percentile of ray lengths to use for height estimate

    Returns:
    - height_map: per-pixel estimated height
    - blob_id_map: labeled valid shadow blobs
    """
    azimuth_rad = np.radians((azimuth_deg + 180) % 360)
    dx_unit = np.sin(azimuth_rad)
    dy_unit = -np.cos(azimuth_rad)

    mask_bin = (mask > 0.5).astype(np.uint8)
    building_bin = (building_mask > 0.5).astype(np.uint8)

    labeled_shadow, n_shadow = label(mask_bin)
    combined = ((mask_bin | building_bin) > 0).astype(np.uint8)
    labeled_combined, _ = label(combined)

    connected_labels = np.unique(labeled_combined[building_bin > 0])
    if require_connection:
        combined = ((mask_bin | building_bin) > 0).astype(np.uint8)
        labeled_combined, _ = label(combined)
        connected_labels = np.unique(labeled_combined[building_bin > 0])

        connected_shadow_ids = set()
        for blob_id in range(1, n_shadow + 1):
            blob_mask = labeled_shadow == blob_id
            overlap = np.unique(labeled_combined[blob_mask])
            if any(l in connected_labels for l in overlap):
                connected_shadow_ids.add(blob_id)
    else:
        connected_shadow_ids = set(range(1, n_shadow + 1))

    height_map = np.zeros_like(mask, dtype=np.float32)
    blob_id_map = np.zeros_like(mask, dtype=np.uint16)
    new_id = 1

    for blob_id in connected_shadow_ids:
        blob_mask = labeled_shadow == blob_id
        if np.sum(blob_mask) < min_blob_area:
            continue

        indices = np.column_stack(np.where(blob_mask))
        ray_lengths = []

        for y0, x0 in indices:
            ray_vals = []
            for i in range(max_steps):
                y = y0 + dy_unit * i
                x = x0 + dx_unit * i

                if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
                    break

                val = profile_line(mask, (y, x), (y, x), order=1, mode="constant", cval=0.0)[0]
                if val < 0.25:
                    break
                ray_vals.append(val)

            ray_lengths.append(len(ray_vals))

        if ray_lengths:
            ray_len = np.percentile(ray_lengths, percentile)
            shadow_len = ray_len * pixel_size
            height = shadow_len * np.tan(np.radians(solar_elevation))

            height_map[blob_mask] = height
            blob_id_map[blob_mask] = new_id
            new_id += 1

    return height_map, blob_id_map


def normalize_stem(stem):
    return stem.replace("summer", "").replace("winter", "")


def get_image_center(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        # get the row/col of the image center
        row = src.height // 2
        col = src.width // 2
        # map that to CRS coordinates
        x, y = src.transform * (col + 0.5, row + 0.5)

        # if already lat/lon, just return
        if src.crs.to_epsg() == 4326:
            return y, x

        # else reproject that one point into WGS84
        lon_arr, lat_arr = transform(
            src.crs,  # source CRS (e.g. 28992)
            "EPSG:4326",  # target CRS
            [x],
            [y],  # arrays of one element
        )
        # extract the single values
        lon, lat = lon_arr[0], lat_arr[0]
        return lat, lon


def true_stem(p: Path) -> str:
    """Return the filename without *any* suffixes (handles .tif.png etc.)."""
    return p.with_suffix("").stem


def preprocess(
    input_dir,
    shapefile_summer_path,
    shapefile_winter_path,
    max_distance_km=5,
    timezone_offset=1,
    building_min=2,
    building_max=3,
):
    (SOLAR_TEST_MASK_DIR / "truth").mkdir(parents=True, exist_ok=True)

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

    for tif in glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff")):
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
                raster_path=FOOTPRINT_DIR / name,
                azimuth_deg=azimuth,
                length_mapunits=(l_min, l_max),
                decay_type="solid",
                out_tiff=SOLAR_TEST_MASK_DIR / "truth" / name,
                out_shape=(512, 512),
                mask_buildings=True,
                erode_buildings_px=0,
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
                raster_path=FOOTPRINT_DIR / name,
                azimuth_deg=azimuth,
                length_mapunits=(l_min, l_max),
                decay_type="solid",
                out_tiff=SOLAR_TEST_MASK_DIR / "truth" / name,
                out_shape=(512, 512),
                mask_buildings=True,
                erode_buildings_px=0,
            )

        # load out_tiff
        with rasterio.open(SOLAR_TEST_MASK_DIR / "truth" / name) as src:
            mask = src.read(1)

        # Load appropriate building mask
        with rasterio.open(FOOTPRINT_DIR / name) as src:
            building_mask = src.read(1)

        height_map, blob_id_map = subpixel_flood_shadow_height(mask, building_mask, azimuth, elevation, 0.25)
        unique_ids = np.unique(blob_id_map)
        unique_ids = unique_ids[unique_ids != 0]

        with rasterio.open(
            SOLAR_TEST_MASK_DIR / "truth" / f"blobs_{name}",
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=np.float32,
        ) as dst:
            dst.write(blob_id_map, 1)

        # --- Evaluate both methods per blob ---
        results = []
        mask_bin = (mask > 0.5).astype(np.uint8)
        labeled, n_blobs = label(mask_bin)

        valid_blob_ids = np.unique(label(height_map > 0.0)[0])[1:]

        for blob_id in valid_blob_ids:
            blob_mask = blob_id_map == blob_id
            est = np.percentile(height_map[blob_mask], 90)

            error = abs(est - l_max)
            if est <= 0.25:
                continue
            results.append(
                {
                    "image": name,
                    "blob_id": blob_id,
                    "true_height": l_max,
                    "est": est,
                    "error": error,
                }
            )

        # Accumulate results for all images
        if "all_results" not in locals():
            all_results = []
        all_results.extend(results)

    # --- Final results across all images ---------------------------------
    df = pd.DataFrame(all_results)
    per_blob_csv = SOLAR_TEST_MASK_DIR / "height_comparison_metrics.csv"
    df.to_csv(per_blob_csv, index=False)
    print(f"\n Per-blob metrics saved to: {per_blob_csv}")

    # ---------------------------------------------------------------------
    #                aggregate statistics (directional method)
    # ---------------------------------------------------------------------
    if df.empty:
        print("  No valid blobs found - summary CSV will not be written.")
        return

    errs = df["error"].values
    mean_e = errs.mean()
    median_e = np.median(errs)
    std_e = errs.std(ddof=1)
    rmse_e = math.sqrt(np.mean(errs**2))
    trim_e = trim_mean(errs, proportiontocut=0.10)  # 10 % trimmed mean
    p90_e = np.percentile(errs, 90)
    p95_e = np.percentile(errs, 95)
    max_e = errs.max()
    n_blobs = len(errs)

    summary_row = {
        "n_blobs": n_blobs,
        "mean_error": mean_e,
        "median_error": median_e,
        "std_error": std_e,
        "rmse": rmse_e,
        "trim10_mean": trim_e,
        "p90_error": p90_e,
        "p95_error": p95_e,
        "max_error": max_e,
    }

    summary_csv = SOLAR_TEST_MASK_DIR / "height_comparison_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)
    print(f" Summary statistics saved to: {summary_csv}")

    # --- Console recap ----------------------------------------------------
    print("\n===== Height-estimation summary =====")
    for k, v in summary_row.items():
        print(f"{k:>12}: {v:6.3f}" if isinstance(v, (float, int)) else f"{k:>12}: {v}")


def subpixel_flood_shadow_height_vectorized(
    mask,
    building_mask,
    azimuth_deg,
    solar_elevation,
    pixel_size=0.25,
    min_blob_area=30,
    max_steps=100,
    percentile=90,
    require_connection=True,
    threshold=0.25,
    diagnostics=False,
    image_name=None,
    connection_strictness=None,
    diag_dir=None,
    file_name=None,
    gdf=None,
    raster_crs=None,
    raster_transform=None,
):
    """
    Vectorized height estimation via ray-casting, removing blobs fully underneath the building mask.
    Also returns the line length in meters before converting to height.

    Always returns the 4-tuple ``(height_map, blob_id_map, shadow_lengths,
    diagnostics_data)``; ``diagnostics_data`` is a per-blob list of dicts (with
    ``truncated`` / ``ray_len`` / ``shadow_length_m`` / ``percentile_ray_length_px``)
    when ``diagnostics=True``, else an empty list. Diagnostic plots are only written
    when ``diag_dir`` is given. ``file_name`` / ``gdf`` / ``raster_crs`` /
    ``raster_transform`` are accepted for call-site compatibility and unused here.
    """
    import os
    from pathlib import Path

    import numpy as np
    from scipy.ndimage import label, map_coordinates

    azimuth_rad = np.radians((azimuth_deg + 180) % 360)
    dx_unit = np.sin(azimuth_rad)
    dy_unit = -np.cos(azimuth_rad)

    mask_bin = (mask > 0.5).astype(np.uint8)
    building_bin = (building_mask > 0.5).astype(np.uint8)

    labeled_shadow, n_shadow = label(mask_bin)
    combined = ((mask_bin | building_bin) > 0).astype(np.uint8)
    labeled_combined, _ = label(combined)

    if require_connection:
        if connection_strictness == "strict":
            connected_labels = np.unique(labeled_combined[building_bin > 0])
            connected_mask = np.isin(labeled_combined, connected_labels)
            connected_shadow_ids = set()
            for blob_id in range(1, n_shadow + 1):
                blob_mask = labeled_shadow == blob_id
                if np.any(blob_mask & connected_mask):
                    connected_shadow_ids.add(blob_id)
        else:
            connected_labels = np.unique(labeled_combined[building_bin > 0])
            connected_shadow_ids = set()
            for blob_id in range(1, n_shadow + 1):
                blob_mask = labeled_shadow == blob_id
                overlap = np.unique(labeled_combined[blob_mask])
                if any(label in connected_labels for label in overlap):
                    connected_shadow_ids.add(blob_id)

    # require_connection=False: keep only blobs that, marched a few steps back along
    # the reverse-sun vector from any pixel, reach a building before another shadow
    # pixel - then relabel the surviving pixels (thesis assign-and-break companion).
    connection_steps = 5
    if not require_connection:
        candidate_mask = np.zeros_like(mask_bin, dtype=bool)
        for blob_id in range(1, n_shadow + 1):
            blob_mask = labeled_shadow == blob_id
            if np.all(building_bin[blob_mask]):
                continue

            y_indices, x_indices = np.nonzero(blob_mask)
            found_connection = False
            for y, x in zip(y_indices, x_indices):
                for step in range(1, connection_steps + 1):
                    y_back = int(round(y - dy_unit * step))
                    x_back = int(round(x - dx_unit * step))
                    if y_back < 0 or y_back >= mask.shape[0] or x_back < 0 or x_back >= mask.shape[1]:
                        break
                    if building_bin[y_back, x_back]:
                        candidate_mask |= blob_mask
                        found_connection = True
                        break
                    if mask_bin[y_back, x_back]:
                        break
                if found_connection:
                    break
        labeled_shadow, n_shadow = label(candidate_mask)
        connected_shadow_ids = set(range(1, n_shadow + 1))

    height_map = np.zeros_like(mask, dtype=np.float32)
    blob_id_map = np.zeros_like(mask, dtype=np.uint16)
    shadow_lengths = []
    diagnostics_data = []

    new_id = 1

    for blob_id in connected_shadow_ids:
        blob_mask = labeled_shadow == blob_id

        if np.all(building_bin[blob_mask]):
            continue

        if blob_mask.sum() < min_blob_area:
            continue

        # Sun-facing edge pixels: a blob pixel counts as an edge if, marched backward
        # along the reverse-sun vector, it reaches a building before another shadow pixel.
        edge_mask = np.zeros_like(blob_mask, dtype=bool)
        y_indices, x_indices = np.nonzero(blob_mask)
        reverse_dx = -dx_unit
        reverse_dy = -dy_unit
        for y, x in zip(y_indices, x_indices):
            for step in range(1, max_steps):
                y_back = int(round(y + reverse_dy * step))
                x_back = int(round(x + reverse_dx * step))
                if y_back < 0 or y_back >= mask.shape[0] or x_back < 0 or x_back >= mask.shape[1]:
                    break
                if building_bin[y_back, x_back]:
                    edge_mask[y, x] = True
                    break  # hit the building - this is a sun-facing edge pixel
                if mask_bin[y_back, x_back]:
                    break  # blocked by another shadow pixel first
        yx_coords = np.argwhere(edge_mask)

        if yx_coords.size == 0:
            continue  # skip blob if no adjacent pixels found

        truncated = (
            np.any(yx_coords[:, 0] == 0)
            or np.any(yx_coords[:, 0] == mask.shape[0] - 1)
            or np.any(yx_coords[:, 1] == 0)
            or np.any(yx_coords[:, 1] == mask.shape[1] - 1)
        )

        y0 = yx_coords[:, 0]
        x0 = yx_coords[:, 1]

        steps = np.arange(max_steps)
        y_rays = y0[:, None] + dy_unit * steps
        x_rays = x0[:, None] + dx_unit * steps

        ray_valid_steps = (y_rays >= 0) & (y_rays < mask.shape[0]) & (x_rays >= 0) & (x_rays < mask.shape[1])

        first_out_of_bounds_idx = (~ray_valid_steps).argmax(axis=1)
        first_out_of_bounds_idx[ray_valid_steps.all(axis=1)] = max_steps

        values = map_coordinates(
            mask,
            [np.clip(y_rays, 0, mask.shape[0] - 1).ravel(), np.clip(x_rays, 0, mask.shape[1] - 1).ravel()],
            order=0,
            mode="constant",
            cval=0.0,
        ).reshape(y_rays.shape)

        valid_shadow = values >= threshold
        shadow_end_idx = (~valid_shadow).argmax(axis=1)
        shadow_end_idx[valid_shadow.all(axis=1)] = max_steps

        building_values = map_coordinates(
            building_mask,
            [np.clip(y_rays, 0, mask.shape[0] - 1).ravel(), np.clip(x_rays, 0, mask.shape[1] - 1).ravel()],
            order=0,
            mode="constant",
            cval=0.0,
        ).reshape(y_rays.shape)
        blocked_building = building_values > 0
        building_end_idx = blocked_building.argmax(axis=1)
        building_end_idx[~blocked_building.any(axis=1)] = max_steps

        final_ray_lengths = np.minimum.reduce([shadow_end_idx, first_out_of_bounds_idx, building_end_idx])

        touched_bounds = np.any(first_out_of_bounds_idx < max_steps)
        ray_lengths = final_ray_lengths
        valid_ray_mask = ray_lengths > 0

        if not np.any(valid_ray_mask):
            continue

        ray_lengths = ray_lengths[valid_ray_mask]

        longest_ray_idx = np.argmax(ray_lengths)
        sorted_indices = np.argsort(ray_lengths)
        n = len(ray_lengths)
        median_ray_idx = sorted_indices[n // 2] if n % 2 == 1 else sorted_indices[(n // 2) - 1]
        median_ray_len = ray_lengths[median_ray_idx]

        longest_ray_len = ray_lengths[longest_ray_idx]

        ray_len = np.percentile(ray_lengths, percentile)
        shadow_len = ray_len * pixel_size
        est_height = shadow_len * np.tan(np.radians(solar_elevation))

        height_map[blob_mask] = est_height
        blob_id_map[blob_mask] = new_id
        shadow_lengths.append(shadow_len)

        percentile_ray_idx = np.argmin(np.abs(ray_lengths - ray_len))
        percentile_x_start = x0[percentile_ray_idx]
        percentile_y_start = y0[percentile_ray_idx]
        percentile_x_end = percentile_x_start + dx_unit * ray_lengths[percentile_ray_idx]
        percentile_y_end = percentile_y_start + dy_unit * ray_lengths[percentile_ray_idx]

        if diagnostics:
            diagnostics_data.append(
                {
                    "blob_id": new_id,
                    "longest_ray_length_px": int(longest_ray_len),
                    "median_ray_length_px": int(median_ray_len),
                    "shadow_length_m": shadow_len,
                    "est_height_m": est_height,
                    "truncated": truncated,
                    "touched_bounds": touched_bounds,
                    "longest_ray_start": (float(x0[longest_ray_idx]), float(y0[longest_ray_idx])),
                    "longest_ray_end": (
                        float(x0[longest_ray_idx] + dx_unit * longest_ray_len),
                        float(y0[longest_ray_idx] + dy_unit * longest_ray_len),
                    ),
                    "percentile_ray_start": (percentile_x_start, percentile_y_start),
                    "percentile_ray_end": (percentile_x_end, percentile_y_end),
                    "percentile_ray_length_px": float(ray_lengths[percentile_ray_idx]),
                    "ray_len": float(ray_len),
                }
            )

        new_id += 1
    if diagnostics and diag_dir is not None and diagnostics_data:
        from physhade.height.diagnostics import render_ray_plot

        name = (image_name or file_name or "tile").replace(os.sep, "_")
        render_ray_plot(
            mask,
            building_mask,
            diagnostics_data,
            pixel_size,
            azimuth_deg,
            Path(diag_dir) / f"{name}_rays.png",
            percentile,
        )
    return height_map, blob_id_map, shadow_lengths, diagnostics_data


def run_sweep(
    footprint_dir: Path,
    out_dir: Path,
    azimuths: list[float],
    percentiles: list[float],
    true_length_m: float = 12.0,
    solar_elevation: float = 30.0,
    pixel_size: float = 0.25,
) -> Path:
    """Azimuth x ray-length-percentile synthetic control (thesis Table 16).

    Solid synthetic shadows of a known length are cast from each footprint at
    every azimuth, run through the *shipped* method (``enforce_shadow_gap`` +
    ``subpixel_flood_shadow_height_vectorized``), and the recovered shadow
    length is compared to ``true_length_m``. Isolates the algorithm's own error
    and shows why the p99 ray-length statistic is used.
    """
    import tempfile

    import matplotlib

    matplotlib.use("Agg")

    from physhade.height.blob_separation import enforce_shadow_gap

    out_dir.mkdir(parents=True, exist_ok=True)
    footprints = sorted(Path(footprint_dir).glob("*.tif"))
    if not footprints:
        raise SystemExit(f"no footprint rasters in {footprint_dir}")

    rows = []
    for fp in footprints:
        with rasterio.open(fp) as src:
            building = (src.read(1) > 0).astype("float32")
        if building.sum() == 0:
            continue
        for az in azimuths:
            with tempfile.TemporaryDirectory() as td:
                syn = Path(td) / "syn.tif"
                smear.multi_shift_shadow_raster(
                    raster_path=str(fp),
                    azimuth_deg=float(az),
                    length_mapunits=(2.0, true_length_m),
                    decay_type="solid",
                    out_tiff=str(syn),
                    out_shape=building.shape,
                    mask_buildings=True,
                    erode_buildings_px=0,
                )
                with rasterio.open(syn) as src:
                    shadow = src.read(1).astype("float32")
            if shadow.sum() == 0:
                continue
            gap = enforce_shadow_gap(shadow, building, float(az))
            for pct in percentiles:
                _, _, _, diag = subpixel_flood_shadow_height_vectorized(
                    gap,
                    building,
                    float(az),
                    solar_elevation,
                    pixel_size=pixel_size,
                    min_blob_area=10,
                    max_steps=100,
                    percentile=float(pct),
                    require_connection=True,
                    diagnostics=True,
                )
                for d in diag:
                    rows.append(
                        {
                            "footprint": fp.stem,
                            "azimuth_deg": int(az),
                            "percentile": int(pct),
                            "blob_id": d["blob_id"],
                            "shadow_length_m": d["shadow_length_m"],
                            "truncated": d["truncated"],
                            "longest_ray_length_px": d["longest_ray_length_px"],
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "azimuth_sweep_blob_metrics.csv", index=False)
    if df.empty:
        print("no blobs produced - check the footprint dir")
        return out_dir

    valid = df[~df["truncated"]].assign(err=lambda x: x["shadow_length_m"] - true_length_m)
    detail = (
        valid.groupby(["footprint", "azimuth_deg", "percentile"], as_index=False)
        .agg(
            num_blobs=("blob_id", "count"),
            mean_error=("err", "mean"),
            rmse=("err", lambda e: float(np.sqrt((e**2).mean()))),
            median_error=("err", "median"),
            max_error=("err", "max"),
        )
        .round(4)
    )
    detail.to_csv(out_dir / "azimuth_sweep_error_metrics.csv", index=False)

    summary = (
        valid.assign(abs_err=lambda x: x["err"].abs())
        .groupby("percentile", as_index=False)
        .agg(
            num_blobs=("blob_id", "count"),
            mean_abs_error=("abs_err", "mean"),
            rmse=("abs_err", lambda e: float(np.sqrt((e**2).mean()))),
            std=("abs_err", "std"),
            median=("abs_err", "median"),
            p90=("abs_err", lambda e: float(np.percentile(e, 90))),
            p95=("abs_err", lambda e: float(np.percentile(e, 95))),
            max=("abs_err", "max"),
        )
        .sort_values("percentile", ascending=False)
        .round(4)
    )
    summary.to_csv(out_dir / "azimuth_sweep_summary_metrics.csv", index=False)
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    for pct, grp in detail.groupby("percentile"):
        m = grp.groupby("azimuth_deg")["mean_error"].mean()
        ax.plot(m.index, m.to_numpy(), marker="o", label=f"p{pct}")
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("mean shadow-length error (m)")
    ax.set_title(f"synthetic sweep - true length {true_length_m:g} m, elevation {solar_elevation:g} deg")
    ax.axhline(0, color="grey", lw=0.8)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_error_vs_azimuth.png", dpi=150)
    plt.close(fig)
    print(f"-> {out_dir}")
    return out_dir


def build_parser():
    shp = DATA_DIR / "main_model" / "shpfiles"
    p = argparse.ArgumentParser(
        description="Synthetic-shadow control for the height algorithm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, default=IMAGE_DIR)
    p.add_argument("--shp-summer", type=Path, default=shp / "2023_beeldmiddenoverzicht_lrl.shp")
    p.add_argument("--shp-winter", type=Path, default=shp / "2023_beeldmiddenoverzicht_hrl.shp")
    p.add_argument("--timezone-offset", type=int, default=0)
    p.add_argument("--max-distance-km", type=float, default=5.0)
    p.add_argument("--building-min", type=float, default=2.0)
    p.add_argument("--building-max", type=float, default=3.0)
    p.add_argument(
        "--sweep",
        action="store_true",
        help="azimuth x ray-length-percentile sweep on the shipped method (thesis Table 16)",
    )
    p.add_argument("--footprint-dir", type=Path, default=DATA_DIR / "val_set" / "building_footprint")
    p.add_argument("--azimuths", default="0,60,120,180,240,300", help="--sweep: comma-separated degrees")
    p.add_argument("--percentiles", default="100,99,95,90,75", help="--sweep: comma-separated percentiles")
    p.add_argument("--true-length", type=float, default=12.0, help="--sweep: synthetic shadow length (m)")
    p.add_argument("--solar-elevation", type=float, default=30.0, help="--sweep: solar elevation (deg)")
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.sweep:
        out = args.out_dir or OUTPUT_DIR / "main_model" / "height" / "synthetic_sweep"
        run_sweep(
            footprint_dir=args.footprint_dir,
            out_dir=out,
            azimuths=[float(a) for a in str(args.azimuths).split(",") if a.strip()],
            percentiles=[float(p) for p in str(args.percentiles).split(",") if p.strip()],
            true_length_m=args.true_length,
            solar_elevation=args.solar_elevation,
        )
        return
    preprocess(
        input_dir=str(args.input_dir),
        shapefile_summer_path=str(args.shp_summer),
        shapefile_winter_path=str(args.shp_winter),
        max_distance_km=args.max_distance_km,
        timezone_offset=args.timezone_offset,
        building_min=args.building_min,
        building_max=args.building_max,
    )


if __name__ == "__main__":
    main()
