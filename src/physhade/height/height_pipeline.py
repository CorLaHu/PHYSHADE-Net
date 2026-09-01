"""Stage 8 / 8b: building-height estimation from real shadow masks (thesis method).

    python -m physhade.height.height_pipeline --from annotated          # annotated shadows
    python -m physhade.height.height_pipeline --from checkpoint         # one model's predictions
    python -m physhade.height.height_pipeline --from runs               # every model x fold in runs.csv
    python -m physhade.height.height_pipeline --val --from runs --final # the held-out val tiles, final models

Per tile:
  1. get the shadow mask - the human annotation (``--from annotated``), a single model
     prediction (``--from checkpoint``), or every ``exp_id x fold`` in a ``runs.csv``
     (``--from runs``, with ``--final`` for ``Output/final_model/runs.csv``);
  2. per-tile solar azimuth / elevation from the PDOK flight shapefiles (pvlib);
  3. ``enforce_shadow_gap`` (assign-and-break) relabels the shadow raster into
     per-building blobs separated by a 1 px gap;
  4. ``subpixel_flood_shadow_height_vectorized`` casts rays from the sun-facing edge
     pixels of each blob, takes the 99th-percentile ray length -> per-blob height,
     and flags blobs whose edge touches the tile border (``truncated``, dropped);
  5. each surviving blob is matched to a 3D BAG building by marching its centroid
     toward the sun; ``delta_height = est - true`` (``actual_height`` /
     ``b3_h_70p - dtm_70p``);
  6. after all tiles: flag ``is_highest_blob`` (the tallest blob per building) and write
     ``per_blob_metrics.csv`` + blob-weighted ``weighted_*`` metric tables.

The BAG GeoPackage defaults to ``heights_training_area.gpkg`` (or
``heights_validation_area.gpkg`` with ``--val``); ``--no-bag`` skips the comparison.
"""

from __future__ import annotations

import argparse
import math
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform as warp_transform
from scipy.ndimage import label as nd_label
from skimage.measure import regionprops
from sklearn.neighbors import KDTree
from tqdm import tqdm

from physhade.config import DATA_DIR, OUTPUT_DIR, SOURCES_DIR, resolve_checkpoint
from physhade.height.blob_separation import enforce_shadow_gap
from physhade.height.height_estimation import subpixel_flood_shadow_height_vectorized

PIXEL_SIZE = 0.25

# tiles whose nearest flight-overview point is wrong; force this photo instead (thesis)
_WINTER_SOLAR_OVERRIDE = {"Winter_Tile8": "2023_346072221_hrl", "Winter_Tile3_Clipped": "2023_346072221_hrl"}

_METRIC_COLS = [
    "mean_est_height",
    "mean_true_height",
    "mean_error",
    "mean_absolute_error",
    "rmse",
    "std_residuals",
    "min_est_height",
    "max_est_height",
]

_PER_BLOB_RENAME = {
    "image": "Image",
    "exp_id": "Model",
    "fold": "Fold",
    "ckpt": "Checkpoint",
    "blob_id": "Blob ID",
    "azimuth": "Azimuth",
    "solar_elevation": "Solar Elevation",
    "n_pixels": "Pixel Area",
    "percentile_ray_length_px": "Closest Real Ray",
    "ray_len": "Interp. Ray Length",
    "shadow_length": "Shadow Length",
    "est_height": "Est. Height",
    "true_height": "True Height",
    "delta_height": "Error",
    "matched_building_id": "Matched Building",
    "is_highest_blob": "Is Highest Blob?",
}


def load_bag_heights(gpkg: Path) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    """3D BAG polygons + ``{identificatie: above-ground height (m)}``.

    Prefers the ``actual_height`` column, falling back to ``b3_h_70p - dtm_70p``
    (with a constant median DTM where the per-building DTM is missing).
    """
    g = gpd.read_file(gpkg)[["identificatie", "b3_h_70p", "dtm_70p", "actual_height", "geometry"]]
    dtm_const = g["dtm_70p"][g["dtm_70p"].between(-10, 50)].median()
    h = g["actual_height"].where(g["actual_height"].between(0, 150))
    b3 = g["b3_h_70p"].where(g["b3_h_70p"].between(-10, 200))
    dtm = g["dtm_70p"].where(g["dtm_70p"].between(-10, 50)).fillna(dtm_const)
    h = h.fillna(b3 - dtm)
    g["height_m"] = h
    g = g[g["height_m"].between(0.5, 150)]
    lookup = g.groupby("identificatie")["height_m"].mean().to_dict()
    return g, lookup


def _tile_centre_lonlat(path: Path) -> tuple[float, float]:
    with rasterio.open(path) as src:
        b = src.bounds
        cx, cy = (b.left + b.right) / 2, (b.top + b.bottom) / 2
        if src.crs and src.crs.to_epsg() != 4326:
            lon, lat = warp_transform(src.crs, "EPSG:4326", [cx], [cy])
            return lon[0], lat[0]
        return cx, cy


def solar_angle_lookup(image_dir: Path, shp_summer: Path, shp_winter: Path, tz_offset: int = 0) -> dict:
    """``{stem: (azimuth_deg, elevation_deg)}`` for every tile in ``image_dir``."""
    tz = timezone(timedelta(hours=tz_offset))
    pts_s = gpd.read_file(shp_summer).to_crs("EPSG:4326")
    pts_w = gpd.read_file(shp_winter).to_crs("EPSG:4326")
    pts_w["fotodatum"] = pd.to_datetime(pts_w["fotodatum"])
    pts_w["fototyd"] = pts_w["fototyd"].astype(str)
    tree_s = KDTree(np.column_stack([pts_s.geometry.x, pts_s.geometry.y]))
    tree_w = KDTree(np.column_stack([pts_w.geometry.x, pts_w.geometry.y]))

    out = {}
    for tif in sorted(image_dir.glob("*.tif*")):
        stem = tif.name.split(".")[0]
        lon, lat = _tile_centre_lonlat(tif)
        winter = "winter" in tif.name.lower()
        if winter:
            override = _WINTER_SOLAR_OVERRIDE.get(stem)
            hit = (
                pts_w[pts_w["fotonaam"] == override] if override and "fotonaam" in pts_w else pts_w.iloc[0:0]
            )
            if not hit.empty:
                row = hit.iloc[0]
            else:
                _, idx = tree_w.query([[lon, lat]], k=1)
                row = pts_w.iloc[idx.item()]
            d = row["fotodatum"].date()
            t = datetime.strptime(row["fototyd"], "%H:%M:%S").time()
            dt = datetime.combine(d, t).replace(tzinfo=tz)
        else:
            _, idx = tree_s.query([[lon, lat]], k=1)
            row = pts_s.iloc[idx.item()]
            dt = datetime.strptime(
                str(int(row["OPNAMEDATU"])).zfill(6) + str(int(row["OPNAMETIJD"])).zfill(6),
                "%y%m%d%H%M%S",
            ).replace(tzinfo=tz)
        sp = pvlib.solarposition.get_solarposition(
            time=pd.DatetimeIndex([dt]), latitude=lat, longitude=lon
        ).iloc[0]
        out[stem] = (float(sp.azimuth), float(sp.apparent_elevation))
    return out


def layout(data_root: Path) -> dict[str, Path]:
    """Resolve the {image, smear, shadow, footprint} dirs for a train/ or val_set/ tree."""
    if (data_root / "georef_image").is_dir():  # Dataset/val_set
        return {
            "image": data_root / "georef_image",
            "smear": data_root / "smear_shadow",
            "shadow": data_root / "annotated_shadows",
            "footprint": data_root / "building_footprint",
        }
    return {
        "image": data_root / "image",
        "smear": data_root / "smear_shadow",
        "shadow": data_root / "annotated_shadow",
        "footprint": data_root / "building_footprint",
    }


# --------------------------------------------------------------------------- models


def _load_seg_model(channels: int, ckpt: Path, device: str):
    """Load a segmentation model, adapting the first conv if the checkpoint's input
    channel count differs."""
    import torch

    from physhade.models.basemodel import PHYSHADENet, UNet

    net = (
        UNet(in_channels=channels, out_channels=1)
        if channels == 3
        else PHYSHADENet(in_channels=channels, out_channels=1)
    )
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    adapted = {}
    for k, v in state.items():
        if k == "down1.conv.0.weight" and v.shape[1] != channels:
            if channels == 4 and v.shape[1] == 3:
                adapted[k] = torch.cat([v, v.mean(1, keepdim=True)], dim=1)
            elif channels == 3 and v.shape[1] == 4:
                adapted[k] = v[:, :3, :, :]
            else:
                raise RuntimeError(f"cannot adapt {k}: ckpt {v.shape[1]}ch vs model {channels}ch")
        else:
            adapted[k] = v
    net.load_state_dict(adapted, strict=False)
    net = net.eval().to(device)
    return net.half() if device == "cuda" else net.float()


def _predict_shadow(net, channels: int, image_p: Path, smear_p: Path | None, device: str) -> np.ndarray:
    import torch

    with rasterio.open(image_p) as s:
        rgb = s.read([1, 2, 3]).astype("float32") / 255.0
    x = rgb
    if channels == 4:
        if smear_p is not None and smear_p.exists():
            with rasterio.open(smear_p) as s:
                smear = s.read(1).astype("float32")
        else:
            smear = np.zeros(rgb.shape[1:], "float32")
        if smear.max() > 1.0:
            smear /= 255.0
        x = np.concatenate([rgb, smear[None]], 0)
    t = torch.from_numpy(x)[None].to(device)
    t = t.half() if device == "cuda" else t.float()
    with torch.no_grad():
        prob = torch.sigmoid(net(t))[0, 0].float().cpu().numpy()
    return (prob > 0.5).astype("float32")


# --------------------------------------------------------------------- per-tile core


def build_dissolved_buildings(bag: gpd.GeoDataFrame, fp_path: Path) -> gpd.GeoDataFrame:
    """Group BAG polygons by the footprint raster's connected components, dissolve
    each group, and carry the largest member's height."""
    with rasterio.open(fp_path) as s:
        building_mask = s.read(1)
        bf_transform = s.transform
        bf_crs = s.crs
    labeled, _ = nd_label(building_mask > 0)

    g = bag.to_crs(bf_crs).copy()
    g["footprint_area"] = g.geometry.area
    group_ids = []
    for geom in g.geometry:
        fp = rasterize(
            [(geom.buffer(0.25), 1)],
            out_shape=building_mask.shape,
            transform=bf_transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )
        groups = labeled[fp == 1]
        groups = groups[groups != 0]
        if groups.size:
            vals, counts = np.unique(groups, return_counts=True)
            group_ids.append(int(vals[counts.argmax()]))
        else:
            group_ids.append(np.nan)
    g["raster_group_id"] = group_ids
    g = g.dropna(subset=["raster_group_id"])
    if g.empty:
        return g

    largest = (
        g.sort_values("footprint_area", ascending=False)
        .groupby("raster_group_id")
        .first()
        .reset_index()[["raster_group_id", "height_m"]]
    )
    buf = 0.2
    g = g.copy()
    g["geometry"] = g.geometry.buffer(buf)
    dissolved = g.dissolve(
        by="raster_group_id",
        as_index=False,
        aggfunc={"identificatie": lambda x: ",".join(x.astype(str))},
    )
    dissolved["geometry"] = dissolved.geometry.buffer(-buf)
    return dissolved.merge(largest, on="raster_group_id", how="left")


def _estimate_tile(
    shadow: np.ndarray,
    fp_mask: np.ndarray,
    transform,
    dissolved: gpd.GeoDataFrame | None,
    az: float,
    elev: float,
    px: float,
    percentile: float,
    rgb: np.ndarray | None = None,
    diag_stub: Path | None = None,
) -> tuple[list[dict], int, int]:
    """One shadow raster -> per-blob height rows, (total blobs, truncated blobs).

    When ``diag_stub`` is given, also write ``<diag_stub>_map.png`` and
    ``<diag_stub>_rays.png`` diagnostic figures.
    """
    gap = enforce_shadow_gap(shadow, fp_mask, az, elev)
    h_map, id_map, _, diag = subpixel_flood_shadow_height_vectorized(
        gap,
        fp_mask,
        az,
        elev,
        pixel_size=px,
        percentile=percentile,
        require_connection=False,
        diagnostics=True,
    )
    if not diag:
        return [], 0, 0
    n_trunc = sum(1 for d in diag if d["truncated"])

    out_shape = shadow.shape
    if dissolved is not None and not dissolved.empty:
        shapes = ((geom, i + 1) for i, geom in enumerate(dissolved.geometry))
        building_raster = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype="uint32")
        id_lookup = dict(enumerate(dissolved["identificatie"], start=1))
        height_lookup = dict(zip(dissolved["identificatie"], dissolved["height_m"]))
    else:
        building_raster = np.zeros(out_shape, "uint32")
        id_lookup, height_lookup = {}, {}

    match_map = np.zeros(out_shape, dtype=np.uint32)
    azr = math.radians(az)
    dxs, dys = math.sin(azr), -math.cos(azr)  # march the centroid toward the sun
    h, w = out_shape
    for prop in regionprops(id_map.astype("int32")):
        for step in range(100):
            y = int(round(prop.centroid[0] + dys * step))
            x = int(round(prop.centroid[1] + dxs * step))
            if not (0 <= y < h and 0 <= x < w):
                break
            if building_raster[y, x]:
                match_map[id_map == prop.label] = building_raster[y, x]
                break

    if diag_stub is not None:
        from physhade.height.diagnostics import render_blob_map, render_ray_plot

        trunc_ids = {d["blob_id"] for d in diag if d["truncated"]}
        valid = np.isin(id_map, np.unique(id_map[h_map > 0]))
        map_ids = np.where(valid, id_map, 0).astype("int32")
        rgb_hw = (
            np.asarray(rgb).transpose(1, 2, 0)
            if rgb is not None and np.asarray(rgb).ndim == 3 and np.asarray(rgb).shape[0] == 3
            else rgb
        )
        if rgb_hw is not None:
            render_blob_map(
                rgb_hw, map_ids, match_map, dissolved, transform, trunc_ids, az, f"{diag_stub}_map.png"
            )
        render_ray_plot(gap, fp_mask, diag, px, az, f"{diag_stub}_rays.png", percentile)

    rows = []
    for d in diag:
        if d["truncated"]:
            continue
        bid = d["blob_id"]
        blob = id_map == bid
        if not blob.any():
            continue
        est_h = float(h_map[blob][0])
        hits = np.unique(match_map[blob])
        hits = hits[hits != 0]
        if hits.size:
            building_id = id_lookup.get(int(hits[0]))
            true_h = float(height_lookup.get(building_id, np.nan))
            delta = est_h - true_h
        else:
            building_id, true_h, delta = None, np.nan, np.nan
        rows.append(
            {
                "blob_id": int(bid),
                "n_pixels": int(blob.sum()),
                "percentile_ray_length_px": d["percentile_ray_length_px"],
                "ray_len": d["ray_len"],
                "shadow_length": d["shadow_length_m"],
                "est_height": est_h,
                "true_height": true_h,
                "delta_height": delta,
                "matched_building_id": building_id,
            }
        )
    return rows, len(diag), n_trunc


# ------------------------------------------------------------------- mask sourcing


def _run_inference(
    runs_csv: Path,
    image_dir: Path,
    smear_dir: Path,
    mask_root: Path,
    device: str,
    final: bool,
    luo_baseline: Path | None,
) -> None:
    """Predict every ``exp_id x fold`` in ``runs_csv`` onto ``mask_root``."""
    df = pd.read_csv(runs_csv)
    df = df[df["fold"].astype(str) != "avg"].dropna(subset=["best_ckpt"])
    if not final:
        df = df[df["exp_id"].astype(str).str.startswith(("RGBS_", "HYB_"))]

    tifs = sorted(image_dir.glob("*.tif"))
    jobs = [(r["exp_id"], str(r["fold"]), runs_csv.parent / str(r["best_ckpt"]), r) for _, r in df.iterrows()]
    for exp_id, fold, ckpt, r in tqdm(jobs, desc="height/infer"):
        if not ckpt.exists():
            warnings.warn(f"checkpoint missing, skipping {exp_id} fold {fold}: {ckpt}", stacklevel=2)
            continue
        with_smear = str(r.get("use_prior_channel", "True")).upper() == "TRUE"
        channels = 4 if with_smear else 3
        net = _load_seg_model(channels, ckpt, device)
        odir = mask_root / str(exp_id) / f"fold_{fold}" / ckpt.stem
        odir.mkdir(parents=True, exist_ok=True)
        for ip in tifs:
            shadow = _predict_shadow(net, channels, ip, smear_dir / f"{ip.stem}.tif", device)
            with rasterio.open(ip) as s:
                meta = s.meta.copy()
            meta.update(count=1, dtype="float32")
            with rasterio.open(odir / f"{ip.stem}.tif", "w", **meta) as dst:
                dst.write(shadow, 1)

    if luo_baseline is not None and luo_baseline.exists():
        net = _load_seg_model(3, luo_baseline, device)
        odir = mask_root / "LUO_UNET" / "fold_0" / luo_baseline.stem
        odir.mkdir(parents=True, exist_ok=True)
        for ip in tifs:
            shadow = _predict_shadow(net, 3, ip, None, device)
            with rasterio.open(ip) as s:
                meta = s.meta.copy()
            meta.update(count=1, dtype="float32")
            with rasterio.open(odir / f"{ip.stem}.tif", "w", **meta) as dst:
                dst.write(shadow, 1)


def _iter_mask_sources(source, checkpoint, lay, device, mask_root):
    """Yield ``(exp_id, fold, ckpt_name, {stem: shadow_path})`` tuples."""
    image_dir, smear_dir, shadow_dir = lay["image"], lay["smear"], lay["shadow"]
    if source == "annotated":
        yield "annotated", "", "", {p.stem: p for p in sorted(shadow_dir.glob("*.tif"))}
    elif source == "checkpoint":
        net = _load_seg_model(4, checkpoint, device)
        preds = {}
        tmp = mask_root / "checkpoint" / checkpoint.stem
        tmp.mkdir(parents=True, exist_ok=True)
        for ip in sorted(image_dir.glob("*.tif")):
            shadow = _predict_shadow(net, 4, ip, smear_dir / f"{ip.stem}.tif", device)
            with rasterio.open(ip) as s:
                meta = s.meta.copy()
            meta.update(count=1, dtype="float32")
            outp = tmp / f"{ip.stem}.tif"
            with rasterio.open(outp, "w", **meta) as dst:
                dst.write(shadow, 1)
            preds[ip.stem] = outp
        yield "checkpoint", "", checkpoint.stem, preds
    else:  # runs - iterate the <exp_id>/fold_<fold>/<ckpt_stem>/ dirs written by _run_inference
        for d in sorted(p for p in mask_root.glob("*/fold_*/*") if p.is_dir()):
            exp_id, fold, ckpt_name = d.parents[1].name, d.parent.name.replace("fold_", ""), d.name
            yield exp_id, fold, ckpt_name, {p.stem: p for p in sorted(d.glob("*.tif"))}


# ------------------------------------------------------------------- aggregation


def _weighted_table(df_acc: pd.DataFrame, keys: list[str], path: Path) -> None:
    if df_acc.empty:
        return
    agg = df_acc.groupby(keys, as_index=False).agg(
        n_accepted_blobs=("blob_id", "size"),
        mean_est_height=("est_height", "mean"),
        mean_true_height=("true_height", "mean"),
        mean_error=("delta_height", "mean"),
        mean_absolute_error=("delta_height", lambda x: x.abs().mean()),
        rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
        std_residuals=("delta_height", "std"),
        min_est_height=("est_height", "min"),
        max_est_height=("est_height", "max"),
    )
    weights = agg["n_accepted_blobs"].where(
        agg.get("exp_id", pd.Series("", index=agg.index)) != "LUO_UNET", 0
    )
    total = weights.sum()
    if total:
        wrow = (agg[_METRIC_COLS].multiply(weights, axis=0).sum() / total).to_dict()
        wrow.update({k: "" for k in keys})
        wrow[keys[0]] = "AVERAGE_WEIGHTED"
        wrow["n_accepted_blobs"] = int(weights.sum())
        agg = pd.concat([agg, pd.DataFrame([wrow])], ignore_index=True)
    agg.round(4).to_csv(path, index=False)


def _aggregate_and_write(records: list[dict], errors: list[dict], out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(records)
    pd.DataFrame(errors).to_csv(out_dir / "errors.csv", index=False)
    if df.empty:
        print("no shadow blobs found")
        return df

    df["squared_error"] = (df["est_height"] - df["true_height"]) ** 2
    df["is_highest_blob"] = False
    for keys, grp in df.groupby(["image", "exp_id", "fold", "matched_building_id"], dropna=False):
        if keys[-1] is None or (isinstance(keys[-1], float) and pd.isna(keys[-1])):
            continue
        df.loc[grp["est_height"].idxmax(), "is_highest_blob"] = True

    df.rename(columns=_PER_BLOB_RENAME).round(4).to_csv(out_dir / "per_blob_metrics.csv", index=False)

    df_acc = df[df["true_height"].notna() & df["is_highest_blob"]].copy()
    _weighted_table(df_acc, ["image"], out_dir / "weighted_per_image_metrics.csv")
    _weighted_table(df_acc, ["exp_id", "image"], out_dir / "weighted_model_image_metrics.csv")
    _weighted_table(df_acc, ["exp_id"], out_dir / "weighted_model_metrics.csv")
    return df


# ------------------------------------------------------------------------- driver


def run(
    source: str,
    data_root: Path,
    solar: dict,
    out_dir: Path,
    *,
    checkpoint: Path | None = None,
    bag_gpkg: Path | None = None,
    limit: int | None = None,
    downsample: int = 1,
    percentile: float = 99.0,
    runs_csv: Path | None = None,
    final: bool = False,
    luo_baseline: Path | None = None,
    diag: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = max(1, downsample)
    px = PIXEL_SIZE * d
    lay = layout(data_root)
    fp_dir, image_dir = lay["footprint"], lay["image"]

    device = "cpu"
    if source in ("checkpoint", "runs"):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    bag = None
    if bag_gpkg is not None:
        bag, _ = load_bag_heights(bag_gpkg)
        with rasterio.open(next(fp_dir.glob("*.tif"))) as s:
            bag = bag.to_crs(s.crs)
        print(f"BAG: {len(bag)} buildings with a height, from {bag_gpkg.name}")

    mask_root = out_dir / "inferred_masks"
    if source == "runs":
        assert runs_csv is not None
        _run_inference(runs_csv, lay["image"], lay["smear"], mask_root, device, final, luo_baseline)

    records: list[dict] = []
    errors: list[dict] = []
    dissolved_cache: dict[str, gpd.GeoDataFrame | None] = {}

    for exp_id, fold, ckpt_name, stem_to_path in _iter_mask_sources(
        source, checkpoint, lay, device, mask_root
    ):
        stems = [s for s in stem_to_path if s in solar]
        if limit:
            stems = stems[:limit]
        for stem in tqdm(stems, desc=f"height/{exp_id}:{fold or '-'}", leave=False):
            az, elev = solar[stem]
            if elev <= 0:
                continue
            sp = stem_to_path[stem]
            with rasterio.open(sp) as s:
                shadow = (s.read(1) > 0).astype("float32")
            fp_path = fp_dir / f"{stem}.tif"
            if not fp_path.exists():
                continue
            with rasterio.open(fp_path) as s:
                fp_mask = (s.read(1) > 0).astype("float32")
                transform = s.transform
            if d > 1:
                shadow, fp_mask = shadow[::d, ::d], fp_mask[::d, ::d]
                transform = transform * transform.scale(d, d)

            if bag is not None and stem not in dissolved_cache:
                dissolved_cache[stem] = build_dissolved_buildings(bag, fp_path)
            dissolved = dissolved_cache.get(stem)

            rgb = diag_stub = None
            if diag:
                ip = image_dir / f"{stem}.tif"
                if ip.exists():
                    with rasterio.open(ip) as s:
                        rgb = s.read([1, 2, 3]).astype("float32") / 255.0
                    if d > 1:
                        rgb = rgb[:, ::d, ::d]
                diag_dir = out_dir / "diag" / str(exp_id) / (fold or "-") / ckpt_name
                diag_dir.mkdir(parents=True, exist_ok=True)
                diag_stub = diag_dir / stem

            rows, n_total, n_trunc = _estimate_tile(
                shadow, fp_mask, transform, dissolved, az, elev, px, percentile, rgb=rgb, diag_stub=diag_stub
            )
            for r in rows:
                r.update(
                    image=stem,
                    exp_id=exp_id,
                    fold=fold,
                    ckpt=ckpt_name,
                    azimuth=az,
                    solar_elevation=elev,
                )
                records.append(r)
            errors.append(
                {
                    "uid": stem,
                    "exp_id": exp_id,
                    "fold": fold,
                    "total_blobs": n_total,
                    "truncated_blobs": n_trunc,
                }
            )

    df = _aggregate_and_write(records, errors, out_dir)
    csv = out_dir / "per_blob_metrics.csv"
    if df.empty:
        return csv

    matched = df[df["true_height"].notna()]
    acc = df[df["true_height"].notna() & df["is_highest_blob"]]
    print(f"{len(df)} blobs / {matched.shape[0]} matched / {acc.shape[0]} accepted (highest-per-building)")
    if not acc.empty:
        e = acc["delta_height"]
        rmse = float(np.sqrt((e**2).mean()))
        print(
            f"  accepted: ME {e.mean():+.2f} m  MAE {e.abs().mean():.2f} m  RMSE {rmse:.2f} m  median|e| {e.abs().median():.2f} m"
        )
    print(f"  -> {csv}")
    return csv


# ---------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    shp = DATA_DIR / "main_model" / "shpfiles"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="source", choices=["annotated", "checkpoint", "runs"], default="annotated")
    p.add_argument("--checkpoint", type=Path, default=None, help="main-model checkpoint (--from checkpoint)")
    p.add_argument("--runs-csv", type=Path, default=None, help="runs.csv to loop (--from runs)")
    p.add_argument(
        "--final", action="store_true", help="--from runs: use every row (final models), not just RGBS_/HYB_"
    )
    p.add_argument(
        "--luo-baseline", type=Path, default=None, help="--from runs: also run this 3-ch UNet as LUO_UNET"
    )
    p.add_argument(
        "--val", action="store_true", help="run on Dataset/val_set against heights_validation_area.gpkg"
    )
    p.add_argument(
        "--data-dir", type=Path, default=None, help="default: main_model/train, or val_set with --val"
    )
    p.add_argument("--shp-summer", type=Path, default=shp / "2023_beeldmiddenoverzicht_lrl.shp")
    p.add_argument("--shp-winter", type=Path, default=shp / "2023_beeldmiddenoverzicht_hrl.shp")
    p.add_argument("--timezone-offset", type=int, default=0)
    p.add_argument("--bag-gpkg", type=Path, default=None, help="3D BAG GeoPackage (default per --val)")
    p.add_argument("--no-bag", action="store_true", help="skip the BAG comparison entirely")
    p.add_argument("--percentile", type=float, default=99.0, help="ray-length percentile (thesis: 99)")
    p.add_argument(
        "--limit", type=int, default=None, help="only the first N tiles per mask source (smoke test)"
    )
    p.add_argument(
        "--downsample", type=int, default=1, help="decimate masks by this factor before raycasting"
    )
    p.add_argument(
        "--diag",
        action="store_true",
        help="also write per-tile <out>/diag/.../<uid>_{map,rays}.png diagnostic figures",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir or (DATA_DIR / "val_set" if args.val else DATA_DIR / "main_model" / "train")

    if args.bag_gpkg is not None:
        bag = None if args.no_bag else args.bag_gpkg
    else:
        name = "heights_validation_area.gpkg" if args.val else "heights_training_area.gpkg"
        default_bag = SOURCES_DIR / name
        bag = None if (args.no_bag or not default_bag.exists()) else default_bag

    ckpt = None
    if args.source == "checkpoint":
        ckpt = resolve_checkpoint(args.checkpoint, subdir="main_model", pattern="final_*/*/epoch*.pth")

    runs_csv = args.runs_csv
    if args.source == "runs" and runs_csv is None:
        runs_csv = OUTPUT_DIR / ("final_model" if args.final else "main_model") / "runs.csv"
    if args.source == "runs" and not runs_csv.exists():
        raise SystemExit(f"runs.csv not found: {runs_csv} - train some models first or pass --runs-csv")

    tag = "val" if args.val else args.source
    out = args.out_dir or OUTPUT_DIR / "main_model" / "height" / f"{tag}_{datetime.now():%Y_%m_%d_%H_%M}"
    solar = solar_angle_lookup(
        layout(data_dir)["image"], args.shp_summer, args.shp_winter, args.timezone_offset
    )
    run(
        args.source,
        data_dir,
        solar,
        out,
        checkpoint=ckpt,
        bag_gpkg=bag,
        limit=args.limit,
        downsample=args.downsample,
        percentile=args.percentile,
        runs_csv=runs_csv,
        final=args.final,
        luo_baseline=args.luo_baseline,
        diag=args.diag,
    )


if __name__ == "__main__":
    main()
