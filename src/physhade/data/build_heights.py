"""Build the 3D-BAG ground-truth height GeoPackage (thesis section 4.5.2).

    python -m physhade.data.build_heights --bag 3dbag.gpkg --dtm-dir ./dtm --area validation

For every building polygon: take its bounding box extended by 5 m, sample the
covering AHN DTM tile, and set ``actual_height = b3_h_70p - dtm_70p`` where
``dtm_70p`` is the 70th percentile of the DTM over that box. Writes a GeoPackage
(layer ``output_heights``) with the columns ``height_pipeline.load_bag_heights``
expects: ``identificatie, b3_h_70p, dtm_70p, actual_height, geometry`` (plus
``b3_h_50p/max/min`` when the BAG layer carries them).

Raw inputs (see DATA.md): 3D BAG (`3dbag.nl`, ``lod12_2d`` layer, attribute
``b3_h_70p`` = 70th-pct roof height above NAP) and AHN DTM raster tiles
(EPSG:28992, e.g. the AHN ``M_*.TIF`` half-metre grid).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

from physhade.config import SOURCES_DIR

_PASSTHROUGH = ["identificatie", "b3_h_50p", "b3_h_70p", "b3_h_max", "b3_h_min"]


def _extend_bbox(geom, buffer_m: float):
    minx, miny, maxx, maxy = geom.bounds
    return box(minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)


def _sample_dtm(bbox, handles: list[rasterio.DatasetReader], pct: float) -> float:
    """70th-percentile DTM value over ``bbox`` from the first covering tile."""
    minx, miny, maxx, maxy = bbox.bounds
    for src in handles:
        left, bottom, right, top = src.bounds
        if not (maxx > left and minx < right and maxy > bottom and miny < top):
            continue
        try:
            out, _ = rio_mask(src, [bbox], crop=True)
        except ValueError:
            continue
        data = out[0].ravel()
        if src.nodata is not None:
            data = data[data != src.nodata]
        data = data[np.isfinite(data)]
        if data.size:
            return float(np.percentile(data, pct))
    return float("nan")


def _clip_geometry(clip: Path):
    """Union of geometries in a GeoPackage, or of raster footprints in a directory."""
    if clip.is_dir():
        polys = []
        for tif in sorted(clip.glob("*.tif")) + sorted(clip.glob("*.tiff")):
            with rasterio.open(tif) as s:
                b = s.bounds
            polys.append(box(b.left, b.bottom, b.right, b.top))
        return unary_union(polys) if polys else None
    return unary_union(gpd.read_file(clip).geometry.tolist())


def run(
    bag_path: Path,
    dtm_dir: Path,
    out_path: Path,
    *,
    bag_layer: str | None = None,
    clip_path: Path | None = None,
    buffer_m: float = 5.0,
    dtm_pct: float = 70.0,
) -> Path:
    g = gpd.read_file(bag_path, layer=bag_layer) if bag_layer else gpd.read_file(bag_path)
    if "b3_h_70p" not in g.columns:
        raise SystemExit(f"{bag_path} (layer {bag_layer}) has no 'b3_h_70p' column - wrong BAG layer?")
    keep = [c for c in _PASSTHROUGH if c in g.columns]
    g = g[[*keep, "geometry"]].copy()

    dtm_tiles = sorted(dtm_dir.glob("*.tif")) + sorted(dtm_dir.glob("*.TIF"))
    if not dtm_tiles:
        raise SystemExit(f"no DTM rasters (*.tif / *.TIF) in {dtm_dir}")
    handles = [rasterio.open(p) for p in dtm_tiles]
    dtm_crs = handles[0].crs
    if g.crs is not None and dtm_crs is not None and g.crs != dtm_crs:
        g = g.to_crs(dtm_crs)

    if clip_path is not None:
        aoi = _clip_geometry(clip_path)
        if aoi is not None:
            before = len(g)
            g = g[g.geometry.intersects(aoi)].reset_index(drop=True)
            print(f"clip: {before} -> {len(g)} buildings intersecting {clip_path.name}")

    g["dtm_70p"] = [
        _sample_dtm(_extend_bbox(geom, buffer_m), handles, dtm_pct)
        for geom in tqdm(g.geometry, desc="sample DTM")
    ]
    for src in handles:
        src.close()

    g["actual_height"] = np.where(np.isfinite(g["dtm_70p"]), g["b3_h_70p"] - g["dtm_70p"], np.nan)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    g.to_file(out_path, driver="GPKG", layer="output_heights")
    n_ok = int(np.isfinite(g["actual_height"]).sum())
    print(
        f"wrote {out_path}  ({n_ok}/{len(g)} buildings with a height, median {g['actual_height'].median():.2f} m)"
    )
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", type=Path, required=True, help="3D BAG GeoPackage (needs b3_h_70p + geometry)")
    p.add_argument("--bag-layer", default=None, help="layer in --bag (default: first / only)")
    p.add_argument("--dtm-dir", type=Path, required=True, help="dir of AHN DTM rasters (*.tif / *.TIF)")
    p.add_argument(
        "--clip",
        type=Path,
        default=None,
        help="restrict BAG to polygons intersecting this gpkg / footprint-raster dir",
    )
    p.add_argument("--area", default="area", help="name for the default output (heights_<area>_area.gpkg)")
    p.add_argument(
        "--buffer-m", type=float, default=5.0, help="bbox extension before DTM sampling (thesis: 5)"
    )
    p.add_argument(
        "--dtm-percentile",
        type=float,
        default=70.0,
        help="DTM percentile for the terrain height (thesis: 70)",
    )
    p.add_argument("--out", type=Path, default=None, help="default: data_sources/heights_<area>_area.gpkg")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = args.out or SOURCES_DIR / f"heights_{args.area}_area.gpkg"
    run(
        args.bag,
        args.dtm_dir,
        out,
        bag_layer=args.bag_layer,
        clip_path=args.clip,
        buffer_m=args.buffer_m,
        dtm_pct=args.dtm_percentile,
    )


if __name__ == "__main__":
    main()
