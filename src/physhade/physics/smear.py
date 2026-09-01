import logging

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from affine import Affine
from rasterio.features import rasterize
from scipy.ndimage import binary_erosion, generate_binary_structure
from skimage.transform import AffineTransform, warp  # sub-pixel shifts

log = logging.getLogger(__name__)


def rasterize_buildings(gdf: gpd.GeoDataFrame, out_shape: tuple, transform: Affine) -> np.ndarray:
    """
    Rasterize building footprints into a binary mask.
    1 = building, 0 = no building.
    """
    building_mask = rasterize(
        [(geom, 1) for geom in gdf.geometry if not geom.is_empty],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    return building_mask


def multi_shift_shadow_raster(
    # gdf: gpd.GeoDataFrame,
    raster_path: str,
    azimuth_deg: float,
    length_mapunits,
    decay_type: str,
    out_tiff: str,
    out_shape: tuple,
    mask_buildings: bool = False,
    erode_buildings_px: int = 0,
):
    """
    Raster‐based shadow generator that exactly matches the vector ShadowGen
    in both direction (azimuth) and length.

    Parameters
    ----------
    gdf : GeoDataFrame of building footprints.
    azimuth_deg : Sun azimuth ( deg clockwise from north).
    length_mapunits : Desired max shadow length in map units.
    out_tiff : Path to write the GeoTIFF.
    out_shape : (rows, cols) of the output raster.
    transform : Affine transform for the raster.
    solid_shadow : If True, no fade (binary OR smear).
    mask_buildings : If True, zero out shadows on buildings.
    erode_buildings_px : If >0, erode building footprint before masking.
    """
    with rasterio.open(raster_path) as src:
        building_mask = (src.read(1) > 0).astype(np.float32)
        log.debug(
            "footprint %s: unique=%s shape=%s", raster_path, np.unique(building_mask), building_mask.shape
        )

        transform = src.transform
        # 2) get pixel resolutions
        px_w = transform.a  # east‐west resolution (map‐units per col)
        px_h = abs(transform.e)  # north‐south resolution (map‐units per row)

        # 3) compute unit‐vector for shadows in map space
        shadow_bearing = (azimuth_deg + 180) % 360
        rad = np.radians(shadow_bearing)
        dx_unit = np.sin(rad)  # east‐component
        dy_unit = np.cos(rad)  # north‐component

        log.debug("shadow unit vector dx=%.4f dy=%.4f", dx_unit, dy_unit)

        # 4) how many steps so that steps * pixel‐length_along_shadow ~ length_mapunits?
        #    effective pixel length along the ray:

        if isinstance(length_mapunits, (tuple, list)):
            l_min, l_max = map(float, length_mapunits)
        else:
            l_min = 2.0
            l_max = float(length_mapunits)

        l_min = max(0.01, abs(l_min))
        l_max = max(l_min + 0.01, abs(l_max))

        res_along = np.hypot(px_w * dx_unit, px_h * dy_unit)
        steps = max(1, int(np.ceil(l_max / res_along)))
        step_dist = l_max / steps

        # 5) accumulate shadow
        shadow_accum = np.zeros_like(building_mask, dtype=np.float32)
        for i in range(1, steps + 1):
            # fade weight
            dist = step_dist * i

            if decay_type == "solid":
                w = 1.0 if dist <= l_max else 0.0

            elif decay_type == "delayed_gradient":
                if dist <= l_min:
                    w = 1.0
                elif dist >= l_max:
                    w = 0.0
                else:
                    w = 1 - (dist - l_min) / (l_max - l_min)
            else:
                raise ValueError("decay_type must be 'solid' or 'delayed_gradient'")

            if w == 0.0:
                continue

            # compute map‐space offset
            dx_map = dx_unit * dist
            dy_map = dy_unit * dist

            # convert to pixel offsets
            shift_c = dx_map / px_w  # cols  east
            shift_r = -dy_map / px_h  # rows  north (inverted)

            # warp & accumulate
            t = AffineTransform(translation=(shift_c, shift_r))
            shifted = warp(
                building_mask,
                t.inverse,
                order=0,
                mode="constant",
                cval=0,
            )

            if decay_type == "solid":
                shadow_accum = np.maximum(shadow_accum, shifted)
            else:
                # shadow_accum += w * shifted
                shadow_accum = np.maximum(shadow_accum, w * shifted)

        log.debug("max prior value: %s", shadow_accum.max())
        # 6) normalize if fading
        if decay_type != "solid":
            m = shadow_accum.max()
            if m > 0:
                shadow_accum /= m
            shadow_accum = np.clip(shadow_accum, 0, 1)

        # 7) mask out buildings if requested
        if mask_buildings:
            mask = building_mask >= 0.5
            if erode_buildings_px > 0:
                strel = generate_binary_structure(2, 1)
                eroded_buildings = binary_erosion(
                    building_mask >= 0.5, structure=strel, iterations=erode_buildings_px
                )
                shadow_accum[eroded_buildings] = 0.0
            else:
                shadow_accum[building_mask >= 0.5] = 0.0
            shadow_accum[mask] = 0.0

        # 8) write GeoTIFF
        meta = {
            "driver": "GTiff",
            "height": out_shape[0],
            "width": out_shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": src.crs.to_string() if src.crs else None,
            "transform": transform,
        }
        with rasterio.open(out_tiff, "w", **meta) as dst:
            dst.write(shadow_accum, 1)

        log.debug("shadow raster written to %s", out_tiff)


def get_transform_from_extent(gdf: gpd.GeoDataFrame, out_shape: tuple) -> Affine:
    """
    Computes an affine transform based on the extent of the GeoDataFrame and
    the desired output raster dimensions.

    Parameters:
        gdf (gpd.GeoDataFrame): GeoDataFrame with building footprints.
        out_shape (tuple): (rows, cols) for the output raster.

    Returns:
        Affine: The computed affine transformation.
    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    nrows, ncols = out_shape  # rows, cols
    # Calculate pixel resolution: extent divided by number of columns (x) or rows (y).
    xres = (xmax - xmin) / ncols
    yres = (ymax - ymin) / nrows
    # The origin is set at the top-left corner: (xmin, ymax)
    transform = Affine.translation(xmin, ymax) * Affine.scale(xres, -yres)
    return transform


def ShadowGen(gdf: gpd.GeoDataFrame, distance: float, solar_azimuth: float):
    azimuth_rads = np.radians(90 - solar_azimuth + 180)

    dx = np.cos(azimuth_rads) * distance
    dy = np.sin(azimuth_rads) * distance

    shadows_polygons = []
    for geom in gdf.geometry:
        if geom.geom_type != "Polygon":
            continue

        points = list(geom.exterior.coords)
        quads = []
        for i in range(len(points)):
            if i == len(points) - 1:
                quad_points = [
                    points[i],
                    points[0],
                    shapely.Point(points[0][0] + dx, points[0][1] + dy),
                    shapely.Point(points[i][0] + dx, points[i][1] + dy),
                ]
            else:
                quad_points = [
                    points[i],
                    points[i + 1],
                    shapely.Point(points[i + 1][0] + dx, points[i + 1][1] + dy),
                    shapely.Point(points[i][0] + dx, points[i][1] + dy),
                ]
            quads.append(shapely.geometry.Polygon(quad_points))

        shadow = shapely.unary_union(quads)
        shadows_polygons.append(shadow)
    shadow_gdf = gpd.GeoDataFrame(geometry=shadows_polygons, crs=gdf.crs)
    return shadow_gdf


# This module is a library. For a runnable demo of the smear accumulation see
# `python -m physhade.physics.smear_showcase --help`.
