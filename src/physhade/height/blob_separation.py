"""Assign-and-break: separate one connected shadow smear into per-building blobs.

Port of ``enforce_shadow_gap`` / ``enforce_pixel_gap`` from the thesis working
tree, with the GIF / matplotlib recording stripped out.

The thesis stamps every shadow pixel with the id of the *nearest* building along
the reverse-sun vector (footprints stepped back onto the shadow raster, closest
step wins), then zeros any pixel that borders a different building's shadow so
the blobs end up separated by a 1 px gap. Marching each shadow pixel outward and
taking the first footprint hit is the same operation, vectorised.
"""

from __future__ import annotations

import numpy as np


def enforce_pixel_gap(classification_raster: np.ndarray) -> np.ndarray:
    """Zero every pixel that has an 8-neighbour carrying a different non-zero id.

    Vectorised equivalent of the thesis's per-label ``scipy.ndimage.generic_filter``
    pass (see :func:`_enforce_pixel_gap_reference`).
    """
    c = np.asarray(classification_raster)
    h, w = c.shape
    remove = np.zeros((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(c)
            src_y = slice(max(0, dy), h + min(0, dy))
            src_x = slice(max(0, dx), w + min(0, dx))
            dst_y = slice(max(0, -dy), h + min(0, -dy))
            dst_x = slice(max(0, -dx), w + min(0, -dx))
            shifted[dst_y, dst_x] = c[src_y, src_x]
            remove |= (shifted != 0) & (shifted != c) & (c != 0)
    out = c.copy()
    out[remove] = 0
    return out


def _enforce_pixel_gap_reference(classification_raster: np.ndarray) -> np.ndarray:
    """Slow, literal port of the thesis loop - used only to check :func:`enforce_pixel_gap`."""
    import scipy.ndimage as ndi

    c = np.asarray(classification_raster)
    out = c.copy()
    for lab in np.unique(c[c > 0]):
        mask = c == lab
        neighbors = ndi.generic_filter(
            c,
            lambda x, _lab=lab: np.any((x != _lab) & (x != 0)),
            size=(3, 3),
            mode="constant",
            cval=0,
        )
        out[mask & (neighbors > 0)] = 0
    return out


def enforce_shadow_gap(
    mask: np.ndarray,
    building_mask: np.ndarray,
    azimuth_deg: float,
    solar_elevation: float | None = None,
    pixel_size: float = 0.25,
    max_steps: int = 150,
) -> np.ndarray:
    """Relabel the shadow raster so each blob carries its nearest building's id.

    Returns a ``uint16`` raster: 0 = background / building / unattributed shadow,
    otherwise the ``scipy.ndimage.label`` id of the building the pixel's shadow
    belongs to, with 1 px gaps between different buildings' shadows.

    ``solar_elevation`` and ``pixel_size`` are unused (kept for signature parity
    with the thesis call sites); direction is set by ``azimuth_deg`` alone.
    """
    from scipy.ndimage import label as nd_label

    mask = np.asarray(mask)
    building_mask = np.asarray(building_mask)
    h, w = mask.shape

    az = np.radians((float(azimuth_deg) + 180.0) % 360.0)
    dy_u, dx_u = np.cos(az), -np.sin(az)

    shadow = (mask > 0.25) & (building_mask == 0)
    labeled_buildings, _ = nd_label(building_mask > 0)

    classification = np.zeros((h, w), dtype=np.uint16)
    sy, sx = np.nonzero(shadow)
    remaining = np.ones(sy.shape, dtype=bool)

    for s in range(1, max_steps + 1):
        idx = np.nonzero(remaining)[0]
        if idx.size == 0:
            break
        yy = np.round(sy[idx] + dy_u * s).astype(np.intp)
        xx = np.round(sx[idx] + dx_u * s).astype(np.intp)
        in_bounds = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        remaining[idx[~in_bounds]] = False  # ray left the raster - never comes back
        idx, yy, xx = idx[in_bounds], yy[in_bounds], xx[in_bounds]
        lab = labeled_buildings[yy, xx]
        hit = lab > 0
        classification[sy[idx[hit]], sx[idx[hit]]] = lab[hit].astype(np.uint16)
        remaining[idx[hit]] = False  # nearest building found

    return enforce_pixel_gap(classification)
