"""Per-tile diagnostic figures for the height pipeline.

Two renderers, matplotlib lazy-imported so importing this module stays cheap:

- ``render_blob_map``  - blob-ID raster + RGB with the per-blob colour overlay,
  matched-building boundaries coloured by their blob, truncated blobs dashed.
- ``render_ray_plot``  - each blob's percentile ray drawn over the shadow +
  building underlay with an ``S: X.XX m`` label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _free_label(
    reserved: np.ndarray, mx: float, my: float, lw: int = 22, lh: int = 30
) -> tuple[float, float]:
    """Find a label anchor near ``(mx, my)`` not overlapping ``reserved`` (updated in place)."""
    h, w = reserved.shape
    for radius in range(20, 80, 10):
        for angle in range(0, 360, 30):
            cr = int(my - radius * np.sin(np.radians(angle)))
            cc = int(mx + radius * np.cos(np.radians(angle)))
            r0, r1 = max(0, cr - lh // 2), min(h, cr + lh // 2)
            c0, c1 = max(0, cc - lw // 2), min(w, cc + lw // 2)
            if r1 > r0 and c1 > c0 and not reserved[r0:r1, c0:c1].any():
                reserved[r0:r1, c0:c1] = True
                return cc, cr
    return mx, my


def render_ray_plot(
    mask: np.ndarray,
    building_mask: np.ndarray,
    diagnostics_data: list[dict],
    pixel_size: float,
    azimuth_deg: float,
    out_path: str | Path,
    percentile: float = 99,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(mask, cmap="gray", alpha=0.5)
    ax.imshow(np.asarray(building_mask) > 0, cmap="Reds", alpha=0.3)

    reserved = np.zeros(mask.shape, dtype=bool)
    for d in diagnostics_data:
        (xs, ys), (xe, ye) = d["percentile_ray_start"], d["percentile_ray_end"]
        n = max(2, int(np.hypot(xe - xs, ye - ys)))
        xi = np.clip(np.linspace(xs, xe, n).astype(int), 0, mask.shape[1] - 1)
        yi = np.clip(np.linspace(ys, ye, n).astype(int), 0, mask.shape[0] - 1)
        rm = np.zeros_like(reserved)
        rm[yi, xi] = True
        reserved |= binary_dilation(rm, structure=np.ones((10, 10)))

    for d in diagnostics_data:
        (xs, ys), (xe, ye) = d["percentile_ray_start"], d["percentile_ray_end"]
        s_len = d["percentile_ray_length_px"] * pixel_size
        ax.plot([xs, xe], [ys, ye], lw=2, label=f"blob {d['blob_id']}")
        mx, my = (xs + xe) / 2, (ys + ye) / 2
        lx, ly = _free_label(reserved, mx, my)
        ax.annotate(
            f"S: {s_len:.2f} m",
            xy=(mx, my),
            xytext=(lx, ly),
            fontsize=8,
            color="yellow",
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.5),
            arrowprops=dict(arrowstyle="-", linestyle=":", linewidth=0.75, color="black"),
        )

    ax.set_title(f"p{percentile:g} ray per blob  (azimuth {float(azimuth_deg):.1f} deg)")
    if diagnostics_data:
        ax.legend(fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_blob_map(
    rgb: np.ndarray,
    blob_id_map: np.ndarray,
    match_map: np.ndarray,
    dissolved,
    transform,
    truncated_ids: set[int],
    azimuth_deg: float,
    out_path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from skimage import measure
    from skimage.measure import regionprops

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob_id_map = np.asarray(blob_id_map).astype("int32")
    n = max(1, int(blob_id_map.max()))
    base = plt.get_cmap("tab20", n)
    cmap = ListedColormap([base(i) for i in range(n)])
    coloured = np.where(blob_id_map > 0, blob_id_map, np.nan)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"wspace": 0.03})

    axl.imshow(np.zeros_like(blob_id_map), cmap="gray")
    axl.imshow(coloured, cmap=cmap, vmin=1, vmax=n)
    for p in regionprops(blob_id_map):
        axl.text(
            p.centroid[1],
            p.centroid[0],
            str(p.label),
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
        )
    axl.set_title("blob id")
    axl.axis("off")

    axr.imshow(np.clip(np.asarray(rgb, float), 0, 1))
    axr.imshow(coloured, cmap=cmap, vmin=1, vmax=n, alpha=0.85)

    # building -> its blob (most common non-zero blob id under the building's match pixels)
    building_to_blob: dict[int, int] = {}
    for b_id in np.unique(match_map):
        if b_id == 0:
            continue
        vals = blob_id_map[match_map == b_id]
        vals = vals[vals != 0]
        if vals.size:
            ids, counts = np.unique(vals, return_counts=True)
            building_to_blob[int(b_id)] = int(ids[np.argmax(counts)])

    if dissolved is not None and not dissolved.empty:
        inv = ~transform
        for raster_id, geom in enumerate(dissolved.geometry, start=1):
            blob = building_to_blob.get(raster_id)
            if not blob or geom is None or geom.is_empty:
                continue
            colour = base(blob - 1)
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                xs, ys = poly.exterior.xy
                px, py = zip(*(inv * (x, y) for x, y in zip(xs, ys)), strict=True)
                axr.plot(px, py, color=colour, lw=1.6, ls="--")

    for bid in truncated_ids:
        m = blob_id_map == bid
        if not m.any():
            continue
        for c in measure.find_contours(m.astype(float), 0.5):
            axr.plot(c[:, 1], c[:, 0], color="black", lw=1.0, ls=":")

    axr.set_title("RGB + blobs (matched-building outline; truncated dotted)")
    axr.axis("off")
    fig.suptitle(f"azimuth {float(azimuth_deg):.1f} deg", y=0.99)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
