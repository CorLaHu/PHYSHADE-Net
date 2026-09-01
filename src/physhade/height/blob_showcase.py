"""Animate the 'assign-and-break' blob separation as a looping GIF.

    python -m physhade.height.blob_showcase --tile Summer_Tile6

Building footprints slide in along the reverse-sun vector; each shifted copy
stamps its colour onto the shadow pixels it sweeps, so one connected shadow
smear is progressively partitioned into per-building blobs. The last frames
apply the 1 px gap (:func:`physhade.height.blob_separation.enforce_pixel_gap`)
that separates neighbouring buildings' shadows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from scipy.ndimage import label as nd_label
from scipy.ndimage import shift as nd_shift
from skimage import measure

from physhade.config import DATA_DIR, OUTPUT_DIR
from physhade.height.blob_separation import enforce_pixel_gap

_FIGSIZE = (4.3, 4.6)
_DPI = 72


def _frame(raster, footprint_edges, base_rgb, cmap, n_buildings, title, sun_vec) -> np.ndarray:
    """One animation frame as an ``(H, W, 3)`` uint8 array (fixed size)."""
    h, w = raster.shape
    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.90))
    ax.imshow(base_rgb, interpolation="nearest")
    ax.imshow(
        np.where(raster > 0, raster.astype(float), np.nan),
        cmap=cmap,
        vmin=1,
        vmax=max(n_buildings, 1),
        interpolation="nearest",
    )
    if footprint_edges is not None:
        for contour in measure.find_contours(footprint_edges.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color="white", lw=0.7, alpha=0.9)
    bx, by = 0.09 * w, 0.9 * h
    ax.annotate(
        "",
        xy=(bx + sun_vec[0] * 40, by + sun_vec[1] * 40),
        xytext=(bx, by),
        arrowprops=dict(arrowstyle="-|>", color="gold", lw=2.0),
    )
    ax.text(bx, by + 18, "sun", color="gold", fontsize=9, ha="center", va="top")
    fig.text(0.5, 0.955, title, fontsize=11, ha="center", va="center")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def visualize_assign_and_break(
    shadow: np.ndarray,
    building_mask: np.ndarray,
    azimuth_deg: float,
    *,
    rgb: np.ndarray | None = None,
    max_steps: int = 120,
    step: int = 4,
    fps: int = 12,
    hold_frames: int = 14,
    out_path: str | Path = "assign_and_break.gif",
) -> Path:
    """Write the assign-and-break animation for one tile to ``out_path`` (a GIF)."""
    plt.switch_backend("Agg")
    import imageio.v2 as imageio

    shadow = np.asarray(shadow, dtype=float)
    building_mask = np.asarray(building_mask)
    h, w = shadow.shape

    if rgb is not None:  # desaturated + dimmed so the coloured blobs carry the frame
        gray = np.asarray(rgb, dtype=float) @ np.array([0.299, 0.587, 0.114])
        base_rgb = np.repeat((gray * 0.55 + 0.05)[..., None], 3, axis=2)
    else:
        base_rgb = np.full((h, w, 3), 0.11)
        base_rgb[shadow > 0.25] = 0.2
    base_rgb = np.clip(base_rgb, 0, 1)

    az_shadow = np.radians((float(azimuth_deg) + 180.0) % 360.0)
    dy_u, dx_u = np.cos(az_shadow), -np.sin(az_shadow)
    shadow_bin = (shadow > 0.25) & (building_mask == 0)
    labeled, n_buildings = nd_label(building_mask > 0)
    footprints = [labeled == b for b in range(1, n_buildings + 1)]

    palette = plt.get_cmap("tab10" if n_buildings <= 10 else "tab20", max(n_buildings, 1))
    cmap = ListedColormap([palette(i) for i in range(max(n_buildings, 1))])

    sun = np.radians(float(azimuth_deg))
    sun_vec = (float(np.sin(sun)), float(-np.cos(sun)))  # points toward the sun

    progressive = np.zeros((h, w), dtype=np.uint16)
    raw: list[tuple[np.ndarray, np.ndarray]] = []  # (progressive_copy, footprint_union)
    for s in range(max_steps, 0, -step):
        union = np.zeros((h, w), dtype=bool)
        for bid, footprint in enumerate(footprints, 1):
            moved = nd_shift(footprint.astype(float), (-dy_u * s, -dx_u * s), order=0, cval=0.0) > 0.5
            union |= moved
            progressive[moved & shadow_bin] = bid
        raw.append((progressive.copy(), union))

    # keep only the stretch where the assignment is actually changing (+ a little padding)
    changed = [k for k in range(1, len(raw)) if not np.array_equal(raw[k][0], raw[k - 1][0])]
    if changed:
        raw = raw[max(0, changed[0] - 3) : min(len(raw), changed[-1] + 4)]

    frames = [
        _frame(prog, union, base_rgb, cmap, n_buildings, f"assign  ({k}/{len(raw)})", sun_vec)
        for k, (prog, union) in enumerate(raw, 1)
    ]

    separated = enforce_pixel_gap(progressive)
    frames += [
        _frame(separated, None, base_rgb, cmap, n_buildings, "break  (1 px gap between buildings)", sun_vec)
    ] * hold_frames

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, format="GIF", fps=fps, loop=0, subrectangles=True, palettesize=64)
    return out_path


def _pick_tile(val_root: Path) -> str:
    """The val tile with the most (buildings x shadow coverage)."""
    best, best_score = "Summer_Tile6", -1.0
    for fp in sorted((val_root / "building_footprint").glob("*.tif")):
        with rasterio.open(fp) as s:
            n = nd_label(s.read(1) > 0)[1]
        sh_path = val_root / "annotated_shadows" / fp.name
        if not sh_path.exists():
            continue
        with rasterio.open(sh_path) as s:
            frac = float((s.read(1) > 0).mean())
        if (score := n * frac) > best_score:
            best, best_score = fp.stem, score
    return best


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR / "val_set", help="a val_set-style tree")
    p.add_argument("--tile", default=None, help="tile stem (default: auto-pick the busiest)")
    p.add_argument("--no-rgb", action="store_true", help="use a plain silhouette instead of the aerial image")
    p.add_argument("--max-steps", type=int, default=130)
    p.add_argument(
        "--step", type=int, default=3, help="stride over the 1..max_steps sweep (higher = fewer frames)"
    )
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "figures" / "assign_and_break.gif")
    return p


def main(argv: list[str] | None = None) -> None:
    plt.switch_backend("Agg")
    args = build_parser().parse_args(argv)
    root = args.data_dir
    tile = args.tile or _pick_tile(root)

    with rasterio.open(root / "annotated_shadows" / f"{tile}.tif") as s:
        shadow = (s.read(1) > 0).astype("float32")
    with rasterio.open(root / "building_footprint" / f"{tile}.tif") as s:
        building = (s.read(1) > 0).astype("uint8")

    rgb = None
    img_path = root / "georef_image" / f"{tile}.tif"
    if not args.no_rgb and img_path.exists():
        with rasterio.open(img_path) as s:
            rgb = s.read([1, 2, 3]).transpose(1, 2, 0).astype("float32") / 255.0

    from physhade.height.height_pipeline import layout, solar_angle_lookup

    shp = DATA_DIR / "main_model" / "shpfiles"
    az = 135.0
    try:
        solar = solar_angle_lookup(
            layout(root)["image"],
            shp / "2023_beeldmiddenoverzicht_lrl.shp",
            shp / "2023_beeldmiddenoverzicht_hrl.shp",
        )
        az = solar.get(tile, (135.0, 0.0))[0]
    except Exception:  # noqa: BLE001 - the shapefiles are optional for the animation
        print(f"solar shapefiles unavailable; using azimuth {az} deg for {tile}")

    out = visualize_assign_and_break(
        shadow,
        building,
        az,
        rgb=rgb,
        max_steps=args.max_steps,
        step=args.step,
        fps=args.fps,
        out_path=args.out,
    )
    print(f"wrote {out}  (tile {tile}, azimuth {az:.1f} deg)")


if __name__ == "__main__":
    main()
