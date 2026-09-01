"""Render a step-by-step mosaic of the pseudo-shadow 'smear' accumulation.

python -m physhade.physics.smear_showcase --raster footprint.tif --azimuth 116 --length 2 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import binary_erosion, generate_binary_structure
from skimage.transform import AffineTransform, warp

from physhade.config import DATA_DIR, OUTPUT_DIR


def visualize_shadow_mosaic(
    raster_path: str,
    azimuth_deg: float,
    length_mapunits,
    out_shape: tuple = (512, 512),
    decay_type: str = "solid",
    mask_buildings: bool = False,
    erode_buildings_px: int = 0,
    out_path: str | Path | None = None,
    show: bool = False,
):
    max_steps_to_show = 7

    with rasterio.open(raster_path) as src:
        building_mask = (src.read(1) > 0).astype(np.float32)
        transform = src.transform
        px_w = transform.a
        px_h = abs(transform.e)

        shadow_bearing = (azimuth_deg + 180) % 360
        rad = np.radians(shadow_bearing)
        dx_unit = np.sin(rad)
        dy_unit = np.cos(rad)

        if isinstance(length_mapunits, (tuple, list)):
            l_min, l_max = map(float, length_mapunits)
        else:
            l_min = 2.0
            l_max = float(length_mapunits)

        l_min = max(0.01, abs(l_min))
        l_max = max(l_min + 0.01, abs(l_max))

        res_along = np.hypot(px_w * dx_unit, px_h * dy_unit)
        total_steps = max(1, int(np.ceil(l_max / res_along)))
        step_dist = l_max / total_steps

        # pick exactly 7 DISTINCT steps via linspace + uniquify + pad
        raw = np.linspace(1, total_steps, max_steps_to_show)
        rounded = np.round(raw).astype(int)
        final_indices = []
        for i in rounded:
            if i not in final_indices:
                final_indices.append(i)
        candidate = 1
        while len(final_indices) < max_steps_to_show:
            if candidate not in final_indices and 1 <= candidate <= total_steps:
                final_indices.append(candidate)
            candidate += 1
        final_indices = sorted(final_indices)

        shadow_accum = np.zeros_like(building_mask, dtype=np.float32)
        shadow_images = []
        titles = []

        for i in range(1, total_steps + 1):
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
                raise ValueError("Invalid decay_type.")
            if w == 0.0:
                continue

            dx_map = dx_unit * dist
            dy_map = dy_unit * dist
            shift_c = dx_map / px_w
            shift_r = -dy_map / px_h

            t = AffineTransform(translation=(shift_c, shift_r))
            shifted = warp(building_mask, t.inverse, order=0, mode="constant", cval=0)

            if decay_type == "solid":
                shadow_accum = np.maximum(shadow_accum, shifted)
            else:
                shadow_accum = np.maximum(shadow_accum, w * shifted)

            if mask_buildings:
                mask = building_mask >= 0.5
                if erode_buildings_px > 0:
                    strel = generate_binary_structure(2, 1)
                    eroded = binary_erosion(mask, structure=strel, iterations=erode_buildings_px)
                    shadow_accum[eroded] = 0.0
                else:
                    shadow_accum[mask] = 0.0

            if i in final_indices:
                if decay_type != "solid":
                    m = shadow_accum.max()
                    norm_shadow = shadow_accum / m if m > 0 else shadow_accum
                else:
                    norm_shadow = shadow_accum.copy()
                shadow_images.append(norm_shadow)
                titles.append(f"Step {i} | Dist {dist:.1f} | Wt {w:.2f}")

        # === 2x3 (now 2x4) Horizontal Mosaic with tight spacing ===
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={"wspace": 0.15, "hspace": 0.04})
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < len(shadow_images):
                im = ax.imshow(shadow_images[idx], cmap="gray")
                ax.set_title(titles[idx], fontsize=12)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
            else:
                ax.axis("off")

        fig.tight_layout()
        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raster",
        type=Path,
        default=DATA_DIR / "main_model" / "all" / "building_footprint" / "data_winter_loc1_5.tif",
        help="binary building-footprint raster",
    )
    p.add_argument("--azimuth", type=float, default=116.0, help="solar azimuth (deg from north)")
    p.add_argument("--length", type=float, nargs=2, metavar=("MIN", "MAX"), default=[2.0, 15.0])
    p.add_argument("--decay", choices=["solid", "delayed_gradient"], default="delayed_gradient")
    p.add_argument("--no-mask-buildings", action="store_true")
    p.add_argument("--erode-px", type=int, default=1)
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "figures" / "smear_showcase.png")
    p.add_argument("--show", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")
    out = visualize_shadow_mosaic(
        raster_path=str(args.raster),
        azimuth_deg=args.azimuth,
        length_mapunits=tuple(args.length),
        decay_type=args.decay,
        mask_buildings=not args.no_mask_buildings,
        erode_buildings_px=args.erode_px,
        out_path=args.out,
        show=args.show,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
