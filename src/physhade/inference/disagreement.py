"""Stage B5e: where do two models' shadow predictions disagree?

    python -m physhade.inference.disagreement \
        --masks-root Output/main_model/height/<run>/inferred_masks \
        --image-dir Dataset/val_set/georef_image \
        --pairs HYB_BCE_PHYS10:RGBS_BCE50_DICE50,HYB_BCE_PHYS10:LUO_UNET

For each ``A:B`` pair, overlays A-only / B-only / agreement regions on the RGB
tile and writes ``<A>_vs_<B>/<uid>_disagreement.png`` plus a per-uid stats CSV
(predicted / disagreement / agreement pixel counts and fractions). Reads the
``inferred_masks/<exp>/fold_*/<ckpt>/<uid>.tif`` tree that
``height_pipeline --from runs`` produces; ``--runs-csv`` runs that inference
first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from physhade.config import DATA_DIR, OUTPUT_DIR

_A_RGB = (0.30, 0.60, 1.00)
_B_RGB = (1.00, 0.55, 0.10)
_AGREE_RGB = (0.20, 0.90, 0.35)


def _masks_for(masks_root: Path, exp_id: str) -> dict[str, Path]:
    """``{uid: mask_path}`` for the first fold / checkpoint of ``exp_id``."""
    exp_dir = masks_root / exp_id
    fold_dirs = sorted(d for d in exp_dir.glob("fold_*") if d.is_dir()) or [exp_dir]
    for fold_dir in fold_dirs:
        ckpt_dirs = sorted(d for d in fold_dir.glob("*") if d.is_dir())
        for ckpt_dir in ckpt_dirs or [fold_dir]:
            tifs = {p.stem: p for p in sorted(ckpt_dir.glob("*.tif"))}
            if tifs:
                return tifs
    return {}


def run(masks_root: Path, image_dir: Path, pairs: list[tuple[str, str]], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for a, b in pairs:
        masks_a, masks_b = _masks_for(masks_root, a), _masks_for(masks_root, b)
        common = sorted(set(masks_a) & set(masks_b))
        if not common:
            print(f"skipping {a} vs {b} - no shared inferred masks under {masks_root}")
            continue
        pair_dir = out_dir / f"{a}_vs_{b}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        stats = []

        for uid in common:
            img_path = next(image_dir.glob(f"{uid}.tif*"), None)
            if img_path is None:
                continue
            with rasterio.open(masks_a[uid]) as s:
                pa = s.read(1) > 0
            with rasterio.open(masks_b[uid]) as s:
                pb = s.read(1) > 0
            with rasterio.open(img_path) as s:
                rgb = np.clip(s.read([1, 2, 3]).transpose(1, 2, 0).astype("float32") / 255.0, 0, 1)

            disagree = np.logical_xor(pa, pb)
            agree = np.logical_and(pa, pb)
            merged = np.logical_or(pa, pb)
            total = int(merged.sum()) or 1
            stats.append(
                {
                    "uid": uid,
                    "predicted_pixels": total,
                    "disagreement_pixels": int(disagree.sum()),
                    "agreement_pixels": int(agree.sum()),
                    "disagreement_fraction": round(disagree.sum() / total, 4),
                    "agreement_fraction": round(agree.sum() / total, 4),
                }
            )

            overlay = rgb.copy()
            a_only, b_only = pa & ~pb, pb & ~pa
            overlay[a_only] = overlay[a_only] * 0.35 + np.array(_A_RGB) * 0.65
            overlay[b_only] = overlay[b_only] * 0.35 + np.array(_B_RGB) * 0.65
            overlay[agree] = overlay[agree] * 0.35 + np.array(_AGREE_RGB) * 0.65

            fig, ax = plt.subplots(figsize=(9, 9))
            ax.imshow(overlay)
            ax.set_title(
                f"{uid}\n{a} vs {b}\n"
                f"disagreement {disagree.sum() / total:.1%}  |  agreement {agree.sum() / total:.1%}",
                fontsize=9,
            )
            ax.axis("off")
            ax.legend(
                handles=[
                    mpatches.Patch(color=_A_RGB, label=f"{a} only"),
                    mpatches.Patch(color=_B_RGB, label=f"{b} only"),
                    mpatches.Patch(color=_AGREE_RGB, label="both"),
                ],
                loc="lower right",
                fontsize=8,
                framealpha=0.7,
            )
            fig.tight_layout()
            fig.savefig(pair_dir / f"{uid}_disagreement.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

        if stats:
            csv = pair_dir / f"{a}_vs_{b}_disagreement_stats.csv"
            pd.DataFrame(stats).to_csv(csv, index=False)
            written.append(csv)
            print(f"{a} vs {b}: {len(stats)} tiles -> {pair_dir}")

    return written


def _parse_pairs(spec: str) -> list[tuple[str, str]]:
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, _, b = chunk.partition(":")
        if not b:
            raise SystemExit(f"bad --pairs entry {chunk!r} - expected 'A:B'")
        out.append((a.strip(), b.strip()))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--masks-root", type=Path, required=True, help="an inferred_masks/ tree")
    p.add_argument("--image-dir", type=Path, default=DATA_DIR / "val_set" / "georef_image")
    p.add_argument("--pairs", required=True, help="comma-separated A:B model-id pairs")
    p.add_argument("--runs-csv", type=Path, default=None, help="run inference into --masks-root first")
    p.add_argument("--smear-dir", type=Path, default=DATA_DIR / "val_set" / "smear_shadow")
    p.add_argument("--final", action="store_true", help="--runs-csv: use every row, not just RGBS_/HYB_")
    p.add_argument("--out-dir", type=Path, default=None, help="default: Output/main_model/disagreement")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.runs_csv is not None:
        import torch

        from physhade.height.height_pipeline import _run_inference

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _run_inference(
            args.runs_csv, args.image_dir, args.smear_dir, args.masks_root, device, args.final, None
        )

    out = args.out_dir or OUTPUT_DIR / "main_model" / "disagreement"
    written = run(args.masks_root, args.image_dir, _parse_pairs(args.pairs), out)
    print(f"wrote {len(written)} stats CSVs under {out}")


if __name__ == "__main__":
    main()
