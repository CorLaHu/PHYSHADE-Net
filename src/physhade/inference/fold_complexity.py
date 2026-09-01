"""Stage B5d: per-fold image-complexity analysis (thesis appendix).

    python -m physhade.inference.fold_complexity

For every training tile, in the fold split ``mainmodel_training`` uses: edge
density (Canny), shadow coverage, and shadow-vs-background grey contrast.
Writes a per-image CSV, a per-fold summary, per-metric boxplots and a
correlation heatmap to ``Output/main_model/fold_complexity/`` - the analysis
that explains fold-to-fold Dice variance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from physhade.config import DATA_DIR, OUTPUT_DIR

_METRICS = ["Edge Density", "Shadow Coverage", "Shadow-Background Contrast"]


def _tile_metrics(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    import cv2

    gray = rgb @ np.array([0.299, 0.587, 0.114])
    edges = cv2.Canny((np.clip(gray, 0, 1) * 255).astype("uint8"), 100, 200)
    edge_density = float((edges > 0).mean())
    shadow_coverage = float(mask.mean())
    fg, bg = gray[mask], gray[~mask]
    contrast = float(abs(fg.mean() - bg.mean())) if fg.size and bg.size else np.nan
    return edge_density, shadow_coverage, contrast


def _folds(data_dir: Path, k: int, seed: int) -> list[list[str]]:
    """The exact grouped k-fold split ``mainmodel_training`` uses."""
    import physhade.train.mainmodel_training as mmt

    mmt.ROOT = data_dir
    mmt.MASK_DIR = data_dir / "annotated_shadow"
    return mmt.load_grouped_folds(data_dir / "pairs.txt", data_dir / "singles.txt", k=k, seed=seed)


def run(data_dir: Path, out_dir: Path, k: int, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir, mask_dir = data_dir / "image", data_dir / "annotated_shadow"

    rows = []
    for fold_idx, uids in enumerate(_folds(data_dir, k, seed), start=1):
        for uid in uids:
            ip, mp = image_dir / f"{uid}.tif", mask_dir / f"{uid}.tif"
            if not (ip.exists() and mp.exists()):
                continue
            with rasterio.open(ip) as s:
                rgb = s.read([1, 2, 3]).transpose(1, 2, 0).astype("float32") / 255.0
            with rasterio.open(mp) as s:
                mask = s.read(1) > 0
            ed, sc, ct = _tile_metrics(rgb, mask)
            rows.append(
                {
                    "Fold": fold_idx,
                    "Image": uid,
                    "Edge Density": ed,
                    "Shadow Coverage": sc,
                    "Shadow-Background Contrast": ct,
                }
            )

    df = pd.DataFrame(rows)
    df.round(4).to_csv(out_dir / "image_complexity_metrics.csv", index=False)
    if df.empty:
        print(f"no tiles found under {data_dir}")
        return out_dir

    summary = df.drop(columns="Image").groupby("Fold").agg(["mean", "std"]).round(4)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out_dir / "fold_complexity_summary.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        for metric in _METRICS:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df, x="Fold", y=metric)
            plt.title(f"{metric} by fold")
            plt.grid(True)
            plt.tight_layout()
            slug = metric.lower().replace(" ", "_").replace("-", "_")
            plt.savefig(out_dir / f"{slug}_boxplot.png", dpi=120)
            plt.close()

        plt.figure(figsize=(5, 4))
        sns.heatmap(df[_METRICS].corr().round(2), annot=True, cmap="coolwarm", square=True)
        plt.title("complexity-metric correlation")
        plt.tight_layout()
        plt.savefig(out_dir / "complexity_metric_correlation.png", dpi=120)
        plt.close()
    except ImportError:
        print("matplotlib/seaborn unavailable - skipping plots")

    print(f"-> {out_dir}  ({len(df)} tiles across {df['Fold'].nunique()} folds)")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR / "main_model" / "train")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None, help="default: Output/main_model/fold_complexity")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not (args.data_dir / "pairs.txt").exists():
        raise SystemExit(f"{args.data_dir}/pairs.txt not found - run training_preprocessing first")
    out = args.out_dir or OUTPUT_DIR / "main_model" / "fold_complexity"
    run(args.data_dir, out, args.folds, args.seed)


if __name__ == "__main__":
    main()
