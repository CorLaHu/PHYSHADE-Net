"""Stage B5: qualitative evaluation of the ablation runs (reads runs.csv).

    python -m physhade.inference.qual_analysis --runs-csv Output/main_model/runs.csv

For the best fold of each experiment, writes per-tile RGB | error-map | smear
strips (0.5 & Otsu) plus best/worst mosaics and a summary CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from skimage.filters import threshold_otsu
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from physhade.config import DATA_DIR, OUTPUT_DIR

matplotlib.use("Agg")
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_OUT = OUTPUT_DIR / "main_model"
DATA_ROOT = DATA_DIR / "main_model" / "train"
IMAGE_DIR = DATA_ROOT / "image"
SMEAR_DIR = DATA_ROOT / "smear_shadow"
MASK_DIR = DATA_ROOT / "annotated_shadow"
BUILDING_DIR = DATA_ROOT / "building_footprint"
PAIR_FILE = DATA_ROOT / "pairs.txt"
SINGLE_FILE = DATA_ROOT / "singles.txt"
OUT_ROOT = BASE_OUT / "qualitative"

#  helpers


def load_best_ckpt(row, in_ch: int):
    from physhade.models import PHYSHADENet

    net = PHYSHADENet(in_channels=in_ch, out_channels=1)
    state = torch.load(BASE_OUT / row["best_ckpt"], map_location=device)["model_state_dict"]
    net.load_state_dict(state, strict=False)
    net = net.eval().to(device)
    return net.half() if device == "cuda" else net.float()


def dice_np(pred, gt, eps=1e-6):
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + eps)


def get_error_map(pred, gt, rgb):
    err = np.zeros_like(rgb)
    tp = np.logical_and(pred, gt == 1)
    fp = np.logical_and(pred, gt == 0)
    fn = np.logical_and(~pred, gt == 1)
    err[tp] = (0.0, 1, 0)  # Greenish teal for true positives
    err[fp] = (1, 0, 0)  # Magenta for false positives
    err[fn] = (0, 0, 1)  # Orange for false negatives
    return err


def save_lineup(rgb, smear, gt, pred, err, out_dir, uid, building=None):
    h = rgb.shape[0]
    spacer = np.ones((h, 1, 3), dtype=np.float32)
    # Visualize smear mask in grayscale
    smear_vis = np.repeat(smear.squeeze()[..., None], 3, axis=2)
    # Overlay building footprints in yellow for context
    if building is not None:
        smear_vis[building] = (0.8667, 0.8667, 0.8667)
    # Ground truth and prediction strips
    gt_vis = np.repeat(gt.astype("float32")[..., None], 3, axis=2)
    pred_vis = np.repeat(pred.astype("float32")[..., None], 3, axis=2)
    # Combine strips: RGB | spacer | error map | spacer | smear (with buildings)
    strip = np.concatenate([rgb, spacer, err, spacer, smear_vis], axis=1)
    plt.imsave(out_dir / f"{uid}_lineup.png", strip, vmin=0, vmax=1)


def make_mosaic(img_paths, out_path, nrow=2):
    imgs = [torch.from_numpy(plt.imread(p).transpose(2, 0, 1)) for p in img_paths]
    grid = make_grid(imgs, nrow=nrow)
    save_image(grid, out_path)


def _load_uids() -> list[str]:
    uids: set[str] = set()
    for f in (PAIR_FILE, SINGLE_FILE):
        if f.exists():
            for line in f.read_text().splitlines():
                uids.update(line.split())
    return sorted(uids)


def run(runs_csv: Path) -> None:
    global OUT_ROOT
    uids = _load_uids()

    df = pd.read_csv(runs_csv)
    fold_rows = df[df["fold"] != "avg"].copy()
    best_rows = fold_rows.sort_values("dice", ascending=False).groupby("exp_id", as_index=False).first()

    summary = []

    for _, row in tqdm(best_rows.iterrows(), total=len(best_rows), desc="Experiments"):
        exp_id = row["exp_id"]
        use_prior = str(row["use_prior_channel"]).lower() in ("true", "1", "yes")
        in_ch = 4 if use_prior else 3
        net = load_best_ckpt(row, in_ch)

        per_tile_dir = OUT_ROOT / "per_tile" / exp_id
        lineup_dir_fixed = OUT_ROOT / "lineups_thresh_0.5" / exp_id
        lineup_dir_otsu = OUT_ROOT / "lineups_otsu" / exp_id

        per_tile_dir.mkdir(parents=True, exist_ok=True)
        lineup_dir_fixed.mkdir(parents=True, exist_ok=True)
        lineup_dir_otsu.mkdir(parents=True, exist_ok=True)

        scores = []
        for uid in uids:
            if not (IMAGE_DIR / f"{uid}.tif").exists() or not (MASK_DIR / f"{uid}.tif").exists():
                continue

            # Load ground truth mask
            with rasterio.open(MASK_DIR / f"{uid}.tif") as src:
                gt = src.read(1) > 0
            if gt.sum() == 0:
                continue

            # Load RGB tile
            with rasterio.open(IMAGE_DIR / f"{uid}.tif") as src:
                rgb = src.read([1, 2, 3]).transpose(1, 2, 0) / 255.0
            # Load smear mask
            with rasterio.open(SMEAR_DIR / f"{uid}.tif") as src:
                smear = src.read(1).astype("float32")[None]
                if smear.max() > 1.001:
                    smear /= 255.0
            # Load building footprints mask
            building = None
            bld_path = BUILDING_DIR / f"{uid}.tif"
            if bld_path.exists():
                with rasterio.open(bld_path) as src:
                    building = src.read(1) > 0

            # Prepare input tensor
            x_np = np.concatenate([rgb.transpose(2, 0, 1), smear], 0) if use_prior else rgb.transpose(2, 0, 1)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device).half()
            with torch.no_grad(), torch.amp.autocast(device):
                logits = torch.sigmoid(net(x))[0, 0].cpu().numpy()

            # Thresholding
            pr_fixed = logits > 0.5
            thresh_otsu = threshold_otsu(logits)
            pr_otsu = logits > thresh_otsu

            # Dice score
            d = dice_np(pr_fixed, gt)
            scores.append((uid, d))

            # Error maps
            err_fixed = get_error_map(pr_fixed, gt, rgb)
            err_otsu = get_error_map(pr_otsu, gt, rgb)

            # Save lineups with building overlay on smear
            save_lineup(rgb, smear, gt, pr_fixed, err_fixed, lineup_dir_fixed, uid, building)
            save_lineup(rgb, smear, gt, pr_otsu, err_otsu, lineup_dir_otsu, uid, building)

            plt.imsave(per_tile_dir / f"{uid}_rgb.png", rgb, vmin=0, vmax=1)
        if not scores:
            continue

        # Compile best/worst summary and mosaics
        scores.sort(key=lambda t: t[1])
        worst = [u for u, _ in scores[:4]]
        best = [u for u, _ in scores[-4:]]

        mos_dir = OUT_ROOT / "mosaics"
        mos_dir.mkdir(exist_ok=True)
        make_mosaic([per_tile_dir / f"{u}_rgb.png" for u in best], mos_dir / f"{exp_id}_best.png")
        make_mosaic([per_tile_dir / f"{u}_rgb.png" for u in worst], mos_dir / f"{exp_id}_worst.png")

        summary.append(
            {
                "exp_id": exp_id,
                "num_tiles": len(scores),
                "dice_mean": np.mean([d for _, d in scores]),
                "dice_std": np.std([d for _, d in scores], ddof=1),
                "best_uid": best[-1],
                "best_dice": scores[-1][1],
                "worst_uid": worst[0],
                "worst_dice": scores[0][1],
                "ckpt_path": row["best_ckpt"],
            }
        )

    csv_dir = OUT_ROOT / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(csv_dir / "qualitative_summary.csv", index=False)
    print(f"done - results under {OUT_ROOT}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-csv", type=Path, default=BASE_OUT / "runs.csv")
    p.add_argument("--out-dir", type=Path, default=None, help="default: Output/main_model/qualitative")
    return p


def main(argv=None) -> None:
    global OUT_ROOT
    args = build_parser().parse_args(argv)
    if args.out_dir:
        OUT_ROOT = args.out_dir
    if not args.runs_csv.exists():
        raise SystemExit(f"runs.csv not found: {args.runs_csv} - run mainmodel_training first")
    run(args.runs_csv)


if __name__ == "__main__":
    main()
