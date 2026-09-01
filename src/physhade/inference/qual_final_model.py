"""Stage B5b: qualitative evaluation of the final model.

    python -m physhade.inference.qual_final_model --checkpoint epoch120.pth

Per tile writes ``RGB | smear(+footprints) | error-map`` strips (0.5 and Otsu
thresholds) plus best/worst mosaics under ``Output/main_model/qualitative/finalmodel``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from skimage.filters import threshold_otsu
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from physhade.config import DATA_DIR, OUTPUT_DIR, resolve_checkpoint
from physhade.models.basemodel import PHYSHADENet

matplotlib.use("Agg")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(path: Path, in_ch: int) -> torch.nn.Module:
    net = PHYSHADENet(in_channels=in_ch, out_channels=1)
    state = torch.load(path, map_location=_DEVICE)["model_state_dict"]
    net.load_state_dict(state, strict=False)
    net = net.eval().to(_DEVICE)
    return net.half() if _DEVICE == "cuda" else net.float()


def dice_np(pred, gt, eps=1e-6):
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + eps)


def get_error_map(pred, gt, rgb):
    err = np.zeros_like(rgb)
    err[np.logical_and(pred, gt == 1)] = (0.0, 1, 0)
    err[np.logical_and(pred, gt == 0)] = (1, 0, 0)
    err[np.logical_and(~pred, gt == 1)] = (0, 0, 1)
    return err


def save_lineup(rgb, smear, gt, pred, err, out_dir, uid, building=None):
    spacer = np.ones((rgb.shape[0], 1, 3), dtype=np.float32)
    smear_vis = np.repeat(smear.squeeze()[..., None], 3, axis=2)
    if building is not None:
        smear_vis[building] = (0.8667, 0.8667, 0.8667)
    strip = np.concatenate([rgb, spacer, smear_vis, spacer, err], axis=1)
    plt.imsave(out_dir / f"{uid}_lineup.png", strip, vmin=0, vmax=1)


def make_mosaic(img_paths, out_path, nrow=2):
    imgs = [torch.from_numpy(plt.imread(str(p)).transpose(2, 0, 1)) for p in img_paths]
    save_image(make_grid(imgs, nrow=nrow), out_path)


def run(checkpoint: Path, data_root: Path, out_root: Path) -> None:
    image_dir, smear_dir = data_root / "image", data_root / "smear_shadow"
    mask_dir, footprint_dir = data_root / "annotated_shadow", data_root / "building_footprint"
    per_tile = out_root / "per_tile"
    lineup_05, lineup_ots, mos = (out_root / d for d in ("lineups_thresh_0.5", "lineups_otsu", "mosaics"))
    for d in (per_tile, lineup_05, lineup_ots, mos):
        d.mkdir(parents=True, exist_ok=True)

    uids = set()
    for f in (data_root / "pairs.txt", data_root / "singles.txt"):
        if f.exists():
            for line in f.read_text().splitlines():
                uids.update(line.split())
    uids = sorted(uids)

    net = load_checkpoint(checkpoint, in_ch=4)
    scores = []
    for uid in tqdm(uids, desc="tiles"):
        img_p, mask_p, smear_p = image_dir / f"{uid}.tif", mask_dir / f"{uid}.tif", smear_dir / f"{uid}.tif"
        if not (img_p.exists() and mask_p.exists() and smear_p.exists()):
            continue
        with rasterio.open(mask_p) as s:
            gt = s.read(1) > 0
        if gt.sum() == 0:
            continue
        with rasterio.open(img_p) as s:
            rgb = s.read([1, 2, 3]).transpose(1, 2, 0) / 255.0
        with rasterio.open(smear_p) as s:
            smear = s.read(1).astype("float32")[None]
            if smear.max() > 1.0:
                smear /= 255.0
        building = None
        if (footprint_dir / f"{uid}.tif").exists():
            with rasterio.open(footprint_dir / f"{uid}.tif") as s:
                building = s.read(1) > 0

        plt.imsave(per_tile / f"{uid}_rgb.png", rgb, vmin=0, vmax=1)
        x = torch.from_numpy(np.concatenate([rgb.transpose(2, 0, 1), smear], axis=0)).unsqueeze(0).to(_DEVICE)
        x = x.half() if _DEVICE == "cuda" else x.float()
        with torch.no_grad():
            logits = torch.sigmoid(net(x))[0, 0].float().cpu().numpy()

        pr_fixed, pr_otsu = logits > 0.5, logits > threshold_otsu(logits)
        scores.append((uid, dice_np(pr_fixed, gt)))
        save_lineup(rgb, smear, gt, pr_fixed, get_error_map(pr_fixed, gt, rgb), lineup_05, uid, building)
        save_lineup(rgb, smear, gt, pr_otsu, get_error_map(pr_otsu, gt, rgb), lineup_ots, uid, building)

    scores.sort(key=lambda x: x[1])
    if scores:
        make_mosaic([per_tile / f"{u}_rgb.png" for u, _ in scores[-4:]], mos / "best.png")
        make_mosaic([per_tile / f"{u}_rgb.png" for u, _ in scores[:4]], mos / "worst.png")
        print(f"mean Dice {np.mean([d for _, d in scores]):.4f} over {len(scores)} tiles -> {out_root}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None, help="default: newest final_*/*/epoch*.pth")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR / "main_model" / "train")
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    ckpt = resolve_checkpoint(args.checkpoint, subdir="main_model", pattern="final_*/*/epoch*.pth")
    out = (
        args.out_dir
        or OUTPUT_DIR / "main_model" / "qualitative" / f"finalmodel_{datetime.now():%Y_%m_%d_%H_%M}"
    )
    run(ckpt, args.data_dir, out)


if __name__ == "__main__":
    main()
