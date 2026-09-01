"""Run the RGB U-Net baseline over a folder and dump metrics + lineup panels.

    python -m physhade.inference.quick_inference --checkpoint best_model.pth \
        --img-dir tiles/ --mask-dir masks/ --out-dir Output/inference/run1
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

matplotlib.use("Agg")

from physhade.config import DATA_DIR, OUTPUT_DIR, resolve_checkpoint
from physhade.models.basemodel import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5
IMG_EXTENSIONS = (".png", ".tif", ".jpg", ".tiff")
IMG_SIZE = 512


def load_model(checkpoint_path, in_channels=3, out_channels=1):
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    state = torch.load(checkpoint_path, map_location=DEVICE)["model_state_dict"]
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def preprocess(img):
    transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
    return transform(img)


def get_error_map(pred, gt, rgb):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    err = np.zeros_like(rgb)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    err[tp] = (0.0, 1, 0)  # green
    err[fp] = (1, 0, 0)  # red
    err[fn] = (0, 0, 1)  # blue
    return err


def dice_coeff(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 2 * inter / denom if denom > 0 else 1.0


def precision(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 1.0


def recall(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def find_mask(mask_dir, uid):
    for ext in IMG_EXTENSIONS:
        path = mask_dir / f"{uid}{ext}"
        if path.exists():
            return path
    return None


def save_lineup(rgb, smear, gt, pred, err_map, out_path):
    h = rgb.shape[0]
    spacer = np.ones((h, 2, 3), dtype=np.float32)

    def vis(x):
        if x.ndim == 2:
            return np.repeat(x[..., None], 3, axis=2)
        return x

    strip = np.concatenate([vis(rgb), spacer, vis(err_map)], axis=1)

    plt.imsave(out_path, strip, vmin=0, vmax=1)


def run(img_dir, mask_dir, ckpt_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = load_model(ckpt_path)

    # lists for per-image metrics
    image_ids = []
    dice_scores = []
    precision_scores = []
    recall_scores = []

    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])

    for img_path in tqdm(images, desc="Processing"):
        uid = img_path.stem
        mask_path = find_mask(mask_dir, uid)
        if mask_path is None:
            print(f"  Mask not found for {uid}")
            continue

        # load image and mask
        img = Image.open(img_path).convert("RGB")
        mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_bin = mask_np > 0  # binary mask

        # inference
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(img_tensor)
            prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # threshold prediction
        pred_thresh = prob_map > THRESHOLD

        # compute metrics
        dice_val = dice_coeff(pred_thresh, mask_bin)
        prec_val = precision(pred_thresh, mask_bin)
        rec_val = recall(pred_thresh, mask_bin)

        image_ids.append(uid)
        dice_scores.append(dice_val)
        precision_scores.append(prec_val)
        recall_scores.append(rec_val)

        # save lineup visualization
        rgb_np = np.array(img) / 255.0
        smear_np = np.zeros_like(mask_bin, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
        save_lineup(
            rgb_np,
            smear_np,
            mask_bin.astype(np.float32),
            pred_thresh.astype(np.float32),
            get_error_map(pred_thresh, mask_bin, rgb_np),
            output_dir / f"{uid}_lineup_thresh{THRESHOLD}.png",
        )

    # save per-image metrics to CSV
    df = pd.DataFrame(
        {"ImageID": image_ids, "Dice": dice_scores, "Precision": precision_scores, "Recall": recall_scores}
    )
    csv_path = output_dir / "metrics_per_image.csv"
    df.to_csv(csv_path, index=False)
    print(f"per-image metrics CSV saved to {csv_path}")
    return df["Dice"].mean() if len(df) else float("nan")


def build_parser() -> argparse.ArgumentParser:
    luo = DATA_DIR / "main_model" / "luo"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="base-model checkpoint; default: newest under Output/base_model/",
    )
    p.add_argument("--img-dir", type=Path, default=luo / "img")
    p.add_argument("--mask-dir", type=Path, default=luo / "masks_machine")
    p.add_argument("--out-dir", type=Path, default=None, help="default: Output/inference/<timestamp>")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ckpt = resolve_checkpoint(args.checkpoint, subdir="base_model")
    out_dir = args.out_dir or OUTPUT_DIR / "inference" / datetime.now().strftime("%Y_%m_%d_%H_%M")
    mean_dice = run(args.img_dir, args.mask_dir, ckpt, out_dir)
    print(f"mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
