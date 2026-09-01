"""Stage B3: train the single final model on all folds + the val_set.

    python -m physhade.train.final_training

Loads the newest base-model checkpoint (or --base-checkpoint) and writes
``Output/main_model/final_BCE_DICE_50/<timestamp>/``.
"""

import argparse
import hashlib
import os
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from physhade.config import DATA_DIR, resolve_checkpoint
from physhade.config import ROOT as BASE_DIR
from physhade.models.basemodel import PHYSHADENet, UNet

matplotlib.use("Agg")

RUN_STAMP = datetime.now().strftime("%Y_%m_%d_%H_%M")
ROOT = Path(str(BASE_DIR / "Dataset/main_model/train"))
BASE_OUT = Path(str(BASE_DIR / "Output/main_model"))
OUT_DIR = BASE_OUT / "final_BCE_DICE_50" / RUN_STAMP

IMAGE_DIR = ROOT / "image"
SMEAR_DIR = ROOT / "smear_shadow"
MASK_DIR = ROOT / "annotated_shadow"
PAIRS = ROOT / "pairs.txt"
SINGLES = ROOT / "singles.txt"

VAL_IMAGE_DIR = DATA_DIR / "val_set" / "georef_image"
VAL_MASK_DIR = DATA_DIR / "val_set" / "annotated_shadows"
VAL_SMEAR_DIR = DATA_DIR / "val_set" / "smear_shadow"

PRETRAIN_CKPT: Path | None = None  # set in main() via config.resolve_checkpoint
CHANNELS = 4
EPOCHS = 150
BATCH_SIZE = 16
LR = 1e-4
WD = 1e-4
SEED = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 25
MIN_DELTA = 0.001


def save_shadow_overlay(logits, y, filenames, out_dir, epoch=0):
    """
    Save overlay images showing:
        - Left: Raw RGB image
        - Right: Green: True Positive, Red: False Positive, Blue: False Negative
    Only saves for Winter_Tile2_Clipped.tif and Winter_Tile3_Clipped.tif.
    """
    probs = torch.sigmoid(logits).cpu()
    preds = (probs > 0.5).float()
    y = y.cpu()

    for idx, fname in enumerate(filenames):
        base_name = Path(fname).stem.lower()
        if base_name in ["winter_tile2_clipped", "winter_tile3_clipped"]:
            pred_mask = preds[idx, 0]
            true_mask = y[idx, 0]

            tp = (pred_mask * true_mask).numpy()
            fp = (pred_mask * (1 - true_mask)).numpy()
            fn = ((1 - pred_mask) * true_mask).numpy()

            overlay = np.zeros((3, *tp.shape), dtype=np.float32)
            overlay[1] = tp  # Green: True Positive
            overlay[0] = fp  # Red: False Positive
            overlay[2] = fn  # Blue: False Negative
            overlay = np.clip(overlay, 0, 1)

            # Load the RGB image
            rgb_path = VAL_IMAGE_DIR / f"{fname}"
            with rasterio.open(rgb_path) as src:
                rgb = src.read([1, 2, 3]).astype(np.float32)
                rgb = np.clip(rgb / 255.0, 0, 1)  # Normalize to [0,1]
                rgb = np.moveaxis(rgb, 0, -1)  # CHW -> HWC

            # Plot side-by-side
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(rgb)
            axs[0].set_title(f"Raw Image: {fname}")
            axs[0].axis("off")

            axs[1].imshow(np.moveaxis(overlay, 0, -1))
            axs[1].set_title(f"Prediction Overlay: {fname}")
            axs[1].axis("off")

            plt.tight_layout()
            out_path = out_dir / f"{base_name}_overlay_epoch{epoch:02d}.png"
            plt.savefig(out_path)
            plt.close()


#
# UTILITY
#
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def load_pretrained_model(channels: int, ckpt_path: Path) -> nn.Module:
    unet = UNet(in_channels=3, out_channels=1)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    unet.load_state_dict(ckpt["model_state_dict"], strict=True)

    model = PHYSHADENet(in_channels=channels, out_channels=1)
    new_state = OrderedDict()
    for k, v in unet.state_dict().items():
        if k == "down1.conv.0.weight" and channels == 4:
            extra = v.mean(1, keepdim=True)
            new_state[k] = torch.cat([v, extra], dim=1)
        else:
            new_state[k] = v.clone()
    model.load_state_dict(new_state, strict=False)
    return model.to(DEVICE)


#
# DATASET
#
class ShadowTiles(Dataset):
    def __init__(
        self, ids, rgb_dir, smear_dir, mask_dir, return_smear_in_x, augment=False, seed=0, multiply=1
    ):
        self.base_ids = ids
        self.rgb_dir = rgb_dir
        self.smear_dir = smear_dir
        self.mask_dir = mask_dir
        self.return_smear_in_x = return_smear_in_x
        self.augment = augment
        self.seed = seed
        self.multiply = multiply

        # Expand dataset via repeated augmentation
        if multiply > 1:
            self.ids = [(uid, i) for uid in ids for i in range(multiply)]
        else:
            self.ids = [(uid, 0) for uid in ids]  # match format
        self.len = len(self.ids)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        uid, aug_idx = self.ids[idx]
        rng = random.Random(self.seed + hash(uid) + aug_idx)  # deterministic per image-version
        rng = random.Random(self.seed + stable_hash(uid) + aug_idx)

        while True:
            with rasterio.open(self.mask_dir / f"{uid}.tif") as src:
                if src.read(1).sum() > 0:
                    break
            # fallback
            idx = rng.randrange(len(self.ids))
            uid, aug_idx = self.ids[idx]

        with rasterio.open(self.rgb_dir / f"{uid}.tif") as src:
            rgb = src.read([1, 2, 3]).astype("float32") / 255.0

        with rasterio.open(self.smear_dir / f"{uid}.tif") as src:
            smear = src.read(1).astype("float32")[None, ...]

        with rasterio.open(self.mask_dir / f"{uid}.tif") as src:
            mask = src.read(1).astype("float32")[None, ...]

        if mask.sum() == 0:
            idx = rng.randrange(len(self.ids))
            return self.__getitem__(idx)

        rgb_t = torch.from_numpy(rgb)
        smear_t = torch.from_numpy(smear)
        y_t = torch.from_numpy((mask > 0).astype("float32"))

        if self.return_smear_in_x:
            x_t = torch.cat([rgb_t, smear_t], 0)
            aux = smear_t
        else:
            x_t = rgb_t
            aux = smear_t

        if self.augment:
            x_t, y_t = self._augment(x_t, y_t, rng)
        filename = f"{uid}.tif"  # get the image filename
        return x_t, y_t, aux, filename

    def _augment(self, x, y, rng):
        if x.shape[0] == 4:
            rgb = x[:3]
            smear = x[3:]
        else:
            rgb = x
            smear = None

        # --- Deterministic geometric augmentation based on aug_idx ---
        # Use aug_idx % 8 to get one of 8 spatial variations
        geom_ops = [
            lambda r, y: (r, y),  # identity
            lambda r, y: (TF.hflip(r), TF.hflip(y)),
            lambda r, y: (TF.vflip(r), TF.vflip(y)),
            lambda r, y: (TF.rotate(r, 90), TF.rotate(y, 90)),
            lambda r, y: (TF.rotate(r, 180), TF.rotate(y, 180)),
            lambda r, y: (TF.rotate(r, 270), TF.rotate(y, 270)),
            lambda r, y: (TF.hflip(TF.rotate(r, 90)), TF.hflip(TF.rotate(y, 90))),
            lambda r, y: (TF.vflip(TF.rotate(r, 90)), TF.vflip(TF.rotate(y, 90))),
        ]
        op = geom_ops[rng.randint(0, len(geom_ops) - 1)]
        rgb, y = op(rgb, y)
        if smear is not None:
            smear, _ = op(smear, y)

        # --- Always apply photometric jitter (on RGB only) ---
        brightness = 1 + (rng.random() - 0.5) * 0.4  # ±20%
        contrast = 1 + (rng.random() - 0.5) * 0.4
        saturation = 1 + (rng.random() - 0.5) * 0.4

        rgb = TF.adjust_brightness(rgb, brightness)
        rgb = TF.adjust_contrast(rgb, contrast)
        rgb = TF.adjust_saturation(rgb, saturation)

        if smear is not None:
            x = torch.cat([rgb, smear], dim=0)
        else:
            x = rgb

        return x, y


def custom_collate(batch):
    data = list(zip(*batch))
    x = default_collate(data[0])
    y = default_collate(data[1])
    sm = default_collate(data[2])
    filenames = list(data[3])  # keep as list
    return x, y, sm, filenames


#
# LOSS
#
class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.1, weight_dice=0.9):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-6) / (probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-6)
        dice_loss = 1 - dice.mean()

        # Combined
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


class Dice_Loss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (preds_flat * targets_flat).sum(1)
        dice = (2.0 * intersection + self.smooth) / (preds_flat.sum(1) + targets_flat.sum(1) + self.smooth)
        return 1 - dice.mean()  # Dice Loss


class BCE_Dice_Physics_Dice(nn.Module):
    def __init__(self, weight_phys=0.3, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.supLoss = BCEDiceLoss(weight_bce, weight_dice)
        self.weight_phys = weight_phys
        self.dice_phys = Dice_Loss()

    def forward(self, logits, targets, smear):
        sup = self.supLoss(logits, targets)  # supervised term
        smear = smear.to(logits.device)
        phys = self.dice_phys(logits, smear)  # Dice loss for physics term
        return sup + self.weight_phys * phys


#
# MAIN
#
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--base-checkpoint", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv=None):
    global ROOT, IMAGE_DIR, SMEAR_DIR, MASK_DIR, PAIRS, SINGLES, OUT_DIR
    global PRETRAIN_CKPT, EPOCHS, BATCH_SIZE, SEED
    args = build_parser().parse_args(argv)
    ROOT = args.data_dir
    IMAGE_DIR, SMEAR_DIR, MASK_DIR = ROOT / "image", ROOT / "smear_shadow", ROOT / "annotated_shadow"
    PAIRS, SINGLES = ROOT / "pairs.txt", ROOT / "singles.txt"
    OUT_DIR = args.out_dir or (BASE_OUT / "final_BCE_DICE_50" / RUN_STAMP)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EPOCHS, BATCH_SIZE, SEED = args.epochs, args.batch_size, args.seed
    PRETRAIN_CKPT = resolve_checkpoint(args.base_checkpoint, subdir="base_model")

    set_seed(SEED)

    # Training set loading (unchanged)
    uids = []
    for a, b in (line.split() for line in open(PAIRS) if line.strip()):
        uids.extend([a, b])
    uids.extend(line.strip() for line in open(SINGLES) if line.strip())
    uids = sorted(set(uids))

    good_ids = []
    for uid in uids:
        with rasterio.open(MASK_DIR / f"{uid}.tif") as src:
            if src.read(1).sum() > 0:
                good_ids.append(uid)

    train_ds = ShadowTiles(
        good_ids, IMAGE_DIR, SMEAR_DIR, MASK_DIR, augment=True, return_smear_in_x=True, seed=SEED, multiply=8
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    # NEW: Validation dataset (no smear channel assumed)
    val_ids = [f.stem for f in VAL_IMAGE_DIR.glob("*.tif")]
    val_ds = ShadowTiles(
        val_ids, VAL_IMAGE_DIR, VAL_SMEAR_DIR, VAL_MASK_DIR, augment=False, return_smear_in_x=True, seed=SEED
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    model = load_pretrained_model(CHANNELS, PRETRAIN_CKPT).to(memory_format=torch.channels_last)
    criterion = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)
    # criterion = BCE_Dice_Physics_Dice(weight_phys=0.3, weight_bce=0.5, weight_dice=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    loss_history = []
    dice_history = []
    val_loss_history = []
    val_dice_history = []

    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for x, y, sm, _ in train_loader:
            x, y, sm = x.to(DEVICE), y.to(DEVICE), sm.to(DEVICE)
            optimizer.zero_grad()

            with amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)

                # Dice score computation
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                intersection = (preds * y).sum(dim=(2, 3))
                union = preds.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
                dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
                dice_score = dice.mean().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            running_dice += dice_score * x.size(0)

        avg_loss = running_loss / len(train_ds)
        avg_dice = running_dice / len(train_ds)
        loss_history.append(avg_loss)
        dice_history.append(avg_dice)

        # Validation phase
        model.eval()
        val_loss = 0.0

        # Initialize global counters
        TP = FP = FN = 0

        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=torch.float16):
            for x, y, _sm, filenames in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                # Accumulate confusion matrix counts globally
                TP += (preds * y).sum().item()
                FP += (preds * (1 - y)).sum().item()
                FN += ((1 - preds) * y).sum().item()

                val_loss += loss.item() * x.size(0)

                # Print per-image Dice scores
                intersection = (preds * y).sum(dim=(2, 3))
                union = preds.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
                dice_per_image = (2.0 * intersection + 1e-6) / (union + 1e-6)
                for fname, d in zip(filenames, dice_per_image.cpu().numpy()):
                    print(f"Validation - File: {fname} - Dice Score: {float(d):.4f}")

                # Save overlays
                save_shadow_overlay(logits, y, filenames, OUT_DIR, epoch)

        avg_val_loss = val_loss / len(val_ds)
        global_val_dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
        val_loss_history.append(avg_val_loss)
        val_dice_history.append(global_val_dice)

        scheduler.step(avg_val_loss)

        print(
            f"[Epoch {epoch:02d}/{EPOCHS}]  "
            f"Train Loss: {avg_loss:.4f}  Train Dice: {avg_dice:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}  Val Global Dice: {global_val_dice:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        torch.save({"model_state_dict": model.state_dict()}, OUT_DIR / f"epoch{epoch:02d}.pth")

        # Early stopping on validation loss
        if best_val_loss - avg_val_loss > MIN_DELTA:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n Early stopping: No improvement in {PATIENCE} epochs (Δ < {MIN_DELTA})")
                break

    # Plot validation Dice
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        x=list(range(1, len(val_dice_history) + 1)), y=val_dice_history, label="Validation Dice Score"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Score over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "val_dice_plot.png")
    plt.close()
    print(f"done - results in {OUT_DIR}")


if __name__ == "__main__":
    main()
