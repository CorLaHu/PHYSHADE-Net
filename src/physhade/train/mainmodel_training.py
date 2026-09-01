"""Stage B2: 5-fold cross-validated loss-function ablation (the thesis workhorse).

    python -m physhade.train.mainmodel_training                 # the 6 hybrid configs (DEFAULT_CONFIGS)
    python -m physhade.train.mainmodel_training --configs all   # the full 26-config ablation
    python -m physhade.train.mainmodel_training --configs HYB_DICE_PHYS10 --folds 1 --epochs 2

Loads the newest base-model checkpoint (or --base-checkpoint) and writes per-config
per-fold runs into Output/main_model/<timestamp>/ plus rows in runs.csv.
"""

import argparse
import os
import random
import shutil
import signal
import sys
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# rasterio is imported before torch/torchvision on purpose: on Windows the two ship
# clashing GDAL DLLs and load order decides which wins (E402 is ignored for this file).
import rasterio  # isort: skip
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from kornia.losses import DiceLoss
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from physhade.config import ROOT as BASE_DIR
from physhade.config import resolve_checkpoint
from physhade.models.basemodel import AttentivePHYSHADENet, PHYSHADENet, UNet

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _handle(sig, frame):
    print("Interrupted, cleaning up...")
    if "tb_fold" in globals():
        tb_fold.close()  # noqa: F821 (set as global in the training loop)
    sys.exit(0)


RUN_STAMP = datetime.now().strftime("%Y_%m_%d_%H_%M")
ROOT = Path(str(BASE_DIR / "Dataset/main_model/train"))
BASE_OUT = Path(str(BASE_DIR / "Output/main_model"))
RUN_DIR = BASE_OUT / RUN_STAMP
OUTPUT_DIR = RUN_DIR  # alias; load_grouped_folds() writes split_pairs here
BASE_CKPT: Path | None = None  # set in main() via config.resolve_checkpoint
config: dict | None = None  # current loss config; set in main()'s loop, read by train_fold()
loss_root: Path | None = None  # current config's output dir; set in main()'s loop
IMAGE_DIR = ROOT / "image"
SMEAR_DIR = ROOT / "smear_shadow"
MASK_DIR = ROOT / "annotated_shadow"
PAIR_FILE = ROOT / "pairs.txt"
SINGLE_FILE = ROOT / "singles.txt"
NUM_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 16
BASE_SEED = 42
AUGMENT_MULTIPLY = 8
AUGMENT_EXPANSION_ENABLED = True

# The full thesis ablation: 26 loss / input configurations, each run over 5 folds.
# `--configs` picks a subset by exp_id (comma-separated); `--configs all` runs the
# lot; with neither, the six hybrid configs below (DEFAULT_CONFIGS) run - the ones
# the thesis carried forward to the final comparison. Every `name` resolves in the
# criterion switch in train_fold().
LOSS_CONFIGS: list[dict] = [
    # 1. Baseline - no prior channel, no physics term
    {"exp_id": "BASE_BCE", "name": "BCEWithLogitsLoss", "params": {}, "use_prior_channel": False},
    {
        "exp_id": "BASE_DICE",
        "name": "BCEDiceLoss",
        "params": {"weight_bce": 0, "weight_dice": 1},
        "use_prior_channel": False,
    },
    # 2. Supervised only - BCE/Dice mix, with (RGBS) and without (RGB) the prior channel
    {
        "exp_id": "RGB_BCE30_DICE70",
        "name": "BCEDiceLoss",
        "params": {"weight_bce": 0.3, "weight_dice": 0.7},
        "use_prior_channel": False,
    },
    {
        "exp_id": "RGB_BCE50_DICE50",
        "name": "BCEDiceLoss",
        "params": {"weight_bce": 0.5, "weight_dice": 0.5},
        "use_prior_channel": False,
    },
    {
        "exp_id": "RGB_BCE70_DICE30",
        "name": "BCEDiceLoss",
        "params": {"weight_bce": 0.7, "weight_dice": 0.3},
        "use_prior_channel": False,
    },
    {
        "exp_id": "RGBS_BCE30_DICE70",
        "name": "BCEDiceLoss",
        "params": {"weight_bce": 0.3, "weight_dice": 0.7},
        "use_prior_channel": True,
    },
    {
        "exp_id": "RGBS_BCE50_DICE50",
        "name": "BCEDiceLoss",
        "params": {"weight_bce": 0.5, "weight_dice": 0.5},
        "use_prior_channel": True,
    },
    {
        "exp_id": "RGBS_BCE70_DICE30",
        "name": "BCEDiceLoss",
        "params": {"weight_bce": 0.7, "weight_dice": 0.3},
        "use_prior_channel": True,
    },
    # 3. Physics-guided loss only (no prior channel)
    {
        "exp_id": "PHYS_ATT_0.1",
        "name": "AttentiveBCEDiceLoss",
        "params": {"attention_weight": 0.1, "weight_bce": 0.5, "weight_dice": 0.5},
        "use_prior_channel": False,
    },
    {
        "exp_id": "PHYS_ATT_0.5",
        "name": "AttentiveBCEDiceLoss",
        "params": {"attention_weight": 0.5, "weight_bce": 0.5, "weight_dice": 0.5},
        "use_prior_channel": False,
    },
    {
        "exp_id": "PHYS_ATT_1.0",
        "name": "AttentiveBCEDiceLoss",
        "params": {"attention_weight": 1, "weight_bce": 0.5, "weight_dice": 0.5},
        "use_prior_channel": False,
    },
    {
        "exp_id": "PHYS_BCE_10",
        "name": "BCEDicePhysicsBCE",
        "params": {"weight_phys": 0.1, "weight_bce": 0.45, "weight_dice": 0.45},
        "use_prior_channel": False,
    },
    {
        "exp_id": "PHYS_BCE_33",
        "name": "BCEDicePhysicsBCE",
        "params": {"weight_phys": 0.3, "weight_bce": 0.3, "weight_dice": 0.3},
        "use_prior_channel": False,
    },
    {
        "exp_id": "PHYS_BCE_50",
        "name": "BCEDicePhysicsBCE",
        "params": {"weight_phys": 0.5, "weight_bce": 0.25, "weight_dice": 0.25},
        "use_prior_channel": False,
    },
    {
        "exp_id": "PHYS_DICE_10",
        "name": "BCEDicePhysicsDice",
        "params": {"weight_phys": 0.1, "weight_bce": 0.45, "weight_dice": 0.45},
        "use_prior_channel": False,
    },
    {
        "exp_id": "PHYS_DICE_33",
        "name": "BCEDicePhysicsDice",
        "params": {"weight_phys": 0.3, "weight_bce": 0.3, "weight_dice": 0.3},
        "use_prior_channel": False,
    },
    {
        "exp_id": "PHYS_DICE_50",
        "name": "BCEDicePhysicsDice",
        "params": {"weight_phys": 0.5, "weight_bce": 0.25, "weight_dice": 0.25},
        "use_prior_channel": False,
    },
    # 4. Hybrid - prior channel + physics-guided loss
    {
        "exp_id": "HYB_BCE_PHYS10",
        "name": "HybridBCE",
        "params": {"weight_phys": 0.1, "weight_bce": 0.45, "weight_dice": 0.45},
        "use_prior_channel": True,
    },
    {
        "exp_id": "HYB_BCE_PHYS30",
        "name": "HybridBCE",
        "params": {"weight_phys": 0.3, "weight_bce": 0.3, "weight_dice": 0.3},
        "use_prior_channel": True,
    },
    {
        "exp_id": "HYB_BCE_PHYS50",
        "name": "HybridBCE",
        "params": {"weight_phys": 0.5, "weight_bce": 0.25, "weight_dice": 0.25},
        "use_prior_channel": True,
    },
    {
        "exp_id": "HYB_DICE_PHYS10",
        "name": "HybridDice",
        "params": {"weight_phys": 0.1, "weight_bce": 0.45, "weight_dice": 0.45},
        "use_prior_channel": True,
    },
    {
        "exp_id": "HYB_DICE_PHYS30",
        "name": "HybridDice",
        "params": {"weight_phys": 0.3, "weight_bce": 0.3, "weight_dice": 0.3},
        "use_prior_channel": True,
    },
    {
        "exp_id": "HYB_DICE_PHYS50",
        "name": "HybridDice",
        "params": {"weight_phys": 0.5, "weight_bce": 0.25, "weight_dice": 0.25},
        "use_prior_channel": True,
    },
    {
        "exp_id": "HYB_ATT_0.1",
        "name": "HybridAttentiveBCEDice",
        "params": {"attention_weight": 0.1, "weight_bce": 0.5, "weight_dice": 0.5},
        "use_prior_channel": True,
    },
    {
        "exp_id": "HYB_ATT_0.5",
        "name": "HybridAttentiveBCEDice",
        "params": {"attention_weight": 0.5, "weight_bce": 0.5, "weight_dice": 0.5},
        "use_prior_channel": True,
    },
    {
        "exp_id": "HYB_ATT_1.0",
        "name": "HybridAttentiveBCEDice",
        "params": {"attention_weight": 1, "weight_bce": 0.5, "weight_dice": 0.5},
        "use_prior_channel": True,
    },
]

DEFAULT_CONFIGS = (
    "HYB_DICE_PHYS10",
    "HYB_DICE_PHYS30",
    "HYB_DICE_PHYS50",
    "HYB_ATT_0.1",
    "HYB_ATT_0.5",
    "HYB_ATT_1.0",
)

loss_configs: list[dict] = []  # the selected subset; filled in main() from --configs


import hashlib
import json
import re


def is_mask_nonempty(uid, mask_dir):
    try:
        with rasterio.open(mask_dir / f"{uid}.tif") as src:
            return src.read(1).sum() > 0
    except Exception:
        print(f"[Warning] Could not read mask for {uid}")
        return False


def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def generate_clean_txts(pair_file, single_file, mask_dir, output_dir=None):
    with open(pair_file) as f:
        raw_pairs = [tuple(line.strip().split()) for line in f if line.strip()]
    with open(single_file) as f:
        raw_singles = [line.strip() for line in f if line.strip()]

    cleaned_pairs = []
    cleaned_singles = []

    for a, b in raw_pairs:
        a_valid = is_mask_nonempty(a, mask_dir)
        b_valid = is_mask_nonempty(b, mask_dir)

        if a_valid and b_valid:
            cleaned_pairs.append((a, b))
        elif a_valid:
            cleaned_singles.append(a)
            print(f"[Pair split] {a} kept as single (b was empty)")
        elif b_valid:
            cleaned_singles.append(b)
            print(f"[Pair split] {b} kept as single (a was empty)")
        else:
            print(f"[Skipping pair] {a}, {b} - both empty")

    for uid in raw_singles:
        if is_mask_nonempty(uid, mask_dir):
            cleaned_singles.append(uid)
        else:
            print(f"[Skipping single] {uid} due to empty mask")

    output_dir = output_dir or mask_dir
    cleaned_pairs_path = output_dir / "pairs_cleaned.txt"
    cleaned_singles_path = output_dir / "singles_cleaned.txt"

    with open(cleaned_pairs_path, "w") as f:
        for a, b in cleaned_pairs:
            f.write(f"{a} {b}\n")

    with open(cleaned_singles_path, "w") as f:
        for uid in cleaned_singles:
            f.write(f"{uid}\n")

    print(f"\n Cleaned pairs saved to: {cleaned_pairs_path}")
    print(f" Cleaned singles saved to: {cleaned_singles_path}")
    print(f"Total good pairs: {len(cleaned_pairs)}")
    print(f"Total good singles: {len(cleaned_singles)}")

    return cleaned_pairs_path, cleaned_singles_path


def make_slug(params: dict, max_len: int = 40) -> str:
    """
    Convert {'weight_bce': 0.3, 'weight_dice': 0.7}
    -> 'w_bce0p3_w_dice0p7'.  Guaranteed to be file-system friendly.
    If the string would be longer than `max_len`, fall back to a 6-char hash.
    """
    if not params:
        return "default"
    parts = []
    for k, v in sorted(params.items()):
        key = re.sub(r"^weight_", "w_", k)  # shorter
        key = re.sub(r"[^0-9A-Za-z_-]", "", key)  # strip odd chars
        val = str(v).replace(".", "p")  # 0.3 -> 0p3
        parts.append(f"{key}{val}")
    slug = "_".join(parts)
    if len(slug) > max_len:
        slug = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return slug


def filter_nonempty_ids(ids, mask_dir):
    good_ids = []
    for uid in ids:
        with rasterio.open(mask_dir / f"{uid}.tif") as src:
            if src.read(1).sum() > 0:
                good_ids.append(uid)
    return good_ids


def set_seed(seed=15):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def set_seed_all(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # turn off autotune
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(
        True
    )  # error-out if a nondet op is hit :contentReference[oaicite:0]{index=0}
    # needed for deterministic GEMM kernels on CUDA >=11.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def prepare_pretraining(
    checkpoint_path: str, channels: int, encoder_freezing: bool = False, use_attention: bool = False
):
    unet = UNet(in_channels=3, out_channels=1)
    ckpt = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    unet.load_state_dict(ckpt["model_state_dict"], strict=True)

    if use_attention:
        retrained_model = AttentivePHYSHADENet(in_channels=channels, out_channels=1)
    else:
        retrained_model = PHYSHADENet(in_channels=channels, out_channels=1)

    new_state = OrderedDict()
    for k, v in unet.state_dict().items():
        if k == "down1.conv.0.weight":  # first conv: [64, in_channels, 3, 3]
            if channels == 3:
                new_state[k] = v.clone()
            elif channels == 4:
                # seed the 4th (prior) channel with the mean of the RGB weights
                extra = v.mean(1, keepdim=True)
                v_new = torch.cat([v, extra], dim=1)
                new_state[k] = v_new
            else:
                raise ValueError(f"Unsupported channel expansion: trying to load 3->{channels}")
        else:
            new_state[k] = v.clone()

    retrained_model.load_state_dict(new_state, strict=False)

    if encoder_freezing:
        for name, param in retrained_model.named_parameters():
            if name.split(".")[0].startswith("down"):
                param.requires_grad_(False)

    return retrained_model


class AttentiveBCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, attention_weight=1.0):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.attention_weight = attention_weight
        self.eps = 1e-6

    def forward(self, logits, targets, smear):
        targets = targets.float()
        smear = smear.float()

        # --- BCE with attention ---
        bce_raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)  # still used for dice        # Emphasize regions where smear > 0
        attn_bce = bce_raw * (1.0 + self.attention_weight * smear)
        bce_loss = attn_bce.mean()

        # --- Dice with attention ---
        intersection = ((probs * targets) * (1.0 + self.attention_weight * smear)).sum(dim=(2, 3))
        union = ((probs + targets) * (1.0 + self.attention_weight * smear)).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean()

        # --- Combined loss ---
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
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


class BCE_Dice_Physics_MSE(nn.Module):
    def __init__(self, weight_phys=0.3, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.supLoss = BCEDiceLoss(weight_bce, weight_dice)
        self.weight_phys = weight_phys
        self.mse = nn.MSELoss()

    def forward(self, logits, targets, smear):
        sup = self.supLoss(logits, targets)  # supervised term
        smear = smear.to(logits.device)
        phys = self.mse(torch.sigmoid(logits), smear)  # physics term
        return sup + self.weight_phys * phys


class BCE_Dice_Physics_BCE(nn.Module):
    def __init__(self, weight_phys=0.3, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.supLoss = BCEDiceLoss(weight_bce, weight_dice)
        self.weight_phys = weight_phys
        self.bce_phys = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, smear):
        sup = self.supLoss(logits, targets)  # supervised term
        smear = smear.to(logits.device)
        phys = self.bce_phys(logits, smear)  # physics term
        return sup + self.weight_phys * phys


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
            x_t, y_t = self._augment(x_t, y_t, rng, aug_idx)

        return x_t, y_t, aux

    def _augment(self, x, y, rng, aug_idx):
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
        op = geom_ops[aug_idx % len(geom_ops)]
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


def dump_fold_images(folds, fold_idx, loss_root, use_prior_channel):
    print(f"Dumping images for fold {fold_idx + 1}")
    train_ids = [uid for j, f in enumerate(folds) if j != fold_idx for uid in f]
    val_ids = folds[fold_idx]

    train_ids = filter_nonempty_ids(train_ids, MASK_DIR)
    val_ids = filter_nonempty_ids(val_ids, MASK_DIR)

    multiply = AUGMENT_MULTIPLY if AUGMENT_EXPANSION_ENABLED else 1

    train_ds = ShadowTiles(
        train_ids,
        IMAGE_DIR,
        SMEAR_DIR,
        MASK_DIR,
        augment=True,
        return_smear_in_x=use_prior_channel,
        seed=BASE_SEED + fold_idx,
        multiply=multiply,
    )
    val_ds = ShadowTiles(
        val_ids,
        IMAGE_DIR,
        SMEAR_DIR,
        MASK_DIR,
        augment=False,
        return_smear_in_x=use_prior_channel,
        seed=BASE_SEED + fold_idx,
        multiply=1,
    )

    aug_train_dir = loss_root / "fold_images" / f"fold_{fold_idx + 1}" / "augmented_training"
    aug_val_dir = loss_root / "fold_images" / f"fold_{fold_idx + 1}" / "augmented_validation"
    aug_train_dir.mkdir(parents=True, exist_ok=True)
    aug_val_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(train_ds)):
        x_t, y_t, _ = train_ds[idx]
        save_image(x_t[:3], aug_train_dir / f"{idx:04d}.png")
        save_image(y_t, aug_train_dir / f"{idx:04d}_mask.png")

    for idx in range(len(val_ds)):
        x_t, y_t, _ = val_ds[idx]
        save_image(x_t[:3], aug_val_dir / f"{idx:04d}.png")
        save_image(y_t, aug_val_dir / f"{idx:04d}_mask.png")


def drop_to_divisible_units(units, num_folds, seed=0):
    rng = random.Random(seed)
    total_images = sum(len(u) for u in units)
    remainder = total_images % num_folds
    if remainder == 0:
        return units

    print(f"[Trim] Need to drop {remainder} image(s) to divide evenly across {num_folds} folds.")

    # Try dropping singles first
    singles = [u for u in units if len(u) == 1]
    if len(singles) >= remainder:
        to_drop = set(rng.sample(singles, remainder))
        kept_units = [u for u in units if u not in to_drop]
        return kept_units

    # Not enough singles; drop pairs (or a combo)
    pairs = [u for u in units if len(u) == 2]
    needed = (remainder + 1) // 2  # drop enough pairs to reach or exceed remainder
    if len(pairs) >= needed:
        to_drop = set(rng.sample(pairs, needed))
        kept_units = [u for u in units if u not in to_drop]
        # Check new remainder; if over-dropped, you could re-balance by adding singles back, but that's messy.
        return kept_units

    print(" Not enough singles or pairs to fully balance folds; proceeding with uneven folds.")
    return units


def drop_to_divisible(ids, num_folds, seed=0):
    """
    Drops the minimum number of IDs to make len(ids) divisible by num_folds.
    Drops are random but deterministic via `seed`.
    """
    remainder = len(ids) % num_folds
    if remainder == 0:
        return ids  # no drop needed

    random.seed(seed)
    print(f"[Trim] Dropping {remainder} sample(s) to evenly divide into {num_folds} folds.")

    # Pick `remainder` random items to drop
    drop_ids = set(random.sample(ids, remainder))
    kept_ids = [x for x in ids if x not in drop_ids]
    return kept_ids


def trim_pairs_and_singles(valid_pairs, valid_singles, k, seed=0):
    """
    Trims singles first, then splits pairs if needed,
    to get total number of base images divisible by k.
    """
    random.seed(seed)
    total_images = 2 * len(valid_pairs) + len(valid_singles)
    remainder = total_images % k

    if remainder == 0:
        print(f"[Trim] Total images: {total_images} - already divisible by {k}.")
        return valid_pairs, valid_singles

    print(f"[Trim] Dropping {remainder} sample(s) to evenly divide into {k} folds.")

    singles_to_drop = min(remainder, len(valid_singles))
    if singles_to_drop > 0:
        singles_drop = random.sample(valid_singles, singles_to_drop)
        valid_singles = [uid for uid in valid_singles if uid not in singles_drop]
        remainder -= singles_to_drop
        print(f"[Trim] Dropped {singles_to_drop} singles.")

    while remainder > 0 and len(valid_pairs) > 0:
        # Pick a random pair to split
        pair_idx = random.randrange(len(valid_pairs))
        pair = valid_pairs.pop(pair_idx)
        # Keep one image as a single, drop the other
        kept_image = random.choice(pair)
        valid_singles.append(kept_image)
        remainder -= 1
        print(f"[Trim] Split pair {pair} - kept {kept_image} as single.")

    total_images = 2 * len(valid_pairs) + len(valid_singles)
    assert total_images % k == 0, (
        f"Failed to trim dataset properly: total={total_images} not divisible by {k}."
    )

    print(f"[Trim] Final count: {total_images} images.")
    return valid_pairs, valid_singles


def load_grouped_folds(pair_file, single_file, k=5, seed=0):
    """
    Distribute pairs to folds without exceeding the target size,
    then fill remaining slots with singles.
    Finally, break and distribute leftover pairs to folds with space left.
    Prints helpful statistics.
    """
    with open(pair_file) as f:
        pairs = [tuple(line.strip().split()) for line in f if line.strip()]
    with open(single_file) as f:
        singles = [line.strip() for line in f if line.strip()]

    # Validate pairs
    valid_pairs = []
    for a, b in pairs:
        if is_mask_nonempty(a, MASK_DIR) and is_mask_nonempty(b, MASK_DIR):
            valid_pairs.append((a, b))

    valid_singles = [uid for uid in singles if is_mask_nonempty(uid, MASK_DIR)]

    total_images = 2 * len(valid_pairs) + len(valid_singles)
    target_fold_size = total_images // k
    print(f"[Trim] Total images after filtering: {total_images} (Target per fold: {target_fold_size})")

    random.seed(seed)
    folds = [[] for _ in range(k)]
    fold_sizes = [0] * k

    # 1. Shuffle pairs
    random.shuffle(valid_pairs)
    pairs_to_split = []

    # 2. Distribute pairs first
    for pair in valid_pairs:
        # Try to find a fold with space for both images
        assigned = False
        for idx in range(k):
            if fold_sizes[idx] + 2 <= target_fold_size:
                folds[idx].extend(pair)
                fold_sizes[idx] += 2
                assigned = True
                break
        if not assigned:
            pairs_to_split.append(pair)  # Save for later splitting

    # 3. Distribute singles next
    random.shuffle(valid_singles)
    for single in valid_singles:
        for idx in range(k):
            if fold_sizes[idx] < target_fold_size:
                folds[idx].append(single)
                fold_sizes[idx] += 1
                break

    singles_from_pairs = []
    split_pairs = []  # NEW: to store split pairs for logging

    for pair in pairs_to_split:
        split_pairs.append(pair)  # Log this pair as split
        singles_from_pairs.extend(pair)

    random.shuffle(singles_from_pairs)
    for single in singles_from_pairs:
        for idx in range(k):
            if fold_sizes[idx] < target_fold_size:
                folds[idx].append(single)
                fold_sizes[idx] += 1
                break

    # Print summary
    print("\n=== Fold Distribution Summary ===")
    print(f"Total images distributed: {sum(fold_sizes)}")
    print(f"  • Pairs kept together: {len(valid_pairs) - len(pairs_to_split)}")
    print(f"  • Singles distributed from original singles: {len(valid_singles)}")
    print(f"  • Singles distributed from split pairs: {len(singles_from_pairs)}")
    for idx, fold in enumerate(folds):
        print(f"Fold {idx + 1}: {len(fold)} images")

    # Save split pair info to a file (NEW)
    if OUTPUT_DIR:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        split_pairs_path = OUTPUT_DIR / "split_pairs_in_folds.txt"
        with open(split_pairs_path, "w") as f:
            for a, b in split_pairs:
                f.write(f"{a} {b}\n")
        print(f" Split pairs saved to: {split_pairs_path}")

    return folds


class MSEProbLoss(nn.Module):
    """
    Mean‐squared error between predicted probabilities and targets,
    i.e. MSELoss(torch.sigmoid(logits), targets).
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return self.mse(probs, targets)


def train_fold(
    fold_idx, train_ids, val_ids, writer, loss_fn_name="BCEWithLogitsLoss", loss_fn_params=None, top_k=3
):
    print(f"=== Fold {fold_idx + 1} ===")
    # Dynamic loss function
    if loss_fn_name == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "DiceLoss":
        criterion = DiceLoss(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "MSELoss":
        criterion = MSEProbLoss()
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "BCEDiceLoss":
        criterion = BCEDiceLoss(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "BCEDicePhysicsBCE":
        criterion = BCE_Dice_Physics_BCE(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "BCEDicePhysicsMSE":
        criterion = BCE_Dice_Physics_MSE(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "BCEDicePhysicsDice":
        criterion = BCE_Dice_Physics_Dice(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "AttentiveBCEDiceLoss":
        criterion = AttentiveBCEDiceLoss(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "HybridMSE":
        criterion = BCE_Dice_Physics_MSE(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "HybridBCE":
        criterion = BCE_Dice_Physics_BCE(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "HybridDice":
        criterion = BCE_Dice_Physics_Dice(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    elif loss_fn_name == "HybridAttentiveBCEDice":
        criterion = AttentiveBCEDiceLoss(**(loss_fn_params or {}))
        use_prior_channel = config.get("use_prior_channel", False)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")

    use_phys = loss_fn_name in {
        "BCEDicePhysicsMSE",
        "BCEDicePhysicsDice",
        "BCEDicePhysicsBCE",
        "HybridMSE",
        "HybridBCE",
        "HybridDice",
        "AttentiveBCEDiceLoss",
        "HybridAttentiveBCEDice",
    }
    use_attention = config.get("use_attention", False)
    if BASE_CKPT is None:
        raise SystemExit("BASE_CKPT not set - run via `python -m physhade.train.mainmodel_training`")
    model = prepare_pretraining(
        str(BASE_CKPT),
        channels=3 + use_prior_channel,
        use_attention=use_attention,
    ).cuda()
    model = model.to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="default", fullgraph=True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    train_ids = filter_nonempty_ids(train_ids, MASK_DIR)
    multiply = AUGMENT_MULTIPLY if AUGMENT_EXPANSION_ENABLED else 1
    train_ds = ShadowTiles(
        train_ids,
        IMAGE_DIR,
        SMEAR_DIR,
        MASK_DIR,
        augment=True,
        return_smear_in_x=use_prior_channel,
        seed=BASE_SEED + fold_idx,
        multiply=multiply,
    )
    x_sample, y_sample, smear_sample = train_ds[0]
    print(x_sample.shape)

    g = torch.Generator().manual_seed(BASE_SEED + fold_idx)
    val_ids = filter_nonempty_ids(val_ids, MASK_DIR)
    val_ds = ShadowTiles(
        val_ids,
        IMAGE_DIR,
        SMEAR_DIR,
        MASK_DIR,
        augment=False,
        return_smear_in_x=use_prior_channel,
        seed=BASE_SEED + fold_idx,
        multiply=1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=g,
        prefetch_factor=4,
        drop_last=True,
        multiprocessing_context="spawn",
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, generator=g, num_workers=0)

    with torch.no_grad():
        fixed_img, fixed_gt, fixed_smear = next(iter(val_loader))
        fixed_img, fixed_gt, fixed_smear = (t.cuda() for t in (fixed_img, fixed_gt, fixed_smear))

    fold_dir = loss_root / f"fold_{fold_idx + 1}"  ### NEW
    fold_dir.mkdir(exist_ok=True)
    debug_dir = fold_dir / "debug"
    ckpt_dir = fold_dir / "ckpt"
    # debug_dir = OUTPUT_DIR / "debug" / f"{loss_fn_name}_fold{fold_idx + 1}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # <- add this line
    dice_scores = []
    val_losses = []

    saved_ckpts = deque()
    scaler = amp.GradScaler("cuda")  # instead of torch.cuda.amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_dice = 0
    early_stop_counter = 0
    early_stop_patience = 25
    best_epoch = -1
    best_ckpt = None

    for epoch in range(EPOCHS):
        model.train()
        for x, y, smear in train_loader:
            x, y, smear = x.cuda(non_blocking=True), y.cuda(non_blocking=True), smear.cuda(non_blocking=True)

            with amp.autocast(
                device_type="cuda", dtype=torch.float16
            ):  # instead of torch.cuda.amp.autocast()
                logits = model(x)
                loss = criterion(logits, y, smear) if use_phys else criterion(logits, y)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # Validation
        model.eval()
        TP = FP = FN = total_loss = 0
        with torch.no_grad():
            for x, y, smear in val_loader:
                x, y, smear = x.cuda(), y.cuda(), smear.cuda()

                logits = model(x)
                pr = torch.sigmoid(logits) > 0.5

                if use_phys:
                    loss_val = criterion(logits, y, smear)
                else:
                    loss_val = criterion(logits, y)

                total_loss += loss_val.item() * x.size(0)

                TP += (pr & (y == 1)).sum().item()
                FP += (pr & (y == 0)).sum().item()
                FN += ((~pr) & (y == 1)).sum().item()

        val_loss = total_loss / len(val_ds)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
        scheduler.step(val_loss)
        dice_scores.append(dice)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:02d}  Dice: {dice:.4f}  Loss: {val_loss:.4f}")

        if writer is not None:
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("dice/val", dice, epoch)
            writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        torch.cuda.empty_cache()

        # Save checkpoint if Dice is top-k
        if len(saved_ckpts) < top_k or dice > min(d for d, _ in saved_ckpts):
            ckpt_path = ckpt_dir / f"ep{epoch:03d}_dice{dice:.4f}.pth"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            saved_ckpts.append((dice, ckpt_path))
            if len(saved_ckpts) > top_k:
                lowest = min(saved_ckpts, key=lambda x: x[0])
                saved_ckpts.remove(lowest)
                os.remove(str(lowest[1]))

            model.eval()
            with torch.no_grad(), amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = (torch.sigmoid(model(fixed_img))[0, 0] > 0.5).cpu().numpy()

            rgb = fixed_img[0, :3].permute(1, 2, 0).cpu().numpy()
            smear = fixed_smear[0, 0].cpu().numpy()  # (H,W)
            gt = fixed_gt[0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[0].axis("off")
            axes[1].imshow(smear, cmap="gray")
            axes[1].set_title("Smear")
            axes[1].axis("off")
            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title(f"Pred @ ep{epoch}")
            axes[2].axis("off")
            axes[3].imshow(gt, cmap="gray")
            axes[3].set_title("Ground-truth")
            axes[3].axis("off")
            fig.tight_layout()
            fig.savefig(debug_dir / f"{loss_fn_name}_ep{epoch:02d}_dice{dice:.3f}.png", dpi=150)
            plt.close(fig)

        if len(saved_ckpts) > top_k:
            lowest = min(saved_ckpts, key=lambda x: x[0])
            saved_ckpts.remove(lowest)
            os.remove(str(lowest[1]))
            # also remove its debug image
            for png in debug_dir.glob(f"*dice{lowest[0]:.3f}.png"):
                png.unlink()

        if saved_ckpts:
            best_path = max(saved_ckpts, key=lambda t: t[0])[1]
            best_epoch = int(best_path.stem.split("_")[0][2:])  # ep### -> ###
            best_ckpt = best_path

            tgt = fold_dir / f"best_ep{best_epoch:03d}.pth"
            shutil.copy(best_path, tgt)

        if epoch % 10 == 0:
            img = x[:1]
            pr = (torch.sigmoid(model(img))[0, 0].cpu() > 0.5).numpy()
            plt.imsave(fold_dir / f"vis_epoch{epoch}.png", pr, cmap="gray")

        if dice > best_dice:
            best_dice = dice
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} - no improvement in {early_stop_patience} epochs.")
            break

    # Save learning curve
    plt.plot(range(1, len(dice_scores) + 1), dice_scores, marker="o")
    plt.title(f"Fold {fold_idx + 1} Dice Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.grid(True)
    plt.savefig(fold_dir / "learning_curve.png")
    plt.close()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    return (
        max(dice_scores),
        min(val_losses),
        len(dice_scores),
        best_epoch,
        str(best_ckpt.relative_to(BASE_OUT)),
        precision,
        recall,
    )  # return both best dice, best (lowest) val loss


def check_leakage(folds):
    all_ids = [uid for fold in folds for uid in fold]
    unique_ids = set(all_ids)
    if len(all_ids) != len(unique_ids):
        print(" Overlap detected! Some IDs appear in multiple folds.")
    else:
        print(" No data leakage detected.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT, help="main_model/train dir")
    p.add_argument("--out-dir", type=Path, default=None, help="default: Output/main_model/<timestamp>")
    p.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="default: newest Output/base_model/*/best_model.pth",
    )
    p.add_argument("--folds", type=int, default=NUM_FOLDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--augment-multiply", type=int, default=AUGMENT_MULTIPLY)
    p.add_argument(
        "--configs",
        default="",
        help="comma-separated exp_ids, or 'all' (default: the 6 hybrid configs)",
    )
    return p


def main(argv=None):
    # these are read as module globals by load_grouped_folds() / train_fold()
    global ROOT, IMAGE_DIR, SMEAR_DIR, MASK_DIR, PAIR_FILE, SINGLE_FILE, OUTPUT_DIR
    global RUN_DIR, NUM_FOLDS, EPOCHS, BATCH_SIZE, BASE_SEED, AUGMENT_MULTIPLY, BASE_CKPT
    global loss_configs, config, loss_root
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    ROOT = args.data_dir
    IMAGE_DIR, SMEAR_DIR, MASK_DIR = ROOT / "image", ROOT / "smear_shadow", ROOT / "annotated_shadow"
    PAIR_FILE, SINGLE_FILE = ROOT / "pairs.txt", ROOT / "singles.txt"
    RUN_DIR = args.out_dir or (BASE_OUT / RUN_STAMP)
    NUM_FOLDS, EPOCHS, BATCH_SIZE, BASE_SEED = args.folds, args.epochs, args.batch_size, args.seed
    AUGMENT_MULTIPLY = args.augment_multiply
    BASE_CKPT = resolve_checkpoint(args.base_checkpoint, subdir="base_model")
    if args.configs.strip().lower() == "all":
        selected = [c["exp_id"] for c in LOSS_CONFIGS]
    elif args.configs.strip():
        selected = [c.strip() for c in args.configs.split(",")]
    else:
        selected = list(DEFAULT_CONFIGS)
    loss_configs = [c for c in LOSS_CONFIGS if c["exp_id"] in set(selected)]
    if not loss_configs:
        raise SystemExit(
            f"no loss configs match {sorted(selected)}; available: {[c['exp_id'] for c in LOSS_CONFIGS]}"
        )

    OUTPUT_DIR = RUN_DIR  # noqa: F841 (module global, see top)
    timestamp = RUN_STAMP
    runs_csv_path = BASE_OUT / "runs.csv"
    dumped_folds = set()
    folds = load_grouped_folds(PAIR_FILE, SINGLE_FILE, k=NUM_FOLDS, seed=BASE_SEED)
    check_leakage(folds)
    if loss_configs:
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not runs_csv_path.exists():
        df = pd.DataFrame(
            columns=[
                "exp_id",
                "timestamp",
                "fold",
                "epoch",
                "ran_epoch",
                "dice",
                "precision",
                "recall",
                "loss",
                "loss_fn",
                "loss_hyperparams",
                "time",
                "aug_mul",
            ]
        )
        df.to_csv(runs_csv_path, index=False)
    else:
        df = pd.read_csv(runs_csv_path)

    for config in loss_configs:
        set_seed_all(BASE_SEED)

        start_time_config = time.time()
        loss_fn_name = config["name"]
        loss_fn_params = config["params"]
        exp_id = config.get("exp_id", f"{loss_fn_name}_{make_slug(loss_fn_params)}")

        param_slug = make_slug(loss_fn_params)
        loss_root = RUN_DIR / exp_id
        loss_root.mkdir(parents=True, exist_ok=True)
        print(
            f"=== Training with loss function {loss_fn_name}, using hyperparameters {loss_fn_params}, using prior channel = {str(config.get('use_prior_channel', False))} ==="
        )
        folds = load_grouped_folds(PAIR_FILE, SINGLE_FILE, k=NUM_FOLDS, seed=BASE_SEED)
        scores = []
        losses = []
        epochs = []
        epochs_ran = []
        best_epochs = []
        best_ckpts = []
        fold_timings = []
        precisions = []
        recalls = []

        loss_root = RUN_DIR / config.get("exp_id", f"{loss_fn_name}_{make_slug(loss_fn_params)}")
        loss_root.mkdir(parents=True, exist_ok=True)

        for i in range(NUM_FOLDS):
            if i not in dumped_folds:
                dump_fold_images(folds, i, RUN_DIR, use_prior_channel=config.get("use_prior_channel", False))
                dumped_folds.add(i)

        for i in range(NUM_FOLDS):
            start_time_fold = time.time()
            val_ids = folds[i]
            train_ids = [uid for j, f in enumerate(folds) if j != i for uid in f]

            _ = make_slug(loss_fn_params)
            tb_root = RUN_DIR / "tb" / exp_id  # <- clean path
            tb_fold = SummaryWriter(log_dir=str(tb_root / f"fold_{i + 1}"))

            set_seed_all(BASE_SEED + i)
            dice, loss, ran_epochs, b_ep, b_ckpt, precision, recall = train_fold(
                i, train_ids, val_ids, tb_fold, loss_fn_name=loss_fn_name, loss_fn_params=loss_fn_params
            )
            best_epochs.append(b_ep)
            best_ckpts.append(b_ckpt)
            scores.append(dice)
            losses.append(loss)
            epochs.append(EPOCHS)
            epochs_ran.append(ran_epochs)
            precisions.append(precision)
            recalls.append(recall)

            end_time_fold = time.time()
            fold_timings.append(end_time_fold - start_time_fold)
            tb_fold.close()  # noqa: F821 (set as global in the training loop)  # flush and close

        config_end_time = time.time()
        config_time = config_end_time - start_time_config
        # Create a batch DataFrame for this loss function
        new_data = pd.DataFrame(
            {
                "exp_id": [exp_id] * NUM_FOLDS,
                "timestamp": [timestamp] * NUM_FOLDS,
                "loss_fn": [loss_fn_name] * NUM_FOLDS,
                "fold": [i + 1 for i in range(NUM_FOLDS)],
                "dice": scores,
                "precision": precisions,
                "recall": recalls,
                "loss": losses,
                "epoch": epochs,
                "ran_epoch": epochs_ran,
                "best_epoch": best_epochs,
                "best_ckpt": best_ckpts,
                "loss_hyperparams": [json.dumps(loss_fn_params)] * NUM_FOLDS,
                "param_slug": [param_slug] * NUM_FOLDS,
                "loss_dir": [
                    str((loss_root / f"fold_{i + 1}").relative_to(BASE_OUT)) for i in range(NUM_FOLDS)
                ],
                "use_prior_channel": str(config.get("use_prior_channel", False)),
                "batch_size": BATCH_SIZE,
                "base_seed": BASE_SEED,
                "time": [fold_timings] * NUM_FOLDS,
                "aug_mul": 1 if not AUGMENT_EXPANSION_ENABLED else AUGMENT_MULTIPLY,
            }
        )

        summary_row = pd.DataFrame(
            {
                "exp_id": exp_id,
                "timestamp": [timestamp],
                "loss_fn": [loss_fn_name],
                "fold": ["avg"],  # or 0 or -1
                "dice": [np.mean(scores)],
                "precision": [np.mean(precisions)],
                "recall": [np.mean(recalls)],
                "loss": [np.mean(losses)],
                "epoch": [EPOCHS],
                "ran_epoch": [np.mean(epochs_ran)],
                "best_epoch": [np.mean(best_epochs)],
                "best_ckpt": [None],  # No single ckpt
                "loss_hyperparams": [json.dumps(loss_fn_params)],
                "param_slug": [param_slug],
                "loss_dir": [str((loss_root).relative_to(BASE_OUT))],
                "use_prior_channel": str(config.get("use_prior_channel", False)),
                "batch_size": BATCH_SIZE,
                "base_seed": BASE_SEED,
                "time": [config_time],
                "aug_mul": 1 if not AUGMENT_EXPANSION_ENABLED else AUGMENT_MULTIPLY,
            }
        )

        df = pd.concat([df, new_data, summary_row], ignore_index=True)
        df.to_csv(runs_csv_path, index=False)

        print(f"\n=== Results for loss function {loss_fn_name} ===")
        for i, (dice_score, loss_score) in enumerate(zip(scores, losses)):
            print(f"Fold {i + 1}: Dice {dice_score:.4f}, Loss {loss_score:.4f}")
        print(f"Mean Dice: {np.mean(scores):.4f}, Mean Loss: {np.mean(losses):.4f}")
        print(f"Time taken to train config: {config_end_time - start_time_config:.2f} seconds")

    if "tb_fold" in globals():
        tb_fold.close()  # noqa: F821 (set as global in the training loop)
    print(f"done - results in {RUN_DIR}")


if __name__ == "__main__":
    main()
