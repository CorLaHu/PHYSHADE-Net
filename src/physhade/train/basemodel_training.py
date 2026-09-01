"""Stage B1: pretrain the RGB U-Net baseline on the AISD shadow set.

    python -m physhade.train.basemodel_training --data-dir Dataset/base_model

Writes ``Output/base_model/<timestamp>/best_model.pth`` (auto-discovered by the
downstream stages).
"""

import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: F401  (import before torchvision - Windows GDAL/torch DLL clash)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tqdm import trange  # noqa: E402

from physhade.config import DATA_DIR, OUTPUT_DIR  # noqa: E402
from physhade.models import basemodel  # noqa: E402

matplotlib.use("Agg")

BATCH_SIZE = 8
STRIDE = 128
PATCH_SIZE = 512
BASE_SEED = 15
EPOCHS = 150


def dice_score(preds, targets, eps=1e-6):
    preds = torch.sigmoid(preds) > 0.5
    targets = targets > 0.5

    TP = (preds & targets).sum(dim=(1, 2, 3)).float()
    FP = (preds & ~targets).sum(dim=(1, 2, 3)).float()
    FN = (~preds & targets).sum(dim=(1, 2, 3)).float()

    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    return dice.mean().item()


def set_seed_all(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class BaseModelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, stride=STRIDE, patch_size=PATCH_SIZE):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imagelist = os.listdir(image_dir)
        self.stride = stride
        self.patch_size = patch_size
        self.patches = []

        for img_name in self.imagelist:
            image_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name)

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            for i in range(0, image.height - self.patch_size + 1, self.stride):
                for j in range(0, image.width - self.patch_size + 1, self.stride):
                    img_patch = image.crop((j, i, j + self.patch_size, i + self.patch_size))
                    mask_patch = mask.crop((j, i, j + self.patch_size, i + self.patch_size))
                    self.patches.append((img_patch, mask_patch))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        img_patch, mask_patch = self.patches[index]

        if self.transform:
            img_patch = self.transform(img_patch)
            mask_patch = self.transform(mask_patch)

        # mask_tensor = transforms.ToTensor()(mask_patch)

        return img_patch, mask_patch


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR / "base_model")
    p.add_argument("--out-dir", type=Path, default=None, help="default: Output/base_model/<timestamp>")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    p.add_argument("--stride", type=int, default=STRIDE)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    EPOCHS, BATCH_SIZE, BASE_SEED = args.epochs, args.batch_size, args.seed
    STRIDE, PATCH_SIZE = args.stride, args.patch_size

    set_seed_all(BASE_SEED)

    transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

    now = datetime.now()
    timestamp = f"{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"
    output_dir = str(args.out_dir or (OUTPUT_DIR / "base_model" / timestamp))
    os.makedirs(output_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
    csv_path = os.path.join(output_dir, "training_log.csv")

    # Set up dataset and dataloaders
    g = torch.Generator().manual_seed(BASE_SEED)

    train_dataset = BaseModelDataset(
        image_dir=str(args.data_dir / "train/shadow"),
        mask_dir=str(args.data_dir / "train/mask"),
        transform=transform,
        stride=STRIDE,
        patch_size=PATCH_SIZE,
    )
    val_dataset = BaseModelDataset(
        image_dir=str(args.data_dir / "val/shadow"),
        mask_dir=str(args.data_dir / "val/mask"),
        transform=transform,
        stride=STRIDE,
        patch_size=PATCH_SIZE,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=g)

    # Model, optimizer, criterion, scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = basemodel.UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(base_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Training loop
    train_losses = []
    val_losses = []
    val_dices = []
    best_val_loss = float("inf")
    checkpoint_dict = {}
    log_rows = []

    early_stop_patience = 25
    early_stop_counter = 0
    best_val_dice = 0
    top_k = 3
    saved_ckpts = []

    for epoch in trange(EPOCHS, desc="Training Epochs"):
        base_model.train()
        running_loss = 0.0
        t1 = time.time()

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = base_model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # === Validation ===
        base_model.eval()
        val_loss_total = 0.0
        dice_total = 0.0
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                val_outputs = base_model(val_images)
                val_loss = criterion(val_outputs, val_masks)
                val_loss_total += val_loss.item()

                dice_total += dice_score(val_outputs, val_masks) * val_images.size(0)

        val_loss_avg = val_loss_total / len(val_loader)
        val_dice_avg = dice_total / len(val_dataset)
        val_losses.append(val_loss_avg)
        val_dices.append(val_dice_avg)

        scheduler.step(val_loss_avg)

        # Logging
        tb_writer.add_scalar("Loss/Train", epoch_loss, epoch)
        tb_writer.add_scalar("Loss/Val", val_loss_avg, epoch)
        tb_writer.add_scalar("Dice/Val", val_dice_avg, epoch)
        tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch [{epoch}/{EPOCHS}] - Time: {time.time() - t1:.2f}s")
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss_avg:.4f} | Val Dice: {val_dice_avg:.4f}")

        # Save best checkpoint based on val loss
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss_avg,
                "val_dice": val_dice_avg,
            }
            ckpt_path = os.path.join(output_dir, f"Epoch_{epoch}_val{val_loss_avg:.4f}.pth")
            torch.save(checkpoint, ckpt_path)
            print(f"Checkpoint saved at epoch {epoch}: {ckpt_path}")
            checkpoint_dict[f"Epoch_{epoch}"] = checkpoint

        # Save checkpoint if it's one of the top-K by Dice
        if len(saved_ckpts) < top_k or val_dice_avg > min([d for d, _ in saved_ckpts]):
            ckpt_path = os.path.join(output_dir, f"Epoch_{epoch}_valDice{val_dice_avg:.4f}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss_avg,
                    "val_dice": val_dice_avg,
                },
                ckpt_path,
            )
            print(f" Saved top model at epoch {epoch}: {ckpt_path}")
            saved_ckpts.append((val_dice_avg, ckpt_path))
            saved_ckpts.sort(reverse=True)
            if len(saved_ckpts) > top_k:
                removed = saved_ckpts.pop(-1)
                os.remove(removed[1])
                print(f"  Removed worst checkpoint: {removed[1]}")

        # Early stopping
        if val_dice_avg > best_val_dice:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss_avg,
                    "val_dice": val_dice_avg,
                },
                os.path.join(output_dir, "best_model.pth"),
            )
            best_val_dice = val_dice_avg
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f" Early stopping at epoch {epoch} - no Dice improvement for {early_stop_patience} epochs.")
            break

        # Plot loss curves
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, marker="o", label="Train Loss")
        plt.plot(val_losses, marker="s", label="Val Loss")
        plt.plot(val_dices, marker="^", label="Val Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Training Metrics Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "training_metrics.png"))
        plt.close()

        # Append log for CSV
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_loss": val_loss_avg,
                "val_dice": val_dice_avg,
                "lr": optimizer.param_groups[0]["lr"],
                "batch_size": BATCH_SIZE,
                "stride": STRIDE,
                "patch_size": PATCH_SIZE,
                "seed": BASE_SEED,
            }
        )

    # Final CSV save
    df = pd.DataFrame(log_rows)
    df.to_csv(csv_path, index=False)
    print(f"Training log saved to {csv_path}")

    print("\nTraining complete.")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Best checkpoint saved at: {os.path.join(output_dir, 'best_model.pth')}")
    print(f"Top {top_k} checkpoints also saved based on Dice.")

    # Summary block (indented into __main__ so the module stays importable)
    summary_path = os.path.join(output_dir, "training_summary.txt")
    best_epoch = df.loc[df["val_dice"].idxmax(), "epoch"]
    best_row = df[df["epoch"] == best_epoch].iloc[0]

    with open(summary_path, "w") as f:
        f.write("=== BaseModel Training Summary ===\n")
        f.write(f"Run Directory: {output_dir}\n")
        f.write(f"Best Epoch: {int(best_epoch)}\n")
        f.write(f"Best Val Dice: {best_row['val_dice']:.4f}\n")
        f.write(f"Best Val Loss: {best_row['val_loss']:.4f}\n")
        f.write(f"Best Model Path: {os.path.join(output_dir, 'best_model.pth')}\n")
        f.write(f"Epochs Run: {len(df)} / {EPOCHS}\n")
        f.write("\n--- Config ---\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Patch Size: {PATCH_SIZE}\n")
        f.write(f"Stride: {STRIDE}\n")
        f.write(f"Seed: {BASE_SEED}\n")

    print(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
