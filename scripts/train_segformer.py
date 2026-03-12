#!/usr/bin/env python3
"""
間取り図画像から「壁」を抽出するセマンティックセグメンテーション学習スクリプト。

PyTorch + Hugging Face Transformers (SegFormer) + Albumentations + segmentation_models_pytorch
Kaggle 等での実行を想定。パスは下記 CONFIG で一括変更可能。

依存: torch, transformers, albumentations, opencv-python-headless, segmentation-models-pytorch, tqdm
"""

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, get_linear_schedule_with_warmup
import segmentation_models_pytorch as smp


# =============================================================================
# パス・ハイパーパラメータ設定（Kaggle 等ではここだけ書き換え）
# =============================================================================
CONFIG = {
    "data_root": Path("/kaggle/input/your-dataset/cubicasa5k_processed"),
    "output_dir": Path("/kaggle/working/segformer_wall"),
    "train_split": "train",
    "val_split": "val",
    "checkpoint": "nvidia/mit-b0",
    "num_labels": 1,
    "epochs": 30,
    "batch_size": 8,
    "lr": 1e-4,
    "warmup_ratio": 0.1,
    "num_workers": 4,
    "soft_threshold": 200,
    "image_size": (512, 512),
}

# ローカル実行用の例（必要に応じて上書き）
if "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
    CONFIG["data_root"] = Path(__file__).resolve().parents[1] / "cubicasa5k_processed"
    CONFIG["output_dir"] = Path(__file__).resolve().parents[1] / "output" / "segformer_wall"


# -----------------------------------------------------------------------------
# 1. 薄い線を消す前処理（Dataset 内で画像読み込み時に適用）
# -----------------------------------------------------------------------------
def soft_threshold_preprocess(image_path, threshold=200):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    soft_img = blurred.copy()
    soft_img[blurred > threshold] = 255
    soft_img_rgb = cv2.cvtColor(soft_img, cv2.COLOR_GRAY2BGR)
    return soft_img_rgb


# -----------------------------------------------------------------------------
# 2. Albumentations Data Augmentation（正解マスクに CoarseDropout は適用しない）
# -----------------------------------------------------------------------------
def get_train_transforms(image_size):
    h, w = image_size[0], image_size[1]
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(
            num_holes_range=(5, 15),
            hole_height_range=(15, 60),
            hole_width_range=(15, 60),
            fill_value=0,
            mask_fill_value=None,
            p=0.5,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size):
    h, w = image_size[0], image_size[1]
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class WallSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform, soft_threshold=200, image_size=(512, 512)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.soft_threshold = soft_threshold
        self.image_size = image_size

        self.image_paths = sorted(self.images_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"画像がありません: {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _image_path_to_mask_path(self, image_path):
        name = image_path.name
        if name.endswith("_image.png"):
            mask_name = name.replace("_image.png", "_mask.png")
        else:
            mask_name = name.replace(".png", "_mask.png")
        return self.masks_dir / mask_name

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self._image_path_to_mask_path(image_path)

        # 前処理: 薄い線を消してから Albumentations に渡す
        image = soft_threshold_preprocess(str(image_path), threshold=self.soft_threshold)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"マスクを読み込めません: {mask_path}")
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = mask.unsqueeze(0)
        return image, mask


# -----------------------------------------------------------------------------
# 3. 損失関数（BCE + Dice 1:1）
# -----------------------------------------------------------------------------
class WallSegmentationLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, logits, targets):
        if logits.shape[2:] != targets.shape[2:]:
            logits = torch.nn.functional.interpolate(
                logits, size=targets.shape[2:], mode="bilinear", align_corners=False
            )
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# -----------------------------------------------------------------------------
# 4. モデル（SegFormer, nvidia/mit-b0, num_labels=1）
# -----------------------------------------------------------------------------
def build_model(checkpoint, num_labels=1):
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=num_labels,
    )
    return model


# -----------------------------------------------------------------------------
# 学習ループ
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        logits = outputs.logits

        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Val", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(pixel_values=images)
        logits = outputs.logits
        loss = criterion(logits, masks)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def main():
    cfg = CONFIG
    cfg["data_root"] = Path(cfg["data_root"])
    cfg["output_dir"] = Path(cfg["output_dir"])
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["image_size"]

    train_images = cfg["data_root"] / cfg["train_split"] / "images"
    train_masks = cfg["data_root"] / cfg["train_split"] / "masks"
    val_images = cfg["data_root"] / cfg["val_split"] / "images"
    val_masks = cfg["data_root"] / cfg["val_split"] / "masks"

    train_ds = WallSegmentationDataset(
        train_images,
        train_masks,
        transform=get_train_transforms(image_size),
        soft_threshold=cfg["soft_threshold"],
        image_size=image_size,
    )
    val_ds = WallSegmentationDataset(
        val_images,
        val_masks,
        transform=get_val_transforms(image_size),
        soft_threshold=cfg["soft_threshold"],
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = build_model(cfg["checkpoint"], num_labels=cfg["num_labels"])
    model = model.to(device)

    criterion = WallSegmentationLoss(bce_weight=0.5, dice_weight=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss = validate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        tqdm.write(
            f"Epoch {epoch}/{cfg['epochs']}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_path = cfg["output_dir"] / "best_model"
            out_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_path)
            torch.save(optimizer.state_dict(), cfg["output_dir"] / "best_model" / "optimizer.pt")

    final_path = cfg["output_dir"] / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    print(f"学習完了。ベストモデル: {cfg['output_dir'] / 'best_model'}")

    # 各エポックの train_loss / val_loss を図示
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = cfg["output_dir"] / "loss_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Loss 曲線を保存: {plot_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SegFormer 壁セグメンテーション学習")
    p.add_argument("--data-root", type=Path, default=None, help="データルート (cubicasa5k_processed の親)")
    p.add_argument("--output-dir", type=Path, default=None, help="出力ディレクトリ")
    p.add_argument("--epochs", type=int, default=None, help="エポック数")
    p.add_argument("--batch-size", type=int, default=None, help="バッチサイズ")
    p.add_argument("--lr", type=float, default=None, help="学習率")
    args = p.parse_args()
    if args.data_root is not None:
        CONFIG["data_root"] = Path(args.data_root)
    if args.output_dir is not None:
        CONFIG["output_dir"] = Path(args.output_dir)
    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs
    if args.batch_size is not None:
        CONFIG["batch_size"] = args.batch_size
    if args.lr is not None:
        CONFIG["lr"] = args.lr
    main()
