"""
SegFormer fine-tuning for binary segmentation (background / room interior)
Dataset: cubicasa_segment/{images,masks}/{train,val}/

Diff from train02.py
--------------------
* Preprocessing: RandomResizedCrop / Resize  →  Patch-based crop (512×512)
  - Train: PadIfNeeded (zero-pad to ≥512) + RandomCrop(512, 512)
  - Val:   PadIfNeeded (zero-pad to ≥512) + CenterCrop(512, 512)
  - Image.MAX_IMAGE_PIXELS = None to prevent DecompressionBomb crash on huge floor plans.
  - All other code (loss, model, optimizer, scheduler, metrics) is unchanged from train02.py.
"""

from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # suppress DecompressionBomb warning for large floor plans

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Kaggle 環境判別 ────────────────────────────────────────────────────────────
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "nvidia/segformer-b3-finetuned-ade-512-512"

if IS_KAGGLE:
    DATA_ROOT  = Path("/kaggle/input/cubicasa_segment")
    OUTPUT_DIR = Path("/kaggle/working/exp03")
else:
    DATA_ROOT  = Path(os.environ.get("KAGGLE_DATA_ROOT")) / "cubicasa_segment"
    OUTPUT_DIR = Path("/content/drive/MyDrive/exp03")

BEST_MODEL = OUTPUT_DIR / "best_model.pth"
LOSS_PNG   = OUTPUT_DIR / "loss.png"

IMG_SIZE      = 512
BATCH_SIZE    = 4
NUM_EPOCHS    = 50
LR            = 6e-5
WEIGHT_DECAY  = 1e-4
EARLY_STOP    = 5          # patience
NUM_WORKERS   = 4
NUM_LABELS    = 2          # 0=background, 1=room

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Loss hyperparameters ───────────────────────────────────────────────────────
WEIGHT_FOCAL  = 0.5    # contribution of Focal Loss  (WEIGHT_FOCAL + WEIGHT_DICE should sum to 1)
WEIGHT_DICE   = 0.5    # contribution of Dice Loss

FOCAL_GAMMA   = 2.0    # focusing parameter  (higher → harder examples get more weight)
FOCAL_ALPHA   = 0.25   # balance factor for the positive (room) class
DICE_SMOOTH   = 1.0    # smoothing constant to avoid division by zero


# ── Loss definitions ──────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Binary Focal Loss via BCEWithLogitsLoss.

    Args:
        gamma: focusing parameter γ ≥ 0.  γ=0 reduces to standard BCE.
        alpha: scalar weight for the positive class (room).
    """
    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, H, W) — raw score for the positive (room) class.
            targets: (B, H, W) — float binary mask (0 or 1).
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)                                  # probability of the correct class
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation.

    Args:
        smooth: additive smoothing constant.
    """
    def __init__(self, smooth: float = DICE_SMOOTH):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, H, W) — raw score for the positive (room) class.
            targets: (B, H, W) — float binary mask (0 or 1).
        """
        probs = torch.sigmoid(logits)                          # → [0, 1]
        intersection = (probs * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice_coeff


class HybridLoss(nn.Module):
    """Weighted combination of Focal Loss and Dice Loss.

    Args:
        weight_focal: weight applied to Focal Loss term.
        weight_dice:  weight applied to Dice Loss term.
        gamma, alpha: Focal Loss hyperparameters.
        smooth:       Dice Loss smoothing constant.
    """
    def __init__(
        self,
        weight_focal: float = WEIGHT_FOCAL,
        weight_dice:  float = WEIGHT_DICE,
        gamma:        float = FOCAL_GAMMA,
        alpha:        float = FOCAL_ALPHA,
        smooth:       float = DICE_SMOOTH,
    ):
        super().__init__()
        self.weight_focal = weight_focal
        self.weight_dice  = weight_dice
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice  = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 2, H, W) — full-resolution logits (upsampled before calling).
            targets: (B, H, W)    — int64 binary mask (0=background, 1=room).
        """
        room_logit = logits[:, 1, :, :]      # positive-class channel  (B, H, W)
        targets_f  = targets.float()          # (B, H, W) float for BCE/Dice

        loss_focal = self.focal(room_logit, targets_f)
        loss_dice  = self.dice(room_logit, targets_f)
        return self.weight_focal * loss_focal + self.weight_dice * loss_dice


# ── Dataset ───────────────────────────────────────────────────────────────────
class FloorSegDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.img_dir  = DATA_ROOT / "images" / split
        self.mask_dir = DATA_ROOT / "masks"  / split
        self.names    = sorted(f.name for f in self.img_dir.glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img  = np.array(Image.open(self.img_dir  / name).convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / name).convert("L"))

        # White pixels (≥128) → 1 (room), others → 0 (background)
        mask = (mask >= 128).astype(np.int64)

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"]          # float32 tensor (C,H,W)
            mask = aug["mask"].long()    # (H,W) int64

        return img, mask


# ── changed: patch-based transforms ───────────────────────────────────────────
def get_transforms(split: str):
    """
    Train : PadIfNeeded (zero-pad to ≥512×512) → RandomCrop(512, 512)
    Val   : PadIfNeeded (zero-pad to ≥512×512) → CenterCrop(512, 512)

    PadIfNeeded uses border_mode=cv2.BORDER_CONSTANT (value=0) so that
    black padding pixels are added and do not introduce false gradients.
    """
    pad = A.PadIfNeeded(
        min_height=IMG_SIZE,
        min_width=IMG_SIZE,
        border_mode=0,          # cv2.BORDER_CONSTANT
        value=0,                # black padding for image
        mask_value=0,           # background label for mask padding
    )
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])

    if split == "train":
        return A.Compose([
            pad,
            A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
            normalize,
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            pad,
            A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
            normalize,
            ToTensorV2(),
        ])


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        id2label={0: "background", 1: "room"},
        label2id={"background": 0, "room": 1},
    )
    return model.to(DEVICE)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes=2):
    """Mean IoU over valid classes."""
    ious = []
    for c in range(num_classes):
        pred_c   = (preds   == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum().float()
        union        = (pred_c | target_c).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    return float(np.mean(ious)) if ious else 0.0


# ── Train / Eval loops ────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, is_train: bool):
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_targets = [], []

    phase = "Train" if is_train else "Val"
    pbar = tqdm(loader, desc=phase, leave=False, unit="batch")

    with torch.set_grad_enabled(is_train):
        for imgs, masks in pbar:
            imgs  = imgs.to(DEVICE)   # (B,3,H,W)
            masks = masks.to(DEVICE)  # (B,H,W)

            outputs = model(pixel_values=imgs)
            logits  = outputs.logits  # (B, num_labels, H/4, W/4)

            upsampled = nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )                         # (B, num_labels, H, W)

            loss = criterion(upsampled, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / (pbar.n + 1):.4f}")

            preds = upsampled.argmax(dim=1)  # (B,H,W)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    avg_loss = total_loss / len(loader)
    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    iou = compute_iou(preds_cat, targets_cat, NUM_LABELS)
    return avg_loss, iou


# ── Plot ──────────────────────────────────────────────────────────────────────
def save_curves(train_losses, val_losses, val_ious):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, val_ious, color="green", label="Val mIoU")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU")
    ax2.set_title("Validation mIoU"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(LOSS_PNG, dpi=150)
    plt.close()
    print(f"Saved learning curves → {LOSS_PNG}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Datasets & loaders
    train_ds = FloorSegDataset("train", get_transforms("train"))
    val_ds   = FloorSegDataset("val",   get_transforms("val"))
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = build_model()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    criterion = HybridLoss(
        weight_focal = WEIGHT_FOCAL,
        weight_dice  = WEIGHT_DICE,
        gamma        = FOCAL_GAMMA,
        alpha        = FOCAL_ALPHA,
        smooth       = DICE_SMOOTH,
    )
    print(
        f"Loss: {WEIGHT_FOCAL:.2f}×FocalLoss(γ={FOCAL_GAMMA}, α={FOCAL_ALPHA})"
        f" + {WEIGHT_DICE:.2f}×DiceLoss(smooth={DICE_SMOOTH})"
    )

    best_iou        = 0.0
    patience_count  = 0
    train_losses, val_losses, val_ious = [], [], []

    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>9}  {'ValIoU':>8}  {'Time':>7}")
    print("-" * 52)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, _        = run_epoch(model, train_loader, optimizer, criterion, is_train=True)
        val_loss, val_iou = run_epoch(model, val_loader,   optimizer, criterion, is_train=False)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {val_loss:>9.4f}  {val_iou:>8.4f}  {elapsed:>6.1f}s")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"         *** New best mIoU={best_iou:.4f} → saved to {BEST_MODEL}")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch} (patience={EARLY_STOP})")
                break

    save_curves(train_losses, val_losses, val_ious)
    print(f"\nDone. Best Val mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
