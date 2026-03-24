"""
Mask2Former instance segmentation fine-tuning on CubiCasa room dataset.
exp03: Differential learning rates (backbone vs head) + Linear warmup.

exp01 からの変更点
------------------
* Backbone (Swin encoder) の学習率を head / decoder より小さくする
  (BACKBONE_LR_SCALE 倍)。事前学習済み特徴量を壊さず、新しい head を
  素早く収束させるため。
* Cosine annealing の前に Linear warmup (WARMUP_EPOCHS エポック) を挿入。
  学習開始直後の不安定な勾配による発散を防ぐ。
* 上記パラメータはすべて CONFIG セクションに集約。
  CLI からも --backbone_lr_scale / --warmup_epochs / --warmup_start_factor で
  オーバーライド可能。

Key design choices (inherited from exp01)
------------------
* Images larger than MAX_W × MAX_H (5000 × 4000) are skipped.
* Memory-efficient index mask strategy for albumentations.
* Mask2FormerForUniversalSegmentation computes Hungarian-matching loss internally.
* HuggingFace accelerate for multi-GPU (DDP), mixed-precision, gradient accumulation.

Usage (single GPU):
  python train03.py --batch_size 2 --epochs 50 --fp16

Usage (multi-GPU, e.g. T4×2 on Kaggle):
  accelerate launch train03.py --batch_size 2 --epochs 50 --fp16
"""

from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None   # suppress DecompressionBomb for large floor plans

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from accelerate import Accelerator

from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig,
)
import albumentations as A

# ─────────────────────────────────────────────────────────────────────────────
# Kaggle 環境判別
# ─────────────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "facebook/mask2former-swin-small-coco-instance"

if IS_KAGGLE:
    DATA_ROOT  = Path("/kaggle/input/cubicasa-mask2former")
    OUTPUT_DIR = Path("/kaggle/working/exp03")
else:
    DATA_ROOT  = Path(os.environ.get("KAGGLE_DATA_ROOT")) / "cubicasa_mask2former"
    OUTPUT_DIR = Path("/content/drive/MyDrive/mask2former_exp03")

ANNO_TRAIN    = DATA_ROOT / "annotations" / "instances_train.json"
ANNO_VAL      = DATA_ROOT / "annotations" / "instances_val.json"
IMG_DIR_TRAIN = DATA_ROOT / "images" / "train"
IMG_DIR_VAL   = DATA_ROOT / "images" / "val"

BEST_MODEL = OUTPUT_DIR / "best_model.pth"
LAST_MODEL = OUTPUT_DIR / "last_model.pth"
LOSS_PNG   = OUTPUT_DIR / "loss.png"

# Image size limits
MAX_W = 5000
MAX_H = 4000

# Letterbox resize target (longest side)
IMG_SIZE = 1024

# Minimum surviving mask pixels to keep an instance after crop
MIN_MASK_PX = 64

# Single class: room
NUM_LABELS      = 1
CAT_ID_TO_LABEL = {1: 0}    # COCO category_id=1 → 0-indexed model label
ID2LABEL        = {0: "room"}
LABEL2ID        = {"room": 0}

# ── Differential LR ──────────────────────────────────────────────────────────
# Backbone (Swin encoder) の学習率は head の BACKBONE_LR_SCALE 倍にする。
# 事前学習済みの特徴量を維持しつつ、新しい head を素早く最適化するため。
BACKBONE_LR_SCALE = 0.2     # backbone LR = lr * BACKBONE_LR_SCALE

# ── Warmup ───────────────────────────────────────────────────────────────────
# Linear warmup: 学習率を WARMUP_START_FACTOR * lr から lr まで線形に増加させる。
# 学習開始直後の大きな勾配による重みの破壊を防ぐ。
WARMUP_EPOCHS       = 2    # warmup のエポック数
WARMUP_START_FACTOR = 0.1   # epoch 0 での学習率倍率 (最終的に 1.0 まで線形増加)


# ─────────────────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask2Former room instance segmentation (exp03)")
    p.add_argument("--model_id",            type=str,   default=DEFAULT_MODEL_ID,
                   help="HuggingFace model ID to fine-tune")
    p.add_argument("--batch_size",          type=int,   default=1,
                   help="Per-GPU batch size (keep small for 1024×1024 + Mask2Former)")
    p.add_argument("--grad_accum",          type=int,   default=8,
                   help="Gradient accumulation steps (effective batch = batch_size × n_gpus × grad_accum)")
    p.add_argument("--epochs",              type=int,   default=50)
    p.add_argument("--lr",                  type=float, default=1e-4,
                   help="Base learning rate (head / decoder)")
    p.add_argument("--weight_decay",        type=float, default=1e-4)
    p.add_argument("--num_workers",         type=int,   default=4)
    p.add_argument("--early_stop",          type=int,   default=10,
                   help="Patience for early stopping (epochs without improvement)")
    p.add_argument("--fp16",                action="store_true",
                   help="Use mixed-precision training (fp16) via accelerate")
    # ── exp03 additions ──────────────────────────────────────────────────────
    p.add_argument("--backbone_lr_scale",   type=float, default=BACKBONE_LR_SCALE,
                   help="Backbone LR = lr * backbone_lr_scale")
    p.add_argument("--warmup_epochs",       type=int,   default=WARMUP_EPOCHS,
                   help="Number of linear warmup epochs before cosine annealing")
    p.add_argument("--warmup_start_factor", type=float, default=WARMUP_START_FACTOR,
                   help="LR scale at epoch 0 of warmup (linearly increases to 1.0)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ────────────────��────────────────────────────────────────────────────────────
def load_coco_items(json_path: Path) -> list[dict]:
    """Load COCO JSON and return list of {image: info, anns: [...]}.

    Width / height from the JSON are trusted (no need to open each file).
    """
    with open(json_path) as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}
    ann_map: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        ann_map[ann["image_id"]].append(ann)

    return [
        {"image": img_info, "anns": ann_map[img_id]}
        for img_id, img_info in img_map.items()
    ]


def filter_huge_images(items: list[dict]) -> tuple[list[dict], int]:
    """Drop items whose image exceeds MAX_W × MAX_H.

    Width / height come from the COCO JSON so no file I/O is needed here.
    """
    kept, skipped = [], 0
    for item in items:
        info = item["image"]
        w, h = info["width"], info["height"]
        if w <= MAX_W and h <= MAX_H:
            kept.append(item)
        else:
            print(f"  Skipped huge image: {info['file_name']} ({w}×{h})")
            skipped += 1
    return kept, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Albumentations transforms
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(split: str) -> A.Compose:
    """
    Returns an A.Compose pipeline with BboxParams.

    BboxParams is used to track which instances survive based on their
    bounding-box visibility.  The index mask (passed as `mask=`) is transformed
    in sync with the image so that the per-pixel instance assignment is always
    spatially consistent.

    Train : LongestMaxSize(1024) → PadIfNeeded(1024×1024) → geometric augments → ColorJitter → Normalize
    Val   : LongestMaxSize(1024) → PadIfNeeded(1024×1024) → Normalize
    """
    letterbox = [
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=0,       # cv2.BORDER_CONSTANT
            value=0,
            mask_value=0,        # background index for the index mask
        ),
    ]
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )
    bbox_params = A.BboxParams(
        format="coco",
        min_area=MIN_MASK_PX,
        min_visibility=0.1,
        label_fields=["category_ids", "ann_ids"],
    )

    if split == "train":
        return A.Compose([
            *letterbox,
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
            normalize,
        ], bbox_params=bbox_params)
    else:
        return A.Compose([
            *letterbox,
            normalize,
        ], bbox_params=bbox_params)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class CubiCasaInstanceDataset(Dataset):
    """
    COCO-format instance segmentation dataset for CubiCasa floor plans.

    Instance mask strategy (memory-efficient)
    ------------------------------------------
    Instead of materialising N full-resolution binary masks (can exceed 2 GB),
    we build one uint8 "index mask" where pixel = instance_id (1-indexed; 0 =
    background).  albumentations crops/flips/rotates this mask identically to
    the RGB image.  After the transform we extract per-instance binary masks from
    the cropped index mask.

    BboxParams linkage
    ------------------
    Bboxes are passed alongside `ann_ids` (0-based original indices).  After the
    crop, `result["ann_ids"]` contains the surviving original indices.  We use
    these to look up the corresponding instance_ids in the cropped index mask.
    """

    def __init__(self, items: list[dict], img_dir: Path, transform: A.Compose):
        self.items     = items
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item     = self.items[idx]
        img_info = item["image"]
        anns     = item["anns"]
        H, W     = img_info["height"], img_info["width"]

        img = np.array(
            Image.open(self.img_dir / img_info["file_name"]).convert("RGB")
        )

        # ── Build index mask and bbox list ────────────────────────────────────
        # index_mask[y, x] = instance_id (1-indexed).
        # Using uint8 is safe: max instances per image in this dataset is 113.
        index_mask  = np.zeros((H, W), dtype=np.uint8)
        bboxes_coco = []   # [(x, y, w, h), ...] in coco format
        cat_ids     = []   # COCO category_id per instance
        ann_ids     = []   # 0-based original index (used to retrieve instance after filtering)

        for ann in anns:
            if ann["iscrowd"] == 1:
                continue
            seg = ann["segmentation"]
            if not isinstance(seg, list) or len(seg) == 0:
                continue   # skip RLE (not present in this dataset, but guard anyway)

            # Draw polygon onto the index mask (instance_id = 1-based position)
            inst_id = len(ann_ids) + 1   # 1-indexed; 0 is reserved for background
            poly    = seg[0]             # single polygon ring per annotation confirmed
            pts     = list(zip(poly[::2], poly[1::2]))
            if len(pts) < 3:
                continue

            pil_m = Image.new("L", (W, H), 0)
            ImageDraw.Draw(pil_m).polygon(pts, fill=1)
            m_arr = np.array(pil_m, dtype=np.uint8)
            if m_arr.sum() == 0:
                continue

            # Write instance_id; later instances overwrite overlapping pixels
            # (rooms in floor plans are non-overlapping in practice)
            index_mask[m_arr > 0] = inst_id

            # Bbox: clamp to image bounds
            x, y, bw, bh = ann["bbox"]
            x  = max(0.0, float(x))
            y  = max(0.0, float(y))
            bw = min(float(bw), W - x)
            bh = min(float(bh), H - y)
            if bw <= 0 or bh <= 0:
                continue

            ann_ids.append(len(ann_ids))   # 0-based index BEFORE appending inst info
            bboxes_coco.append([x, y, bw, bh])
            cat_ids.append(ann["category_id"])

        # ── Apply albumentations transform ────────────────────────────────────
        # `mask` (singular) is the index mask — cropped in sync with the image.
        # `bboxes` + `ann_ids` let BboxParams tell us which instances survived.
        result = self.transform(
            image=img,
            mask=index_mask,
            bboxes=bboxes_coco,
            category_ids=cat_ids,
            ann_ids=ann_ids,
        )

        cropped_index  = result["mask"]            # (1024, 1024) uint8
        surviving_orig = result["ann_ids"]         # original 0-based indices of bbox-survivors
        surviving_cats = result["category_ids"]    # their category_ids

        # ── Extract binary masks for surviving instances ───────────────────────
        # inst_id in the index mask = original_index + 1 (1-indexed).
        # We additionally check that enough pixels remain after the crop.
        final_masks   = []
        final_cat_ids = []
        for orig_i, c in zip(surviving_orig, surviving_cats):
            inst_id     = orig_i + 1   # 1-indexed instance id in the index mask
            binary_mask = (cropped_index == inst_id).astype(np.float32)
            if binary_mask.sum() >= MIN_MASK_PX:
                final_masks.append(binary_mask)
                final_cat_ids.append(c)

        # ── Convert to tensors ────────────────────────────────────────────────
        img_np       = result["image"]                              # (H, W, C) float32
        pixel_values = torch.from_numpy(img_np.transpose(2, 0, 1)) # (3, H, W)

        if final_masks:
            mask_labels  = torch.stack([
                torch.from_numpy(m) for m in final_masks
            ])  # (N, H, W)  float32
            class_labels = torch.tensor(
                [CAT_ID_TO_LABEL[c] for c in final_cat_ids], dtype=torch.long
            )  # (N,)
        else:
            # No valid instances in this crop — return empty targets.
            # Mask2FormerLoss handles num_masks == 0 gracefully.
            mask_labels  = torch.zeros((0, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "mask_labels":  mask_labels,
            "class_labels": class_labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: list[dict]) -> dict:
    """Stack pixel_values; keep mask_labels / class_labels as lists.

    Mask2Former の forward() は mask_labels / class_labels を Python list
    (画像ごとにテンソルが 1 つ) として受け取るため、可変 N のバッチが自然に扱える。
    """
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask_labels":  [b["mask_labels"]  for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(model_id: str) -> Mask2FormerForUniversalSegmentation:
    """Load pretrained Mask2Former and replace the head for num_labels=1.

    デバイスへの転送は accelerator.prepare() に委ねるため、
    ここでは .to(device) を呼ばない。
    """
    config = Mask2FormerConfig.from_pretrained(
        model_id,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,   # head weights differ (80 → 1 class)
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer & Scheduler (exp03: differential LR + warmup)
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(
    model: Mask2FormerForUniversalSegmentation,
    lr: float,
    weight_decay: float,
    backbone_lr_scale: float,
) -> AdamW:
    """AdamW with separate parameter groups for backbone and head/decoder.

    Backbone (Swin encoder): lr * backbone_lr_scale
      → 事前学習済み特徴量を大きく変えないよう抑制する。
    Head / decoder (pixel decoder, transformer decoder, predictor):
      → base lr でより積極的に学習させる。

    Mask2FormerForUniversalSegmentation のモジュール構造:
      model.model.pixel_level_module.encoder  ← Swin backbone (ここだけ低 LR)
      model.model.pixel_level_module.decoder  ← pixel decoder
      model.model.transformer_module          ← transformer decoder
      model.class_predictor                   ← class head
      model.mask_embedder                     ← mask head
    """
    backbone_params    = list(model.model.pixel_level_module.encoder.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    other_params       = [p for p in model.parameters() if id(p) not in backbone_param_ids]

    return AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_scale},
            {"params": other_params,    "lr": lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: AdamW,
    total_epochs: int,
    warmup_epochs: int,
    warmup_start_factor: float,
) -> SequentialLR:
    """LinearLR warmup → CosineAnnealingLR.

    warmup フェーズ:
      epoch 0  : lr * warmup_start_factor
      epoch W-1: lr * 1.0  (= base lr)

    cosine フェーズ:
      epoch W  〜 total_epochs : lr → eta_min (1e-7) で余弦減衰
    """
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-7,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val loops
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    model,
    loader: DataLoader,
    optimizer: AdamW,
    accelerator: Accelerator,
    is_train: bool,
) -> float:
    model.train(is_train)
    total_loss = 0.0
    n_steps    = 0

    phase = "Train" if is_train else "Val  "
    # マルチプロセス環境では main process のみ進捗バーを表示する
    pbar = tqdm(loader, desc=phase, leave=False, unit="batch",
                disable=not accelerator.is_main_process)

    for batch in pbar:
        # pixel_values は accelerate が自動でデバイスに転送する。
        # mask_labels / class_labels は Python list のため自動転送されないので
        # accelerator.device を使って明示的に転送する。
        pixel_values = batch["pixel_values"]
        mask_labels  = [m.to(accelerator.device) for m in batch["mask_labels"]]
        class_labels = [c.to(accelerator.device) for c in batch["class_labels"]]

        if is_train:
            # accelerator.accumulate() が勾配蓄積・DDP の no_sync を管理する。
            with accelerator.accumulate(model):
                outputs = model(
                    pixel_values=pixel_values,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                loss = outputs.loss   # Hungarian-matched CE + mask BCE+Dice + box L1+GIoU

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                outputs = model(
                    pixel_values=pixel_values,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                loss = outputs.loss

        total_loss += loss.item()
        n_steps    += 1
        if accelerator.is_main_process:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{total_loss / n_steps:.4f}",
            )

    return total_loss / max(n_steps, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Curve saving
# ─────────────────────────────────────────────────────────────────────────────
def save_curves(train_losses: list[float], val_losses: list[float]) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mask2Former Loss Curve (exp03)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PNG, dpi=150)
    plt.close()
    print(f"Saved loss curve → {LOSS_PNG}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Accelerator の初期化 ───────────────────────────────────────────────────
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="fp16" if args.fp16 else "no",
    )

    if accelerator.is_main_process:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Device              : {accelerator.device}")
        print(f"Num processes       : {accelerator.num_processes}")
        print(f"Model               : {args.model_id}")
        print(f"Batch size/GPU      : {args.batch_size}")
        print(f"Grad accum steps    : {args.grad_accum}")
        print(f"Effective batch     : {args.batch_size * accelerator.num_processes * args.grad_accum}")
        print(f"Mixed precision     : {args.fp16}")
        print(f"Epochs              : {args.epochs}  (early stop patience={args.early_stop})")
        print(f"Base LR (head)      : {args.lr}")
        print(f"Backbone LR scale   : {args.backbone_lr_scale}  → backbone LR={args.lr * args.backbone_lr_scale:.2e}")
        print(f"Warmup epochs       : {args.warmup_epochs}  (start factor={args.warmup_start_factor})")
        print()

    # ── Load & filter annotations ─────────────────────────────────────────────
    if accelerator.is_main_process:
        print("Loading annotations …")
    train_items = load_coco_items(ANNO_TRAIN)
    val_items   = load_coco_items(ANNO_VAL)

    if accelerator.is_main_process:
        print("Filtering huge images …")
    train_items, tr_skip = filter_huge_images(train_items)
    val_items,   va_skip = filter_huge_images(val_items)

    if accelerator.is_main_process:
        print(f"Train : skipped {tr_skip:3d}, kept {len(train_items)}")
        print(f"Val   : skipped {va_skip:3d}, kept {len(val_items)}")
        print()

    # ── Datasets & DataLoaders ─────────────────────────────────────────────────
    train_ds = CubiCasaInstanceDataset(train_items, IMG_DIR_TRAIN, get_transforms("train"))
    val_ds   = CubiCasaInstanceDataset(val_items,   IMG_DIR_VAL,   get_transforms("val"))

    if accelerator.is_main_process:
        print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Model, optimizer, scheduler ───────────────────────────────────────────
    model     = build_model(args.model_id)
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_scale=args.backbone_lr_scale,
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
    )

    # accelerator.prepare() がデバイス転送・DDP ラップ・DataLoader の
    # DistributedSampler 設定をすべて担う。
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []

    if accelerator.is_main_process:
        print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>9}  {'LR(head)':>10}  {'Time':>7}")
        print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss  = run_epoch(model, train_loader, optimizer, accelerator, is_train=True)
        val_loss = run_epoch(model, val_loader,   optimizer, accelerator, is_train=False)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0

        # ── ログ・保存はメインプロセスのみ ────────────────────────────────────
        if accelerator.is_main_process:
            # param_groups[1] が head / decoder (base lr) のグループ
            current_lr = optimizer.param_groups[1]["lr"]
            print(f"{epoch:>6}  {tr_loss:>10.4f}  {val_loss:>9.4f}  {current_lr:>10.2e}  {elapsed:>6.1f}s")

            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), LAST_MODEL)

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save(unwrapped.state_dict(), BEST_MODEL)
                print(f"         *** New best val loss={best_val_loss:.4f}  → {BEST_MODEL}")
            else:
                patience_count += 1
                if patience_count >= args.early_stop:
                    print(f"\nEarly stopping at epoch {epoch} (patience={args.early_stop})")
                    break

    if accelerator.is_main_process:
        save_curves(train_losses, val_losses)
        print(f"\nDone.  Best val loss: {best_val_loss:.4f}")
        print(f"Best model : {BEST_MODEL}")
        print(f"Last model : {LAST_MODEL}")


if __name__ == "__main__":
    main()
