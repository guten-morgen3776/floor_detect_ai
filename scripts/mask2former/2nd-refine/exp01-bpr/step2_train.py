"""
step2_train.py  ―  BPR Refinement Network 学習

学習設定:
  - Optimizer  : SGD (momentum=0.9, weight_decay=0.0005)
  - LR Scheduler: Poly (power=0.9)
  - Loss       : CrossEntropyLoss (2-class)
  - イテレーション基準 (1 epoch が非常に長いため epoch 基準を使わない)

Usage:
    python step2_train.py
    python step2_train.py --total_iters 80000 --batch_size 128
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model   import LightweightRefineNet
from dataset import BoundaryPatchDataset


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             max_batches: int = 200) -> float:
    """
    パッチ単位の foreground IoU を計算して返す。

    max_batches で検証を早期打ち切りし、待ち時間を抑える。
    """
    model.eval()
    inter_sum = 0.0
    union_sum = 0.0
    n_batches = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        logits  = model(inputs)                      # (B, 2, H, W)
        preds   = logits.argmax(dim=1)               # (B, H, W)

        fg_pred = preds   == 1
        fg_gt   = targets == 1
        inter_sum += (fg_pred & fg_gt).sum().item()
        union_sum += (fg_pred | fg_gt).sum().item()

        n_batches += 1
        if n_batches >= max_batches:
            break

    model.train()
    return inter_sum / (union_sum + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Loss 曲線の保存
# ─────────────────────────────────────────────────────────────────────────────

def save_loss_plot(train_losses: list[float],
                   val_ious:     list[float],
                   val_steps:    list[int],
                   out_path:     Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_title("Train Loss (per 100 iters avg)")
    ax1.set_xlabel("100 iterations")
    ax1.set_ylabel("CrossEntropyLoss")
    ax1.grid(True)

    ax2.plot(val_steps, val_ious, marker="o")
    ax2.set_title("Val Foreground IoU")
    ax2.set_xlabel("iterations")
    ax2.set_ylabel("IoU")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  Loss plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BPR Refinement Network 学習")
    p.add_argument("--total_iters",  type=int,   default=config.TOTAL_ITERS)
    p.add_argument("--batch_size",   type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=config.LR)
    p.add_argument("--num_workers",  type=int,   default=config.NUM_WORKERS)
    p.add_argument("--val_every",    type=int,   default=config.VAL_EVERY_ITERS)
    p.add_argument("--no_cache",     action="store_true",
                   help="パッチインデックスのキャッシュを使わない")
    p.add_argument("--resume",       type=str,   default=None,
                   help="チェックポイントパス (途中から再開)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Total iters : {args.total_iters:,}")
    print(f"Batch size  : {args.batch_size}")
    print(f"LR          : {args.lr}")
    print(f"Val every   : {args.val_every:,} iters")
    print("=" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    use_cache = not args.no_cache

    print("\n[Train Dataset]")
    train_ds = BoundaryPatchDataset(
        img_dir   = config.IMG_DIR_TRAIN,
        pred_dir  = config.PRED_DIR_TRAIN,
        gt_dir    = config.GT_DIR_TRAIN,
        split     = "train",
        use_cache = use_cache,
    )
    print("\n[Val Dataset]")
    val_ds = BoundaryPatchDataset(
        img_dir   = config.IMG_DIR_VAL,
        pred_dir  = config.PRED_DIR_VAL,
        gt_dir    = config.GT_DIR_VAL,
        split     = "val",
        use_cache = use_cache,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LightweightRefineNet(
        in_channels = config.IN_CHANNELS,
        num_classes = config.NUM_CLASSES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {n_params:,}")

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Resumed from: {args.resume}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = args.lr,
        momentum     = config.MOMENTUM,
        weight_decay = config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(
            (1 - (step + start_step) / args.total_iters) ** config.POLY_POWER,
            1e-3,  # 最小 LR 倍率
        ),
    )

    criterion = nn.CrossEntropyLoss()

    # ── 学習ループ ─────────────────────────────────────────────────────────────
    loader_iter    = iter(train_loader)
    train_losses   = []
    val_ious       = []
    val_steps_log  = []
    best_val_iou   = 0.0
    running_loss   = 0.0
    t_start        = time.time()

    model.train()
    pbar = tqdm(range(args.total_iters), desc="Training")

    for step in pbar:
        # DataLoader を無限ループ
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inputs, targets = next(loader_iter)

        inputs  = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)                          # (B, 2, H, W)
        loss   = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # ── 100 iters ごとに loss を記録 ───────────────────────────────────
        if (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            train_losses.append(avg_loss)
            running_loss = 0.0
            elapsed = time.time() - t_start
            lr_now  = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{avg_loss:.4f}",
                             lr=f"{lr_now:.5f}",
                             elapsed=f"{elapsed:.0f}s")

        # ── 検証 & チェックポイント ────────────────────────────────────────
        if (step + 1) % args.val_every == 0:
            val_iou = validate(model, val_loader, device)
            val_ious.append(val_iou)
            val_steps_log.append(step + 1)

            print(f"\n[Step {step+1:,}] val_iou={val_iou:.4f}"
                  f"  best={best_val_iou:.4f}"
                  f"  lr={scheduler.get_last_lr()[0]:.6f}")

            torch.save(model.state_dict(), config.LAST_MODEL)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), config.BEST_MODEL)
                print(f"  → Best model saved: {config.BEST_MODEL}")

            save_loss_plot(train_losses, val_ious, val_steps_log, config.LOSS_PNG)

    # ── 最終チェックポイント & プロット ────────────────────────────────────────
    torch.save(model.state_dict(), config.LAST_MODEL)
    save_loss_plot(train_losses, val_ious, val_steps_log, config.LOSS_PNG)

    elapsed = time.time() - t_start
    print(f"\n=== step2 完了  best_val_iou={best_val_iou:.4f}"
          f"  elapsed={elapsed:.0f}s ===")


if __name__ == "__main__":
    main()
