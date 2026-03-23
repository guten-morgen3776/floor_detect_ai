"""
predict05.py  -  Two-stage segmentation inference with TTA

Stage 1 (Coarse):  train01 weights (SegFormer-b3)
                   Full image resized to 512x512 → global room/background mask
                   TTA: 4 rotations (0°, 90°, 180°, 270°), logits averaged
Stage 2 (Fine):    train03 weights (SegFormer-b2)
                   Sliding window (PATCH_SIZE=1024, STRIDE=512) within coarse-mask ROI only
                   TTA: 4 rotations (0°, 90°, 180°, 270°), logits averaged
Fusion:            fine mask result inside ROI, forced background outside ROI

Usage:
    python predict05.py <input_dir> [--coarse <path>] [--fine <path>] [--output <dir>]
                        [--patch-size {768,1024}] [--dilate-px <int>]
"""

from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # suppress DecompressionBomb for large floor plans

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation

# ─────────────────────────────────────────────
# Kaggle 環境判別
# ─────────────────────────────────────────────
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COARSE_MODEL_ID = "nvidia/segformer-b3-finetuned-ade-512-512"
FINE_MODEL_ID   = "nvidia/segformer-b2-finetuned-ade-512-512"
NUM_LABELS      = 2

if IS_KAGGLE:
    DEFAULT_COARSE_WEIGHTS = Path("/kaggle/working/exp01/best_model.pth")
    DEFAULT_FINE_WEIGHTS   = Path("/kaggle/working/exp03/best_model.pth")
    DEFAULT_OUTPUT_DIR     = Path("/kaggle/working/predict05")
else:
    DEFAULT_COARSE_WEIGHTS = Path("/content/drive/MyDrive/exp01/best_model.pth")
    DEFAULT_FINE_WEIGHTS   = Path("/content/drive/MyDrive/exp03/best_model.pth")
    DEFAULT_OUTPUT_DIR     = Path("/content/drive/MyDrive/predict05")

COARSE_SIZE        = 512    # train01の学習サイズ
DEFAULT_PATCH_SIZE = 1024   # 細推論パッチサイズ (768 or 1024)
DEFAULT_DILATE_PX  = 128    # 粗マスク膨張量 (px, 元解像度基準)
ROI_OVERLAP_THRESH = 0.50   # パッチ面積に対するROI被覆率の閾値（未満はスキップ）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────
def build_model(model_id: str, weights_path: Path) -> nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        id2label={0: "background", 1: "room"},
        label2id={"background": 0, "room": 1},
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


# ─────────────────────────────────────────────
# Stage 1: Coarse inference (TTA)
# ─────────────────────────────────────────────
def coarse_predict(model: nn.Module, image_rgb: np.ndarray) -> np.ndarray:
    """
    TTA: 0°・90°・180°・270° の4枚を推論し、roomロジットを平均して
    元解像度のバイナリマスク(0/1)を返す。

    Args:
        model     : train01モデル (SegFormer-b3, eval mode)
        image_rgb : uint8 RGB画像 (H, W, 3)

    Returns:
        mask (H, W) uint8 — 1=room, 0=background
    """
    H, W = image_rgb.shape[:2]
    avg_logit = np.zeros((H, W), dtype=np.float32)

    for k in range(4):  # k=0:0°, k=1:90°CCW, k=2:180°, k=3:270°CCW
        rot_img = np.rot90(image_rgb, k=k)
        rot_H, rot_W = rot_img.shape[:2]

        resized = A.Resize(COARSE_SIZE, COARSE_SIZE)(image=rot_img)["image"]
        tensor  = NORMALIZE(image=resized)["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(pixel_values=tensor).logits  # (1,2,128,128)
            up = nn.functional.interpolate(
                logits, size=(rot_H, rot_W), mode="bilinear", align_corners=False
            )
            room_logit = up[0, 1].cpu().numpy()  # (rot_H, rot_W)

        # 逆回転して元の向きに戻す
        back_logit = np.rot90(room_logit, k=(4 - k) % 4)
        avg_logit += back_logit

    avg_logit /= 4
    mask = (avg_logit > 0).astype(np.uint8)
    return mask


# ─────────────────────────────────────────────
# ROI生成
# ─────────────────────────────────────────────
def make_roi_mask(coarse_mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """
    粗マスクをdilateしてROIマスク(0/1)を返す。
    粗モデルの境界ズレを吸収するため、room領域を膨張させる。

    Args:
        coarse_mask : uint8 バイナリマスク (H, W) — 1=room
        dilate_px   : 膨張量 (px, 元解像度基準)

    Returns:
        roi_mask (H, W) uint8 — 1=ROI, 0=確実なbackground
    """
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    return cv2.dilate(coarse_mask, kernel)


# ─────────────────────────────────────────────
# Hann窓
# ─────────────────────────────────────────────
def make_hann_window(size: int) -> np.ndarray:
    """2D Hann window (size, size), values in [0, 1]."""
    w1d = np.hanning(size)
    return np.outer(w1d, w1d).astype(np.float32)


# ─────────────────────────────────────────────
# Stage 2: Fine inference の内部実装 (1回転分)
# ─────────────────────────────────────────────
def _fine_predict_logits_in_roi(
    model: nn.Module,
    image_rgb: np.ndarray,
    roi_mask: np.ndarray,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """
    ROIと重なるパッチのみスライディングウィンドウ推論を実行し、
    Hann窓で重み付け平均したロジットマップ (H, W) を返す。
    (TTA用内部関数。閾値処理はしない。)

    Args:
        model      : train03モデル (SegFormer-b2, eval mode)
        image_rgb  : uint8 RGB画像 (H, W, 3)
        roi_mask   : uint8 ROIマスク (H, W) — 1=推論対象
        patch_size : スライディングウィンドウのパッチサイズ (768 or 1024)
        stride     : スライディングウィンドウのストライド

    Returns:
        avg_logit (H, W) float32 — roomロジット平均値 (ROIなし領域は -999.0)
    """
    H, W = image_rgb.shape[:2]

    # ── パディング (reflect) ──────────────────────────────────────────────
    def padded_dim(dim: int) -> int:
        if dim <= patch_size:
            return patch_size
        n = int(np.ceil((dim - patch_size) / stride))
        return patch_size + n * stride

    H_pad = padded_dim(H)
    W_pad = padded_dim(W)

    img_pad = np.pad(
        image_rgb,
        ((0, H_pad - H), (0, W_pad - W), (0, 0)),
        mode="reflect",
    )
    roi_pad = np.pad(
        roi_mask,
        ((0, H_pad - H), (0, W_pad - W)),
        mode="constant", constant_values=0,
    )

    # ── 累積配列 ─────────────────────────────────────────────────────────
    logit_sum  = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_sum = np.zeros((H_pad, W_pad), dtype=np.float32)
    hann2d     = make_hann_window(patch_size)

    y_starts = list(range(0, H_pad - patch_size + 1, stride))
    x_starts = list(range(0, W_pad - patch_size + 1, stride))
    total_patches = len(y_starts) * len(x_starts)
    executed = 0

    # ── スライディングウィンドウ ──────────────────────────────────────────
    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                # ROI被覆率が閾値未満のパッチはスキップ
                roi_patch = roi_pad[y:y + patch_size, x:x + patch_size]
                if roi_patch.mean() < ROI_OVERLAP_THRESH:
                    continue

                patch  = img_pad[y:y + patch_size, x:x + patch_size]
                tensor = NORMALIZE(image=patch)["image"].unsqueeze(0).to(DEVICE)

                logits = model(pixel_values=tensor).logits  # (1,2,P/4,P/4)
                up = nn.functional.interpolate(
                    logits,
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
                room_logit = up[0, 1].cpu().numpy()  # (patch_size, patch_size)

                logit_sum [y:y + patch_size, x:x + patch_size] += room_logit * hann2d
                weight_sum[y:y + patch_size, x:x + patch_size] += hann2d
                executed += 1

    print(f"  Patches: {executed}/{total_patches} executed "
          f"({total_patches - executed} skipped by ROI filter)")

    # weight_sum=0 の領域 (ROIスキップ箇所) は -999 にして background 扱い
    avg_logit = np.where(weight_sum > 0, logit_sum / weight_sum, -999.0)
    return avg_logit[:H, :W]


# ─────────────────────────────────────────────
# Stage 2: Fine inference (ROI内のみ、TTA)
# ─────────────────────────────────────────────
def fine_predict_in_roi(
    model: nn.Module,
    image_rgb: np.ndarray,
    roi_mask: np.ndarray,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """
    TTA: 0°・90°・180°・270° の4枚に対してスライディングウィンドウ推論を行い、
    各推論結果を元の向きに逆回転した後、ロジットを平均して最終マスクを返す。

    Args:
        model      : train03モデル (SegFormer-b2, eval mode)
        image_rgb  : uint8 RGB画像 (H, W, 3)
        roi_mask   : uint8 ROIマスク (H, W) — 1=推論対象
        patch_size : スライディングウィンドウのパッチサイズ (768 or 1024)
        stride     : スライディングウィンドウのストライド

    Returns:
        fine_mask (H, W) uint8 — 1=room, 0=background
    """
    H, W = image_rgb.shape[:2]
    avg_logit = np.zeros((H, W), dtype=np.float32)

    for k in range(4):  # k=0:0°, k=1:90°CCW, k=2:180°, k=3:270°CCW
        print(f"    TTA rotation {k * 90}°…")
        rot_img = np.rot90(image_rgb, k=k)
        rot_roi = np.rot90(roi_mask,  k=k)

        rot_logit = _fine_predict_logits_in_roi(
            model, rot_img, rot_roi, patch_size, stride
        )

        # 逆回転して元の向きに戻す
        back_logit = np.rot90(rot_logit, k=(4 - k) % 4)
        avg_logit += back_logit

    avg_logit /= 4
    fine_mask = (avg_logit > 0).astype(np.uint8)  # logit>0 ↔ prob>0.5
    return fine_mask


# ─────────────────────────────────────────────
# 融合
# ─────────────────────────────────────────────
def fuse(fine_mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """
    ROI内は細マスクの結果を採用、ROI外は強制的にbackground(0)。

    fine_mask AND roi_mask の論理積により、
    粗モデルが確実にbackgroundと判定した領域での誤認を排除する。
    """
    return (fine_mask & (roi_mask > 0)).astype(np.uint8)


# ─────────────────────────────────────────────
# 結果保存
# ─────────────────────────────────────────────
def save_results(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    coarse_mask: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    """バイナリマスク・オーバーレイ・粗マスクを保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ① 最終マスク (room=255, background=0)
    mask_path = output_dir / f"{stem}_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    print(f"  Saved mask       → {mask_path}")

    # ② 粗マスク (デバッグ用)
    coarse_path = output_dir / f"{stem}_coarse.png"
    Image.fromarray((coarse_mask * 255).astype(np.uint8)).save(coarse_path)
    print(f"  Saved coarse     → {coarse_path}")

    # ③ オーバーレイ (room領域を半透明の赤で塗る)
    overlay = image_rgb.copy().astype(np.float32)
    room_area = mask > 0
    alpha = 0.4
    overlay[room_area, 0] = overlay[room_area, 0] * (1 - alpha) + 255 * alpha
    overlay[room_area, 1] = overlay[room_area, 1] * (1 - alpha)
    overlay[room_area, 2] = overlay[room_area, 2] * (1 - alpha)
    overlay_path = output_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(overlay_path)
    print(f"  Saved overlay    → {overlay_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Two-stage segmentation inference with TTA (coarse=train01, fine=train03)"
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing input floor plan images")
    parser.add_argument("--coarse", type=str,
                        default=str(DEFAULT_COARSE_WEIGHTS),
                        help=f"Path to train01 best_model.pth (default: {DEFAULT_COARSE_WEIGHTS})")
    parser.add_argument("--fine", type=str,
                        default=str(DEFAULT_FINE_WEIGHTS),
                        help=f"Path to train03 best_model.pth (default: {DEFAULT_FINE_WEIGHTS})")
    parser.add_argument("--output", type=str,
                        default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--patch-size", type=int,
                        choices=[768, 1024], default=DEFAULT_PATCH_SIZE,
                        help=f"Fine model patch size (default: {DEFAULT_PATCH_SIZE})")
    parser.add_argument("--stride", type=int,
                        default=None,
                        help="Fine model sliding window stride (default: patch-size // 2)")
    parser.add_argument("--dilate-px", type=int,
                        default=DEFAULT_DILATE_PX,
                        help=f"Coarse mask dilation in pixels (default: {DEFAULT_DILATE_PX})")
    args = parser.parse_args()

    input_dir      = Path(args.input_dir)
    output_dir     = Path(args.output)
    coarse_weights = Path(args.coarse)
    fine_weights   = Path(args.fine)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not coarse_weights.exists():
        raise FileNotFoundError(f"Coarse model weights not found: {coarse_weights}")
    if not fine_weights.exists():
        raise FileNotFoundError(f"Fine model weights not found: {fine_weights}")

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    print(f"Device       : {DEVICE}")
    print(f"Input dir    : {input_dir}  ({len(image_paths)} images)")
    print(f"Output dir   : {output_dir}")
    print(f"Coarse model : {coarse_weights}")
    print(f"Fine model   : {fine_weights}")
    stride = args.stride if args.stride is not None else args.patch_size // 2
    print(f"Patch size   : {args.patch_size}  Stride: {stride}")
    print(f"Dilate px    : {args.dilate_px}")
    print(f"TTA          : 4 rotations (0°, 90°, 180°, 270°)")

    # ── モデルロード (1回だけ) ─────────────────────────────────────────────
    print("\nLoading coarse model…")
    coarse_model = build_model(COARSE_MODEL_ID, coarse_weights)

    print("Loading fine model…")
    fine_model = build_model(FINE_MODEL_ID, fine_weights)

    # ── 画像ごとの処理 ────────────────────────────────────────────────────
    for i, input_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {input_path.name}")
        image_rgb = np.array(Image.open(input_path).convert("RGB"))
        H, W = image_rgb.shape[:2]
        print(f"  Image size : {W}×{H}")

        # Stage 1: 粗推論 (TTA)
        print("  [Stage 1] Coarse inference (TTA)…")
        coarse_mask = coarse_predict(coarse_model, image_rgb)
        room_ratio  = coarse_mask.mean()
        print(f"  Coarse room coverage: {room_ratio:.1%}")

        # ROI生成
        roi_mask = make_roi_mask(coarse_mask, dilate_px=args.dilate_px)
        roi_ratio = roi_mask.mean()
        print(f"  ROI coverage (after dilate): {roi_ratio:.1%}")

        # Stage 2: 細推論 (ROI内のみ、TTA)
        print(f"  [Stage 2] Fine inference with TTA (patch={args.patch_size}, stride={stride})…")
        fine_mask = fine_predict_in_roi(
            fine_model, image_rgb, roi_mask, patch_size=args.patch_size, stride=stride
        )

        # 融合
        final_mask = fuse(fine_mask, roi_mask)
        print(f"  Final room coverage: {final_mask.mean():.1%}")

        save_results(image_rgb, final_mask, coarse_mask, output_dir, stem=input_path.stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
