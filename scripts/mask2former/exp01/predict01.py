"""
predict01.py  -  Mask2Former instance segmentation inference

Strategy
--------
全画像を CROP_SIZE × CROP_SIZE (= 1024 × 1024) にリサイズして 1 回推論し、
検出されたインスタンスマスクを元解像度に復元してオーバーレイ画像を保存する。

Outputs (per image)
-------------------
  <stem>_mask.png    : インスタンス ID マスク (uint8, 0=background, 1..N=instance)
  <stem>_overlay.png : 元画像 + 半透明インスタンスカラーオーバーレイ

Usage:
    python predict01.py <input_dir> [--weights <path>] [--output <dir>]
                        [--model_id <str>] [--threshold <float>]
"""

from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None   # suppress DecompressionBomb for large floor plans

import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import albumentations as A
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Kaggle 環境判別
# ─────────────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "facebook/mask2former-swin-small-coco-instance"

NUM_LABELS = 1
ID2LABEL   = {0: "room"}
LABEL2ID   = {"room": 0}

CROP_SIZE  = 1024   # 学習時と同じクロップサイズ
THRESHOLD  = 0.5    # インスタンス confidence 閾値

if IS_KAGGLE:
    DEFAULT_WEIGHTS    = Path("/kaggle/working/exp01/best_model.pth")
    DEFAULT_OUTPUT_DIR = Path("/kaggle/working/predict01")
else:
    DEFAULT_WEIGHTS    = Path("/content/drive/MyDrive/mask2former_exp01/best_model.pth")
    DEFAULT_OUTPUT_DIR = Path("/content/drive/MyDrive/mask2former_predict01")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 学習時と同じ正規化パラメータ
_NORMALIZE = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# インスタンス描画用カラーパレット (RGB) ― 最大 24 インスタンスまで対応
PALETTE = [
    (220,  60,  60), ( 60, 200,  60), ( 60,  60, 220),
    (220, 200,  60), (200,  60, 220), ( 60, 220, 220),
    (220, 130,  60), (130, 220,  60), ( 60, 130, 220),
    (220,  60, 130), (130,  60, 220), ( 60, 220, 130),
    (160,  60,  60), ( 60, 160,  60), ( 60,  60, 160),
    (160, 160,  60), (160,  60, 160), ( 60, 160, 160),
    (200, 100,  60), (100, 200,  60), ( 60, 100, 200),
    (200,  60, 100), (100,  60, 200), ( 60, 200, 100),
]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(
    model_id: str,
    weights_path: Path,
) -> Mask2FormerForUniversalSegmentation:
    """
    学習時と同じ設定でモデルを構築し、保存した state_dict をロードして返す。
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
        ignore_mismatched_sizes=True,
    )
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    """
    uint8 RGB (H, W, 3) → 1024×1024 リサイズ → 正規化 → (1, 3, 1024, 1024) テンソル

    学習時は PadIfNeeded → CenterCrop(1024) で 1024×1024 にしていたが、
    推論では全体情報を保持したいため Resize(1024, 1024) を使用する。
    """
    resized = A.Resize(CROP_SIZE, CROP_SIZE)(image=image_rgb)["image"]
    normed  = _NORMALIZE(image=resized)["image"]                          # (1024, 1024, 3)
    tensor  = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0)   # (1, 3, 1024, 1024)
    return tensor.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict_instances(
    model: Mask2FormerForUniversalSegmentation,
    image_rgb: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[float], list[int]]:
    """
    元画像に対してインスタンスセグメンテーションを実行する。

    Flow:
        1. 1024×1024 にリサイズして forward
        2. class_queries_logits → softmax → no-object 以外の最大スコアを confidence に
        3. masks_queries_logits → sigmoid → bilinear で元解像度にアップサンプリング
        4. threshold 以上のクエリのみ残す

    Args:
        model      : eval モードの Mask2FormerForUniversalSegmentation
        image_rgb  : uint8 RGB (orig_H, orig_W, 3)
        threshold  : confidence 閾値

    Returns:
        masks  : (N, orig_H, orig_W) bool ndarray
        scores : [N] float list
        labels : [N] int list  (0-indexed class label)
    """
    orig_H, orig_W = image_rgb.shape[:2]
    pixel_values   = preprocess(image_rgb)   # (1, 3, 1024, 1024)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # ── クラス確率 ────────────────────────────────────────────────────────
    # class_queries_logits : (1, num_queries, num_labels + 1)
    #   最後の次元が no-object クラス
    class_logits = outputs.class_queries_logits[0]           # (Q, num_labels+1)
    class_probs  = class_logits.softmax(dim=-1)              # (Q, num_labels+1)
    scores, pred_labels = class_probs[:, :-1].max(dim=-1)    # (Q,)  no-object を除外

    # ── マスクを元解像度にアップサンプリング ─────────────────────────────
    # masks_queries_logits : (1, Q, H_model/4, W_model/4) = (1, Q, 256, 256)
    mask_logits = outputs.masks_queries_logits[0]            # (Q, 256, 256)
    mask_probs  = mask_logits.sigmoid()                      # (Q, 256, 256)

    mask_probs_full = F.interpolate(
        mask_probs.unsqueeze(0),                             # (1, Q, 256, 256)
        size=(orig_H, orig_W),
        mode="bilinear",
        align_corners=False,
    )[0]                                                     # (Q, orig_H, orig_W)

    # ── confidence フィルタ ────────────────────────────────────────────────
    keep = (scores >= threshold).cpu()                       # (Q,) bool

    filtered_scores  = scores[keep].cpu().tolist()
    filtered_labels  = pred_labels[keep].cpu().tolist()
    filtered_masks   = (mask_probs_full[keep] > 0.5).cpu().numpy()   # (N, H, W) bool

    return filtered_masks, filtered_scores, filtered_labels


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_instance_id_mask(masks: np.ndarray, orig_H: int, orig_W: int) -> np.ndarray:
    """
    masks (N, H, W) bool → インスタンス ID マスク (H, W) uint8
    0 = background, 1..N = instance (後勝ちで重複を上書き)
    """
    id_mask = np.zeros((orig_H, orig_W), dtype=np.uint8)
    for i, m in enumerate(masks):
        id_mask[m] = i + 1
    return id_mask


def make_overlay(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    scores: list[float],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    元画像にインスタンスマスクを半透明で重ねたオーバーレイ画像を返す。

    各インスタンスは PALETTE の色で塗られ、重心位置に "#番号 score" テキストを描画する。

    Args:
        image_rgb : (H, W, 3) uint8
        masks     : (N, H, W) bool ndarray
        scores    : [N] confidence スコア
        alpha     : マスク不透明度 (0=透明, 1=不透明)

    Returns:
        overlay (H, W, 3) uint8
    """
    overlay = image_rgb.astype(np.float32).copy()

    for i, (m, score) in enumerate(zip(masks, scores)):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        overlay[m, 0] = overlay[m, 0] * (1 - alpha) + color[0] * alpha
        overlay[m, 1] = overlay[m, 1] * (1 - alpha) + color[1] * alpha
        overlay[m, 2] = overlay[m, 2] * (1 - alpha) + color[2] * alpha

    # テキストラベルを各インスタンスの重心に描画 (PIL)
    pil = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for i, (m, score) in enumerate(zip(masks, scores)):
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        label_text = f"#{i + 1} {score:.2f}"
        # 白背景の影で視認性を確保
        draw.text((cx - 1, cy - 1), label_text, fill=(0, 0, 0),   font=font)
        draw.text((cx,     cy    ), label_text, fill=(255, 255, 255), font=font)

    return np.array(pil)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
def save_results(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    scores: list[float],
    output_dir: Path,
    stem: str,
) -> None:
    """
    インスタンス ID マスクとオーバーレイ画像を output_dir に保存する。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_H, orig_W = image_rgb.shape[:2]

    # ① インスタンス ID マスク
    id_mask    = make_instance_id_mask(masks, orig_H, orig_W)
    mask_path  = output_dir / f"{stem}_mask.png"
    Image.fromarray(id_mask).save(mask_path)
    print(f"  Saved mask    → {mask_path}")

    # ② オーバーレイ
    if len(masks) > 0:
        overlay = make_overlay(image_rgb, masks, scores)
    else:
        overlay = image_rgb.copy()
    overlay_path = output_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    print(f"  Saved overlay → {overlay_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Mask2Former instance segmentation inference  "
            "(resize to 1024×1024 → infer → restore to original resolution)"
        )
    )
    p.add_argument("input_dir",   type=str,
                   help="Directory containing input floor plan images")
    p.add_argument("--weights",   type=str, default=str(DEFAULT_WEIGHTS),
                   help=f"Path to best_model.pth  (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--output",    type=str, default=str(DEFAULT_OUTPUT_DIR),
                   help=f"Output directory  (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--model_id",  type=str, default=DEFAULT_MODEL_ID,
                   help=f"HuggingFace model ID used during training  (default: {DEFAULT_MODEL_ID})")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help=f"Instance confidence threshold  (default: {THRESHOLD})")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output)
    weights    = Path(args.weights)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    print(f"Device     : {DEVICE}")
    print(f"Model ID   : {args.model_id}")
    print(f"Weights    : {weights}")
    print(f"Input dir  : {input_dir}  ({len(image_paths)} images)")
    print(f"Output dir : {output_dir}")
    print(f"Crop size  : {CROP_SIZE}×{CROP_SIZE}")
    print(f"Threshold  : {args.threshold}")

    print("\nLoading model …")
    model = build_model(args.model_id, weights)
    print("Model loaded.\n")

    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name}")
        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        H, W = image_rgb.shape[:2]
        print(f"  Image size : {W}×{H}")

        masks, scores, labels = predict_instances(model, image_rgb, threshold=args.threshold)
        print(f"  Instances detected : {len(masks)}")
        for j, (s, lb) in enumerate(zip(scores, labels)):
            print(f"    #{j + 1:3d}  label={ID2LABEL.get(int(lb), lb)}  score={s:.3f}")

        save_results(image_rgb, masks, scores, output_dir, stem=img_path.stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
