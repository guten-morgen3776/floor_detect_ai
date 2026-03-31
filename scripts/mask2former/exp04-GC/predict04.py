"""
predict04.py  -  Mask2Former instance segmentation inference + morphological post-processing

Strategy
--------
画像を Letterbox リサイズ (LongestMaxSize → PadIfNeeded) で 1024×1024 に変換して
1 回推論し、検出されたインスタンスマスクを元解像度に復元してオーバーレイ画像を保存する。

predict03.py との差分:
  - train04.py (exp04-GC) で学習したモデルに対応。
  - デフォルトの weights / output パスを exp04 に変更。
  - 前処理・推論・後処理ロジックは predict03.py と同一。

Outputs (per image)
-------------------
  <stem>_mask.png    : インスタンス ID マスク (uint8, 0=background, 1..N=instance)
  <stem>_overlay.png : 元画像 + 半透明インスタンスカラーオーバーレイ

Usage:
    python predict04.py <input_dir> [--weights <path>] [--output <dir>]
                        [--model_id <str>] [--threshold <float>]
"""

from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None   # suppress DecompressionBomb for large floor plans

import os
import argparse
import numpy as np
from pathlib import Path

import cv2
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

IMG_SIZE   = 1024   # 学習時と同じ Letterbox サイズ
THRESHOLD  = 0.5    # インスタンス confidence 閾値

if IS_KAGGLE:
    DEFAULT_WEIGHTS    = Path("/kaggle/working/exp04/best_model.pth")
    DEFAULT_OUTPUT_DIR = Path("/kaggle/working/predict04")
else:
    DEFAULT_WEIGHTS    = Path("/content/drive/MyDrive/mask2former_exp04/best_model.pth")
    DEFAULT_OUTPUT_DIR = Path("/content/drive/MyDrive/mask2former_predict04")

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
# Post-processing CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# モルフォロジー変換のカーネルサイズ (奇数推奨)
MORPH_OPENING_KERNEL_SIZE = 5   # Opening (ノイズ除去) のカーネルサイズ
MORPH_CLOSING_KERNEL_SIZE = 11  # Closing (穴埋め)   のカーネルサイズ
# カーネル形状: cv2.MORPH_RECT / cv2.MORPH_ELLIPSE / cv2.MORPH_CROSS
MORPH_KERNEL_SHAPE        = cv2.MORPH_ELLIPSE
# 各操作の反復回数
MORPH_OPENING_ITERATIONS  = 1
MORPH_CLOSING_ITERATIONS  = 1

# ─────────────────────────────────────────────────────────────────────────────
# Post-processing CONFIG (overlap resolution)
# ─────────────────────────────────────────────────────────────────────────────
# True  : 複数マスクが重なるピクセルを面積最大のインスタンスだけに割り当てる
# False : 重複をそのまま残す (従来動作)
RESOLVE_OVERLAPS_BY_AREA = True

# ─────────────────────────────────────────────────────────────────────────────
# Post-processing CONFIG (area filtering)
# ─────────────────────────────────────────────────────────────────────────────
# 被り解消後のマスク面積がこの値 [px] 未満のインスタンスを除去する。
# 0 に設定するとフィルタを無効化する。
MIN_INSTANCE_AREA = 500


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
def preprocess(image_rgb: np.ndarray) -> tuple[torch.Tensor, int, int, int, int]:
    """
    Letterbox リサイズ: LongestMaxSize(1024) → PadIfNeeded(1024×1024) → 正規化

    train04.py の学習時前処理と同じ変換を適用する。
    PadIfNeeded のデフォルトは中央配置 (画像を 1024×1024 の中心に置き、
    パディングを上下・左右に均等に分配) であるため、逆変換時に必要な
    pad_top / pad_left を合わせて返す。

    Args:
        image_rgb : uint8 RGB (H, W, 3)

    Returns:
        tensor    : (1, 3, 1024, 1024) float32 テンソル (DEVICE 上)
        scaled_H  : Letterbox 後、パディング前の高さ
        scaled_W  : Letterbox 後、パディング前の幅
        pad_top   : PadIfNeeded で付与された上パディング量 (px)
        pad_left  : PadIfNeeded で付与された左パディング量 (px)
    """
    orig_H, orig_W = image_rgb.shape[:2]

    # スケール後サイズを計算 (パディング前)
    # LongestMaxSize と同じロジック: 長辺が IMG_SIZE になるようにスケール
    scale    = IMG_SIZE / max(orig_H, orig_W)
    scaled_H = round(orig_H * scale)
    scaled_W = round(orig_W * scale)

    # PadIfNeeded デフォルトの中央配置によるパディングオフセット
    # pad_h = IMG_SIZE - scaled_H  →  上 = pad_h // 2, 下 = pad_h - pad_h // 2
    pad_top  = (IMG_SIZE - scaled_H) // 2
    pad_left = (IMG_SIZE - scaled_W) // 2

    # albumentations による Letterbox 変換
    letterbox = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=0,   # cv2.BORDER_CONSTANT
            value=0,
        ),
    ])
    resized = letterbox(image=image_rgb)["image"]   # (1024, 1024, 3) uint8

    normed = _NORMALIZE(image=resized)["image"]                          # (1024, 1024, 3)
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0)   # (1, 3, 1024, 1024)

    return tensor.to(DEVICE), scaled_H, scaled_W, pad_top, pad_left


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
        1. Letterbox リサイズ (1024×1024) して forward
        2. class_queries_logits → softmax → no-object 以外の最大スコアを confidence に
        3. masks_queries_logits → sigmoid → bilinear で 1024×1024 にアップサンプリング
        4. Letterbox 逆変換: パディング除去 (pad_top:pad_top+scaled_H, pad_left:pad_left+scaled_W) → 元解像度にリサイズ
        5. threshold 以上のクエリのみ残す

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
    pixel_values, scaled_H, scaled_W, pad_top, pad_left = preprocess(image_rgb)   # (1, 3, 1024, 1024)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # ── クラス確率 ────────────────────────────────────────────────────────
    # class_queries_logits : (1, num_queries, num_labels + 1)
    #   最後の次元が no-object クラス
    class_logits = outputs.class_queries_logits[0]           # (Q, num_labels+1)
    class_probs  = class_logits.softmax(dim=-1)              # (Q, num_labels+1)
    scores, pred_labels = class_probs[:, :-1].max(dim=-1)    # (Q,)  no-object を除外

    # ── マスクを元解像度にアップサンプリング (Letterbox 逆変換) ──────────
    # masks_queries_logits : (1, Q, H_model/4, W_model/4) = (1, Q, 256, 256)
    mask_logits = outputs.masks_queries_logits[0]            # (Q, 256, 256)
    mask_probs  = mask_logits.sigmoid()                      # (Q, 256, 256)

    # Step 1: 1024×1024 (Letterbox 空間) にアップサンプリング
    mask_probs_lb = F.interpolate(
        mask_probs.unsqueeze(0),                             # (1, Q, 256, 256)
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0]                                                     # (Q, 1024, 1024)

    # Step 2: パディングを除去
    # PadIfNeeded のデフォルトは中央配置のため、上に pad_top・左に pad_left の
    # パディングが付与されている。画像コンテンツ領域を正確にクロップする。
    mask_probs_cropped = mask_probs_lb[
        :,
        pad_top : pad_top + scaled_H,
        pad_left: pad_left + scaled_W,
    ]                                                        # (Q, scaled_H, scaled_W)

    # Step 3: 元解像度にリサイズ
    mask_probs_full = F.interpolate(
        mask_probs_cropped.unsqueeze(0),                     # (1, Q, scaled_H, scaled_W)
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
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────
def apply_morphology(masks: np.ndarray) -> np.ndarray:
    """
    各インスタンスマスクに Opening → Closing のモルフォロジー変換を適用する。

    Opening (erosion → dilation): 細いノイズ・孤立点を除去する
    Closing (dilation → erosion): 内部の小さな穴・途切れを埋める

    カーネルサイズ・形状・反復回数は CONFIG セクションで変更できる。

    Args:
        masks : (N, H, W) bool ndarray

    Returns:
        processed_masks : (N, H, W) bool ndarray
    """
    kernel_open  = cv2.getStructuringElement(
        MORPH_KERNEL_SHAPE,
        (MORPH_OPENING_KERNEL_SIZE, MORPH_OPENING_KERNEL_SIZE),
    )
    kernel_close = cv2.getStructuringElement(
        MORPH_KERNEL_SHAPE,
        (MORPH_CLOSING_KERNEL_SIZE, MORPH_CLOSING_KERNEL_SIZE),
    )

    processed = []
    for m in masks:
        m_uint8 = m.astype(np.uint8) * 255
        m_uint8 = cv2.morphologyEx(
            m_uint8, cv2.MORPH_OPEN,  kernel_open,
            iterations=MORPH_OPENING_ITERATIONS,
        )
        m_uint8 = cv2.morphologyEx(
            m_uint8, cv2.MORPH_CLOSE, kernel_close,
            iterations=MORPH_CLOSING_ITERATIONS,
        )
        processed.append(m_uint8 > 0)

    return np.stack(processed, axis=0) if len(processed) > 0 else masks


def resolve_overlaps_by_area(masks: np.ndarray) -> np.ndarray:
    """
    複数のマスクが重なるピクセルに対してのみ、面積が最大のインスタンスを残す。
    重複していないピクセルは変更しない。

    Args:
        masks : (N, H, W) bool ndarray

    Returns:
        resolved : (N, H, W) bool ndarray
    """
    if len(masks) == 0:
        return masks

    N = len(masks)
    areas = masks.sum(axis=(1, 2))                                    # (N,) 各インスタンスの面積

    # 2つ以上のマスクが重なるピクセルを特定
    overlap = masks.sum(axis=0) > 1                                   # (H, W) bool
    if not overlap.any():
        return masks

    resolved = masks.copy()

    # area_map[i, y, x] = そのピクセルをマスク i がカバーしていれば areas[i]、していなければ 0
    area_map = masks.astype(np.float32) * areas[:, np.newaxis, np.newaxis]  # (N, H, W)
    best = area_map.argmax(axis=0)                                    # (H, W) 最大面積インスタンスの index

    # 重複ピクセルはいったん全マスクを False に
    resolved[:, overlap] = False

    # 最大面積インスタンスのみ True に戻す
    for i in range(N):
        resolved[i][overlap & (best == i)] = True

    return resolved


def filter_by_area(
    masks: np.ndarray,
    scores: list[float],
    labels: list[int],
    min_area: int,
) -> tuple[np.ndarray, list[float], list[int]]:
    """
    マスク面積が min_area px 未満のインスタンスを除去する。
    被り解消後の最終マスクに対して適用する。

    Args:
        masks    : (N, H, W) bool ndarray
        scores   : [N] confidence スコア
        labels   : [N] クラスラベル
        min_area : この値 [px] 未満のインスタンスを除去 (0 なら無効)

    Returns:
        filtered_masks  : (M, H, W) bool ndarray  (M <= N)
        filtered_scores : [M] float list
        filtered_labels : [M] int list
    """
    if min_area <= 0 or len(masks) == 0:
        return masks, scores, labels

    areas = masks.sum(axis=(1, 2))          # (N,) 各インスタンスのピクセル数
    keep  = areas >= min_area               # (N,) bool

    filtered_masks  = masks[keep]
    filtered_scores = [s for s, k in zip(scores, keep) if k]
    filtered_labels = [lb for lb, k in zip(labels, keep) if k]

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
        draw.text((cx - 1, cy - 1), label_text, fill=(0, 0, 0),      font=font)
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
            "(Letterbox resize to 1024×1024 → infer → restore to original resolution)"
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
    print(f"IMG size   : {IMG_SIZE}×{IMG_SIZE} (Letterbox)")
    print(f"Threshold  : {args.threshold}")
    print(f"Morph open : kernel={MORPH_OPENING_KERNEL_SIZE}, iter={MORPH_OPENING_ITERATIONS}")
    print(f"Morph close: kernel={MORPH_CLOSING_KERNEL_SIZE}, iter={MORPH_CLOSING_ITERATIONS}")
    print(f"Overlap res: {'enabled (largest area wins)' if RESOLVE_OVERLAPS_BY_AREA else 'disabled'}")
    print(f"Area filter: {f'>= {MIN_INSTANCE_AREA} px' if MIN_INSTANCE_AREA > 0 else 'disabled'}")

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

        # ── モルフォロジー後処理 ──────────────────────────────────────────
        if len(masks) > 0:
            masks = apply_morphology(masks)

        # ── 重複解消: 重なりピクセルを面積最大インスタンスに割り当て ──────
        if RESOLVE_OVERLAPS_BY_AREA and len(masks) > 0:
            masks = resolve_overlaps_by_area(masks)

        # ── 面積フィルタ: 被り解消後の小さいインスタンスを除去 ────────────
        if MIN_INSTANCE_AREA > 0 and len(masks) > 0:
            before = len(masks)
            masks, scores, labels = filter_by_area(masks, scores, labels, MIN_INSTANCE_AREA)
            removed = before - len(masks)
            if removed:
                print(f"  Area filter : removed {removed} instance(s) (< {MIN_INSTANCE_AREA} px)")

        save_results(image_rgb, masks, scores, output_dir, stem=img_path.stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
