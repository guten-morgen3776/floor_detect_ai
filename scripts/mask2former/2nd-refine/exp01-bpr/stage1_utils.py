"""
stage1_utils.py  ―  Mask2Former (Stage1) の推論・後処理ユーティリティ

predict04.py の処理をこのフォルダ内に自己完結させるために移植したもの。
step1_generate_pred_masks.py と step3_inference.py の両方から使用する。
"""

from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None   # 大きな間取り図の DecompressionBomb 警告を抑制

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

import config


# ─────────────────────────────────────────────────────────────────────────────
# Mask2Former 設定
# ─────────────────────────────────────────────────────────────────────────────
NUM_LABELS = 1
ID2LABEL   = {0: "room"}
LABEL2ID   = {"room": 0}

IMG_SIZE   = config.STAGE1_IMG_SIZE
THRESHOLD  = config.STAGE1_THRESHOLD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_NORMALIZE = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# インスタンス描画用カラーパレット (RGB, 最大 24 インスタンス)
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
# モルフォロジー後処理 CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MORPH_OPENING_KERNEL_SIZE  = 5
MORPH_CLOSING_KERNEL_SIZE  = 11
MORPH_KERNEL_SHAPE         = cv2.MORPH_ELLIPSE
MORPH_OPENING_ITERATIONS   = 1
MORPH_CLOSING_ITERATIONS   = 1

# 重複解消: True = 重複ピクセルを面積最大インスタンスに割り当てる
RESOLVE_OVERLAPS_BY_AREA = True

# 面積フィルタ: この値 [px] 未満のインスタンスを除去 (0 = 無効)
MIN_INSTANCE_AREA = 500


# ─────────────────────────────────────────────────────────────────────────────
# モデル構築
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_id: str,
    weights_path: Path,
) -> Mask2FormerForUniversalSegmentation:
    """
    学習時と同じ設定でモデルを構築し、保存済み state_dict をロードして返す。
    """
    cfg = Mask2FormerConfig.from_pretrained(
        model_id,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        config=cfg,
        ignore_mismatched_sizes=True,
    )
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 前処理
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    image_rgb: np.ndarray,
) -> tuple[torch.Tensor, int, int, int, int]:
    """
    Letterbox リサイズ: LongestMaxSize(IMG_SIZE) → PadIfNeeded(IMG_SIZE×IMG_SIZE) → 正規化

    Returns
    -------
    tensor    : (1, 3, IMG_SIZE, IMG_SIZE) float32 (DEVICE 上)
    scaled_H  : パディング前の高さ
    scaled_W  : パディング前の幅
    pad_top   : 上パディング量 (px)
    pad_left  : 左パディング量 (px)
    """
    orig_H, orig_W = image_rgb.shape[:2]

    scale    = IMG_SIZE / max(orig_H, orig_W)
    scaled_H = round(orig_H * scale)
    scaled_W = round(orig_W * scale)
    pad_top  = (IMG_SIZE - scaled_H) // 2
    pad_left = (IMG_SIZE - scaled_W) // 2

    letterbox = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=0,
            fill=0,           # albumentations >= 2.0 では value → fill
        ),
    ])
    resized = letterbox(image=image_rgb)["image"]
    normed  = _NORMALIZE(image=resized)["image"]
    tensor  = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0)

    return tensor.to(DEVICE), scaled_H, scaled_W, pad_top, pad_left


# ─────────────────────────────────────────────────────────────────────────────
# 推論
# ─────────────────────────────────────────────────────────────────────────────

def predict_instances(
    model: Mask2FormerForUniversalSegmentation,
    image_rgb: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[float], list[int]]:
    """
    元画像に対してインスタンスセグメンテーションを実行する。

    Returns
    -------
    masks  : (N, orig_H, orig_W) bool ndarray
    scores : [N] float list
    labels : [N] int list
    """
    orig_H, orig_W = image_rgb.shape[:2]
    pixel_values, scaled_H, scaled_W, pad_top, pad_left = preprocess(image_rgb)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    class_logits = outputs.class_queries_logits[0]
    class_probs  = class_logits.softmax(dim=-1)
    scores, pred_labels = class_probs[:, :-1].max(dim=-1)

    mask_logits    = outputs.masks_queries_logits[0]
    mask_probs     = mask_logits.sigmoid()
    del outputs  # 中間出力を即解放

    # ── Step1: 1024×1024 にアップサンプリング ─────────────────────────────────
    mask_probs_lb = F.interpolate(
        mask_probs.unsqueeze(0),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0]
    del mask_probs  # アップサンプリング後は不要

    # ── Step2: パディング除去 ──────────────────────────────────────────────────
    # .clone() でビューを切り、直後に mask_probs_lb を解放できるようにする
    mask_probs_cropped = mask_probs_lb[
        :,
        pad_top : pad_top + scaled_H,
        pad_left: pad_left + scaled_W,
    ].clone()
    del mask_probs_lb  # ビューを断ち切ったので解放可能

    # ── Step3: confidence フィルタを先に適用してクエリ数を削減 ─────────────────
    # 全クエリを元解像度にアップサンプリングするとメモリが爆発するため、
    # 閾値を超えたクエリのみを orig_H×orig_W にアップサンプリングする。
    keep = (scores >= threshold).cpu()

    filtered_scores = scores[keep].cpu().tolist()
    filtered_labels = pred_labels[keep].cpu().tolist()

    if keep.sum() == 0:
        del mask_probs_cropped
        torch.cuda.empty_cache()
        return np.zeros((0, orig_H, orig_W), dtype=bool), [], []

    # ── Step4: 残ったマスクだけ元解像度にアップサンプリング ──────────────────
    kept_probs = mask_probs_cropped[keep.to(mask_probs_cropped.device)]  # (N, sH, sW)
    del mask_probs_cropped  # フィルタ後は不要

    mask_probs_full = F.interpolate(
        kept_probs.unsqueeze(0),
        size=(orig_H, orig_W),
        mode="bilinear",
        align_corners=False,
    )[0]                                                                  # (N, H, W)
    del kept_probs  # アップサンプリング後は不要

    filtered_masks = (mask_probs_full > 0.5).cpu().numpy()               # (N, H, W) bool
    del mask_probs_full
    torch.cuda.empty_cache()

    return filtered_masks, filtered_scores, filtered_labels


# ─────────────────────────────────────────────────────────────────────────────
# 後処理
# ─────────────────────────────────────────────────────────────────────────────

def apply_morphology(masks: np.ndarray) -> np.ndarray:
    """各インスタンスマスクに Opening → Closing のモルフォロジー変換を適用する。"""
    kernel_open  = cv2.getStructuringElement(
        MORPH_KERNEL_SHAPE,
        (MORPH_OPENING_KERNEL_SIZE, MORPH_OPENING_KERNEL_SIZE),
    )
    kernel_close = cv2.getStructuringElement(
        MORPH_KERNEL_SHAPE,
        (MORPH_CLOSING_KERNEL_SIZE, MORPH_CLOSING_KERNEL_SIZE),
    )

    # リストを使わず事前割り当てして余分な (N,H,W) コピーを削減
    result = np.empty_like(masks)
    for i, m in enumerate(masks):
        m_uint8 = m.astype(np.uint8) * 255
        m_uint8 = cv2.morphologyEx(m_uint8, cv2.MORPH_OPEN,  kernel_open,
                                   iterations=MORPH_OPENING_ITERATIONS)
        m_uint8 = cv2.morphologyEx(m_uint8, cv2.MORPH_CLOSE, kernel_close,
                                   iterations=MORPH_CLOSING_ITERATIONS)
        result[i] = m_uint8 > 0

    return result


def resolve_overlaps_by_area(masks: np.ndarray) -> np.ndarray:
    """
    複数マスクが重なるピクセルを面積最大のインスタンスだけに割り当てる。
    重複していないピクセルは変更しない。
    """
    if len(masks) == 0:
        return masks

    areas    = masks.sum(axis=(1, 2))
    overlap  = masks.sum(axis=0) > 1
    if not overlap.any():
        return masks

    # (N,H,W) float32 の area_map を避け、(H,W) int32 の winner マップに置き換える
    # → ピーク RAM を N 分の 1 に削減（例: N=20 なら 960 MB → 48 MB）
    order  = np.argsort(areas)                                  # 面積昇順のインデックス
    winner = np.full(masks.shape[1:], -1, dtype=np.int32)      # (H, W) のみ
    for i in order:
        winner[overlap & masks[i]] = i                          # 大きい面積で上書き

    resolved = masks.copy()
    resolved[:, overlap] = False
    for i in range(len(masks)):
        resolved[i][overlap & (winner == i)] = True

    return resolved


def filter_by_area(
    masks: np.ndarray,
    scores: list[float],
    labels: list[int],
    min_area: int,
) -> tuple[np.ndarray, list[float], list[int]]:
    """min_area px 未満のインスタンスを除去する。"""
    if min_area <= 0 or len(masks) == 0:
        return masks, scores, labels

    areas = masks.sum(axis=(1, 2))
    keep  = areas >= min_area

    filtered_masks  = masks[keep]
    filtered_scores = [s  for s,  k in zip(scores, keep) if k]
    filtered_labels = [lb for lb, k in zip(labels, keep) if k]

    return filtered_masks, filtered_scores, filtered_labels


# ─────────────────────────────────────────────────────────────────────────────
# 可視化 & 保存
# ─────────────────────────────────────────────────────────────────────────────

def make_instance_id_mask(masks: np.ndarray, orig_H: int, orig_W: int) -> np.ndarray:
    """masks (N, H, W) bool → インスタンス ID マスク (H, W) uint8 (0=bg, 1..N=instance)"""
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
    """元画像にインスタンスマスクを半透明で重ねたオーバーレイ画像を返す。"""
    overlay = image_rgb.astype(np.float32).copy()

    for i, (m, score) in enumerate(zip(masks, scores)):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        overlay[m, 0] = overlay[m, 0] * (1 - alpha) + color[0] * alpha
        overlay[m, 1] = overlay[m, 1] * (1 - alpha) + color[1] * alpha
        overlay[m, 2] = overlay[m, 2] * (1 - alpha) + color[2] * alpha

    pil  = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()

    for i, (m, score) in enumerate(zip(masks, scores)):
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        cy, cx     = int(ys.mean()), int(xs.mean())
        label_text = f"#{i + 1} {score:.2f}"
        draw.text((cx - 1, cy - 1), label_text, fill=(0,   0,   0),   font=font)
        draw.text((cx,     cy    ), label_text, fill=(255, 255, 255), font=font)

    return np.array(pil)


def save_results(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    scores: list[float],
    output_dir: Path,
    stem: str,
) -> None:
    """インスタンス ID マスクとオーバーレイ画像を output_dir に保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_H, orig_W = image_rgb.shape[:2]

    id_mask   = make_instance_id_mask(masks, orig_H, orig_W)
    mask_path = output_dir / f"{stem}_mask.png"
    Image.fromarray(id_mask).save(mask_path)
    print(f"  Saved mask    → {mask_path}")

    overlay      = make_overlay(image_rgb, masks, scores) if len(masks) > 0 else image_rgb.copy()
    overlay_path = output_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    print(f"  Saved overlay → {overlay_path}")
