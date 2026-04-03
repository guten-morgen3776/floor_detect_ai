"""
boundary_patch.py  ―  BPR パッチ処理ユーティリティ

含まれる処理:
  - 境界ピクセル抽出 (dilate - erode)
  - Dense sliding window によるパッチ候補生成
  - NMS によるパッチフィルタリング
  - IoU 計算 (単体 / バッチ)
  - 精緻化済みパッチの再統合 (reassemble)
"""

import cv2
import numpy as np

from config import DILATION_RADIUS, PATCH_SIZE, PATCH_STRIDE, INPUT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# 境界ピクセル抽出
# ─────────────────────────────────────────────────────────────────────────────

def extract_boundary_pixels(mask: np.ndarray,
                            dilation_radius: int = DILATION_RADIUS) -> np.ndarray:
    """
    バイナリマスクから境界ピクセルを抽出する。

    Parameters
    ----------
    mask : (H, W) bool or uint8 (0 or 1)
    dilation_radius : int
        膨張・収縮カーネルの半径 (px)

    Returns
    -------
    boundary_map : (H, W) bool
    """
    k = 2 * dilation_radius + 1
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m        = mask.astype(np.uint8)
    dilated  = cv2.dilate(m, kernel)
    eroded   = cv2.erode(m, kernel)
    return (dilated - eroded).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# パッチ候補生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_patch_candidates(boundary_map: np.ndarray,
                              patch_size: int = PATCH_SIZE,
                              stride: int = PATCH_STRIDE) -> list[dict]:
    """
    境界ピクセル上に Dense sliding window でパッチ候補を生成する。

    Parameters
    ----------
    boundary_map : (H, W) bool
    patch_size   : int  切り出しパッチの辺長 (px, 正方形)
    stride       : int  境界ピクセルのサンプリングストライド

    Returns
    -------
    candidates : list of {'x1', 'y1', 'x2', 'y2', 'score'}
    """
    H, W   = boundary_map.shape
    half   = patch_size // 2
    bys, bxs = np.where(boundary_map)

    if len(bys) == 0:
        return []

    # 境界ピクセルを stride 間隔でサブサンプリング
    idx = np.arange(0, len(bys), max(1, stride // 4))
    bys, bxs = bys[idx], bxs[idx]

    candidates = []
    for cy, cx in zip(bys, bxs):
        x1 = int(max(0, cx - half))
        y1 = int(max(0, cy - half))
        x2 = int(min(W, x1 + patch_size))
        y2 = int(min(H, y1 + patch_size))

        # パッチが端にぶつかってサイズが縮んだ場合は始点を補正
        if x2 - x1 < patch_size:
            x1 = max(0, x2 - patch_size)
        if y2 - y1 < patch_size:
            y1 = max(0, y2 - patch_size)

        # スコア: パッチ内境界ピクセル密度 (NMS 用)
        patch_boundary = boundary_map[y1:y2, x1:x2]
        score = float(patch_boundary.mean())
        candidates.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score})

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────────────────────────────────────

def nms_patches(candidates: list[dict],
                iou_threshold: float = 0.25) -> list[dict]:
    """
    パッチ候補に NMS を適用して重複を削減する。

    Parameters
    ----------
    candidates    : list of {'x1', 'y1', 'x2', 'y2', 'score'}
    iou_threshold : float  IoU がこの値を超えるペアを除去

    Returns
    -------
    kept : list of dict (NMS 後のパッチ)
    """
    if not candidates:
        return []

    boxes  = np.array([[c["x1"], c["y1"], c["x2"], c["y2"]] for c in candidates],
                      dtype=np.float32)
    scores = np.array([c["score"] for c in candidates], dtype=np.float32)
    order  = scores.argsort()[::-1]
    kept   = []

    while order.size > 0:
        i = order[0]
        kept.append(candidates[i])
        if order.size == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = ((boxes[order[1:], 2] - boxes[order[1:], 0]) *
                  (boxes[order[1:], 3] - boxes[order[1:], 1]))
        iou = inter / (area_i + area_j - inter + 1e-6)

        order = order[1:][iou <= iou_threshold]

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# IoU 計算
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """1 対 1 のバイナリマスク IoU を計算する。"""
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / (float(union) + 1e-6)


def compute_iou_batch(pred: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """
    pred (H, W) と gt_masks (N, H, W) の間の IoU を一括計算する。

    Returns
    -------
    ious : (N,) float32
    """
    p    = pred.astype(bool)
    gts  = gt_masks.astype(bool)
    inter = (p[np.newaxis] & gts).sum(axis=(1, 2)).astype(np.float32)
    union = (p[np.newaxis] | gts).sum(axis=(1, 2)).astype(np.float32)
    return inter / (union + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# パッチ再統合
# ─────────────────────────────────────────────────────────────────────────────

def reassemble_patches(original_mask: np.ndarray,
                       refined_patches: list[dict],
                       input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    精緻化済みパッチを元マスクに統合する。

    Parameters
    ----------
    original_mask   : (H, W) bool or uint8
    refined_patches : list of {
        'box'      : (x1, y1, x2, y2),
        'fg_logit' : np.ndarray (input_size, input_size) float32  foreground のロジット
    }
    input_size : int  refinement network の入力サイズ

    Returns
    -------
    final_mask : (H, W) uint8  (0 or 1)
    """
    H, W = original_mask.shape

    logit_sum  = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    for patch in refined_patches:
        x1, y1, x2, y2 = patch["box"]
        ph, pw = y2 - y1, x2 - x1
        if ph <= 0 or pw <= 0:
            continue

        fg_logit_resized = cv2.resize(
            patch["fg_logit"].astype(np.float32),
            (pw, ph),
            interpolation=cv2.INTER_LINEAR,
        )
        logit_sum[y1:y2, x1:x2]  += fg_logit_resized
        weight_map[y1:y2, x1:x2] += 1.0

    patch_region = weight_map > 0
    avg_logits   = np.where(
        patch_region,
        logit_sum / (weight_map + 1e-6),
        -1e9,
    )

    # logit > 0 ⟺ sigmoid > 0.5 → foreground
    refined_binary = (avg_logits > 0).astype(np.uint8)

    final_mask = original_mask.astype(np.uint8).copy()
    final_mask[patch_region] = refined_binary[patch_region]
    return final_mask
