"""
step3_inference.py  ―  End-to-End 推論パイプライン

Mask2Former (Stage1) → BPR Refinement (Stage2) → 最終マスク保存

入力: 建築図面画像 (PNG/JPG)
出力:
  {stem}_mask.png    : インスタンス ID マスク (uint8, 0=bg, 1..N=instance)
  {stem}_overlay.png : オーバーレイ可視化

Usage:
    python step3_inference.py <input_dir> [--output <dir>]
    python step3_inference.py <input_dir> --nms_iou 0.55  # 高精度モード
    python step3_inference.py <input_dir> --no_refine      # Stage1 のみ (ablation 用)
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import cv2
import torch

import config
import stage1_utils as s1
from model          import LightweightRefineNet
from boundary_patch import (
    extract_boundary_pixels,
    generate_patch_candidates,
    nms_patches,
    reassemble_patches,
)


# ─────────────────────────────────────────────────────────────────────────────
# 正規化定数 (ImageNet, dataset.py と同一)
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# BPR による 1 インスタンスマスクの精緻化
# ─────────────────────────────────────────────────────────────────────────────

def refine_mask(mask: np.ndarray,
                image: np.ndarray,
                bpr_model: torch.nn.Module,
                device: torch.device,
                nms_iou: float = config.NMS_IOU_INFER,
                input_size: int = config.INPUT_SIZE) -> np.ndarray:
    """
    1 インスタンスのバイナリマスクを BPR モデルで精緻化する。

    Parameters
    ----------
    mask      : (H, W) bool
    image     : (H, W, 3) uint8 RGB
    bpr_model : eval モードの LightweightRefineNet
    device    : torch.device

    Returns
    -------
    refined : (H, W) uint8  (0 or 1)
    """
    boundary    = extract_boundary_pixels(mask, config.DILATION_RADIUS)
    candidates  = generate_patch_candidates(boundary, config.PATCH_SIZE, config.PATCH_STRIDE)
    patch_boxes = nms_patches(candidates, nms_iou)

    if not patch_boxes:
        return mask.astype(np.uint8)

    # ── パッチバッチ作成 ───────────────────────────────────────────────────
    inputs_list = []
    for box in patch_boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        img_p  = image[y1:y2, x1:x2]
        mask_p = mask[y1:y2, x1:x2].astype(np.float32)

        img_r  = cv2.resize(img_p,  (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask_p, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

        img_norm  = (img_r.astype(np.float32) / 255.0 - _MEAN) / _STD
        mask_norm = (mask_r - 0.5) / 0.5

        inp = np.concatenate([
            img_norm.transpose(2, 0, 1),
            mask_norm[np.newaxis],
        ], axis=0)
        inputs_list.append(inp)

    # ── BPR 推論 ─────────────────────────────────────────────────────────────
    batch = torch.from_numpy(np.stack(inputs_list)).float().to(device)
    with torch.no_grad():
        logits = bpr_model(batch)
    fg_logits = logits[:, 1, :, :].cpu().numpy()

    # ── パッチ再統合 ──────────────────────────────────────────────────────────
    refined_patches = [
        {
            "box":      (b["x1"], b["y1"], b["x2"], b["y2"]),
            "fg_logit": fg_logits[i],
        }
        for i, b in enumerate(patch_boxes)
    ]
    return reassemble_patches(mask, refined_patches, input_size)


# ─────────────────────────────────────────────────────────────────────────────
# 1 画像の全インスタンスを精緻化
# ─────────────────────────────────────────────────────────────────────────────

def refine_all_masks(masks: np.ndarray,
                     image: np.ndarray,
                     bpr_model: torch.nn.Module,
                     device: torch.device,
                     nms_iou: float = config.NMS_IOU_INFER) -> np.ndarray:
    """(N, H, W) bool の全インスタンスマスクを BPR で精緻化する。"""
    refined = [refine_mask(mask, image, bpr_model, device, nms_iou) for mask in masks]
    return np.stack(refined).astype(bool) if refined else masks


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    default_out = (
        "/kaggle/working/bpr_predict"
        if config.IS_KAGGLE
        else "/content/drive/MyDrive/bpr_predict"
    )
    p = argparse.ArgumentParser(description="Mask2Former + BPR の End-to-End 推論")
    p.add_argument("input_dir",         type=str,
                   help="入力画像ディレクトリ")
    p.add_argument("--stage1_weights",  type=str,
                   default=str(config.STAGE1_WEIGHTS),
                   help="Mask2Former 重みファイルパス")
    p.add_argument("--stage2_weights",  type=str,
                   default=str(config.BEST_MODEL),
                   help="BPR モデル重みファイルパス")
    p.add_argument("--model_id",        type=str,
                   default=config.STAGE1_MODEL_ID)
    p.add_argument("--output",          type=str,
                   default=default_out,
                   help="出力ディレクトリ")
    p.add_argument("--threshold",       type=float,
                   default=config.STAGE1_THRESHOLD,
                   help="Mask2Former confidence 閾値")
    p.add_argument("--nms_iou",         type=float,
                   default=config.NMS_IOU_INFER,
                   help="BPR NMS IoU 閾値 (0.25〜0.55, 高いほど精緻化パッチ増)")
    p.add_argument("--no_refine",       action="store_true",
                   help="BPR を使わず Stage1 のみで出力 (ablation 用)")
    return p.parse_args()


# ─────��───────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    device     = s1.DEVICE
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Mask2Former ─────────────────────────────────────────────────
    stage1_weights = Path(args.stage1_weights)
    if not stage1_weights.exists():
        raise FileNotFoundError(f"Stage1 重みが見つかりません: {stage1_weights}")

    print("Stage1 モデルをロード中...")
    stage1_model = s1.build_model(args.model_id, stage1_weights)
    print("Stage1 ロード完了")

    # ── Stage 2: BPR ─────────────────────────────────────────────────────────
    bpr_model = None
    if not args.no_refine:
        stage2_weights = Path(args.stage2_weights)
        if not stage2_weights.exists():
            raise FileNotFoundError(f"Stage2 重みが見つかりません: {stage2_weights}")
        print("Stage2 (BPR) モデルをロード中...")
        bpr_model = LightweightRefineNet(
            in_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
        )
        bpr_model.load_state_dict(torch.load(stage2_weights, map_location=device))
        bpr_model.to(device).eval()
        print("Stage2 ロード完了\n")

    # ── 画像一覧 ─────────────────────────────────────────────────────────────
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"画像が見つかりません: {input_dir}")

    print(f"入力画像数 : {len(image_paths)}")
    print(f"出力先     : {output_dir}")
    print(f"BPR NMS    : {args.nms_iou}")
    print()

    # ── 推論ループ ─────────────────────────────────────────────────────────────
    for img_path in tqdm(image_paths, desc="Inference"):
        image_rgb = np.array(Image.open(img_path).convert("RGB"))

        # Stage 1: Mask2Former
        masks, scores, labels = s1.predict_instances(
            stage1_model, image_rgb, args.threshold
        )

        if len(masks) == 0:
            s1.save_results(image_rgb, masks, scores, output_dir, img_path.stem)
            continue

        # Stage 1 後処理
        masks = s1.apply_morphology(masks)
        if s1.RESOLVE_OVERLAPS_BY_AREA:
            masks = s1.resolve_overlaps_by_area(masks)
        if s1.MIN_INSTANCE_AREA > 0:
            masks, scores, labels = s1.filter_by_area(
                masks, scores, labels, s1.MIN_INSTANCE_AREA
            )

        # Stage 2: BPR 精緻化
        if bpr_model is not None and len(masks) > 0:
            masks = refine_all_masks(masks, image_rgb, bpr_model, device, args.nms_iou)
            if s1.RESOLVE_OVERLAPS_BY_AREA:
                masks = s1.resolve_overlaps_by_area(masks)
            if s1.MIN_INSTANCE_AREA > 0:
                masks, scores, labels = s1.filter_by_area(
                    masks, scores, labels, s1.MIN_INSTANCE_AREA
                )

        s1.save_results(image_rgb, masks, scores, output_dir, img_path.stem)

    print("\n=== step3 完了 ===")


if __name__ == "__main__":
    main()
