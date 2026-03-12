#!/usr/bin/env python3
"""
学習済み SegFormer を用いて間取り図画像から「壁」を予測する推論スクリプト。

・学習済みモデルは train_segformer.py の CONFIG['output_dir'] / 'best_model' を参照。
・推論対象: cubicasa5k_processed/test/images
・前処理: soft_threshold_preprocess（画像読み込み直後）、Normalize + ToTensorV2 のみ（Augmentation なし）
・出力: マスク画像（uint8）と、元画像＋予測壁（赤半透明）の重ね合わせ画像。

依存: torch, transformers, albumentations, opencv-python-headless, tqdm
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm


# =============================================================================
# パス・パラメータ設定（変更時はここを編集）
# =============================================================================
CONFIG = {
    # 学習済みモデルディレクトリ（train_segformer.py の output_dir / "best_model" または "final_model"）
    "model_path": Path(__file__).resolve().parents[1] / "output" / "segformer_wall" / "best_model",
    # 推論対象: cubicasa5k_processed の test セット
    "data_root": Path(__file__).resolve().parents[1] / "cubicasa5k_processed",
    "test_split": "test",
    # 推論結果の保存先
    "output_dir": Path(__file__).resolve().parents[1] / "output" / "segformer_predictions",
    "save_masks": True,   # マスク画像（uint8）を保存するか
    "save_overlays": True,  # 元画像＋予測壁の重ね合わせを保存するか
    # 前処理（train_segformer.py と同一）
    "soft_threshold": 200,
    "image_size": (512, 512),
    # 推論
    "THRESHOLD": 0.5,  # sigmoid 出力がこの値以上なら壁(255)、未満なら背景(0)
    "device": None,    # None なら auto (cuda/cpu)
}


# -----------------------------------------------------------------------------
# 薄い線を消す前処理（train_segformer.py と同一）
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
# 推論用変換（Resize + Normalize + ToTensorV2 のみ、Augmentation なし）
# -----------------------------------------------------------------------------
def get_inference_transforms(image_size):
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
# リサイズ（拡大）→ sigmoid → 閾値 → uint8 マスク (255=壁, 0=背景)
# -----------------------------------------------------------------------------
def logits_to_mask(logits, threshold, original_shape):
    """
    logits: (1, 1, H, W) or (1, H, W) のテンソル
    threshold: この値以上を壁(255)とする
    original_shape: (H_orig, W_orig) に先にリサイズしてから sigmoid / 閾値 を適用
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)  # (1, H, W)
    # 先に元解像度へリサイズ（拡大）
    if logits.shape[1:] != original_shape:
        logits = torch.nn.functional.interpolate(
            logits.unsqueeze(1),
            size=original_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    probs = torch.sigmoid(logits)
    mask_np = (probs.cpu().numpy() >= threshold).astype(np.uint8)
    mask_np = (mask_np * 255).astype(np.uint8)  # 1 -> 255
    return mask_np.squeeze(0)


# -----------------------------------------------------------------------------
# 元画像に予測壁を赤半透明で重ねた画像を生成
# -----------------------------------------------------------------------------
def create_overlay(original_bgr, mask_uint8, alpha=0.5):
    """
    original_bgr: (H, W, 3) BGR, 元画像（前処理後で可）
    mask_uint8: (H, W) 0 or 255
    alpha: 赤の重ね合わせの強さ (0~1)
    """
    overlay = original_bgr.copy()
    red = np.array([0, 0, 255], dtype=np.uint8)  # BGR で赤
    wall = (mask_uint8 > 0)
    overlay[wall] = (
        (1 - alpha) * overlay[wall].astype(np.float32) + alpha * red.astype(np.float32)
    ).astype(np.uint8)
    return overlay


# -----------------------------------------------------------------------------
# モデル読み込み
# -----------------------------------------------------------------------------
def load_model(model_path, num_labels=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(model_path),
        num_labels=num_labels,
    )
    model = model.to(device)
    model.eval()
    return model, device


def main():
    cfg = CONFIG.copy()
    cfg["data_root"] = Path(cfg["data_root"])
    cfg["model_path"] = Path(cfg["model_path"])
    cfg["output_dir"] = Path(cfg["output_dir"])

    if not cfg["model_path"].exists():
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません: {cfg['model_path']}\n"
            "train_segformer.py で学習後、CONFIG['model_path'] を確認してください。"
        )

    test_images_dir = cfg["data_root"] / cfg["test_split"] / "images"
    if not test_images_dir.exists():
        raise FileNotFoundError(f"推論対象ディレクトリがありません: {test_images_dir}")

    image_paths = sorted(test_images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"画像がありません: {test_images_dir}")

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    if cfg["save_masks"]:
        (cfg["output_dir"] / "masks").mkdir(parents=True, exist_ok=True)
    if cfg["save_overlays"]:
        (cfg["output_dir"] / "overlays").mkdir(parents=True, exist_ok=True)

    device = cfg["device"]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model, device = load_model(
        cfg["model_path"],
        num_labels=1,
        device=device,
    )
    transform = get_inference_transforms(cfg["image_size"])
    image_size = cfg["image_size"]
    threshold = cfg["THRESHOLD"]
    soft_threshold = cfg["soft_threshold"]

    for image_path in tqdm(image_paths, desc="Predict"):
        # 画像読み込み直後に soft_threshold_preprocess を適用
        image_bgr = soft_threshold_preprocess(str(image_path), threshold=soft_threshold)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_bgr.shape[:2]

        # 推論用変換（Resize + Normalize + ToTensorV2 のみ）
        transformed = transform(image=image_rgb)
        image_tensor = transformed["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

        with torch.no_grad():
            outputs = model(pixel_values=image_tensor)
            logits = outputs.logits  # (1, 1, h, w) など

        # リサイズ（拡大）→ sigmoid → THRESHOLD で 255/0 の uint8 マスク
        mask_uint8 = logits_to_mask(
            logits,
            threshold=threshold,
            original_shape=(orig_h, orig_w),
        )

        # マスク保存
        if cfg["save_masks"]:
            mask_path = cfg["output_dir"] / "masks" / (image_path.stem + "_pred_mask.png")
            cv2.imwrite(str(mask_path), mask_uint8)

        # 重ね合わせ: 元画像 + 予測壁を赤半透明
        if cfg["save_overlays"]:
            overlay = create_overlay(image_bgr, mask_uint8, alpha=0.5)
            overlay_path = cfg["output_dir"] / "overlays" / (image_path.stem + "_overlay.png")
            cv2.imwrite(str(overlay_path), overlay)

    print(f"推論完了: {len(image_paths)} 枚")
    print(f"出力先: {cfg['output_dir']}")
    if cfg["save_masks"]:
        print(f"  マスク: {cfg['output_dir'] / 'masks'}")
    if cfg["save_overlays"]:
        print(f"  重ね合わせ: {cfg['output_dir'] / 'overlays'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SegFormer 壁セグメンテーション推論")
    p.add_argument("--model-path", type=Path, default=None, help="学習済みモデルディレクトリ")
    p.add_argument("--data-root", type=Path, default=None, help="cubicasa5k_processed のパス")
    p.add_argument("--output-dir", type=Path, default=None, help="推論結果の保存先")
    p.add_argument("--threshold", type=float, default=None, help="壁判定の閾値 (0~1)")
    args = p.parse_args()
    if args.model_path is not None:
        CONFIG["model_path"] = Path(args.model_path)
    if args.data_root is not None:
        CONFIG["data_root"] = Path(args.data_root)
    if args.output_dir is not None:
        CONFIG["output_dir"] = Path(args.output_dir)
    if args.threshold is not None:
        CONFIG["THRESHOLD"] = args.threshold
    main()
