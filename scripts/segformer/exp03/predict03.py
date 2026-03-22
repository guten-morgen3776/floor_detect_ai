"""
Sliding-window inference for the SegFormer binary segmentation model
trained in train03.py.

Usage:
    python predict03.py <input_dir> <output_dir> [--model <model_path>]

Outputs (saved to <output_dir>):
    mask.png    : binary mask  (room=255, background=black=0)
    overlay.png : original image with room region overlaid in semi-transparent red
"""

from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # prevent DecompressionBomb crash on huge floor plans

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
MODEL_ID   = "nvidia/segformer-b2-finetuned-ade-512-512"
NUM_LABELS = 2

if IS_KAGGLE:
    DEFAULT_MODEL_PATH = Path("/kaggle/working/exp03/best_model.pth")
    DEFAULT_OUTPUT_DIR = Path("/kaggle/working/predict03")
else:
    DEFAULT_MODEL_PATH = Path("/content/drive/MyDrive/exp03/best_model.pth")
    DEFAULT_OUTPUT_DIR = Path("/content/drive/MyDrive/predict03")

PATCH_SIZE = 512
STRIDE     = 384

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Val-time preprocessing (identical to train03.py)
# ─────────────────────────────────────────────
VAL_TRANSFORM = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def build_model(model_path: Path) -> nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        id2label={0: "background", 1: "room"},
        label2id={"background": 0, "room": 1},
    )
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def make_hann_window(size: int) -> np.ndarray:
    """2D Hann window of shape (size, size), values in [0, 1]."""
    w1d = np.hanning(size)           # (size,)
    return np.outer(w1d, w1d)        # (size, size)


def predict(model: nn.Module, image_rgb: np.ndarray) -> np.ndarray:
    """
    Full-image sliding-window inference.

    Args:
        model      : loaded SegFormer model (eval mode).
        image_rgb  : uint8 RGB image (H, W, 3).

    Returns:
        mask (H, W) binary uint8 array — 255=room, 0=background.
    """
    H, W = image_rgb.shape[:2]

    # ── 1. Pad image so patches reach every corner ────────────────────────────
    # Compute padded size as the smallest value ≥ original that fits an integer
    # number of strides (with PATCH_SIZE hanging over the last stride).
    def padded_dim(dim: int) -> int:
        if dim <= PATCH_SIZE:
            return PATCH_SIZE
        n = int(np.ceil((dim - PATCH_SIZE) / STRIDE))
        return PATCH_SIZE + n * STRIDE

    H_pad = padded_dim(H)
    W_pad = padded_dim(W)
    pad_b = H_pad - H   # bottom padding
    pad_r = W_pad - W   # right padding

    img_pad = np.pad(
        image_rgb,
        ((0, pad_b), (0, pad_r), (0, 0)),
        mode="reflect",
    )  # (H_pad, W_pad, 3)

    # ── 2. Prepare accumulation arrays ───────────────────────────────────────
    # logit_sum : accumulated weighted room-class logits
    # weight_sum: accumulated Hann window weights (denominator)
    logit_sum  = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_sum = np.zeros((H_pad, W_pad), dtype=np.float32)

    hann2d = make_hann_window(PATCH_SIZE).astype(np.float32)  # (512, 512)

    # ── 3. Sliding window loop ────────────────────────────────────────────────
    y_starts = list(range(0, H_pad - PATCH_SIZE + 1, STRIDE))
    x_starts = list(range(0, W_pad - PATCH_SIZE + 1, STRIDE))

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                patch = img_pad[y : y + PATCH_SIZE, x : x + PATCH_SIZE]  # (512,512,3)

                # Val-time normalise → tensor
                aug    = VAL_TRANSFORM(image=patch)
                tensor = aug["image"].unsqueeze(0).to(DEVICE)             # (1,3,512,512)

                outputs = model(pixel_values=tensor)
                logits  = outputs.logits                                   # (1,2,128,128)

                upsampled = nn.functional.interpolate(
                    logits,
                    size=(PATCH_SIZE, PATCH_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )  # (1,2,512,512)

                room_logit = upsampled[0, 1].cpu().numpy()  # (512,512)

                logit_sum [y : y + PATCH_SIZE, x : x + PATCH_SIZE] += room_logit * hann2d
                weight_sum[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += hann2d

    # ── 4. Average and threshold ──────────────────────────────────────────────
    # Avoid division by zero (shouldn't happen with valid padding, but be safe)
    avg_logit = np.where(weight_sum > 0, logit_sum / weight_sum, 0.0)

    # Threshold at logit=0 (equivalent to probability=0.5)
    pred_full = (avg_logit > 0).astype(np.uint8)  # (H_pad, W_pad)

    # ── 5. Crop back to original size ─────────────────────────────────────────
    pred = pred_full[:H, :W]  # (H, W)

    return (pred * 255).astype(np.uint8)


def save_results(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    """Save binary mask and overlay image."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── ① Binary mask ────────────────────────────────────────────────────────
    mask_path = output_dir / f"{stem}_mask.png"
    Image.fromarray(mask).save(mask_path)
    print(f"Saved mask    → {mask_path}")

    # ── ② Overlay ────────────────────────────────────────────────────────────
    overlay = image_rgb.copy().astype(np.float32)
    room_area = mask > 127  # boolean (H, W)

    alpha = 0.4
    # Paint room pixels with semi-transparent red (R=255, G=0, B=0)
    overlay[room_area, 0] = overlay[room_area, 0] * (1 - alpha) + 255 * alpha
    overlay[room_area, 1] = overlay[room_area, 1] * (1 - alpha)
    overlay[room_area, 2] = overlay[room_area, 2] * (1 - alpha)
    overlay = overlay.clip(0, 255).astype(np.uint8)

    overlay_path = output_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    print(f"Saved overlay → {overlay_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sliding-window inference with SegFormer (train03 weights)"
    )
    parser.add_argument("input_dir",  type=str, help="Directory containing input floor plan images")
    parser.add_argument("output_dir", type=str, help="Directory to save results")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to best_model.pth (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    print(f"Device     : {DEVICE}")
    print(f"Model      : {model_path}")
    print(f"Input dir  : {input_dir}  ({len(image_paths)} images)")
    print(f"Output dir : {output_dir}")
    print(f"Patch size : {PATCH_SIZE}  Stride: {STRIDE}")

    # ── Load model (once) ─────────────────────────────────────────────────────
    print("Loading model…")
    model = build_model(model_path)

    # ── Process each image ────────────────────────────────────────────────────
    for i, input_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {input_path.name}")
        image_rgb = np.array(Image.open(input_path).convert("RGB"))
        H, W = image_rgb.shape[:2]
        print(f"  Image size : {W}×{H}")

        print("  Running sliding-window inference…")
        mask = predict(model, image_rgb)

        save_results(image_rgb, mask, output_dir, stem=input_path.stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
