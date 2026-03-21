"""
SegFormer inference — loads best_model.pth and runs prediction on target images.

Usage:
    python predict01.py --input /path/to/images --output /path/to/output

Input directory should contain .png files.
Output directory will contain binary mask PNGs (0=background, 255=room).
"""

import argparse
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Config (must match train01.py) ────────────────────────────────────────────
MODEL_ID   = "nvidia/segformer-b3-finetuned-ade-512-512"
SCRIPT_DIR = Path(__file__).resolve().parent
BEST_MODEL = SCRIPT_DIR / "best_model.pth"
IMG_SIZE   = 512
NUM_LABELS = 2
BATCH_SIZE = 4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transform (same as val transform in train01.py) ───────────────────────────
TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(weights_path: Path) -> torch.nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        id2label={0: "background", 1: "room"},
        label2id={"background": 0, "room": 1},
    )
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded weights: {weights_path}")
    return model


# ── Single image prediction ────────────────────────────────────────────────────
def predict_image(model: torch.nn.Module, img_path: Path) -> np.ndarray:
    """Returns a binary mask array (H_orig x W_orig), values 0 or 1."""
    orig = Image.open(img_path).convert("RGB")
    orig_size = (orig.height, orig.width)   # (H, W)

    img_np = np.array(orig)
    aug    = TRANSFORM(image=img_np)
    tensor = aug["image"].unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    with torch.no_grad():
        outputs  = model(pixel_values=tensor)
        logits   = outputs.logits  # (1, num_labels, H/4, W/4)
        upsampled = nn.functional.interpolate(
            logits,
            size=orig_size,
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

    return pred.astype(np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SegFormer inference with best_model.pth")
    parser.add_argument("--input",   type=str, required=True,
                        help="Directory containing input .png images")
    parser.add_argument("--output",  type=str, required=True,
                        help="Directory to save predicted mask PNGs")
    parser.add_argument("--weights", type=str, default=str(BEST_MODEL),
                        help=f"Path to model weights (default: {BEST_MODEL})")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    weights    = Path(args.weights)

    assert input_dir.exists(),  f"Input directory not found: {input_dir}"
    assert weights.exists(),    f"Weights file not found: {weights}"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.png"))
    if not image_paths:
        image_paths = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG/JPG images found in {input_dir}")

    print(f"Device      : {DEVICE}")
    print(f"Input dir   : {input_dir}  ({len(image_paths)} images)")
    print(f"Output dir  : {output_dir}")
    print(f"Weights     : {weights}")
    print()

    model = build_model(weights)

    t_start = time.time()
    for img_path in tqdm(image_paths, desc="Predicting", unit="img"):
        mask = predict_image(model, img_path)

        # Save as grayscale PNG: 0=background, 255=room
        out_mask = (mask * 255).astype(np.uint8)
        out_path = output_dir / img_path.name
        Image.fromarray(out_mask, mode="L").save(out_path)

    elapsed = time.time() - t_start
    per_img = elapsed / len(image_paths)
    print(f"\nDone. {len(image_paths)} images in {elapsed:.1f}s ({per_img:.2f}s/img)")
    print(f"Masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
