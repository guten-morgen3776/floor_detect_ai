"""
cubicasa_segment/ の images と masks を 5 件サンプリングして重ね合わせ確認。
マスク領域を薄い赤で表示する。
"""

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATASET_DIR = Path("/Users/aokitenju/floor_detect_ai-1/cubicasa_segment")
OUTPUT_DIR  = Path("/Users/aokitenju/floor_detect_ai-1/scripts/mask_overlay_check")
NUM_SAMPLES = 5
SEED        = 42
ALPHA       = 0.35          # 赤マスクの透明度
MASK_COLOR  = (80, 80, 255) # BGR: 薄い赤


def overlay_mask(img_path: Path, msk_path: Path) -> np.ndarray:
    img  = cv2.imread(str(img_path))
    mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)

    colored = img.copy()
    roi = mask == 255
    colored[roi] = MASK_COLOR

    result = cv2.addWeighted(colored, ALPHA, img, 1 - ALPHA, 0)
    return result


def collect_pairs():
    pairs = []
    for split in ("train", "val"):
        img_dir = DATASET_DIR / "images" / split
        msk_dir = DATASET_DIR / "masks"  / split
        for img_path in img_dir.glob("*.png"):
            msk_path = msk_dir / img_path.name
            if msk_path.exists():
                pairs.append((img_path, msk_path))
    return pairs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs()
    print(f"対応ペア数: {len(pairs)}")

    random.seed(SEED)
    samples = random.sample(pairs, min(NUM_SAMPLES, len(pairs)))

    _, axes = plt.subplots(1, NUM_SAMPLES, figsize=(6 * NUM_SAMPLES, 6))
    if NUM_SAMPLES == 1:
        axes = [axes]

    for ax, (img_path, msk_path) in zip(axes, samples):
        result_bgr = overlay_mask(img_path, msk_path)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        out_path = OUTPUT_DIR / f"mask_overlay_{img_path.stem}.png"
        cv2.imwrite(str(out_path), result_bgr)
        print(f"保存: {out_path}")

        ax.imshow(result_rgb)
        ax.set_title(img_path.stem, fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    summary_path = OUTPUT_DIR / "mask_overlay_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"\n一覧画像: {summary_path}")
    plt.show()


if __name__ == "__main__":
    main()
