"""
壁マスク生成スクリプト
cubicasa_segment/masks の部屋マスクから壁マスクを生成する。

アルゴリズム:
  1. 元画像の明るいピクセル(白系)を連結成分で解析
  2. 画像端に接するor最大の白連結成分 → 外部(exterior)
  3. 壁 = 部屋でない & 外部でない

出力先: cubicasa_wall_masks/{train,val}/
"""

import os
import warnings
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # 大きな画像に対応

# ─────────────────────────────────────────────
# パス設定
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
IMG_DIR  = BASE_DIR / "cubicasa_segment" / "images"
MASK_DIR = BASE_DIR / "cubicasa_segment" / "masks"
OUT_DIR  = BASE_DIR / "cubicasa_wall_masks" / "masks"

SPLITS = ["train", "val"]
BRIGHT_THRESH = 200  # この値以上のRGB全チャンネルを「白/外部候補」とする


def generate_wall_mask(img_path: Path, mask_path: Path) -> np.ndarray:
    """
    部屋マスクと元画像から壁マスク(uint8, 0/255)を生成する。

    Parameters
    ----------
    img_path  : 元のRGB画像パス
    mask_path : 部屋マスクパス (grayscale, 255=room, 0=non-room)

    Returns
    -------
    wall_mask : np.ndarray shape=(H,W) dtype=uint8, 255=wall, 0=other
    """
    orig = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    r, g, b = orig[:, :, 0], orig[:, :, 1], orig[:, :, 2]
    h, w = orig.shape[:2]

    # 明るいピクセルマスク（白系）
    is_bright = (r > BRIGHT_THRESH) & (g > BRIGHT_THRESH) & (b > BRIGHT_THRESH)
    bright_u8 = is_bright.astype(np.uint8) * 255

    # 連結成分解析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_u8, connectivity=8)

    exterior = np.zeros((h, w), dtype=bool)

    if num_labels > 1:
        border_labels: set[int] = set()

        for lbl in range(1, num_labels):
            lx = stats[lbl, cv2.CC_STAT_LEFT]
            ly = stats[lbl, cv2.CC_STAT_TOP]
            lw = stats[lbl, cv2.CC_STAT_WIDTH]
            lh = stats[lbl, cv2.CC_STAT_HEIGHT]
            # 画像境界（±1ピクセル余裕）に接しているか
            if lx <= 1 or ly <= 1 or (lx + lw) >= w - 1 or (ly + lh) >= h - 1:
                border_labels.add(lbl)

        # 境界接触なし → 最大連結成分を外部とみなす（緑枠等のケース）
        if not border_labels:
            areas = [(stats[lbl, cv2.CC_STAT_AREA], lbl) for lbl in range(1, num_labels)]
            _, largest_lbl = max(areas)
            border_labels.add(largest_lbl)

        for lbl in border_labels:
            exterior |= (labels == lbl)

    # 壁 = 部屋でない & 外部でない
    wall = (mask == 0) & (~exterior)
    return (wall.astype(np.uint8) * 255)


def main():
    for split in SPLITS:
        img_split  = IMG_DIR  / split
        mask_split = MASK_DIR / split
        out_split  = OUT_DIR  / split
        out_split.mkdir(parents=True, exist_ok=True)

        fnames = sorted(os.listdir(mask_split))
        total = len(fnames)
        print(f"\n[{split}] {total} images")

        skip = 0
        for i, fname in enumerate(fnames):
            if not fname.endswith(".png"):
                continue

            img_path  = img_split  / fname
            mask_path = mask_split / fname
            out_path  = out_split  / fname

            if not img_path.exists():
                skip += 1
                continue

            try:
                wall_mask = generate_wall_mask(img_path, mask_path)
                Image.fromarray(wall_mask).save(out_path)
            except Exception as e:
                print(f"  ERROR {fname}: {e}")
                skip += 1
                continue

            if (i + 1) % 200 == 0 or (i + 1) == total:
                print(f"  {i+1}/{total} done")

        print(f"  Skipped: {skip}")

    print("\nDone. Wall masks saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
