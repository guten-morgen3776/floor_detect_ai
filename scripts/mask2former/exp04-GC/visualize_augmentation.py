"""
Visualize train augmentation from train04.py on 10 training images.
Saves original + augmented side-by-side into augmentation_test/.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import albumentations as A
from albumentations import ImageOnlyTransform


# ─────────────────────────────────────────────
# Paths (local)
# ─────────────────────────────────────────────
DATA_ROOT  = Path(os.environ.get("KAGGLE_DATA_ROOT", "")) / "cubicasa_mask2former" \
             if os.environ.get("KAGGLE_DATA_ROOT") else \
             Path.home() / "floor_detect_ai-1" / "cubicasa_mask2former"

IMG_DIR    = DATA_ROOT / "images" / "train"
OUTPUT_DIR = Path(__file__).parent / "augmentation_test"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE   = 1024
MIN_MASK_PX = 64
NUM_IMAGES  = 10
NUM_AUG_PER_IMAGE = 3   # 1枚につき何パターン保存するか


# ─────────────────────────────────────────────
# Custom transform (train04.py と同一)
# ─────────────────────────────────────────────
class RandomDashedLines(ImageOnlyTransform):
    def __init__(
        self,
        num_lines=(1, 4),
        color=(160, 160, 160),
        thickness=1,
        dash_len=10,
        gap_len=6,
        p=0.3,
    ):
        super().__init__(p=p)
        self.num_lines = num_lines
        self.color     = color
        self.thickness = thickness
        self.dash_len  = dash_len
        self.gap_len   = gap_len

    def apply(self, img, **_params):
        img  = img.copy()
        H, W = img.shape[:2]
        n    = np.random.randint(self.num_lines[0], self.num_lines[1] + 1)
        step = self.dash_len + self.gap_len
        for _ in range(n):
            if np.random.rand() < 0.5:
                y  = np.random.randint(0, H)
                x0 = np.random.randint(0, W)
                x1 = np.random.randint(x0, W)
                x  = x0
                while x < x1:
                    xe = min(x + self.dash_len, x1)
                    cv2.line(img, (x, y), (xe, y), self.color, self.thickness)
                    x += step
            else:
                x  = np.random.randint(0, W)
                y0 = np.random.randint(0, H)
                y1 = np.random.randint(y0, H)
                y  = y0
                while y < y1:
                    ye = min(y + self.dash_len, y1)
                    cv2.line(img, (x, y), (x, ye), self.color, self.thickness)
                    y += step
        return img

    def get_transform_init_args_dict(self):
        return {
            "num_lines": self.num_lines,
            "color":     self.color,
            "thickness": self.thickness,
            "dash_len":  self.dash_len,
            "gap_len":   self.gap_len,
        }


def get_train_transform() -> A.Compose:
    """train04.py の train 用パイプライン（Normalize は除く）"""
    letterbox = [
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=0,
            fill=0,
            fill_mask=0,
        ),
    ]
    return A.Compose([
        *letterbox,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
        A.GaussNoise(std_range=(5.0 / 255.0, 30.0 / 255.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4),
                        hole_height_range=(8, 32),
                        hole_width_range=(8, 32),
                        fill=0, p=0.2),
        RandomDashedLines(num_lines=(1, 4), color=(160, 160, 160),
                          thickness=1, dash_len=10, gap_len=6, p=0.3),
        # Normalize は視覚確認には不要なので省略
    ])


def letterbox_only() -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                      border_mode=0, fill=0),
    ])


def save_comparison(orig: np.ndarray, augs: list[np.ndarray], out_path: Path):
    """orig と複数の aug を横並びにして保存。"""
    imgs = [orig] + augs
    # 全て IMG_SIZE x IMG_SIZE になっているはずだが念のため揃える
    imgs = [cv2.resize(i, (IMG_SIZE, IMG_SIZE)) for i in imgs]

    # ラベルテキストを貼る
    labeled = []
    for k, im in enumerate(imgs):
        canvas = im.copy()
        label  = "original" if k == 0 else f"aug_{k}"
        cv2.putText(canvas, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
        labeled.append(canvas)

    combined = np.concatenate(labeled, axis=1)
    cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def main():
    img_paths = sorted(IMG_DIR.glob("*.png"))[:NUM_IMAGES]
    if not img_paths:
        img_paths = sorted(IMG_DIR.glob("*.jpg"))[:NUM_IMAGES]
    if not img_paths:
        print(f"No images found in {IMG_DIR}")
        sys.exit(1)

    print(f"Found {len(img_paths)} images → saving to {OUTPUT_DIR}")

    lb   = letterbox_only()
    aug  = get_train_transform()

    for img_path in img_paths:
        img_rgb = np.array(Image.open(img_path).convert("RGB"))

        # オリジナルをリサイズ（letterbox のみ）
        orig_resized = lb(image=img_rgb)["image"]

        # 複数パターンの augmentation を生成
        aug_results = []
        for _ in range(NUM_AUG_PER_IMAGE):
            result = aug(image=img_rgb)
            aug_results.append(result["image"])

        out_name = img_path.stem + "_aug.png"
        out_path = OUTPUT_DIR / out_name
        save_comparison(orig_resized, aug_results, out_path)
        print(f"  Saved: {out_name}")

    print("Done.")


if __name__ == "__main__":
    main()
