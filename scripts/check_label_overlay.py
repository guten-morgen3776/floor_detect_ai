"""
images と labels の対応を視覚的に確認するスクリプト。
labels 内の正規化座標 (0~1) を元画像サイズに戻してポリゴンを重ね合わせる。
"""

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATASET_DIR = Path("/Users/aokitenju/floor_detect_ai-1/my_cubicasa_dataset")
OUTPUT_DIR = Path("/Users/aokitenju/floor_detect_ai-1/scripts/overlay_check")
NUM_SAMPLES = 5
SEED = 42

# ポリゴンごとに色を変える（視認性向上）
COLORS = [
    (255, 80,  80),
    (80,  200, 80),
    (80,  80,  255),
    (255, 200, 0),
    (0,   200, 255),
    (200, 0,   200),
    (255, 140, 0),
    (0,   180, 180),
]


def parse_label(label_path: Path):
    """YOLOv8-seg フォーマットのラベルを読み込む。
    各行: class_id x1 y1 x2 y2 ... (正規化座標)
    戻り値: list of (class_id, [(x,y), ...])
    """
    polygons = []
    with open(label_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            class_id = int(tokens[0])
            coords = list(map(float, tokens[1:]))
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            polygons.append((class_id, points))
    return polygons


def draw_overlay(image_path: Path, label_path: Path) -> np.ndarray:
    """画像にポリゴンを重ね描きした BGR 配列を返す。"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")

    h, w = img.shape[:2]
    overlay = img.copy()

    polygons = parse_label(label_path)
    for idx, (class_id, points) in enumerate(polygons):
        # 正規化座標 → ピクセル座標
        pts = np.array([(int(x * w), int(y * h)) for x, y in points], dtype=np.int32)
        color = COLORS[idx % len(COLORS)]

        # 半透明塗りつぶし
        cv2.fillPoly(overlay, [pts], color)

    # 元画像と合成（alpha blend）
    result = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # 輪郭線を上書き
    for idx, (class_id, points) in enumerate(polygons):
        pts = np.array([(int(x * w), int(y * h)) for x, y in points], dtype=np.int32)
        color = COLORS[idx % len(COLORS)]
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)

    return result


def collect_pairs():
    """train / val 両方から (image_path, label_path) ペアを収集。"""
    pairs = []
    for split in ("train", "val"):
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        for img_path in img_dir.glob("*.png"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
    return pairs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs()
    print(f"対応ペア数: {len(pairs)}")

    random.seed(SEED)
    samples = random.sample(pairs, min(NUM_SAMPLES, len(pairs)))

    fig, axes = plt.subplots(1, NUM_SAMPLES, figsize=(6 * NUM_SAMPLES, 6))
    if NUM_SAMPLES == 1:
        axes = [axes]

    for ax, (img_path, lbl_path) in zip(axes, samples):
        result_bgr = draw_overlay(img_path, lbl_path)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        # 個別ファイル保存
        out_path = OUTPUT_DIR / f"overlay_{img_path.stem}.png"
        cv2.imwrite(str(out_path), result_bgr)
        print(f"保存: {out_path}")

        n_polys = len(parse_label(lbl_path))
        ax.imshow(result_rgb)
        ax.set_title(f"{img_path.stem}\n({n_polys} polygons)", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    summary_path = OUTPUT_DIR / "overlay_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"\n一覧画像を保存: {summary_path}")
    plt.show()


if __name__ == "__main__":
    main()
