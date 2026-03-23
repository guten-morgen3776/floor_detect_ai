"""
YOLO segmentation → COCO instance segmentation 変換スクリプト
Input : my_cubicasa_dataset/  (YOLOv8-seg format)
Output: cubicasa_mask2former/ (COCO instance segmentation format)
"""

import json
import os
import shutil
from pathlib import Path

from PIL import Image


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SRC_ROOT = Path("/Users/aokitenju/floor_detect_ai-1/my_cubicasa_dataset")
DST_ROOT = Path("/Users/aokitenju/floor_detect_ai-1/cubicasa_mask2former")

CATEGORIES = [{"id": 1, "name": "room", "supercategory": "floor"}]
SPLITS = ["train", "val"]


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
def polygon_area(xs: list[float], ys: list[float]) -> float:
    """Shoelace formula for polygon area."""
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def parse_yolo_seg_line(line: str, width: int, height: int):
    """
    YOLO-seg 1行をパース。
    Returns (category_id_0indexed, segmentation_pixels, bbox_xywh, area)
    segmentation_pixels: [x1, y1, x2, y2, ...] (pixel coords, flat list)
    """
    tokens = line.strip().split()
    class_id = int(tokens[0])
    coords = list(map(float, tokens[1:]))

    # 正規化座標 → ピクセル座標
    xs = [coords[i] * width for i in range(0, len(coords), 2)]
    ys = [coords[i] * height for i in range(1, len(coords), 2)]

    # 重複端点のクリーニング（最後の点が最初の点と同じ場合は除去）
    while len(xs) > 3 and abs(xs[-1] - xs[0]) < 1e-3 and abs(ys[-1] - ys[0]) < 1e-3:
        xs.pop()
        ys.pop()

    seg_flat = []
    for x, y in zip(xs, ys):
        seg_flat.extend([round(x, 2), round(y, 2)])

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    bbox = [round(xmin, 2), round(ymin, 2), round(xmax - xmin, 2), round(ymax - ymin, 2)]

    area = polygon_area(xs, ys)

    return class_id, seg_flat, bbox, area


def convert_split(split: str):
    print(f"\n=== {split} ===")
    img_src_dir = SRC_ROOT / "images" / split
    lbl_src_dir = SRC_ROOT / "labels" / split
    img_dst_dir = DST_ROOT / "images" / split
    ann_dst_dir = DST_ROOT / "annotations"

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    ann_dst_dir.mkdir(parents=True, exist_ok=True)

    images_list = []
    annotations_list = []
    ann_id = 1

    img_files = sorted(img_src_dir.glob("*.png"))
    total = len(img_files)
    print(f"  画像数: {total}")

    for img_id, img_path in enumerate(img_files, start=1):
        # 画像をコピー
        dst_img_path = img_dst_dir / img_path.name
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)

        # 画像サイズ取得
        with Image.open(img_path) as img:
            width, height = img.size

        images_list.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        # アノテーション読み込み
        lbl_path = lbl_src_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue

        with open(lbl_path) as f:
            lines = [l for l in f.readlines() if l.strip()]

        for line in lines:
            try:
                class_id, seg_flat, bbox, area = parse_yolo_seg_line(line, width, height)
            except Exception as e:
                print(f"  [WARN] パース失敗 {lbl_path.name}: {e}")
                continue

            if len(seg_flat) < 6:  # ポリゴンが3点未満は無効
                continue

            annotations_list.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id + 1,  # YOLO 0-indexed → COCO 1-indexed
                    "segmentation": [seg_flat],
                    "area": round(area, 2),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        if img_id % 500 == 0:
            print(f"  {img_id}/{total} 処理済み")

    coco_dict = {
        "info": {
            "description": "CubiCasa5k Room Segmentation (converted from YOLOv8-seg)",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "categories": CATEGORIES,
        "images": images_list,
        "annotations": annotations_list,
    }

    out_json = ann_dst_dir / f"instances_{split}.json"
    with open(out_json, "w") as f:
        json.dump(coco_dict, f)

    print(f"  アノテーション数: {len(annotations_list)}")
    print(f"  出力: {out_json}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        convert_split(split)

    print("\n完了!")
    print(f"出力先: {DST_ROOT}")
    print("  cubicasa_mask2former/")
    print("    images/train/  ... PNG画像")
    print("    images/val/    ... PNG画像")
    print("    annotations/instances_train.json")
    print("    annotations/instances_val.json")
