"""
my_cubicasa_dataset の images + labels から
セグメンテーション学習用データセットを生成する。

出力先: /Users/aokitenju/floor_detect_ai-1/cubicasa_segment/
  images/train/, images/val/  … 元画像のコピー
  masks/train/,  masks/val/   … 2値マスク PNG (0=背景, 255=部屋)

既存の train/val 分割(約 80:20)をそのまま踏襲する。
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SRC_DIR = Path("/Users/aokitenju/floor_detect_ai-1/my_cubicasa_dataset")
DST_DIR = Path("/Users/aokitenju/floor_detect_ai-1/cubicasa_segment")


def parse_label(label_path: Path):
    """YOLOv8-seg ラベルを読み込む。
    各行: class_id x1 y1 x2 y2 ... (正規化座標)
    戻り値: list of np.ndarray shape=(N,2) dtype=float32
    """
    polygons = []
    with open(label_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            coords = list(map(float, tokens[1:]))
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            polygons.append(pts)
    return polygons


def make_mask(image_path: Path, label_path: Path) -> np.ndarray:
    """元画像と同サイズの 2値マスク (uint8) を生成して返す。"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    for pts_norm in parse_label(label_path):
        # 正規化座標 → ピクセル座標
        pts_px = (pts_norm * np.array([w, h], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [pts_px], color=255)

    return mask


def process_split(split: str):
    src_img_dir = SRC_DIR / "images" / split
    src_lbl_dir = SRC_DIR / "labels" / split
    dst_img_dir = DST_DIR / "images" / split
    dst_msk_dir = DST_DIR / "masks" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_msk_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(src_img_dir.glob("*.png"))
    ok = skip = 0

    for img_path in tqdm(img_paths, desc=f"{split:5s}", unit="img"):
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            skip += 1
            continue

        # 画像コピー
        shutil.copy2(img_path, dst_img_dir / img_path.name)

        # マスク生成・保存
        mask = make_mask(img_path, lbl_path)
        mask_out = dst_msk_dir / (img_path.stem + ".png")
        cv2.imwrite(str(mask_out), mask)
        ok += 1

    print(f"  {split}: {ok} 件完了, {skip} 件スキップ (ラベル欠損)")


def main():
    print(f"出力先: {DST_DIR}\n")
    for split in ("train", "val"):
        process_split(split)

    # 統計
    for split in ("train", "val"):
        n_img = len(list((DST_DIR / "images" / split).glob("*.png")))
        n_msk = len(list((DST_DIR / "masks"  / split).glob("*.png")))
        print(f"{split:5s} | images: {n_img}, masks: {n_msk}, 一致: {n_img == n_msk}")


if __name__ == "__main__":
    main()
