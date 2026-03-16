"""
CubiCasa5k を走査し、YOLOv8-seg 学習用データセットを構築するスクリプト。

- model.svg と対応画像（F1_scaled.png 優先）のペアを収集
- Space ポリゴンを抽出し画像サイズで正規化
- 80% train / 20% val に分割
- my_cubicasa_dataset/ に images/, labels/, data.yaml を出力
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# 大きな図面画像でも読み込みできるようにする（CubiCasa5k の高解像度画像用）
Image.MAX_IMAGE_PIXELS = None

# 同一ディレクトリの extract_room_polygons を利用（scripts から実行時）
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from extract_room_polygons import extract_room_polygons_from_svg

# 画像ファイルの優先順位（先にマッチしたものを使用）
IMAGE_PRIORITY = ("F1_scaled.png", "F1_original.png")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def find_image_for_svg(svg_path: Path) -> Path | None:
    """
    model.svg と同じディレクトリから対応画像を探す。
    F1_scaled.png を優先し、なければ F1_original.png、それもなければ任意の画像。
    """
    parent = svg_path.parent
    for name in IMAGE_PRIORITY:
        candidate = parent / name
        if candidate.is_file():
            return candidate
    for f in parent.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
            return f
    return None


def get_unique_prefix(svg_path: Path, root: Path) -> str:
    """
    名前衝突を防ぐため、ルートからの相対パスをアンダースコア区切りにしたプレフィックスを返す。
    例: root=.../cubicasa5k, svg_path=.../cubicasa5k/high_quality/17/model.svg -> "high_quality_17"
    """
    try:
        rel = svg_path.parent.relative_to(root)
    except ValueError:
        rel = svg_path.parent
    if rel.parts:
        return "_".join(rel.parts)
    return svg_path.parent.name or "sample"


def normalize_polygons(
    polygons: List[List[Tuple[float, float]]],
    width: float,
    height: float,
) -> List[List[Tuple[float, float]]]:
    """
    ポリゴン座標を画像の幅・高さで割り、0.0〜1.0 に正規化する。
    """
    if width <= 0 or height <= 0:
        return []
    out: List[List[Tuple[float, float]]] = []
    for poly in polygons:
        normalized = []
        for x, y in poly:
            nx = max(0.0, min(1.0, x / width))
            ny = max(0.0, min(1.0, y / height))
            normalized.append((nx, ny))
        if len(normalized) >= 3:  # YOLO seg は最低3点必要
            out.append(normalized)
    return out


def polygon_to_yolo_line(class_id: int, polygon: List[Tuple[float, float]]) -> str:
    """
    1つのポリゴンを YOLO セグメンテーション形式の1行に変換する。
    形式: "class_id x1 y1 x2 y2 ..."
    """
    parts = [str(class_id)]
    for x, y in polygon:
        parts.append(f"{x:.6f}")
        parts.append(f"{y:.6f}")
    return " ".join(parts)


def collect_samples(root_dir: Path):
    """
    cubicasa5k ルートを走査し、(画像Path, SVG Path, ユニークプレフィックス) のリストを返す。
    画像が無い model.svg はスキップする。
    """
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        return []

    samples: List[Tuple[Path, Path, str]] = []
    for svg_path in sorted(root_dir.rglob("model.svg")):
        if not svg_path.is_file():
            continue
        img_path = find_image_for_svg(svg_path)
        if img_path is None:
            continue
        prefix = get_unique_prefix(svg_path, root_dir)
        # 拡張子を除いたベース名（F1_scaled など）
        stem = img_path.stem
        samples.append((img_path, svg_path, f"{prefix}_{stem}"))
    return samples


def get_image_size(img_path: Path) -> Tuple[int, int]:
    """Pillow で画像の (width, height) を取得する。"""
    with Image.open(img_path) as im:
        return im.size  # (width, height)


def build_dataset(
    cubicasa_root: Path,
    out_root: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """
    データセットを構築する。
    - サンプル収集 → ポリゴン抽出・正規化 → train/val 分割 → 出力
    """
    out_root = out_root.resolve()
    samples = collect_samples(cubicasa_root)
    if not samples:
        print("No (model.svg, image) pairs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    random.seed(seed)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    train_img_dir = out_root / "images" / "train"
    train_lbl_dir = out_root / "labels" / "train"
    val_img_dir = out_root / "images" / "val"
    val_lbl_dir = out_root / "labels" / "val"
    for d in (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    def process_batch(entries: List[Tuple[Path, Path, str]], img_dir: Path, lbl_dir: Path) -> int:
        count = 0
        for img_path, svg_path, base_name in entries:
            try:
                width, height = get_image_size(img_path)
            except Exception as e:
                print(f"Warning: could not read image {img_path}: {e}", file=sys.stderr)
                continue
            try:
                polygons = extract_room_polygons_from_svg(svg_path)
            except Exception as e:
                print(f"Warning: could not parse SVG {svg_path}: {e}", file=sys.stderr)
                continue

            normalized = normalize_polygons(polygons, float(width), float(height))
            if not normalized:
                continue

            # 画像はコピー（拡張子は元のまま）
            out_img = img_dir / f"{base_name}{img_path.suffix}"
            shutil.copy2(img_path, out_img)

            # ラベル: 1部屋1行、クラスID=0
            lines = [polygon_to_yolo_line(0, poly) for poly in normalized]
            out_label = lbl_dir / f"{base_name}.txt"
            out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            count += 1
        return count

    n_train_ok = process_batch(train_samples, train_img_dir, train_lbl_dir)
    n_val_ok = process_batch(val_samples, val_img_dir, val_lbl_dir)

    # data.yaml（path はデータセットルートの絶対パス；学習時は --data で指定）
    yaml_path = out_root / "data.yaml"
    yaml_content = f"""# YOLOv8-seg dataset: CubiCasa5k room segmentation
path: {out_root.resolve().as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: room
"""

    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"Collected {n} samples (train {n_train}, val {n - n_train}).")
    print(f"Written: train {n_train_ok}, val {n_val_ok}.")
    print(f"Output: {out_root}")
    print(f"  images/train, images/val")
    print(f"  labels/train, labels/val")
    print(f"  data.yaml (nc=1, names: 0=room)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build YOLOv8-seg dataset from CubiCasa5k (model.svg + images)."
    )
    parser.add_argument(
        "cubicasa_root",
        type=Path,
        nargs="?",
        default=Path("/Users/aokitenju/Downloads/archive (1)/cubicasa5k/cubicasa5k"),
        help="Root directory of CubiCasa5k (contains subdirs with model.svg and images)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("my_cubicasa_dataset"),
        help="Output dataset root (default: my_cubicasa_dataset)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    args = parser.parse_args()

    build_dataset(
        cubicasa_root=args.cubicasa_root,
        out_root=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
