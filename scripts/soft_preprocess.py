"""
前処理スクリプト: 全学習データに soft_threshold 前処理を一括適用し保存する。

- 入力: /kaggle/input/.../archive-3/ の train/valid/test images/*.jpg
- 出力: /kaggle/working/preprocessed/ に前処理済み画像 + ラベルをそのままコピー
- data.yaml は出力先に合わせてパスを書き換えてコピー

実行例:
  python scripts/soft_preprocess.py
  python scripts/soft_preprocess.py --visualize   # 完了後にサンプル可視化
"""

import argparse
import logging
import shutil
import random
from pathlib import Path

import cv2
import yaml

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
INPUT_BASE = Path(
    "/kaggle/input/datasets/umairinayat/floor-plans-500-annotated-object-detection/archive-3"
)
OUTPUT_BASE = Path("/kaggle/working/preprocessed")
SPLITS = ("train", "valid", "test")

# ---------------------------------------------------------------------------
# 前処理関数
# ---------------------------------------------------------------------------


def soft_threshold_preprocess(image_path, threshold=200):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    soft_img = blurred.copy()
    soft_img[blurred > threshold] = 255
    soft_img_rgb = cv2.cvtColor(soft_img, cv2.COLOR_GRAY2BGR)
    return soft_img_rgb  # BGR (OpenCV) を返す（保存時はそのまま cv2.imwrite で .jpg に保存可能）


# ---------------------------------------------------------------------------
# 一括処理
# ---------------------------------------------------------------------------


def process_split(input_base: Path, output_base: Path, split: str, threshold: int):
    """
    1 split (train/valid/test) の images/*.jpg を前処理し、
    対応する labels/*.txt をそのままコピーする。
    戻り値: (成功数, スキップしたファイル名のリスト)
    """
    img_dir = input_base / split / "images"
    label_dir_in = input_base / split / "labels"
    out_img_dir = output_base / split / "images"
    out_label_dir = output_base / split / "labels"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    jpg_files = list(img_dir.glob("*.jpg"))
    success = 0
    skipped = []

    for jpg_path in jpg_files:
        try:
            out_img = soft_threshold_preprocess(str(jpg_path), threshold=threshold)
            if out_img is None:
                skipped.append(jpg_path.name)
                continue
            out_path = out_img_dir / jpg_path.name
            cv2.imwrite(str(out_path), out_img)

            # 対応する .txt をそのままコピー
            txt_name = jpg_path.stem + ".txt"
            txt_src = label_dir_in / txt_name
            if txt_src.exists():
                shutil.copy2(txt_src, out_label_dir / txt_name)
            success += 1
        except Exception as e:
            logging.warning("スキップ %s: %s", jpg_path.name, e)
            skipped.append(jpg_path.name)

    return success, skipped


def copy_data_yaml_with_rewrite(input_base: Path, output_base: Path):
    """
    入力側の data.yaml を読み、train/val/test パスを出力先に合わせて書き換え、
    output_base/data.yaml に保存する。
    """
    src_yaml = input_base / "data.yaml"
    if not src_yaml.exists():
        logging.warning("data.yaml が見つかりません: %s", src_yaml)
        # 既定の nc/names で新規作成
        base = Path(output_base).resolve()
        data_config = {
            "train": str(base / "train" / "images"),
            "val": str(base / "valid" / "images"),
            "test": str(base / "test" / "images"),
            "nc": 3,
            "names": ["door", "window", "zone"],
        }
    else:
        with open(src_yaml, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f) or {}
        # パスを出力先に書き換え（絶対パスで出力先を指す）
        data_config["train"] = str(Path(output_base).resolve() / "train" / "images")
        data_config["val"] = str(Path(output_base).resolve() / "valid" / "images")
        data_config["test"] = str(Path(output_base).resolve() / "test" / "images")

    output_base.mkdir(parents=True, exist_ok=True)
    dst_yaml = output_base / "data.yaml"
    with open(dst_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return dst_yaml


def run_preprocess(input_base: Path, output_base: Path, threshold: int = 200):
    """
    全 split を処理し、data.yaml をコピー。処理件数とスキップログを返す。
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    all_skipped = []
    counts = {}

    for split in SPLITS:
        success, skipped = process_split(input_base, output_base, split, threshold)
        counts[split] = success
        all_skipped.extend((split, name) for name in skipped)
        for name in skipped:
            logging.info("[SKIP] %s/images/%s", split, name)

    copy_data_yaml_with_rewrite(input_base, output_base)

    return counts, all_skipped


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------


def visualize_samples(
    input_base: Path,
    output_base: Path,
    n_samples: int = 3,
    split: str = "train",
):
    """
    前処理前後の画像をランダムに n_samples 枚選び、横に並べて表示する。
    matplotlib を使用。保存は行わず表示のみ（plt.show()）。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("visualize_samples には matplotlib が必要です。pip install matplotlib")
        return

    img_dir_in = input_base / split / "images"
    img_dir_out = output_base / split / "images"
    if not img_dir_in.exists() or not img_dir_out.exists():
        logging.warning("画像ディレクトリが見つかりません: %s または %s", img_dir_in, img_dir_out)
        return

    jpg_names = [p.name for p in img_dir_in.glob("*.jpg") if (img_dir_out / p.name).exists()]
    if not jpg_names:
        logging.warning("対応する前処理済み画像がありません。")
        return

    chosen = random.sample(jpg_names, min(n_samples, len(jpg_names)))
    n = len(chosen)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(chosen):
        before = cv2.imread(str(img_dir_in / name))
        after = cv2.imread(str(img_dir_out / name))
        if before is not None:
            axes[i, 0].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Before: {name}")
        axes[i, 0].axis("off")
        if after is not None:
            axes[i, 1].imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"After: {name}")
        axes[i, 1].axis("off")

    plt.suptitle(f"Soft threshold preprocess — {split} (random {n} samples)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="全学習データに soft_threshold 前処理を一括適用し preprocessed/ に保存"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_BASE,
        help="入力ルート（archive-3 の親ディレクトリ）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_BASE,
        help="出力ルート（preprocessed/）",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="soft_threshold の閾値",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="処理完了後に train からランダム3枚の前後を表示",
    )
    args = parser.parse_args()

    counts, skipped = run_preprocess(args.input, args.output, args.threshold)

    print("\n--- 処理件数 ---")
    for split in SPLITS:
        print(f"  {split}: {counts[split]} 件")
    print(f"  合計: {sum(counts.values())} 件")
    if skipped:
        print(f"\nスキップ: {len(skipped)} 件（上記ログ参照）")

    if args.visualize:
        visualize_samples(args.input, args.output, n_samples=3, split="train")


if __name__ == "__main__":
    main()
