"""
YOLO11s ファインチューニングスクリプト（家具あり難しい図面 14 件）

- 学習済み重み: /kaggle/working/runs/detect/train/weights/best.pt
- 追加データ: /kaggle/working/difficult_annotation/image/ (.png) + labels/ (.txt, YOLO形式, door/window/zone)
- 学習前処理: soft_threshold_preprocess() を適用（前処理済み画像を一時ディレクトリに保存してから学習）
- Augmentation: Albumentations（RandomBrightnessContrast, OneOf[GaussNoise, ISONoise]）※CoarseDropout は除外
- 出力: /kaggle/working/runs/finetune/ に重みと学習曲線を保存

実行: python scripts/finetune.py
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import yaml
import albumentations as A
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# パス設定（Kaggle 想定）
# ---------------------------------------------------------------------------
WEIGHTS_PATH = Path("/kaggle/working/runs/detect/train/weights/best.pt")
DIFFICULT_IMAGE_DIR = Path("/kaggle/working/difficult_annotation/image")
DIFFICULT_LABEL_DIR = Path("/kaggle/working/difficult_annotation/labels")
PREPROCESSED_BASE = Path("/kaggle/working/difficult_annotation_preprocessed")
FINETUNE_OUTPUT = Path("/kaggle/working/runs/finetune")
KAGGLE_WORKING = Path("/kaggle/working")

# データセット構造: PREPROCESSED_BASE/images/ と PREPROCESSED_BASE/labels/
PREPROCESSED_IMAGES = PREPROCESSED_BASE / "images"
PREPROCESSED_LABELS = PREPROCESSED_BASE / "labels"


# ---------------------------------------------------------------------------
# 前処理関数
# ---------------------------------------------------------------------------
def soft_threshold_preprocess(image_path, threshold=200):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    soft_img = blurred.copy()
    soft_img[blurred > threshold] = 255
    soft_img_rgb = cv2.cvtColor(soft_img, cv2.COLOR_GRAY2BGR)
    return soft_img_rgb


def run_preprocess(image_dir: Path, label_dir: Path, out_images: Path, out_labels: Path, threshold: int = 200):
    """
    画像に soft_threshold_preprocess を適用し、前処理済み画像とラベルを保存する。
    ラベルは label_dir にあればコピー、なければ image と同階層の labels を探す。
    """
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    png_files = list(image_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"画像がありません: {image_dir}")

    for png_path in png_files:
        out_img = soft_threshold_preprocess(str(png_path), threshold=threshold)
        if out_img is None:
            continue
        out_path = out_images / png_path.name
        cv2.imwrite(str(out_path), out_img)

        txt_name = png_path.stem + ".txt"
        txt_src = label_dir / txt_name
        if not txt_src.exists():
            # image の兄弟に labels がある場合（image が image という名前のとき）
            alt = image_dir.parent / "labels" / txt_name
            txt_src = alt if alt.exists() else txt_src
        if txt_src.exists():
            shutil.copy2(txt_src, out_labels / txt_name)

    return len(png_files)


# ---------------------------------------------------------------------------
# Augmentation（CoarseDropout なし）
# ---------------------------------------------------------------------------
def get_finetune_augmentations():
    return [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.35, p=0.5),
        A.OneOf(
            [
                A.GaussNoise(var_limit=(15.0, 60.0), p=1.0),
                A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.15, 0.5), p=1.0),
            ],
            p=0.4,
        ),
    ]


# ---------------------------------------------------------------------------
# 学習曲線のプロット
# ---------------------------------------------------------------------------
def plot_curves(results_csv: Path, save_dir: Path):
    """results.csv から loss と mAP を読み、プロットして save_dir に保存する。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib がないため学習曲線をスキップします")
        return

    if not results_csv.exists():
        print(f"[WARN] {results_csv} が見つかりません。学習曲線をスキップします")
        return

    rows = []
    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    epochs = [int(r.get("epoch", i)) for i, r in enumerate(rows)]
    # loss: train の box/cls/dfl の和を簡易的に使用（列名は環境により微妙に違う可能性あり）
    def get_float(r, *keys, default=0.0):
        for k in keys:
            v = r.get(k)
            if v is not None and v != "":
                try:
                    return float(v)
                except ValueError:
                    pass
        return default

    train_box = [get_float(r, "train/box_loss", "train_box_loss") for r in rows]
    train_cls = [get_float(r, "train/cls_loss", "train_cls_loss") for r in rows]
    train_dfl = [get_float(r, "train/dfl_loss", "train_dfl_loss") for r in rows]
    train_loss = [a + b + c for a, b, c in zip(train_box, train_cls, train_dfl)]

    mAP50 = [get_float(r, "metrics/mAP50(B)", "metrics/mAP50(B)", "mAP50") for r in rows]
    mAP50_95 = [get_float(r, "metrics/mAP50-95(B)", "metrics/mAP50-95(B)", "mAP50-95") for r in rows]

    save_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_loss (box+cls+dfl)", color="C0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "curve_loss.png", dpi=150)
    plt.close(fig)

    # mAP
    fig, ax = plt.subplots(figsize=(8, 5))
    if any(mAP50):
        ax.plot(epochs, mAP50, label="mAP50", color="C1")
    if any(mAP50_95):
        ax.plot(epochs, mAP50_95, label="mAP50-95", color="C2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "curve_mAP.png", dpi=150)
    plt.close(fig)

    print(f"[INFO] 学習曲線を保存しました: {save_dir / 'curve_loss.png'}, {save_dir / 'curve_mAP.png'}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="YOLO11s ファインチューニング（難しい図面 15 件）")
    p.add_argument("--weights", type=Path, default=WEIGHTS_PATH, help="学習済み重み .pt")
    p.add_argument("--image-dir", type=Path, default=DIFFICULT_IMAGE_DIR, help="追加データ画像 dir (.png)")
    p.add_argument("--label-dir", type=Path, default=DIFFICULT_LABEL_DIR, help="追加データラベル dir (.txt)")
    p.add_argument("--preprocessed", type=Path, default=PREPROCESSED_BASE, help="前処理済み出力先")
    p.add_argument("--output", type=Path, default=FINETUNE_OUTPUT, help="ファインチューニング結果の保存先")
    p.add_argument("--freeze", type=int, default=10, help="凍結する低レイヤー数")
    p.add_argument("--lr0", type=float, default=0.001, help="初期学習率")
    p.add_argument("--epochs", type=int, default=50, help="エポック数")
    p.add_argument("--batch", type=int, default=4, help="バッチサイズ（15件なら小さめ推奨）")
    p.add_argument("--imgsz", type=int, default=640, help="入力画像サイズ")
    p.add_argument("--threshold", type=int, default=200, help="soft_threshold_preprocess の閾値")
    return p.parse_args()


def main():
    args = parse_args()

    out_images = args.preprocessed / "images"
    out_labels = args.preprocessed / "labels"

    # 1) 前処理: 画像に soft_threshold_preprocess を適用
    if not args.image_dir.exists():
        raise FileNotFoundError(f"画像ディレクトリがありません: {args.image_dir}")
    n = run_preprocess(args.image_dir, args.label_dir, out_images, out_labels, threshold=args.threshold)
    print(f"[INFO] 前処理済み画像: {n} 件 -> {out_images}")

    # 2) data.yaml（前処理済みディレクトリ向け）
    data_config = {
        "path": str(args.preprocessed.resolve()),
        "train": "images",
        "val": "images",
        "nc": 3,
        "names": ["door", "window", "zone"],
    }
    args.preprocessed.mkdir(parents=True, exist_ok=True)
    data_yaml = args.preprocessed / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    if not args.weights.exists():
        raise FileNotFoundError(f"学習済み重みがありません: {args.weights}")

    # 3) ファインチューニング
    project_dir = args.output.parent  # e.g. /kaggle/working/runs
    name_dir = args.output.name       # finetune
    model = YOLO(str(args.weights))
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_dir),
        name=name_dir,
        freeze=args.freeze,
        lr0=args.lr0,
        augmentations=get_finetune_augmentations(),
    )

    # 4) 学習曲線をプロットして保存
    # Ultralytics は project/name/ に results.csv を保存する
    results_csv = args.output / "results.csv"
    plot_curves(results_csv, args.output)

    print(f"[INFO] ファインチューニング済み重み: {args.output / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    main()
