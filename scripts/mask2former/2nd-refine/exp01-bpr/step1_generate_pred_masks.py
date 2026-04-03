"""
step1_generate_pred_masks.py  ―  Mask2Former で全訓練画像を推論し予測マスクを保存

stage1_utils.py の推論・後処理関数を使用する。
出力: {PRED_DIR_TRAIN|VAL}/{stem}_pred.npz
      keys: masks (N, H, W) bool, scores (N,) float32

Usage:
    python step1_generate_pred_masks.py --split train
    python step1_generate_pred_masks.py --split val
    python step1_generate_pred_masks.py --split all
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import config
import stage1_utils as s1


# ─────────────────────────────────────────────────────────────────────────────
# 画像ステム一覧を取得
# ─────────────────────────────────────────────────────────────────────────────

def get_stems(img_dir: Path) -> list[str]:
    return sorted(p.stem for p in img_dir.glob("*.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 1 枚分の推論 → NPZ 保存
# ─────────────────────────────────────────────────────────────────────────────

def process_image(model, img_path: Path, out_path: Path) -> int:
    """
    1 枚の画像を推論して予測マスクを NPZ に保存する。

    Returns
    -------
    n_instances : int  保存したインスタンス数
    """
    image_rgb = np.array(Image.open(img_path).convert("RGB"))

    masks, scores, labels = s1.predict_instances(model, image_rgb, config.STAGE1_THRESHOLD)

    if len(masks) == 0:
        np.savez_compressed(
            out_path,
            masks=np.zeros((0, *image_rgb.shape[:2]), dtype=bool),
            scores=np.array([], dtype=np.float32),
        )
        return 0

    # 後処理 (stage1_utils の定数に従う)
    masks = s1.apply_morphology(masks)
    if s1.RESOLVE_OVERLAPS_BY_AREA:
        masks = s1.resolve_overlaps_by_area(masks)
    if s1.MIN_INSTANCE_AREA > 0:
        masks, scores, labels = s1.filter_by_area(
            masks, scores, labels, s1.MIN_INSTANCE_AREA
        )

    np.savez_compressed(
        out_path,
        masks=masks.astype(bool),
        scores=np.array(scores, dtype=np.float32),
    )
    return len(masks)


# ─────────────────────────────────────────────────────────────────────────────
# スプリット処理
# ─────────────────────────────────────────────────────────────────────────────

def run_split(model, img_dir: Path, pred_dir: Path, skip_existing: bool) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    stems = get_stems(img_dir)

    if not stems:
        raise FileNotFoundError(f"画像が見つかりません: {img_dir}")

    print(f"  {len(stems)} 枚の画像を処理します: {img_dir}")
    total_instances = 0

    for stem in tqdm(stems, desc=f"  {img_dir.name}"):
        img_path = img_dir / f"{stem}.png"
        out_path = pred_dir / f"{stem}_pred.npz"

        if skip_existing and out_path.exists():
            continue

        n = process_image(model, img_path, out_path)
        total_instances += n

    print(f"  完了: 合計 {total_instances:,} インスタンス保存")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask2Former 全画像推論 → 予測マスク NPZ 保存")
    p.add_argument("--split",         choices=["train", "val", "all"],
                   default="all",     help="処理するスプリット")
    p.add_argument("--weights",       type=str,
                   default=str(config.STAGE1_WEIGHTS),
                   help="Stage1 モデルの重みファイルパス")
    p.add_argument("--model_id",      type=str,
                   default=config.STAGE1_MODEL_ID,
                   help="HuggingFace モデル ID")
    p.add_argument("--skip_existing", action="store_true",
                   help="既存の NPZ ファイルをスキップする")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"Device    : {s1.DEVICE}")
    print(f"Model ID  : {args.model_id}")
    print(f"Weights   : {args.weights}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"重みファイルが見つかりません: {weights_path}")

    print("モデルをロード中...")
    model = s1.build_model(args.model_id, weights_path)
    print("モデルロード完了\n")

    splits = []
    if args.split in ("train", "all"):
        splits.append((config.IMG_DIR_TRAIN, config.PRED_DIR_TRAIN))
    if args.split in ("val", "all"):
        splits.append((config.IMG_DIR_VAL, config.PRED_DIR_VAL))

    for img_dir, pred_dir in splits:
        print(f"\n=== split: {img_dir.name} ===")
        run_split(model, img_dir, pred_dir, args.skip_existing)

    print("\n=== step1 完了 ===")


if __name__ == "__main__":
    main()
