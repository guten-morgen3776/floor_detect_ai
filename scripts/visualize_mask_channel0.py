#!/usr/bin/env python3
"""
full_out のチャンネル0（クラスIDマスク）を取り出し、struct_in と並べて可視化する。
各クラスのピクセル数（面積）をコンソールに出力し、YOLO用の情報として利用できるようにする。

必要なライブラリ: numpy, matplotlib
  pip install numpy matplotlib
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# パス・設定
# -----------------------------------------------------------------------------
BASE_DIR = Path("/Users/aokitenju/Downloads/archive-4")
STRUCT_IN_DIR = BASE_DIR / "modified-swiss-dwellings-v2" / "train" / "struct_in"
FULL_OUT_DIR = BASE_DIR / "modified-swiss-dwellings-v2" / "train" / "full_out"

TARGET_ID = 10000

OUTPUT_IMAGE_PATH = Path("mask_visualization.png")
SHOW_PLOT = os.environ.get("SHOW_PLOT", "1").lower() in ("1", "true", "yes")


def load_struct_in_as_uint8(image_path: Path) -> np.ndarray:
    """struct_in の .npy を読み込み、0〜255 uint8 で返す（黒い間取り図用）。"""
    arr = np.load(image_path).astype(np.float32)
    arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def main():
    target_id = TARGET_ID
    if target_id is None:
        print("Error: TARGET_ID must be set.", file=sys.stderr)
        sys.exit(1)

    full_out_path = FULL_OUT_DIR / f"{target_id}.npy"
    struct_in_path = STRUCT_IN_DIR / f"{target_id}.npy"
    if not full_out_path.exists():
        print(f"Error: Not found: {full_out_path}", file=sys.stderr)
        sys.exit(1)
    if not struct_in_path.exists():
        print(f"Error: Not found: {struct_in_path}", file=sys.stderr)
        sys.exit(1)

    # 1) full_out を読み込み、チャンネル0のみ取り出す
    full_out = np.load(full_out_path)
    mask = np.asarray(full_out[:, :, 0], dtype=np.float64)

    # 2) チャンネル0に含まれるユニークな値（クラスID）を取得
    unique_vals, counts = np.unique(mask.ravel(), return_counts=True)
    n_classes = len(unique_vals)

    # 3) コンソール出力: 各クラスIDのピクセル数（面積）— YOLO用
    total_pixels = mask.size
    print("=" * 60)
    print(f"チャンネル0 クラスマスク — クラス別ピクセル数（面積） [ID={target_id}]")
    print("=" * 60)
    print(f"画像サイズ: {mask.shape[0]} x {mask.shape[1]} = {total_pixels} ピクセル")
    print()
    print("クラスID(値) | ピクセル数(面積) | 割合(%)")
    print("-" * 50)
    for val, cnt in zip(unique_vals, counts):
        pct = 100.0 * cnt / total_pixels
        print(f"  {val:12.4g} | {cnt:16d} | {pct:6.2f}%")
    print("-" * 50)
    print(f"  合計             | {sum(counts):16d} | 100.00%")
    print("=" * 60)

    # マスクをクラスインデックス（0, 1, 2, ...）に変換してカラーマップ用に使う
    index_img = np.zeros(mask.shape, dtype=np.intp)
    for i, v in enumerate(unique_vals):
        index_img[mask == v] = i

    # struct_in を読み込み（左側用）
    struct_img = load_struct_in_as_uint8(struct_in_path)

    # 4) サブプロットで左: struct_in、右: チャンネル0マスク（tab20で色分け）
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # 左: struct_in（黒い間取り図）
    if struct_img.ndim == 3 and struct_img.shape[-1] == 3:
        axes[0].imshow(struct_img, origin="upper")
    else:
        axes[0].imshow(struct_img, origin="upper", cmap="gray")
    axes[0].set_title("struct_in (入力間取り)")
    axes[0].axis("off")

    # 右: チャンネル0マスク（クラスごとに色分け）
    # tab20 は 0..19 を別色にする。クラス数が20超の場合は tab20b 等を重ねるか、ここでは単一カラーマップで表示
    cmap = plt.get_cmap("tab20")
    im = axes[1].imshow(
        index_img,
        origin="upper",
        cmap=cmap,
        vmin=-0.5,
        vmax=max(n_classes - 0.5, 0.5),
        interpolation="nearest",
    )
    axes[1].set_title("full_out チャンネル0 (クラスIDマスク)")
    axes[1].axis("off")
    # カラーバーでクラスID（元の値）を表示
    cbar = plt.colorbar(im, ax=axes[1], ticks=range(n_classes), shrink=0.8)
    cbar.ax.set_yticklabels([f"{v:.4g}" for v in unique_vals], fontsize=8)

    plt.tight_layout()

    # 5) 保存
    out_path = Path(OUTPUT_IMAGE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
