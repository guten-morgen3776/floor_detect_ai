#!/usr/bin/env python3
"""
full_out ディレクトリの .npy 出力画像を読み込み、値域を分析して可視化するスクリプト。

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
FULL_OUT_DIR = BASE_DIR / "modified-swiss-dwellings-v2" / "train" / "full_out"

TARGET_ID = 10000

OUTPUT_IMAGE_PATH = Path("full_out_test.png")
SHOW_PLOT = os.environ.get("SHOW_PLOT", "1").lower() in ("1", "true", "yes")


def analyze_value_distribution(arr: np.ndarray) -> None:
    """配列の値の種類・分布をコンソールに出力し、チャンネルの意味を推測しやすくする。"""
    arr = np.asarray(arr, dtype=np.float64)
    print("=" * 60)
    print("full_out .npy 値域・分布の分析")
    print("=" * 60)
    print(f"形状: {arr.shape}")
    print(f"dtype: {arr.dtype}")
    print()

    # 全体の統計
    print("【全体】")
    print(f"  最小値: {np.nanmin(arr):.6g}")
    print(f"  最大値: {np.nanmax(arr):.6g}")
    print(f"  平均:   {np.nanmean(arr):.6g}")
    print(f"  標準偏差: {np.nanstd(arr):.6g}")
    print()

    # チャンネル別
    if arr.ndim >= 3 and arr.shape[-1] <= 10:
        for c in range(arr.shape[-1]):
            ch = arr[..., c]
            print(f"【チャンネル {c}】")
            print(f"  最小: {np.nanmin(ch):.6g}, 最大: {np.nanmax(ch):.6g}, 平均: {np.nanmean(ch):.6g}")
            uniq = np.unique(ch)
            print(f"  ユニークな値の数: {len(uniq)}")
            if len(uniq) <= 30:
                print(f"  ユニーク値: {uniq.tolist()}")
            else:
                print(f"  ユニーク値の先頭10個: {uniq[:10].tolist()}")
                print(f"  ユニーク値の末尾10個: {uniq[-10:].tolist()}")
            print()
    print()

    # ユニークな値の数が少ない場合 → セグメンテーション／離散ラベルの可能性
    uniq_all = np.unique(arr)
    print("【全体のユニーク値】")
    print(f"  種類数: {len(uniq_all)}")
    if len(uniq_all) <= 50:
        print(f"  値: {uniq_all.tolist()}")
    else:
        print(f"  先頭20個: {uniq_all[:20].tolist()}")
        print(f"  末尾20個: {uniq_all[-20:].tolist()}")
    print()

    # ヒストグラム風の区間別度数（ざっくり）
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi > lo:
        bins = np.linspace(lo, hi, 11)
        hist, _ = np.histogram(arr.ravel(), bins=bins)
        print("【値の分布（区間別ピクセル数）】")
        for i in range(len(bins) - 1):
            print(f"  [{bins[i]:.3g}, {bins[i+1]:.3g}): {hist[i]}")
    print("=" * 60)


def full_out_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    full_out の float16 配列を 0〜255 の uint8 に変換する。
    マイナス値を考慮し、最小値〜最大値で正規化する。
    """
    arr = np.asarray(arr, dtype=np.float64)
    v_min = np.nanmin(arr)
    v_max = np.nanmax(arr)
    span = (v_max - v_min) or 1.0
    normalized = (arr - v_min) / span
    # 0〜1 にクリップ（NaN は 0 に）
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def main():
    target_id = TARGET_ID
    if target_id is None:
        print("Error: TARGET_ID must be set.", file=sys.stderr)
        sys.exit(1)

    npy_path = FULL_OUT_DIR / f"{target_id}.npy"
    if not npy_path.exists():
        print(f"Error: Not found: {npy_path}", file=sys.stderr)
        sys.exit(1)

    # 1) 読み込み
    arr = np.load(npy_path)
    if arr.shape != (512, 512, 3):
        print(f"Warning: Expected shape (512, 512, 3), got {arr.shape}")

    # 2) 値域・分布をコンソールに出力
    analyze_value_distribution(arr)

    # 3) 0〜255 uint8 に変換（最小〜最大で正規化）
    img = full_out_to_uint8(arr)

    # 4) 表示と保存
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, origin="upper")
    ax.set_title(f"full_out 可視化 (ID={target_id})\nmin-max 正規化 → 0-255 uint8")
    ax.axis("off")
    plt.tight_layout()

    out_path = Path(OUTPUT_IMAGE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
