"""
ターゲットデータ (floor-data-sbuild.json) の複雑さスコアを算出し、
学習データ (mask2former_augmented_data) の分布と比較する
"""

import json
import math
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# パス設定
# ─────────────────────────────────────────────
TARGET_JSON  = Path("/Users/aokitenju/floor_detect_ai-1/floor-data-sbuild.json")
TRAIN_JSON   = Path("/Users/aokitenju/floor_detect_ai-1/mask2former_augmented_data/annotations/instances_train.json")
VAL_JSON     = Path("/Users/aokitenju/floor_detect_ai-1/mask2former_augmented_data/annotations/instances_val.json")
OUTPUT_DIR   = Path("/Users/aokitenju/floor_detect_ai-1/mask2former_augmented_data/difficulty_check/lr_data_vs_target")
OUTPUT_DIR.mkdir(exist_ok=True)

HIST_COLOR   = "#4C72B0"   # blue  (学習データ)
TARGET_COLOR = "#DD8452"   # orange (ターゲット)
GREEN_COLOR  = "#55A868"
RED_COLOR    = "#C44E52"
GRID_ALPHA   = 0.3

# ─────────────────────────────────────────────
# ジオメトリ ユーティリティ (analyze_mask2former_difficulty.py と同一)
# ─────────────────────────────────────────────

def polygon_perimeter(coords: list) -> float:
    xs = coords[0::2]
    ys = coords[1::2]
    n = len(xs)
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += math.hypot(xs[j] - xs[i], ys[j] - ys[i])
    return total


def polsby_popper(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return 4 * math.pi * area / (perimeter ** 2)


def bbox_centroid(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def are_adjacent(bbox_a, bbox_b, tol: float = 5.0) -> bool:
    ax1, ay1, aw, ah = bbox_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = bbox_b
    bx2, by2 = bx1 + bw, by1 + bh
    gap_x = max(ax1, bx1) - min(ax2, bx2)
    gap_y = max(ay1, by1) - min(ay2, by2)
    return gap_x <= tol and gap_y <= tol


# ─────────────────────────────────────────────
# 指標計算
# ─────────────────────────────────────────────

def compute_metrics(coco_data: dict) -> list[dict]:
    img_map = {img["id"]: img for img in coco_data["images"]}

    ann_by_img = defaultdict(list)
    for ann in coco_data["annotations"]:
        if not ann.get("iscrowd", False):
            ann_by_img[ann["image_id"]].append(ann)

    records = []
    for img_id, anns in ann_by_img.items():
        img = img_map[img_id]
        W = img.get("width", 1)
        H = img.get("height", 1)
        img_area = W * H

        n_rooms = len(anns)
        vertices_per_room, pp_scores, areas_px, centroids, bboxes = [], [], [], [], []

        for ann in anns:
            seg  = ann["segmentation"]
            area = ann["area"]

            if isinstance(seg, list) and len(seg) > 0:
                # フラット配列 or リスト of ポリゴン を統一処理
                if isinstance(seg[0], list):
                    coords = max(seg, key=len)
                else:
                    coords = seg
                n_verts = len(coords) // 2
                perim   = polygon_perimeter(coords)
                pp      = polsby_popper(area, perim)
            else:
                n_verts, pp = 4, 1.0

            vertices_per_room.append(n_verts)
            pp_scores.append(pp)
            areas_px.append(area)
            cx, cy = bbox_centroid(ann["bbox"])
            centroids.append((cx, cy))
            bboxes.append(ann["bbox"])

        max_vertices = max(vertices_per_room)
        mean_pp      = float(np.mean(pp_scores))
        space_util   = sum(areas_px) / img_area if img_area > 0 else 0.0
        cv           = float(np.std(areas_px) / np.mean(areas_px)) if n_rooms > 1 else 0.0

        adj_count = 0
        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                if are_adjacent(bboxes[i], bboxes[j]):
                    adj_count += 1

        if n_rooms > 1:
            cxs = [c[0] / W for c in centroids]
            cys = [c[1] / H for c in centroids]
            spatial_spread = float(np.std(cxs) ** 2 + np.std(cys) ** 2)
        else:
            spatial_spread = 0.0

        records.append({
            "image_id":      img_id,
            "file_name":     img.get("file_name", str(img_id)),
            "n_rooms":       n_rooms,
            "max_vertices":  max_vertices,
            "mean_pp":       mean_pp,
            "space_util":    space_util,
            "area_cv":       cv,
            "adj_count":     adj_count,
            "spatial_spread": spatial_spread,
        })

    return records


def compute_scores(records: list[dict], ref_records: list[dict] | None = None) -> np.ndarray:
    """
    各指標を ref_records の範囲で正規化してスコア化。
    ref_records=None の場合は records 自身で正規化。

    使用指標 (max_vertices / mean_pp はアノテーション品質依存のため除外):
      n_rooms      0.35  部屋数
      adj_count    0.25  隣接構造の複雑さ
      area_cv      0.25  大小部屋の混在
      space_util   0.10  充填率
      spatial_spread 0.05  空間分布
    """
    src = ref_records if ref_records is not None else records

    def _ref(key):
        return np.array([r[key] for r in src])

    def _norm(arr, ref_arr):
        mn, mx = ref_arr.min(), ref_arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        return np.clip((arr - mn) / (mx - mn), 0.0, 1.0)

    keys = ["n_rooms", "space_util", "area_cv", "adj_count", "spatial_spread"]
    arrs = {k: np.array([r[k] for r in records]) for k in keys}
    refs = {k: _ref(k) for k in keys}

    n_rooms_n = _norm(arrs["n_rooms"],       refs["n_rooms"])
    adj_n     = _norm(arrs["adj_count"],     refs["adj_count"])
    cv_n      = _norm(arrs["area_cv"],       refs["area_cv"])
    util_n    = _norm(arrs["space_util"],    refs["space_util"])
    spread_n  = _norm(arrs["spatial_spread"],refs["spatial_spread"])

    return (0.35 * n_rooms_n + 0.25 * adj_n + 0.25 * cv_n
            + 0.10 * util_n + 0.05 * spread_n)


# ─────────────────────────────────────────────
# 可視化
# ─────────────────────────────────────────────

def plot_comparison_hist(train_arr, target_arr, title, xlabel, filepath,
                         bins=40, vline_train=None, vline_target=None):
    """学習データとターゲットを重ねたヒストグラム"""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(train_arr, bins=bins, color=HIST_COLOR,   alpha=0.6, label="Train/Val data",
            edgecolor="white", linewidth=0.4, density=True)
    ax.hist(target_arr, bins=bins, color=TARGET_COLOR, alpha=0.8, label="Target (sbuild)",
            edgecolor="white", linewidth=0.4, density=True)

    if vline_train is not None:
        ax.axvline(vline_train,  color=HIST_COLOR,   linestyle="--", linewidth=1.5,
                   label=f"Train mean={vline_train:.2f}")
    if vline_target is not None:
        ax.axvline(vline_target, color=TARGET_COLOR,  linestyle="--", linewidth=1.8,
                   label=f"Target mean={vline_target:.2f}")

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def plot_complexity_score_comparison(train_scores, target_scores, target_records, filepath):
    """
    複雑さスコアの比較:
    左: density ヒストグラム重ね
    右: ターゲット各図面をドットでプロット (ランキング順)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── 左: 分布比較 ──
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(train_scores,  bins=bins, color=HIST_COLOR,   alpha=0.6,
            density=True, label="Train/Val data", edgecolor="white", linewidth=0.4)
    ax.hist(target_scores, bins=bins, color=TARGET_COLOR,  alpha=0.8,
            density=True, label="Target (sbuild)", edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(train_scores),  color=HIST_COLOR,   linestyle="--", linewidth=1.5,
               label=f"Train mean={np.mean(train_scores):.3f}")
    ax.axvline(np.mean(target_scores), color=TARGET_COLOR,  linestyle="--", linewidth=1.8,
               label=f"Target mean={np.mean(target_scores):.3f}")
    # 学習データの p25/p75 帯
    p25 = np.percentile(train_scores, 25)
    p75 = np.percentile(train_scores, 75)
    ax.axvspan(p25, p75, alpha=0.08, color=HIST_COLOR, label=f"Train IQR [{p25:.2f}-{p75:.2f}]")
    ax.set_title("Complexity Score Distribution Comparison", fontsize=12)
    ax.set_xlabel("Complexity Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    # ── 右: ターゲット個別スコア ──
    ax = axes[1]
    order = np.argsort(target_scores)[::-1]  # 難しい順
    y_pos = np.arange(len(target_scores))

    colors = []
    for s in target_scores[order]:
        if s > p75:
            colors.append(RED_COLOR)
        elif s > p25:
            colors.append(TARGET_COLOR)
        else:
            colors.append(GREEN_COLOR)

    ax.barh(y_pos, target_scores[order], color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(p25, color=GREEN_COLOR,  linestyle=":", linewidth=1.2, label=f"Train p25={p25:.2f}")
    ax.axvline(p75, color=RED_COLOR,    linestyle=":", linewidth=1.2, label=f"Train p75={p75:.2f}")
    ax.axvline(np.mean(train_scores), color=HIST_COLOR, linestyle="--", linewidth=1.2,
               label=f"Train mean={np.mean(train_scores):.2f}")

    # ファイル名（短縮）をラベルに
    labels = []
    for i in order:
        fname = target_records[i]["file_name"]
        # パスから末尾のファイル名だけ取り出して短縮
        short = Path(fname).stem[:30]
        labels.append(f"{short}  (n={target_records[i]['n_rooms']})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Complexity Score", fontsize=11)
    ax.set_title("Target Images Ranked by Complexity", fontsize=12)

    easy_patch  = mpatches.Patch(color=GREEN_COLOR,  label="Easy (< Train p25)")
    mid_patch   = mpatches.Patch(color=TARGET_COLOR, label="Medium (Train IQR)")
    hard_patch  = mpatches.Patch(color=RED_COLOR,    label="Hard (> Train p75)")
    ax.legend(handles=[hard_patch, mid_patch, easy_patch], fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def plot_radar_comparison(train_records, target_records, filepath):
    """
    指標ごとの平均値をレーダーチャートで比較
    """
    keys   = ["n_rooms", "space_util", "area_cv", "adj_count", "spatial_spread"]
    labels = ["Rooms", "Space\nUtil", "Area\nCV", "Adj\nCount", "Spatial\nSpread"]

    # 各指標を全体(train)の min-max で正規化
    def _norm_arr(key, src_records):
        all_vals = np.array([r[key] for r in train_records + target_records])
        vals     = np.array([r[key] for r in src_records])
        mn, mx   = all_vals.min(), all_vals.max()
        if mx == mn:
            return np.zeros(len(vals))
        return (vals - mn) / (mx - mn)

    train_means  = [float(np.mean(_norm_arr(k, train_records)))  for k in keys]
    target_means = [float(np.mean(_norm_arr(k, target_records))) for k in keys]

    N = len(keys)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    train_means  += train_means[:1]
    target_means += target_means[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles, train_means,  color=HIST_COLOR,   linewidth=2,   label="Train/Val (mean)")
    ax.fill(angles, train_means,  color=HIST_COLOR,   alpha=0.15)
    ax.plot(angles, target_means, color=TARGET_COLOR,  linewidth=2.5, linestyle="--", label="Target (mean)")
    ax.fill(angles, target_means, color=TARGET_COLOR,  alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Complexity Metric Profile\n(normalized, higher = more complex)", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def plot_per_metric_comparison(train_records, target_records, filepath):
    """各指標について学習データとターゲットの箱ひげ図を並べる"""
    keys   = ["n_rooms", "space_util", "area_cv", "adj_count", "spatial_spread"]
    titles = ["Rooms per Plan", "Space Utilization", "Area CV",
              "Adjacency Count", "Spatial Spread"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    axes = axes.flatten()

    for ax, key, title in zip(axes, keys, titles):
        train_vals  = [r[key] for r in train_records]
        target_vals = [r[key] for r in target_records]
        bp = ax.boxplot([train_vals, target_vals],
                        labels=["Train/Val", "Target"],
                        patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2},
                        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
        bp["boxes"][0].set_facecolor(HIST_COLOR);   bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(TARGET_COLOR);  bp["boxes"][1].set_alpha(0.7)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=GRID_ALPHA)

    plt.suptitle("Per-Metric Distribution: Train vs Target", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    print("=== ターゲットデータ 複雑さスコア比較 ===\n")

    # ── データ読み込み ──
    print("学習データ読み込み中...")
    train_recs = []
    for p in [TRAIN_JSON, VAL_JSON]:
        if p.exists():
            with open(p) as f:
                train_recs.extend(compute_metrics(json.load(f)))
    print(f"  学習データ図面数: {len(train_recs):,}")

    print("ターゲットデータ読み込み中...")
    with open(TARGET_JSON) as f:
        target_data = json.load(f)
    target_recs = compute_metrics(target_data)
    print(f"  ターゲット図面数: {len(target_recs)}")

    # ── スコア計算 (学習データのスケールで正規化) ──
    train_scores  = compute_scores(train_recs,  ref_records=None)
    target_scores = compute_scores(target_recs, ref_records=train_recs)

    # ── 統計表示 ──
    print("\n=== 複雑さスコア 比較 ===")
    print(f"  学習データ:  mean={np.mean(train_scores):.3f}  median={np.median(train_scores):.3f}"
          f"  p25={np.percentile(train_scores,25):.3f}  p75={np.percentile(train_scores,75):.3f}")
    print(f"  ターゲット:  mean={np.mean(target_scores):.3f}  median={np.median(target_scores):.3f}"
          f"  p25={np.percentile(target_scores,25):.3f}  p75={np.percentile(target_scores,75):.3f}")

    p25_train = np.percentile(train_scores, 25)
    p75_train = np.percentile(train_scores, 75)
    n_hard   = int(np.sum(target_scores > p75_train))
    n_medium = int(np.sum((target_scores >= p25_train) & (target_scores <= p75_train)))
    n_easy   = int(np.sum(target_scores < p25_train))
    print(f"\n  ターゲット難易度内訳 (学習データ基準):")
    print(f"    Hard   (> Train p75={p75_train:.3f}): {n_hard} 枚 ({n_hard/len(target_recs)*100:.1f}%)")
    print(f"    Medium (Train IQR):                   {n_medium} 枚 ({n_medium/len(target_recs)*100:.1f}%)")
    print(f"    Easy   (< Train p25={p25_train:.3f}): {n_easy} 枚 ({n_easy/len(target_recs)*100:.1f}%)")

    print("\n=== 各指標 比較 ===")
    keys   = ["n_rooms", "space_util", "area_cv", "adj_count", "spatial_spread"]
    labels = ["部屋数", "充填率", "面積 CV", "隣接ペア数", "空間分布"]
    for k, lbl in zip(keys, labels):
        tv = np.array([r[k] for r in train_recs])
        tg = np.array([r[k] for r in target_recs])
        print(f"  {lbl:12s}  Train: mean={np.mean(tv):.2f} median={np.median(tv):.2f}"
              f"  |  Target: mean={np.mean(tg):.2f} median={np.median(tg):.2f}")

    # ── グラフ生成 ──
    print("\n=== グラフ生成 ===")

    # T01. 複雑さスコア 比較 (メイン図)
    plot_complexity_score_comparison(
        train_scores, target_scores, target_recs,
        OUTPUT_DIR / "T01_complexity_score_comparison.png",
    )

    # T02. レーダーチャート
    plot_radar_comparison(
        train_recs, target_recs,
        OUTPUT_DIR / "T02_radar_comparison.png",
    )

    # T03. 各指標 箱ひげ図
    plot_per_metric_comparison(
        train_recs, target_recs,
        OUTPUT_DIR / "T03_per_metric_boxplot.png",
    )

    # T04-T07. 主要指標 個別の重ね合わせヒストグラム
    for tag, key, title, xlabel in [
        ("T04", "n_rooms",      "Rooms per Plan",        "Num Rooms"),
        ("T05", "area_cv",      "Area CV (size hetero)",  "CV"),
        ("T06", "adj_count",    "Adjacency Count",        "Count"),
        ("T07", "space_util",   "Space Utilization",      "Ratio"),
    ]:
        tv = np.array([r[key] for r in train_recs])
        tg = np.array([r[key] for r in target_recs])
        plot_comparison_hist(
            tv, tg, f"{title} — Train vs Target", xlabel,
            OUTPUT_DIR / f"{tag}_{key}_comparison.png",
            bins=40,
            vline_train=float(np.mean(tv)),
            vline_target=float(np.mean(tg)),
        )

    # ── CSV出力 ──
    csv_path = OUTPUT_DIR / "target_difficulty_metrics.csv"
    order = np.argsort(target_scores)[::-1]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "file_name", "n_rooms",
            "space_util", "area_cv", "adj_count", "spatial_spread",
            "complexity_score", "tier",
        ])
        writer.writeheader()
        for rank, i in enumerate(order, 1):
            r  = target_recs[i]
            sc = float(target_scores[i])
            tier = "Hard" if sc > p75_train else ("Easy" if sc < p25_train else "Medium")
            writer.writerow({
                "rank":             rank,
                "file_name":        Path(r["file_name"]).name,
                "n_rooms":          r["n_rooms"],
                "space_util":       f"{r['space_util']:.4f}",
                "area_cv":          f"{r['area_cv']:.4f}",
                "adj_count":        r["adj_count"],
                "spatial_spread":   f"{r['spatial_spread']:.6f}",
                "complexity_score": f"{sc:.4f}",
                "tier":             tier,
            })
    print(f"\nCSV出力: {csv_path}")

    # ── 上位/下位の表示 ──
    print("\n=== ターゲット 難しい順 全図面 ===")
    for rank, i in enumerate(order, 1):
        r  = target_recs[i]
        sc = float(target_scores[i])
        tier = "Hard  " if sc > p75_train else ("Easy  " if sc < p25_train else "Medium")
        print(f"  #{rank:2d} [{tier}] score={sc:.3f}  rooms={r['n_rooms']:3d}"
              f"  cv={r['area_cv']:.3f}  adj={r['adj_count']:3d}"
              f"  util={r['space_util']:.3f}"
              f"  {Path(r['file_name']).name[:50]}")

    print(f"\n出力先: {OUTPUT_DIR}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
