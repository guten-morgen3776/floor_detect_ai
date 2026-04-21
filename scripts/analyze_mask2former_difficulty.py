"""
mask2former_augmented_data 図面複雑さ分析スクリプト
COCO形式アノテーション (instances_train.json / instances_val.json) を解析

計測指標:
  - 部屋数 (rooms_per_plan)
  - 図面の最大頂点数 (max_vertices_per_plan)
  - Polsby–Popper スコア (形状複雑さ)
  - 空間充填率 (space_utilization)
  - 面積変動係数 CV (size_heterogeneity)
  - 隣接部屋ペア数 (adjacency_count)
  - 空間分布の広がり (spatial_spread)
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# パス設定
# ─────────────────────────────────────────────
DATA_ROOT  = Path("/Users/aokitenju/floor_detect_ai-1/mask2former_augmented_data")
ANN_DIR    = DATA_ROOT / "annotations"
OUTPUT_DIR = DATA_ROOT / "difficulty_check"
OUTPUT_DIR.mkdir(exist_ok=True)

SPLITS = ["train", "val"]

HIST_COLOR  = "#4C72B0"
LINE_COLOR  = "#DD8452"
GREEN_COLOR = "#55A868"
RED_COLOR   = "#C44E52"
GRID_ALPHA  = 0.3

# ─────────────────────────────────────────────
# ジオメトリ ユーティリティ
# ─────────────────────────────────────────────

def polygon_perimeter(coords: list[float]) -> float:
    """頂点列 [x1,y1,x2,y2,...] から周囲長を計算"""
    xs = coords[0::2]
    ys = coords[1::2]
    n = len(xs)
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += math.hypot(xs[j] - xs[i], ys[j] - ys[i])
    return total


def polsby_popper(area: float, perimeter: float) -> float:
    """Polsby–Popper スコア: 4π×A / P²  (1=円形, 0=複雑)"""
    if perimeter <= 0:
        return 0.0
    return 4 * math.pi * area / (perimeter ** 2)


def bbox_centroid(bbox: list[float]):
    """COCO bbox [x,y,w,h] → (cx, cy)"""
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def are_adjacent(bbox_a: list[float], bbox_b: list[float], tol: float = 5.0) -> bool:
    """
    2つのbbox [x,y,w,h] が tol ピクセル以内で隣接しているか判定
    """
    ax1, ay1, aw, ah = bbox_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = bbox_b
    bx2, by2 = bx1 + bw, by1 + bh

    gap_x = max(ax1, bx1) - min(ax2, bx2)
    gap_y = max(ay1, by1) - min(ay2, by2)
    return gap_x <= tol and gap_y <= tol


# ─────────────────────────────────────────────
# データ読み込み & 指標計算
# ─────────────────────────────────────────────

def load_coco(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def compute_metrics(coco_data: dict) -> list[dict]:
    """
    画像ごとに指標を計算して list[dict] を返す
    """
    # image_id → image情報 マッピング
    img_map = {img["id"]: img for img in coco_data["images"]}

    # image_id → annotations リスト
    ann_by_img = defaultdict(list)
    for ann in coco_data["annotations"]:
        if ann.get("iscrowd", 0) == 0:
            ann_by_img[ann["image_id"]].append(ann)

    records = []
    for img_id, anns in ann_by_img.items():
        img = img_map[img_id]
        W, H = img["width"], img["height"]
        img_area = W * H

        # ── 部屋ごとの計算 ──
        n_rooms = len(anns)
        vertices_per_room = []
        pp_scores = []
        areas_px = []
        centroids = []
        bboxes = []

        for ann in anns:
            seg = ann["segmentation"]
            area = ann["area"]

            # 頂点数は最長ポリゴンから取得（RLE は除外）
            if isinstance(seg, list) and len(seg) > 0:
                coords = max(seg, key=len)
                n_verts = len(coords) // 2
                perim = polygon_perimeter(coords)
                pp = polsby_popper(area, perim)
            else:
                n_verts = 4
                pp = 1.0  # 不明な場合は矩形扱い

            vertices_per_room.append(n_verts)
            pp_scores.append(pp)
            areas_px.append(area)
            cx, cy = bbox_centroid(ann["bbox"])
            centroids.append((cx, cy))
            bboxes.append(ann["bbox"])

        # ── 図面レベル指標 ──
        max_vertices = max(vertices_per_room)

        # Polsby-Popper: 図面の平均値（低いほど複雑）
        mean_pp = float(np.mean(pp_scores))

        # 空間充填率: 全部屋面積合計 / 画像面積
        space_util = sum(areas_px) / img_area if img_area > 0 else 0.0

        # 面積変動係数 CV = std/mean
        if n_rooms > 1:
            cv = float(np.std(areas_px) / np.mean(areas_px))
        else:
            cv = 0.0

        # 隣接ペア数
        adj_count = 0
        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                if are_adjacent(bboxes[i], bboxes[j]):
                    adj_count += 1

        # 空間分布の広がり: 重心座標の標準偏差の和（正規化）
        if n_rooms > 1:
            cxs = [c[0] / W for c in centroids]
            cys = [c[1] / H for c in centroids]
            spatial_spread = float(np.std(cxs) ** 2 + np.std(cys) ** 2)
        else:
            spatial_spread = 0.0

        records.append({
            "image_id":      img_id,
            "file_name":     img["file_name"],
            "n_rooms":       n_rooms,
            "max_vertices":  max_vertices,
            "mean_pp":       mean_pp,
            "space_util":    space_util,
            "area_cv":       cv,
            "adj_count":     adj_count,
            "spatial_spread": spatial_spread,
        })

    return records


# ─────────────────────────────────────────────
# 統計サマリ
# ─────────────────────────────────────────────

def describe(arr: np.ndarray, label: str) -> dict:
    return {
        "label":  label,
        "count":  int(len(arr)),
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75":    float(np.percentile(arr, 75)),
        "p95":    float(np.percentile(arr, 95)),
        "max":    float(np.max(arr)),
    }


def print_stats(d: dict):
    print(f"  {d['label']}")
    print(f"    count={d['count']:,}  mean={d['mean']:.3f}  std={d['std']:.3f}")
    print(f"    min={d['min']:.3f}  p25={d['p25']:.3f}  median={d['median']:.3f}  p75={d['p75']:.3f}  p95={d['p95']:.3f}  max={d['max']:.3f}")


# ─────────────────────────────────────────────
# グラフ描画ユーティリティ
# ─────────────────────────────────────────────

def save_histogram(arr, title, xlabel, ylabel, filepath,
                   bins=40, log_scale=False, color=HIST_COLOR,
                   vlines: dict | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=bins, color=color, edgecolor="white", linewidth=0.5)
    if log_scale:
        ax.set_yscale("log")
    if vlines:
        for label, val in vlines.items():
            ax.axvline(val, color=LINE_COLOR, linestyle="--", linewidth=1.5,
                       label=f"{label}={val:.3f}")
        ax.legend(fontsize=9)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def save_cumulative(arr, title, xlabel, filepath, color=HIST_COLOR):
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_arr, cdf, color=color, linewidth=2)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Cumulative Ratio", fontsize=11)
    ax.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def save_scatter(x, y, title, xlabel, ylabel, filepath,
                 alpha=0.3, s=8, color=HIST_COLOR):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=s, alpha=alpha, color=color)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def save_correlation_heatmap(records: list[dict], filepath: Path):
    """全指標の相関行列ヒートマップ"""
    keys = ["n_rooms", "max_vertices", "mean_pp", "space_util",
            "area_cv", "adj_count", "spatial_spread"]
    labels = ["Rooms", "Max Verts", "PP Score", "Space Util",
              "Area CV", "Adj Count", "Spatial Spread"]

    mat = np.array([[r[k] for k in keys] for r in records])
    corr = np.corrcoef(mat.T)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Metric Correlation Matrix", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def save_overview_dashboard(records: list[dict], filepath: Path):
    """全7指標を一覧できるダッシュボード図"""
    metrics = [
        ("n_rooms",       "Rooms per Plan",       "Count"),
        ("max_vertices",  "Max Vertices",         "Vertices"),
        ("mean_pp",       "Polsby-Popper Score",  "Score"),
        ("space_util",    "Space Utilization",    "Ratio"),
        ("area_cv",       "Area CV",              "CV"),
        ("adj_count",     "Adjacency Count",      "Count"),
        ("spatial_spread","Spatial Spread",       "Spread"),
    ]
    colors = [HIST_COLOR, GREEN_COLOR, LINE_COLOR, RED_COLOR,
              "#8172B2", "#64B5CD", "#CCB974"]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (key, label, unit) in enumerate(metrics):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        vals = np.array([r[key] for r in records])
        ax.hist(vals, bins=40, color=colors[idx], edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(vals), color="black", linestyle="--", linewidth=1.2,
                   label=f"μ={np.mean(vals):.2f}")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(unit, fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=GRID_ALPHA)

    fig.suptitle("Floor Plan Complexity Metrics Dashboard", fontsize=15, y=1.01)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def save_complexity_score_dist(records: list[dict], filepath: Path):
    """
    簡易複雑さスコア (0〜1 正規化後の加重和) の分布
    weights: n_rooms×0.25 + max_vertices×0.15 + (1-pp)×0.2 + cv×0.2 + adj×0.2
    """
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    n_rooms   = _norm(np.array([r["n_rooms"]       for r in records]))
    max_verts = _norm(np.array([r["max_vertices"]   for r in records]))
    inv_pp    = _norm(1 - np.array([r["mean_pp"]    for r in records]))
    cv        = _norm(np.array([r["area_cv"]         for r in records]))
    adj       = _norm(np.array([r["adj_count"]       for r in records]))
    spread    = _norm(np.array([r["spatial_spread"]  for r in records]))

    score = (0.25 * n_rooms + 0.15 * max_verts + 0.20 * inv_pp
             + 0.15 * cv + 0.15 * adj + 0.10 * spread)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ヒストグラム
    ax = axes[0]
    ax.hist(score, bins=50, color=HIST_COLOR, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(score), color=LINE_COLOR, linestyle="--", linewidth=1.5,
               label=f"μ={np.mean(score):.3f}")
    ax.axvline(np.percentile(score, 25), color=GREEN_COLOR, linestyle=":",
               linewidth=1.2, label=f"p25={np.percentile(score,25):.3f}")
    ax.axvline(np.percentile(score, 75), color=RED_COLOR, linestyle=":",
               linewidth=1.2, label=f"p75={np.percentile(score,75):.3f}")
    ax.set_title("Complexity Score Distribution", fontsize=13)
    ax.set_xlabel("Complexity Score (0=easy, 1=hard)", fontsize=11)
    ax.set_ylabel("Num Floor Plans", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    # CDF
    ax = axes[1]
    sorted_s = np.sort(score)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax.plot(sorted_s, cdf, color=HIST_COLOR, linewidth=2)
    ax.axhline(0.25, color=GREEN_COLOR, linestyle=":", linewidth=1.2, label="25%")
    ax.axhline(0.75, color=RED_COLOR,   linestyle=":", linewidth=1.2, label="75%")
    ax.set_title("Complexity Score CDF", fontsize=13)
    ax.set_xlabel("Complexity Score", fontsize=11)
    ax.set_ylabel("Cumulative Ratio", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath.name}")

    return score


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    print("=== mask2former_augmented_data 図面複雑さ分析 ===\n")
    all_records = []

    for split in SPLITS:
        json_path = ANN_DIR / f"instances_{split}.json"
        if not json_path.exists():
            print(f"[SKIP] {json_path} not found")
            continue
        print(f"[{split}] 読み込み中: {json_path}")
        coco = load_coco(json_path)
        recs = compute_metrics(coco)
        print(f"  → 図面数: {len(recs):,}")
        all_records.extend(recs)

    print(f"\n合計図面数: {len(all_records):,}\n")

    # ── numpy 配列に変換 ──
    n_rooms   = np.array([r["n_rooms"]        for r in all_records])
    max_verts = np.array([r["max_vertices"]    for r in all_records])
    mean_pp   = np.array([r["mean_pp"]         for r in all_records])
    space_util= np.array([r["space_util"]      for r in all_records])
    area_cv   = np.array([r["area_cv"]         for r in all_records])
    adj_count = np.array([r["adj_count"]       for r in all_records])
    spread    = np.array([r["spatial_spread"]  for r in all_records])

    # ── 統計表示 ──
    print("=== 統計サマリ ===")
    for arr, label in [
        (n_rooms,    "部屋数 (rooms_per_plan)"),
        (max_verts,  "図面の最大頂点数 (max_vertices)"),
        (mean_pp,    "PP スコア平均 (mean_pp)  ※低いほど複雑"),
        (space_util, "充填率 (space_util)"),
        (area_cv,    "面積 CV (area_cv)"),
        (adj_count,  "隣接ペア数 (adj_count)"),
        (spread,     "空間分布の広がり (spatial_spread)"),
    ]:
        print_stats(describe(arr, label))
        print()

    # ── グラフ生成 ──
    print("=== グラフ生成 ===")

    # 01. Rooms per plan histogram
    save_histogram(
        n_rooms, "Rooms per Floor Plan (Distribution)", "Num Rooms", "Num Floor Plans",
        OUTPUT_DIR / "01_rooms_per_plan_hist.png",
        bins=range(0, int(n_rooms.max()) + 2),
        vlines={"mean": float(np.mean(n_rooms)), "median": float(np.median(n_rooms))},
    )

    # 02. Rooms per plan CDF
    save_cumulative(n_rooms, "Rooms per Floor Plan (CDF)", "Num Rooms",
                    OUTPUT_DIR / "02_rooms_per_plan_cdf.png")

    # 03. Max vertices histogram
    save_histogram(
        max_verts, "Max Vertices per Floor Plan (Distribution)", "Max Vertices", "Num Floor Plans",
        OUTPUT_DIR / "03_max_vertices_hist.png",
        bins=50,
        vlines={"mean": float(np.mean(max_verts)), "median": float(np.median(max_verts))},
    )

    # 04. Max vertices CDF
    save_cumulative(max_verts, "Max Vertices per Floor Plan (CDF)", "Max Vertices",
                    OUTPUT_DIR / "04_max_vertices_cdf.png")

    # 05. Polsby-Popper score distribution
    save_histogram(
        mean_pp, "Polsby-Popper Score (mean per plan)\n(1=simple rect, 0=complex shape)",
        "PP Score", "Num Floor Plans",
        OUTPUT_DIR / "05_polsby_popper_hist.png",
        bins=50,
        vlines={"mean": float(np.mean(mean_pp)), "median": float(np.median(mean_pp))},
    )

    # 06. Space utilization distribution
    save_histogram(
        space_util, "Space Utilization (total room area / image area)",
        "Utilization Ratio", "Num Floor Plans",
        OUTPUT_DIR / "06_space_utilization_hist.png",
        bins=50,
        vlines={"mean": float(np.mean(space_util)), "median": float(np.median(space_util))},
    )

    # 07. Area CV distribution
    save_histogram(
        area_cv, "Room Area CV (size heterogeneity)\n(higher = more varied room sizes)",
        "CV", "Num Floor Plans",
        OUTPUT_DIR / "07_area_cv_hist.png",
        bins=50,
        vlines={"mean": float(np.mean(area_cv)), "median": float(np.median(area_cv))},
    )

    # 08. Adjacency count distribution
    save_histogram(
        adj_count, "Adjacent Room Pairs (Distribution)", "Adjacency Count", "Num Floor Plans",
        OUTPUT_DIR / "08_adjacency_hist.png",
        bins=50,
        vlines={"mean": float(np.mean(adj_count)), "median": float(np.median(adj_count))},
    )

    # 09. Spatial spread distribution
    save_histogram(
        spread, "Spatial Spread of Room Centroids\n(normalized std^2 of centroid coords)",
        "Spatial Spread", "Num Floor Plans",
        OUTPUT_DIR / "09_spatial_spread_hist.png",
        bins=50,
        vlines={"mean": float(np.mean(spread)), "median": float(np.median(spread))},
    )

    # 10. Rooms vs max vertices scatter
    save_scatter(
        n_rooms, max_verts,
        "Num Rooms vs Max Vertices", "Num Rooms", "Max Vertices",
        OUTPUT_DIR / "10_rooms_vs_max_vertices.png",
    )

    # 11. Rooms vs area CV scatter
    save_scatter(
        n_rooms, area_cv,
        "Num Rooms vs Area CV", "Num Rooms", "Area CV",
        OUTPUT_DIR / "11_rooms_vs_area_cv.png",
    )

    # 12. Rooms vs adjacency count scatter
    save_scatter(
        n_rooms, adj_count,
        "Num Rooms vs Adjacency Count", "Num Rooms", "Adjacency Count",
        OUTPUT_DIR / "12_rooms_vs_adjacency.png",
    )

    # 13. Space utilization vs PP score scatter
    save_scatter(
        space_util, mean_pp,
        "Space Utilization vs PP Score", "Space Utilization", "Mean PP Score",
        OUTPUT_DIR / "13_space_util_vs_pp.png",
        color=GREEN_COLOR,
    )

    # 14. 相関行列ヒートマップ
    save_correlation_heatmap(all_records, OUTPUT_DIR / "14_correlation_heatmap.png")

    # 15. 全指標ダッシュボード
    save_overview_dashboard(all_records, OUTPUT_DIR / "15_overview_dashboard.png")

    # 16. 複雑さスコア分布
    score = save_complexity_score_dist(all_records, OUTPUT_DIR / "16_complexity_score.png")

    # ── 複雑さ上位・下位 サンプルを出力 ──
    print("\n=== 複雑さスコア 上位10図面 ===")
    top_idx = np.argsort(score)[::-1][:10]
    for rank, i in enumerate(top_idx, 1):
        r = all_records[i]
        print(f"  #{rank:2d}  score={score[i]:.3f}  file={r['file_name']}"
              f"  rooms={r['n_rooms']}  max_verts={r['max_vertices']}"
              f"  pp={r['mean_pp']:.3f}  cv={r['area_cv']:.3f}  adj={r['adj_count']}")

    print("\n=== 複雑さスコア 下位10図面 (簡単) ===")
    bot_idx = np.argsort(score)[:10]
    for rank, i in enumerate(bot_idx, 1):
        r = all_records[i]
        print(f"  #{rank:2d}  score={score[i]:.3f}  file={r['file_name']}"
              f"  rooms={r['n_rooms']}  max_verts={r['max_vertices']}"
              f"  pp={r['mean_pp']:.3f}  cv={r['area_cv']:.3f}  adj={r['adj_count']}")

    # ── CSV 出力 ──
    csv_path = OUTPUT_DIR / "difficulty_metrics.csv"
    import csv
    fieldnames = ["file_name", "n_rooms", "max_vertices", "mean_pp",
                  "space_util", "area_cv", "adj_count", "spatial_spread", "complexity_score"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec, sc in zip(all_records, score):
            writer.writerow({
                "file_name":        rec["file_name"],
                "n_rooms":          rec["n_rooms"],
                "max_vertices":     rec["max_vertices"],
                "mean_pp":          f"{rec['mean_pp']:.4f}",
                "space_util":       f"{rec['space_util']:.4f}",
                "area_cv":          f"{rec['area_cv']:.4f}",
                "adj_count":        rec["adj_count"],
                "spatial_spread":   f"{rec['spatial_spread']:.6f}",
                "complexity_score": f"{sc:.4f}",
            })
    print(f"\nCSV出力: {csv_path}")
    print(f"\n出力先: {OUTPUT_DIR}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
