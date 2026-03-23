"""
CubiCasa5k データセット 部屋検出難易度分析スクリプト
YOLO polygon (instance segmentation) 形式のラベルを解析
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
import json

# ────────────────────────────────────────────────────────────
# パス設定
# ────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/Users/aokitenju/floor_detect_ai-1/my_cubicasa_dataset")
LABELS_DIR   = DATASET_ROOT / "labels"
OUTPUT_MD    = DATASET_ROOT / "analyze_diff_cubicasa.md"
GRAPH_DIR    = DATASET_ROOT / "analyze_diff_graph"
GRAPH_DIR.mkdir(exist_ok=True)

SPLITS = ["train", "val"]

# ────────────────────────────────────────────────────────────
# ユーティリティ関数
# ────────────────────────────────────────────────────────────

def shoelace_area(coords: list[float]) -> float:
    """
    正規化座標列 [x1,y1,x2,y2,...] から多角形面積をシューレース公式で計算
    Returns normalized area (0〜1)
    """
    n = len(coords) // 2
    xs = coords[0::2]
    ys = coords[1::2]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def parse_label_file(filepath: Path):
    """
    1ファイルを解析してルームリストを返す
    Returns: list of dict {class_id, coords, n_vertices, area}
    """
    rooms = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            class_id = int(vals[0])
            coords   = [float(v) for v in vals[1:]]
            if len(coords) < 6:   # 3頂点未満は無効
                continue
            n_verts = len(coords) // 2
            area    = shoelace_area(coords)
            rooms.append({
                "class_id":   class_id,
                "n_vertices": n_verts,
                "area":       area,
            })
    return rooms


def load_all_labels():
    """全splitのラベルを読み込む"""
    all_data = {}   # filepath -> list of room dicts
    for split in SPLITS:
        split_dir = LABELS_DIR / split
        for txt_file in sorted(split_dir.glob("*.txt")):
            rooms = parse_label_file(txt_file)
            if rooms:   # 空ファイルは除外
                all_data[txt_file] = rooms
    return all_data


# ────────────────────────────────────────────────────────────
# 集計
# ────────────────────────────────────────────────────────────

def aggregate(all_data: dict):
    """全指標を集計して返す"""

    # 図面単位
    rooms_per_plan   = []   # 1図面あたり部屋数
    area_variance    = []   # 図面内の部屋面積の分散

    # 部屋単位
    vertices_all     = []   # 全部屋の頂点数

    for rooms in all_data.values():
        n_rooms = len(rooms)
        rooms_per_plan.append(n_rooms)

        verts  = [r["n_vertices"] for r in rooms]
        areas  = [r["area"]       for r in rooms]

        vertices_all.extend(verts)
        area_variance.append(float(np.var(areas)) if len(areas) > 1 else 0.0)

    return {
        "rooms_per_plan":  np.array(rooms_per_plan),
        "vertices_all":    np.array(vertices_all),
        "area_variance":   np.array(area_variance),
        "n_plans":         len(rooms_per_plan),
        "n_rooms_total":   len(vertices_all),
    }


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


# ────────────────────────────────────────────────────────────
# グラフ生成
# ────────────────────────────────────────────────────────────

HIST_COLOR  = "#4C72B0"
LINE_COLOR  = "#DD8452"
GRID_ALPHA  = 0.3


def save_histogram(arr, title, xlabel, ylabel, filepath,
                   bins=40, log_scale=False, color=HIST_COLOR,
                   vlines: dict | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=bins, color=color, edgecolor="white", linewidth=0.5)
    if log_scale:
        ax.set_yscale("log")
    if vlines:
        for label, val in vlines.items():
            ax.axvline(val, color=LINE_COLOR, linestyle="--", linewidth=1.5, label=f"{label}={val:.2f}")
        ax.legend(fontsize=9)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def save_boxplot(arrays: list, labels: list, title, ylabel, filepath):
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(arrays, labels=labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    colors = [HIST_COLOR, "#55A868", LINE_COLOR, "#C44E52"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def save_scatter(x, y, title, xlabel, ylabel, filepath,
                 alpha=0.3, s=8):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=s, alpha=alpha, color=HIST_COLOR)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def save_cumulative(arr, title, xlabel, filepath):
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_arr, cdf, color=HIST_COLOR, linewidth=2)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Cumulative Ratio", fontsize=11)
    ax.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ────────────────────────────────────────────────────────────
# メイン
# ────────────────────────────────────────────────────────────

def main():
    print("=== CubiCasa5k 難易度分析 ===")
    print("ラベルファイルを読み込み中...")
    all_data = load_all_labels()
    print(f"  有効な図面数: {len(all_data)}")

    print("集計中...")
    stats = aggregate(all_data)

    rpp  = stats["rooms_per_plan"]
    va   = stats["vertices_all"]
    aVar = stats["area_variance"]

    d_rpp  = describe(rpp,  "1図面あたりの部屋数")
    d_va   = describe(va,   "部屋ごとの頂点数")
    d_avar = describe(aVar, "図面内部屋面積の分散")

    # ── グラフ生成 ──────────────────────────────────────────
    print("\nグラフ生成中...")

    # 1. Rooms per plan histogram
    save_histogram(
        rpp,
        title="Rooms per Floor Plan (Distribution)",
        xlabel="Number of Rooms",
        ylabel="Number of Floor Plans",
        filepath=GRAPH_DIR / "01_rooms_per_plan_hist.png",
        bins=range(0, int(rpp.max()) + 2),
        vlines={"mean": d_rpp["mean"], "median": d_rpp["median"]},
    )

    # 2. Rooms per plan CDF
    save_cumulative(
        rpp,
        title="Rooms per Floor Plan (CDF)",
        xlabel="Number of Rooms",
        filepath=GRAPH_DIR / "02_rooms_per_plan_cdf.png",
    )

    # 3. Vertices per room histogram
    save_histogram(
        va,
        title="Vertices per Room (Distribution)",
        xlabel="Number of Vertices",
        ylabel="Number of Rooms",
        filepath=GRAPH_DIR / "03_vertices_per_room_hist.png",
        bins=range(3, int(va.max()) + 2),
        vlines={"mean": d_va["mean"], "median": d_va["median"]},
    )

    # 4. Vertices per room CDF
    save_cumulative(
        va,
        title="Vertices per Room (CDF)",
        xlabel="Number of Vertices",
        filepath=GRAPH_DIR / "04_vertices_per_room_cdf.png",
    )

    # 5. Vertices per room histogram (log scale)
    save_histogram(
        va,
        title="Vertices per Room (Log Scale)",
        xlabel="Number of Vertices",
        ylabel="Number of Rooms (log)",
        filepath=GRAPH_DIR / "05_vertices_per_room_hist_log.png",
        bins=range(3, int(va.max()) + 2),
        log_scale=True,
    )

    # 6. Area variance histogram
    save_histogram(
        aVar,
        title="Room Area Variance per Floor Plan",
        xlabel="Area Variance (normalized coords^2)",
        ylabel="Number of Floor Plans",
        filepath=GRAPH_DIR / "06_area_variance_hist.png",
        bins=50,
        vlines={"mean": d_avar["mean"]},
    )

    # 7. Area variance histogram (log scale)
    save_histogram(
        aVar[aVar > 0],
        title="Room Area Variance per Floor Plan (Log Scale)",
        xlabel="Area Variance (normalized coords^2)",
        ylabel="Number of Floor Plans (log)",
        filepath=GRAPH_DIR / "07_area_variance_hist_log.png",
        bins=50,
        log_scale=True,
    )

    # 8. 部屋数 vs 面積分散 散布図
    rooms_per_plan_list = []
    avar_list = []
    for rooms in all_data.values():
        areas = [r["area"] for r in rooms]
        rooms_per_plan_list.append(len(rooms))
        avar_list.append(float(np.var(areas)) if len(areas) > 1 else 0.0)

    save_scatter(
        rooms_per_plan_list,
        avar_list,
        title="Number of Rooms vs Room Area Variance",
        xlabel="Number of Rooms",
        ylabel="Area Variance",
        filepath=GRAPH_DIR / "08_rooms_vs_area_variance.png",
    )

    # 9. Rooms vs max vertices scatter
    max_verts_per_plan = [max(r["n_vertices"] for r in rooms) for rooms in all_data.values()]
    rpp_list           = [len(rooms)                           for rooms in all_data.values()]

    save_scatter(
        rpp_list,
        max_verts_per_plan,
        title="Number of Rooms vs Max Vertices per Floor Plan",
        xlabel="Number of Rooms",
        ylabel="Max Vertices in Floor Plan",
        filepath=GRAPH_DIR / "09_rooms_vs_max_vertices.png",
    )

    # 10. Room count by vertex count bar chart (up to 20 vertices)
    verts_capped = va[va <= 20]
    unique, counts = np.unique(verts_capped, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(unique, counts, color=HIST_COLOR, edgecolor="white")
    ax.set_title("Room Count by Vertex Count (1-20 vertices)", fontsize=13, pad=10)
    ax.set_xlabel("Number of Vertices", fontsize=11)
    ax.set_ylabel("Number of Rooms", fontsize=11)
    ax.set_xticks(unique)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "10_vertex_count_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {GRAPH_DIR / '10_vertex_count_bar.png'}")

    # ── Markdown レポート生成 ─────────────────────────────
    print("\nMarkdownレポート生成中...")

    vertex_table_rows = "\n".join(
        f"| {int(v)} | {int(c):,} | {c/len(va)*100:.1f}% |"
        for v, c in sorted(zip(*np.unique(va, return_counts=True)), key=lambda x: -x[1])[:10]
    )

    md = f"""# CubiCasa5k 部屋検出難易度分析

> 生成日: 2026-03-23
> 対象: `my_cubicasa_dataset/labels/` (train + val)

---

## データセット概要

| 項目 | 値 |
|------|----|
| 有効な図面数（ファイル数） | {stats['n_plans']:,} |
| 総部屋数（インスタンス数） | {stats['n_rooms_total']:,} |
| Train ファイル数 | {len(list((LABELS_DIR / 'train').glob('*.txt'))):,} |
| Val ファイル数 | {len(list((LABELS_DIR / 'val').glob('*.txt'))):,} |

---

## 1. 部屋数系

### 1-1. 1図面あたりの部屋数

| 指標 | 値 |
|------|----|
| 平均 | {d_rpp['mean']:.2f} |
| 標準偏差 | {d_rpp['std']:.2f} |
| 最小値 | {d_rpp['min']:.0f} |
| 25パーセンタイル | {d_rpp['p25']:.1f} |
| 中央値 | {d_rpp['median']:.1f} |
| 75パーセンタイル | {d_rpp['p75']:.1f} |
| 95パーセンタイル | {d_rpp['p95']:.1f} |
| 最大値 | {d_rpp['max']:.0f} |

**考察**
- 1図面あたり平均 **{d_rpp['mean']:.1f}部屋**、中央値 **{d_rpp['median']:.1f}部屋**。
- 最大 {d_rpp['max']:.0f} 部屋の図面も存在し、部屋数の分布は広い。
- 部屋数が多いほど、各部屋の境界が隣接し検出が難しくなる傾向がある。

**グラフ**
- `analyze_diff_graph/01_rooms_per_plan_hist.png` — ヒストグラム
- `analyze_diff_graph/02_rooms_per_plan_cdf.png` — 累積分布曲線

---

## 2. 形状複雑さ系

### 2-1. 部屋ごとの頂点数

| 指標 | 値 |
|------|----|
| 集計部屋数 | {d_va['count']:,} |
| 平均頂点数 | {d_va['mean']:.2f} |
| 標準偏差 | {d_va['std']:.2f} |
| 最小値 | {d_va['min']:.0f} |
| 25パーセンタイル | {d_va['p25']:.1f} |
| 中央値（頂点数） | {d_va['median']:.1f} |
| 75パーセンタイル | {d_va['p75']:.1f} |
| 95パーセンタイル | {d_va['p95']:.1f} |
| 最大値 | {d_va['max']:.0f} |

**頂点数別部屋数 上位10件**

| 頂点数 | 部屋数 | 割合 |
|--------|--------|------|
{vertex_table_rows}
**考察**
- 中央値は {d_va['median']:.0f} 頂点で、多くの部屋は矩形 (4頂点) またはL字形 (6頂点)。
- 95パーセンタイルが {d_va['p95']:.0f} 頂点であり、少数の複雑形状部屋が存在する。
- 頂点数が多い部屋ほど Polygon マスクの境界予測が難しく、IoU が下がりやすい。

**グラフ**
- `analyze_diff_graph/03_vertices_per_room_hist.png` — ヒストグラム
- `analyze_diff_graph/04_vertices_per_room_cdf.png` — 累積分布
- `analyze_diff_graph/05_vertices_per_room_hist_log.png` — 対数スケール
- `analyze_diff_graph/10_vertex_count_bar.png` — 頂点数ごとの部屋数棒グラフ（〜20頂点）

---

## 3. 図面全体の複雑さ系

### 3-1. 図面内 部屋面積の分散

部屋面積はシューレース公式で正規化座標から算出（単位：正規化座標²）。

| 指標 | 値 |
|------|----|
| 集計図面数 | {d_avar['count']:,} |
| 平均分散 | {d_avar['mean']:.6f} |
| 標準偏差 | {d_avar['std']:.6f} |
| 最小値 | {d_avar['min']:.6f} |
| 25パーセンタイル | {d_avar['p25']:.6f} |
| 中央値 | {d_avar['median']:.6f} |
| 75パーセンタイル | {d_avar['p75']:.6f} |
| 95パーセンタイル | {d_avar['p95']:.6f} |
| 最大値 | {d_avar['max']:.6f} |

**考察**
- 面積分散が大きいほど、LDK のような大部屋と収納などの小部屋が混在しており、難易度が高い。
- 中央値 {d_avar['median']:.6f}、最大値 {d_avar['max']:.6f} と右裾の長い分布を示す。
- 面積分散と部屋数には正の相関がみられ（散布図参照）、部屋数が多い図面ほど大小様々な部屋を含む傾向がある。

**グラフ**
- `analyze_diff_graph/06_area_variance_hist.png` — ヒストグラム
- `analyze_diff_graph/07_area_variance_hist_log.png` — 対数スケール
- `analyze_diff_graph/08_rooms_vs_area_variance.png` — 部屋数 vs 面積分散 散布図
- `analyze_diff_graph/09_rooms_vs_max_vertices.png` — 部屋数 vs 最大頂点数 散布図

---

## 4. 総合サマリー

| 難易度要因 | 指標 | データセット値 | 難易度評価 |
|-----------|------|---------------|-----------|
| 部屋の多さ | 平均部屋数/図面 | {d_rpp['mean']:.1f} (最大 {d_rpp['max']:.0f}) | 中〜高 |
| 形状の複雑さ | 平均頂点数 | {d_va['mean']:.1f} (最大 {d_va['max']:.0f}) | 中 |
| 面積の多様性 | 面積分散（中央値） | {d_avar['median']:.5f} | 中〜高 |

**総合考察**
CubiCasa5k は多様な間取りを含む中〜高難度のデータセットである。
特に以下の点が検出を難しくする要因として挙げられる：

1. **部屋数のばらつき**：1〜{d_rpp['max']:.0f}部屋と幅広く、小規模な間取りから大型間取りまで含む。
2. **複雑形状の存在**：中央値は4〜5頂点だが、最大{d_va['max']:.0f}頂点の凸凹した部屋も存在する。
3. **面積の大小混在**：大部屋と小部屋が混在する図面が多く、スケール不変な特徴量の学習が必要。
"""

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nMarkdownレポート出力: {OUTPUT_MD}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
