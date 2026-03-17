#!/usr/bin/env python3
"""
CSV の geom 列（WKT ポリゴン）のみを描画してスケール・座標を確認するスクリプト。
画像（.npy）は使わず、空白キャンバスにポリゴンのみプロットする。

必要なライブラリ: pandas, matplotlib, shapely
  pip install pandas matplotlib shapely
"""

import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shapely import wkt

# -----------------------------------------------------------------------------
# パス・設定
# -----------------------------------------------------------------------------
BASE_DIR = Path("/Users/aokitenju/Downloads/archive-4")
CSV_PATH = BASE_DIR / "mds_V2_5.372k.csv"

# 対象サンプルID
TARGET_ID = 10000

# CSV で ID 照合に試す列の順序
ID_CANDIDATE_COLUMNS = [
    "Unnamed: 0",
    "plan_id",
    "floor_id",
    "building_id",
    "unit_id",
    "area_id",
    "apartment_id",
]

# 色分けに使う列（優先順）
LABEL_COLUMN = "roomtype"
LABEL_COLUMN_FALLBACK = "entity_subtype"

OUTPUT_IMAGE_PATH = Path("csv_only_test.png")
SHOW_PLOT = os.environ.get("SHOW_PLOT", "1").lower() in ("1", "true", "yes")


def filter_csv_rows_by_id(csv_path: Path, target_id) -> pd.DataFrame:
    """対象IDに一致する行を抽出する。chunk で読みメモリに配慮。"""
    target_str = str(target_id)
    target_int = None
    try:
        target_int = int(target_id)
    except (TypeError, ValueError):
        pass

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    columns = [c.strip('"').strip() for c in header.rstrip("\n").split(",")]

    id_cols = [c for c in ID_CANDIDATE_COLUMNS if c in columns]
    if not id_cols:
        raise ValueError(
            f"None of the candidate ID columns found in CSV. Columns: {list(columns)}"
        )

    usecols = id_cols + ["geom"]
    for lb in (LABEL_COLUMN, LABEL_COLUMN_FALLBACK):
        if lb in columns and lb not in usecols:
            usecols.append(lb)

    chunks = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=100_000,
        encoding="utf-8",
        on_bad_lines="warn",
    ):
        for col in id_cols:
            if col not in chunk.columns:
                continue
            if target_int is not None and chunk[col].dtype in (np.int64, np.float64):
                mask = chunk[col] == target_int
            else:
                mask = chunk[col].astype(str) == target_str
            if mask.any():
                chunks.append(chunk.loc[mask])
                break
        else:
            continue
        break

    if not chunks:
        raise ValueError(
            f"No rows found for TARGET_ID={target_id} in columns {id_cols}. "
            "Check TARGET_ID or ID_CANDIDATE_COLUMNS."
        )

    return pd.concat(chunks, ignore_index=True)


def get_polygons_from_geom(geom_obj):
    """shapely の geom から exterior 座標リストを yield。Polygon / MultiPolygon 対応。"""
    if hasattr(geom_obj, "exterior") and geom_obj.exterior is not None:
        yield list(geom_obj.exterior.coords)
    elif hasattr(geom_obj, "geoms"):
        for g in geom_obj.geoms:
            if hasattr(g, "exterior") and g.exterior is not None:
                yield list(g.exterior.coords)


def main():
    target_id = TARGET_ID
    if target_id is None:
        print("Error: TARGET_ID must be set (e.g. 10000).", file=sys.stderr)
        sys.exit(1)
    target_id = int(target_id) if str(target_id).isdigit() else target_id

    if not CSV_PATH.exists():
        print(f"Error: CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # 1) CSV を読み、対象 ID の行を全て抽出
    try:
        rows = filter_csv_rows_by_id(CSV_PATH, target_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(rows)} rows for ID = {target_id}.")

    # ラベル列（色分け用）
    label_col = LABEL_COLUMN if LABEL_COLUMN in rows.columns else LABEL_COLUMN_FALLBACK
    if label_col not in rows.columns:
        label_col = None

    # 2) 空白キャンバスを用意し、Y軸を反転（画像座標系に合わせる）
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.invert_yaxis()

    # 用途ごとに色を割り当て
    label_to_color = {}
    colors = plt.cm.tab10(np.linspace(0, 1, 20))
    color_idx = 0
    legend_handles = []

    # 3) 全ポリゴンをループし、WKT をパースしてプロット
    for _, row in rows.iterrows():
        try:
            geom = wkt.loads(row["geom"])
        except Exception as e:
            print(f"Skip row: failed to parse geom ({e})")
            continue
        if geom.is_empty:
            continue

        label = str(row.get(label_col, "")) if label_col else ""
        if label not in label_to_color:
            label_to_color[label] = colors[color_idx % len(colors)]
            if label:
                legend_handles.append(mpatches.Patch(color=label_to_color[label], label=label))
            color_idx += 1
        color = label_to_color.get(label, "gray")

        for ring in get_polygons_from_geom(geom):
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            # 境界線を描画（塗りつぶしは薄く）
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.9)
            ax.fill(xs, ys, color=color, alpha=0.12)

    # 4) 部屋用途ごとの凡例
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    # 5) X軸・Y軸のメモリ（座標値）は表示したまま
    ax.set_xlabel("X (world coords)")
    ax.set_ylabel("Y (world coords)")
    ax.set_title(f"CSV polygons only (ID={target_id})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # 6) 保存
    out_path = Path(OUTPUT_IMAGE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
