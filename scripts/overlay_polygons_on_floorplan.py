#!/usr/bin/env python3
"""
フロアプラン画像（.npy）の上に、CSVメタデータのWKTポリゴン（geom列）を
重畳描画して、座標系の対応関係を視覚的に検証するスクリプト。

必要なライブラリ: numpy, pandas, matplotlib, shapely
  pip install numpy pandas matplotlib shapely

headless 環境では SHOW_PLOT=0 で実行するとウィンドウを開きません。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely import wkt
from shapely.geometry import Polygon

# -----------------------------------------------------------------------------
# パス設定
# -----------------------------------------------------------------------------
BASE_DIR = Path("/Users/aokitenju/Downloads/archive-4")
IMAGE_DIR = BASE_DIR / "modified-swiss-dwellings-v2" / "train" / "struct_in"
CSV_PATH = BASE_DIR / "mds_V2_5.372k.csv"

# 対象サンプルID（None の場合は struct_in 内の最初の .npy ファイル名を自動取得）
TARGET_ID = None  # 例: 10000 や "10000" に変更可能

# 画像サイズ（.npy の形状に合わせる）
IMAGE_H, IMAGE_W = 512, 512

# CSV で ID 照合に試す列の順序（先にマッチした列を使用）
ID_CANDIDATE_COLUMNS = [
    "Unnamed: 0",  # 調査でファイル名と一致する可能性が高い
    "plan_id",
    "floor_id",
    "building_id",
    "unit_id",
    "area_id",
    "apartment_id",
]

# ラベル表示用の列（存在すれば使用）
LABEL_COLUMN = "roomtype"  # なければ entity_subtype にフォールバック
LABEL_COLUMN_FALLBACK = "entity_subtype"

# -----------------------------------------------------------------------------
# 座標変換パラメータ（後から調整しやすいように分離）
# ワールド座標 (WKT) → 画像座標 の変換式:
#   img_x = (world_x - offset_x) * scale_x
#   img_y = (world_y - offset_y) * scale_y  （必要に応じて Y 反転は下記で）
# -----------------------------------------------------------------------------
# ポリゴン群のワールド座標範囲からスケール・オフセットを決める場合に使用
USE_WORLD_BBOX_FOR_TRANSFORM = True  # True: 当該サンプルの全ポリゴンの bbox で正規化

# 固定のスケール・オフセットを使う場合（USE_WORLD_BBOX_FOR_TRANSFORM=False のとき）
SCALE_X = 1.0
SCALE_Y = 1.0
OFFSET_X = 0.0
OFFSET_Y = 0.0

# 画像の Y 軸は上から下が正のため、WKT 座標の Y を反転するか
FLIP_Y = True

# 出力
OUTPUT_IMAGE_PATH = Path("overlay_test.png")
# 環境変数 DISPLAY が無い場合や headless では False にすると安全
SHOW_PLOT = os.environ.get("SHOW_PLOT", "1").lower() in ("1", "true", "yes")


def get_first_npy_stem(image_dir: Path) -> str:
    """画像ディレクトリ内の最初の .npy ファイル名（拡張子なし）を返す。"""
    npy_files = sorted(image_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {image_dir}")
    return npy_files[0].stem


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """.npy を読み込み、0〜255 の uint8 にクリップして返す。"""
    arr = np.load(image_path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (H,W,C), got shape {arr.shape}")
    # マイナスは 0、255 超は 255 にクリップ
    arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def filter_csv_rows_by_id(csv_path: Path, target_id) -> pd.DataFrame:
    """
    対象IDに一致する行を取得する。
    一致する列を動的に探す（plan_id, apartment_id, building_id 等）。
    メモリ対策のため、該当IDの行だけを chunk で読み込む。
    """
    target_str = str(target_id)
    target_int = None
    try:
        target_int = int(target_id)
    except (TypeError, ValueError):
        pass

    # まずヘッダーだけ取得
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    columns = [c.strip('"').strip() for c in header.rstrip("\n").split(",")]

    # 試す ID 列（CSV に存在するものだけ）
    id_cols = [c for c in ID_CANDIDATE_COLUMNS if c in columns]
    if not id_cols:
        raise ValueError(
            f"None of the candidate ID columns {ID_CANDIDATE_COLUMNS} found in CSV. "
            f"Columns: {list(columns)}"
        )

    usecols = id_cols + ["geom"]
    for lb in (LABEL_COLUMN, LABEL_COLUMN_FALLBACK):
        if lb in columns and lb not in usecols:
            usecols.append(lb)

    # chunk で読みながら、いずれかの列が target と一致する行を集める
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


def world_to_image_coords(
    world_x: np.ndarray,
    world_y: np.ndarray,
    world_bounds: tuple,
    image_width: int = IMAGE_W,
    image_height: int = IMAGE_H,
) -> tuple:
    """
    ワールド座標 (world_x, world_y) を画像座標 (img_x, img_y) に変換する。
    変換式をここで一元化し、スケール・オフセットは後から変更しやすいようにする。

    world_bounds: (wx_min, wy_min, wx_max, wy_max)
    """
    wx_min, wy_min, wx_max, wy_max = world_bounds
    if USE_WORLD_BBOX_FOR_TRANSFORM:
        range_x = (wx_max - wx_min) or 1.0
        range_y = (wy_max - wy_min) or 1.0
        scale_x = image_width / range_x
        scale_y = image_height / range_y
        offset_x = wx_min
        offset_y = wy_min
    else:
        scale_x, scale_y = SCALE_X, SCALE_Y
        offset_x, offset_y = OFFSET_X, OFFSET_Y

    img_x = (world_x - offset_x) * scale_x
    img_y = (world_y - offset_y) * scale_y
    if FLIP_Y:
        img_y = image_height - 1 - img_y
    return img_x, img_y


def collect_world_bounds(rows: pd.DataFrame) -> tuple:
    """行の geom から全ポリゴンのワールド座標範囲 (wx_min, wy_min, wx_max, wy_max) を返す。"""
    wx_min = wy_min = np.inf
    wx_max = wy_max = -np.inf
    for _, row in rows.iterrows():
        try:
            geom = wkt.loads(row["geom"])
        except Exception:
            continue
        if geom.is_empty:
            continue
        if hasattr(geom, "exterior") and geom.exterior is not None:
            xs = np.array(geom.exterior.coords.xy[0])
            ys = np.array(geom.exterior.coords.xy[1])
        elif hasattr(geom, "geoms"):
            for g in geom.geoms:
                if hasattr(g, "exterior") and g.exterior is not None:
                    xs = np.array(g.exterior.coords.xy[0])
                    ys = np.array(g.exterior.coords.xy[1])
                    wx_min, wx_max = min(wx_min, xs.min()), max(wx_max, xs.max())
                    wy_min, wy_max = min(wy_min, ys.min()), max(wy_max, ys.max())
            continue
        else:
            continue
        wx_min, wx_max = min(wx_min, xs.min()), max(wx_max, xs.max())
        wy_min, wy_max = min(wy_min, ys.min()), max(wy_max, ys.max())

    if wx_min == np.inf:
        return 0.0, 0.0, 1.0, 1.0
    return wx_min, wy_min, wx_max, wy_max


def main():
    # 1) 対象IDの決定（モジュール変数 TARGET_ID は書き換えず、ローカル変数で扱う）
    if TARGET_ID is None:
        target_id = get_first_npy_stem(IMAGE_DIR)
        print(f"Using first .npy in directory: TARGET_ID = {target_id}")
    else:
        target_id = str(TARGET_ID).strip()
        if target_id.isdigit():
            target_id = int(target_id)
        print(f"TARGET_ID = {target_id}")

    image_path = IMAGE_DIR / f"{target_id}.npy"
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    if not CSV_PATH.exists():
        print(f"Error: CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # 2) 画像の読み込みと前処理
    img = load_and_preprocess_image(image_path)
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h != IMAGE_H or img_w != IMAGE_W:
        print(f"Warning: Image shape {img.shape} != ({IMAGE_H}, {IMAGE_W}). Using actual size for display.")

    # 3) メタデータの読み込みとフィルタリング
    try:
        rows = filter_csv_rows_by_id(CSV_PATH, target_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(rows)} rows for this ID.")

    # ワールド座標範囲を取得（変換用）
    world_bounds = collect_world_bounds(rows)

    # 4) ポリゴンのパースと描画
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, origin="upper", extent=[0, img_w, img_h, 0])
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")

    label_col = LABEL_COLUMN if LABEL_COLUMN in rows.columns else LABEL_COLUMN_FALLBACK
    if label_col not in rows.columns:
        label_col = None

    # 凡例用（ラベル → 色を固定）
    legend_handles = []
    label_to_color = {}
    colors = plt.cm.tab10(np.linspace(0, 1, 20))
    color_idx = 0

    for idx, row in rows.iterrows():
        try:
            geom = wkt.loads(row["geom"])
        except Exception as e:
            print(f"Skip row {idx}: failed to parse geom ({e})")
            continue
        if geom.is_empty:
            continue

        # Polygon または MultiPolygon の exterior を取得
        polygons_to_draw = []
        if hasattr(geom, "exterior") and geom.exterior is not None:
            polygons_to_draw.append(geom)
        elif hasattr(geom, "geoms"):
            for g in geom.geoms:
                if hasattr(g, "exterior") and g.exterior is not None:
                    polygons_to_draw.append(g)

        label = str(row.get(label_col, "")) if label_col else ""
        if label not in label_to_color:
            label_to_color[label] = colors[color_idx % len(colors)]
            if label:
                legend_handles.append(mpatches.Patch(color=label_to_color[label], label=label))
            color_idx += 1
        color = label_to_color[label]

        for poly in polygons_to_draw:
            xs = np.array(poly.exterior.coords.xy[0])
            ys = np.array(poly.exterior.coords.xy[1])
            img_x, img_y = world_to_image_coords(xs, ys, world_bounds, img_w, img_h)
            ax.plot(img_x, img_y, color=color, linewidth=1.5, alpha=0.9)
            ax.fill(img_x, img_y, color=color, alpha=0.15)

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    ax.set_title(f"Floor plan overlay (ID={target_id})")
    plt.tight_layout()

    # 5) 出力
    out_path = Path(OUTPUT_IMAGE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
