"""
YOLO + SAM の分割推論（パッチベース推論）でセグメンテーションマスクを生成するスクリプト。

- 入力画像を SAHI を用いて 2x2 程度にスライス（オーバーラップ 20%）
- 各パッチで YOLO → BBox フィルタ → SAM 推論 → マスクを取得
- スライスごとのマスクをオフセットでキャンバスに論理和で結合
- 結合マスクにモルフォロジー・面積ベース被り除去を適用

既存の predict.py / segment_SAM.py / segment_SAM_post_pri_small.py は改変しない。
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

# -----------------------------------------------------------------------------
# CONFIG: パス・パラメータのまとめ（後から書き換えやすいように冒頭に集約）
# -----------------------------------------------------------------------------
CONFIG = {
    # 画像拡張子
    "image_extensions": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    # デフォルトパス（プロジェクトルート相対）
    "default_weights_rel": "runs/detect/train/weights/best.pt",
    "default_source_rel": "kaggle/input/datasets/umairinayat/floor-plans-500-annotated-object-detection/archive-3/test/images",
    # YOLO
    "imgsz": 640,
    "conf": 0.25,
    # BBox 面積フィルタ：この値以下の zone はノイズとして除外（ピクセル²）
    "MIN_AREA_THRESHOLD": 10000,
    # スライス（2x2, 20% オーバーラップ）
    "overlap_ratio": 0.2,
    "slice_rows": 2,
    "slice_cols": 2,
    # SAM
    "sam_model": "sam_b.pt",
    "zone_margin": 10.0,
    # 後処理（segment_SAM_post_pri_small と同様）
    "morph_kernel": 30,
    "epsilon_rate": 0.002,
    # 出力
    "save_dir": "results",
    "output_dir": "segment_results_sliced",
}

# SAHI 利用時（オプション）: get_slice_bboxes でスライス座標を取得する場合
try:
    from sahi.slicing import get_slice_bboxes as _sahi_get_slice_bboxes
    _SAHI_AVAILABLE = True
except ImportError:
    _SAHI_AVAILABLE = False


# =============================================================================
# 1. キャンバス初期化
# =============================================================================

def create_mask_canvas(height: int, width: int, channels: int = 1) -> np.ndarray:
    """
    マスク結合用のゼロ配列（黒画像キャンバス）を作成する。
    元の入力画像と同じ高さ・幅・チャンネル数で、uint8 のゼロ配列を返す。
    マスクは1チャンネルで十分なため、通常は (H, W) を返す。
    """
    if channels == 1:
        return np.zeros((height, width), dtype=np.uint8)
    return np.zeros((height, width, channels), dtype=np.uint8)


# =============================================================================
# 2. 画像のスライス（SAHI または自前計算）
# =============================================================================

def get_slice_dimensions(
    height: int, width: int,
    overlap_ratio: float,
    slice_rows: int, slice_cols: int,
) -> Tuple[int, int]:
    """
    縦 slice_rows × 横 slice_cols で overlap_ratio のオーバーラップになる
    slice_height, slice_width を計算する。
    ステップ = slice * (1 - overlap_ratio) で、n 枚で全体をカバーするには
    slice * (1 + (n-1)*(1-overlap)) >= size となる。
    """
    # 縦: (slice_rows - 1) ステップ + 1 スライスでカバー
    denom_h = 1.0 + (slice_rows - 1) * (1.0 - overlap_ratio)
    denom_w = 1.0 + (slice_cols - 1) * (1.0 - overlap_ratio)
    slice_height = int(math.ceil(height / denom_h))
    slice_width = int(math.ceil(width / denom_w))
    return slice_height, slice_width


def get_slice_bboxes_fallback(
    height: int, width: int,
    overlap_ratio: float = 0.2,
    slice_rows: int = 2, slice_cols: int = 2,
) -> List[Tuple[int, int, int, int]]:
    """
    SAHI を使わない場合のスライス座標計算。
    返り値: [(x_min, y_min, x_max, y_max), ...] のリスト（画像座標）。
    """
    slice_height, slice_width = get_slice_dimensions(
        height, width, overlap_ratio, slice_rows, slice_cols
    )
    step_h = int(slice_height * (1.0 - overlap_ratio))
    step_w = int(slice_width * (1.0 - overlap_ratio))
    bboxes = []
    for row in range(slice_rows):
        for col in range(slice_cols):
            y_min = row * step_h
            x_min = col * step_w
            y_max = min(y_min + slice_height, height)
            x_max = min(x_min + slice_width, width)
            if x_max > x_min and y_max > y_min:
                bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes


def get_slice_bboxes(
    height: int, width: int,
    overlap_ratio: float = 0.2,
    slice_rows: int = 2, slice_cols: int = 2,
) -> List[Tuple[int, int, int, int]]:
    """
    全体画像における各スライスの開始座標（オフセット）と範囲を取得する。
    SAHI が利用可能な場合は get_slice_bboxes を使用し、否则は自前計算。
    返り値: [(x_min, y_min, x_max, y_max), ...]
    """
    if _SAHI_AVAILABLE:
        slice_height, slice_width = get_slice_dimensions(
            height, width, overlap_ratio, slice_rows, slice_cols
        )
        try:
            # SAHI: (image_height, image_width, slice_height, slice_width, overlap_ratio, ...)
            bboxes = _sahi_get_slice_bboxes(
                image_height=height,
                image_width=width,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
            )
            # SAHI の形式に合わせて (x_min, y_min, x_max, y_max) のリストに変換
            if hasattr(bboxes, "__iter__") and len(bboxes) > 0:
                out = []
                for b in bboxes:
                    if hasattr(b, "minx"):
                        out.append((int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)))
                    else:
                        out.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
                return out
        except Exception:
            pass
    return get_slice_bboxes_fallback(
        height, width, overlap_ratio, slice_rows, slice_cols
    )


def slice_image(
    image: np.ndarray,
    slice_bboxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    入力画像をスライスし、各スライス画像とその (x_min, y_min, x_max, y_max) を返す。
    返り値: [(slice_image, (x_min, y_min, x_max, y_max)), ...]
    """
    result = []
    for (x_min, y_min, x_max, y_max) in slice_bboxes:
        slice_img = image[y_min:y_max, x_min:x_max].copy()
        result.append((slice_img, (x_min, y_min, x_max, y_max)))
    return result


# =============================================================================
# 3. パッチごとの推論（YOLO → BBox フィルタ → SAM）
# =============================================================================

def extract_prompts_from_result(r, zone_margin: float = 10.0):
    """
    YOLO 結果 r から zone_boxes と door/window の negative_points を抽出。
    既存の segment_SAM.py のロジックに準拠。
    返り値: [{"zone_box": [x1,y1,x2,y2], "negative_points": [[x,y], ...]}, ...]
    """
    zone_data = []
    all_door_window = []

    for box in r.boxes:
        cls_id = int(box.cls[0])
        class_name = r.names[cls_id]
        xyxy = box.xyxy[0].tolist()

        if class_name == "zone":
            zone_data.append({"zone_box": xyxy, "negative_points": []})
        elif class_name in ("door", "window"):
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            all_door_window.append([center_x, center_y])

    for zd in zone_data:
        x1, y1, x2, y2 = zd["zone_box"]
        x1_m = x1 - zone_margin
        y1_m = y1 - zone_margin
        x2_m = x2 + zone_margin
        y2_m = y2 + zone_margin
        for pt in all_door_window:
            px, py = pt
            if x1_m <= px <= x2_m and y1_m <= py <= y2_m:
                zd["negative_points"].append(pt)

    return zone_data


def filter_zone_boxes_by_area(
    zone_prompts: List[dict],
    min_area_threshold: float,
) -> List[dict]:
    """
    BBox の面積（幅×高さ）が min_area_threshold 以下の zone を除外する。
    """
    filtered = []
    for zp in zone_prompts:
        x1, y1, x2, y2 = zp["zone_box"]
        area = (x2 - x1) * (y2 - y1)
        if area > min_area_threshold:
            filtered.append(zp)
    return filtered


def select_best_mask(masks: np.ndarray, box_area: float) -> Optional[np.ndarray]:
    """マルチマスク候補から、YOLO Box の面積に最も近いマスクを選択。"""
    if masks is None or masks.size == 0:
        return None
    areas = np.array([np.sum(m > 0) for m in masks])
    idx = np.argmin(np.abs(areas - box_area))
    return masks[idx]


def run_sam_for_one_box(
    predictor: SAMPredictor,
    image: np.ndarray,
    box: list,
    negative_points: list,
) -> Optional[np.ndarray]:
    """
    SAM で 1 つの Bounding Box だけをプロンプトとしてセグメントする。
    複数 Box をまとめて渡すと RuntimeError になるため、必ず要素数4の1個の box のみ渡す。
    box: [x1, y1, x2, y2]
    処理内容は segment_SAM.run_sam_for_zone に揃えている（bboxes / points / labels の渡し方）。
    """
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)

    points = negative_points
    labels = [0] * len(negative_points) if points else None

    # Ultralytics SAM: bboxes は (N,4) 形式、points と labels は任意（segment_SAM.py と同じ）
    bboxes = np.array([box], dtype=np.float32)

    predictor.set_image(image)
    results = predictor(
        bboxes=bboxes,
        points=np.array(points, dtype=np.float32) if points else None,
        labels=np.array(labels, dtype=np.int64) if labels else None,
        multimask_output=True,
    )
    predictor.reset_image()

    if not results or len(results) == 0:
        return None
    r = results[0]
    if r.masks is None or r.masks.data is None:
        return None
    masks = r.masks.data.cpu().numpy()
    best = select_best_mask(masks, box_area)
    return best


def run_patch_inference(
    yolo_model: YOLO,
    sam_predictor: SAMPredictor,
    slice_img: np.ndarray,
    min_area_threshold: float,
    zone_margin: float,
) -> Optional[np.ndarray]:
    """
    1 枚のスライス画像に対して YOLO → BBox フィルタ → SAM を実行し、
    そのスライス内の全ゾーンマスクを論理和で結合したマスクを返す。
    複数 Box を一度に SAM に渡さず、1 Box ずつループで推論し、ローカルキャンバスに重ね合わせる。
    """
    # 3a. YOLO 推論（スライス画像上でローカルな BBox 取得）
    yolo_results = yolo_model.predict(
        slice_img,
        imgsz=CONFIG["imgsz"],
        conf=CONFIG["conf"],
        verbose=False,
    )
    if not yolo_results or len(yolo_results) == 0:
        return None
    r = yolo_results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    zone_prompts = extract_prompts_from_result(r, zone_margin=zone_margin)
    # 3b. 極小ノイズ Box を除外
    filtered_zone_prompts = filter_zone_boxes_by_area(zone_prompts, min_area_threshold)
    if not filtered_zone_prompts:
        return None

    # スライス用のローカルキャンバス（ゼロ配列）。ここに個別マスクを重ね合わせる。
    slice_h, slice_w = slice_img.shape[:2]
    slice_canvas = np.zeros((slice_h, slice_w), dtype=np.uint8)

    # 3c. 1 Box ずつループで SAM 推論し、出力マスクをローカルキャンバスに論理和で合成
    for zp in filtered_zone_prompts:
        box = zp["zone_box"]  # 要素数4の1次元リスト [x1, y1, x2, y2]
        mask = run_sam_for_one_box(
            sam_predictor,
            slice_img,
            box,
            zp["negative_points"],
        )
        if mask is None:
            continue
        mask_bin = (mask > 0).astype(np.uint8)
        slice_canvas = cv2.bitwise_or(slice_canvas, mask_bin)

    # 1 枚もマスクが重ならなかった場合は None
    if np.max(slice_canvas) == 0:
        return None
    return slice_canvas


# =============================================================================
# 4. グローバルマスクへの結合
# =============================================================================

def merge_slice_mask_into_canvas(
    canvas: np.ndarray,
    slice_mask: np.ndarray,
    x_min: int, y_min: int, x_max: int, y_max: int,
) -> None:
    """
    スライスマスクをキャンバス上の正しい位置に貼り付け、のり代部分は論理和でマージする。
    canvas を in-place で更新する。
    """
    sh, sw = slice_mask.shape[:2]
    # キャンバス上の対応領域（スライスと同じサイズにクリップ）
    ch, cw = canvas.shape[:2]
    y_end = min(y_min + sh, ch)
    x_end = min(x_min + sw, cw)
    actual_h = y_end - y_min
    actual_w = x_end - x_min
    if actual_h <= 0 or actual_w <= 0:
        return
    # スライスマスクも右端・下端でクリップする場合がある
    mask_h, mask_w = slice_mask.shape[:2]
    use_mask = slice_mask[:actual_h, :actual_w]
    if use_mask.ndim > 2:
        use_mask = use_mask.squeeze()
    roi = canvas[y_min:y_end, x_min:x_end]
    # 論理和で結合（のり代も自然にマージ）
    merged = cv2.bitwise_or(roi, use_mask)
    canvas[y_min:y_end, x_min:x_end] = merged


# =============================================================================
# 5. 後処理（モルフォロジー・面積ベース被り除去）
# =============================================================================

def apply_morphology_and_fill_holes(
    mask: np.ndarray,
    morph_kernel: int,
) -> np.ndarray:
    """
    モルフォロジー変換（CLOSE + OPEN）と、最大輪郭の内部を塗りつぶして穴をなくす。
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask_uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(solid, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        solid = mask_uint8
    return solid


def postprocess_combined_mask(
    combined_mask: np.ndarray,
    morph_kernel: int,
    epsilon_rate: float,
) -> np.ndarray:
    """
    結合マスクに以下を適用する：
    - モルフォロジー変換・微小平滑化
    - 面積ベースの被り除去（連結成分を面積の小さい順に処理し、NOT(global_occupancy) で引き算してから足し込む）
    """
    # 連結成分（輪郭）ごとに面積の小さい順に処理
    mask_uint8 = (combined_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return combined_mask

    # 面積でソート（小さい順）
    contour_list = [(cv2.contourArea(c), c) for c in contours]
    contour_list.sort(key=lambda x: x[0])

    h, w = combined_mask.shape[:2]
    global_occupancy = np.zeros((h, w), dtype=np.uint8)
    for _area, cnt in contour_list:
        # 輪郭マスクを作成
        blob_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        # モルフォロジー＋穴埋め
        blob_mask = apply_morphology_and_fill_holes(blob_mask, morph_kernel)
        # 被り除去: すでに確定した領域を引き算（NOT 演算）
        resolved = cv2.bitwise_and(blob_mask, cv2.bitwise_not(global_occupancy))
        # 確定領域に足し込む
        global_occupancy = cv2.bitwise_or(global_occupancy, resolved)

    return global_occupancy


def mask_to_polygon_smooth(mask: np.ndarray, epsilon_rate: float) -> list:
    """マスクから外周ポリゴンを取得し、epsilon で平滑化する。"""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    if epsilon_rate > 0:
        perimeter = cv2.arcLength(largest, True)
        epsilon = epsilon_rate * perimeter
        approx = cv2.approxPolyDP(largest, epsilon, True)
    else:
        approx = largest
    return [[int(p[0][0]), int(p[0][1])] for p in approx]


# =============================================================================
# 画像パス収集・デフォルトパス
# =============================================================================

def get_default_weights() -> str:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / CONFIG["default_weights_rel"])


def get_default_source() -> str:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / CONFIG["default_source_rel"])


def collect_image_paths(source: str) -> List[Path]:
    p = Path(source)
    exts = CONFIG["image_extensions"]
    if not p.exists():
        return []
    if p.is_file():
        return [p] if p.suffix.lower() in exts else []
    paths = []
    for ext in exts:
        paths.extend(p.glob(f"*{ext}"))
    return sorted(paths)


# =============================================================================
# メイン処理
# =============================================================================

def process_one_image(
    image_path: Path,
    yolo_model: YOLO,
    sam_predictor: SAMPredictor,
    output_dir: Path,
) -> None:
    """
    1 枚の画像について、スライス → パッチ推論 → 結合 → 後処理 まで実行し、
    オーバーレイ画像とポリゴン JSON を保存する。
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] 読み込み失敗: {image_path}")
        return
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width = image.shape[:2]
    overlap = CONFIG["overlap_ratio"]
    rows, cols = CONFIG["slice_rows"], CONFIG["slice_cols"]

    # ----- 1. キャンバス初期化 -----
    canvas = create_mask_canvas(height, width)

    # ----- 2. スライス座標取得とスライス画像の取得 -----
    slice_bboxes = get_slice_bboxes(height, width, overlap, rows, cols)
    slices_with_offsets = slice_image(image, slice_bboxes)

    # ----- 3–4. パッチごとに YOLO → SAM → マスクをキャンバスに結合 -----
    min_area = CONFIG["MIN_AREA_THRESHOLD"]
    zone_margin = CONFIG["zone_margin"]
    for idx, (slice_img, (x_min, y_min, x_max, y_max)) in enumerate(slices_with_offsets):
        patch_mask = run_patch_inference(
            yolo_model, sam_predictor, slice_img, min_area, zone_margin
        )
        if patch_mask is not None:
            merge_slice_mask_into_canvas(canvas, patch_mask, x_min, y_min, x_max, y_max)

    # ----- 5. 後処理 -----
    combined = postprocess_combined_mask(
        canvas,
        CONFIG["morph_kernel"],
        CONFIG["epsilon_rate"],
    )

    # オーバーレイ描画・保存
    overlay = image.copy()
    color = (100, 255, 100)
    overlay[combined > 0] = (
        overlay[combined > 0] * 0.6 + np.array(color, dtype=np.uint8) * 0.4
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 2)

    stem = image_path.stem
    overlay_path = output_dir / f"{stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"[INFO] オーバーレイ保存: {overlay_path}")

    # マスク保存（オプション）
    mask_path = output_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), combined)

    # ポリゴン JSON
    polygon = mask_to_polygon_smooth(combined, CONFIG["epsilon_rate"])
    json_path = output_dir / f"{stem}_polygons.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"source": str(image_path), "polygon": polygon},
            f, ensure_ascii=False, indent=2,
        )
    print(f"[INFO] ポリゴンJSON保存: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO+SAM 分割推論でセグメンテーションマスクを生成"
    )
    parser.add_argument("--weights", type=str, default=None, help="YOLO 重みパス")
    parser.add_argument("--source", type=str, default=None, help="入力画像フォルダ or 1枚のパス")
    parser.add_argument("--output-dir", type=str, default=CONFIG["output_dir"], help="出力ディレクトリ")
    parser.add_argument("--sam-model", type=str, default=CONFIG["sam_model"], help="SAM モデル")
    parser.add_argument("--imgsz", type=int, default=CONFIG["imgsz"], help="YOLO 推論サイズ")
    parser.add_argument("--conf", type=float, default=CONFIG["conf"], help="YOLO 信頼度閾値")
    parser.add_argument("--min-area", type=float, default=CONFIG["MIN_AREA_THRESHOLD"], help="BBox 面積閾値（これ以下は除外）")
    parser.add_argument("--overlap", type=float, default=CONFIG["overlap_ratio"], help="スライスオーバーラップ率")
    parser.add_argument("--morph-kernel", type=int, default=CONFIG["morph_kernel"], help="モルフォロジーカーネルサイズ")
    parser.add_argument("--epsilon-rate", type=float, default=CONFIG["epsilon_rate"], help="ポリゴン平滑化")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = args.weights or get_default_weights()
    source = args.source or get_default_source()

    if not Path(weights).exists():
        raise FileNotFoundError(f"重みファイルが見つかりません: {weights}")
    image_paths = collect_image_paths(source)
    if not image_paths:
        raise FileNotFoundError(f"画像がありません: {source}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CONFIG を CLI で上書き
    CONFIG["MIN_AREA_THRESHOLD"] = args.min_area
    CONFIG["overlap_ratio"] = args.overlap
    CONFIG["morph_kernel"] = args.morph_kernel
    CONFIG["epsilon_rate"] = args.epsilon_rate
    CONFIG["imgsz"] = args.imgsz
    CONFIG["conf"] = args.conf
    CONFIG["sam_model"] = args.sam_model

    print("[INFO] YOLO モデル読み込み:", weights)
    yolo_model = YOLO(weights)
    print("[INFO] SAM モデル読み込み:", CONFIG["sam_model"])
    sam_predictor = SAMPredictor(overrides=dict(
        conf=0.25, task="segment", mode="predict", imgsz=1024, model=CONFIG["sam_model"],
    ))

    for path in image_paths:
        process_one_image(path, yolo_model, sam_predictor, output_dir)

    print(f"[INFO] セグメント結果を保存しました: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
