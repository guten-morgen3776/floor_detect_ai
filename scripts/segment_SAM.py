"""
YOLO推論結果をプロンプトとしてSAMでセグメントするスクリプト

- predict.py（または run_predict）の結果をメモリ上で受け取り、SAMでゾーンをセグメント
- 出力: オーバーレイ画像 (.jpg), ポリゴン座標 (.json)

実行: python scripts/segment_SAM.py
（predict.py と同様の --weights, --source 等のオプションに対応）
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics.models.sam import Predictor as SAMPredictor

from predict import run_predict


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO推論結果をSAMでセグメント（見取り図用）"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="YOLO学習済み重み（best.pt）のパス",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="推論対象の画像フォルダ（test/images/）",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="YOLO推論結果の保存先（predict の save-dir と揃える）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="segment_results",
        help="SAMセグメント結果（オーバーレイ・JSON）の保存先",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="sam2-b.pt",
        help="SAMモデル（sam_b.pt または sam2-b.pt）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO推論時の画像サイズ",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO検出の信頼度閾値",
    )
    parser.add_argument(
        "--zone-margin",
        type=float,
        default=10.0,
        help="ゾーン内のドア・窓判定用マージン（ピクセル）。この範囲内のドア・窓中心を負のポイントとする",
    )
    return parser.parse_args()


def get_default_weights():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / "runs/detect/train/weights/best.pt")


def get_default_source():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(
        project_root
        / "kaggle/input/datasets/umairinayat/floor-plans-500-annotated-object-detection/archive-3/test/images"
    )


def extract_prompts_from_result(r, zone_margin: float = 10.0):
    """
    YOLO結果 r から zone_boxes と door_window_points（zone内に限定）を抽出。

    返り値: list of dict
        [{"zone_box": [x1,y1,x2,y2], "negative_points": [[x,y], ...]}, ...]
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

    # 各 zone の BBox に内包 or マージン内のドア・窓中心をその zone の negative_points に割り当て
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


def mask_to_polygon(mask: np.ndarray):
    """
    バイナリマスクから外周ポリゴン頂点リストを取得。
    最大面積の輪郭を採用。
    """
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = [[int(p[0][0]), int(p[0][1])] for p in largest]
    return polygon


def select_best_mask(masks: np.ndarray, box_area: float):
    """
    マルチマスク候補から、YOLO Box の面積に最も近い（＝最大面積の）マスクを選択。
    製図線による過分割を防ぐ。
    """
    if masks is None or masks.size == 0:
        return None
    areas = np.array([np.sum(m > 0) for m in masks])
    idx = np.argmin(np.abs(areas - box_area))
    return masks[idx]


def run_sam_for_zone(
    predictor,
    image: np.ndarray,
    zone_box: list,
    negative_points: list,
) -> Optional[np.ndarray]:
    """
    SAMで1ゾーンをセグメント。bbox + 負のポイントを渡す。
    multimask の場合は Box 面積に最も近いマスクを採用。
    """
    x1, y1, x2, y2 = zone_box
    box_area = (x2 - x1) * (y2 - y1)

    points = negative_points
    labels = [0] * len(negative_points) if points else None

    # Ultralytics SAM: bboxes は (N,4) 形式、points と labels は任意
    bboxes = np.array([zone_box], dtype=np.float32)

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


def draw_overlay(
    image: np.ndarray,
    polygon: list,
    color: tuple,
    alpha: float = 0.4,
) -> np.ndarray:
    """半透明でゾーンを塗り、輪郭線を描画。"""
    overlay = image.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
    return image


def main():
    args = parse_args()
    weights = args.weights or get_default_weights()
    source = args.source or get_default_source()

    if not Path(weights).exists():
        raise FileNotFoundError(f"重みファイルが見つかりません: {weights}")
    if not Path(source).exists():
        raise FileNotFoundError(f"推論対象が見つかりません: {source}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] YOLO推論を実行中...")
    results = run_predict(
        weights=weights,
        source=source,
        save_dir=args.save_dir,
        imgsz=args.imgsz,
        conf=args.conf,
    )

    print(f"[INFO] SAMモデルを読み込み中: {args.sam_model}")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        imgsz=1024,
        model=args.sam_model,
    )
    predictor = SAMPredictor(overrides=overrides)

    # 色パレット（ゾーンごとに異なる色）
    colors = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 255, 100),
        (255, 100, 255),
        (100, 255, 255),
    ]

    for i, r in enumerate(results):
        orig_img = r.orig_img
        if orig_img is None:
            continue
        img = orig_img.copy()
        if len(orig_img.shape) == 2:
            img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

        zone_prompts = extract_prompts_from_result(r, zone_margin=args.zone_margin)
        polygons_all = []
        overlay_img = img.copy()

        for j, zp in enumerate(zone_prompts):
            mask = run_sam_for_zone(
                predictor,
                orig_img,
                zp["zone_box"],
                zp["negative_points"],
            )
            if mask is None:
                continue
            polygon = mask_to_polygon(mask)
            if not polygon:
                continue
            polygons_all.append({"zone_index": j, "polygon": polygon})
            color = colors[j % len(colors)]
            draw_overlay(overlay_img, polygon, color, alpha=0.4)

        stem = Path(r.path).stem if r.path else f"image_{i}"
        overlay_path = output_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_img)
        print(f"[INFO] オーバーレイ保存: {overlay_path}")

        json_path = output_dir / f"{stem}_polygons.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": str(r.path) if r.path else f"image_{i}",
                    "zones": polygons_all,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[INFO] ポリゴンJSON保存: {json_path}")

    print(f"[INFO] セグメント結果を保存しました: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
