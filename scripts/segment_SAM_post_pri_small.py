"""
YOLO推論結果をプロンプトとしてSAMでセグメントするスクリプト
- ドア・窓はすべて「塗らない(negative)」として扱う
- モルフォロジー変換(アイロンがけ)で境界を直線化
- 面積が小さいゾーンから順にマスクを確定させ、被り（オーバーラップ）を除去する
- 描画時に「引き算済みのマスク」を使用し、大きな部屋が他の部屋を避けている（穴が開いている）状態を可視化する
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics.models.sam import Predictor as SAMPredictor

# ※環境に合わせて import を調整してください
from predict import run_predict

# ============================================================
# CONFIG: パスやパラメータをここにまとめて記載（書き換え容易化）
# ============================================================
CONFIG = {
    # --- パス（プロジェクトルートからの相対パス） ---
    "weights_relpath": "runs/detect/train/weights/best.pt",
    "source_relpath": (
        "kaggle/input/datasets/umairinayat/"
        "floor-plans-500-annotated-object-detection/archive-3/test/images"
    ),
    "save_dir": "results",
    "output_dir": "segment_results",
    "sam_model": "sam2-b.pt",
    # --- YOLO推論パラメータ ---
    "imgsz": 640,
    "conf": 0.25,
    # --- SAM Predictor パラメータ ---
    "sam_conf": 0.25,
    "sam_imgsz": 1024,
    # --- セグメント処理パラメータ ---
    "zone_margin": 10.0,
    "morph_kernel": 30,
    "epsilon_rate": 0.002,
    # --- 描画パラメータ ---
    "overlay_alpha": 0.4,
    "colors": [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
    ],
}
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO推論結果をSAMでセグメント（見取り図用）"
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="YOLO学習済み重み（best.pt）のパス"
    )
    parser.add_argument(
        "--source", type=str, default=None, help="推論対象の画像フォルダ（test/images/）"
    )
    parser.add_argument(
        "--save-dir", type=str, default=CONFIG["save_dir"], help="YOLO推論結果の保存先"
    )
    parser.add_argument(
        "--output-dir", type=str, default=CONFIG["output_dir"], help="SAMセグメント結果の保存先"
    )
    parser.add_argument(
        "--sam-model", type=str, default=CONFIG["sam_model"], help="SAMモデル"
    )
    parser.add_argument(
        "--imgsz", type=int, default=CONFIG["imgsz"], help="YOLO推論時の画像サイズ"
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIG["conf"], help="YOLO検出の信頼度閾値"
    )
    parser.add_argument(
        "--zone-margin", type=float, default=CONFIG["zone_margin"], help="ゾーン内のドア・窓判定用マージン"
    )
    parser.add_argument(
        "--morph-kernel", type=int, default=CONFIG["morph_kernel"], help="モルフォロジー変換のカーネルサイズ"
    )
    parser.add_argument(
        "--epsilon-rate", type=float, default=CONFIG["epsilon_rate"], help="頂点数削減の平滑化パラメータ"
    )
    return parser.parse_args()


def get_default_weights():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / CONFIG["weights_relpath"])


def get_default_source():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / CONFIG["source_relpath"])


def extract_prompts_from_result(r, zone_margin: float = 10.0):
    """
    YOLO結果からzone_boxesとdoor/windowを抽出。
    ドアも窓も「塗らない（negative_points）」設定にする。
    """
    zone_data = []
    all_door_window = []

    for box in r.boxes:
        cls_id = int(box.cls[0])
        class_name = r.names[cls_id]
        xyxy = box.xyxy[0].tolist()

        if class_name == "zone":
            zone_data.append({"zone_box": xyxy, "positive_points": [], "negative_points": []})
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


def select_best_mask(masks: np.ndarray, box_area: float):
    if masks is None or masks.size == 0:
        return None
    areas = np.array([np.sum(m > 0) for m in masks])
    idx = np.argmin(np.abs(areas - box_area))
    return masks[idx]


def run_sam_for_zone(
    predictor, image: np.ndarray, zone_box: list, positive_points: list, negative_points: list,
) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = zone_box
    box_area = (x2 - x1) * (y2 - y1)

    points = []
    labels = []
    
    for pt in positive_points:
        points.append(pt)
        labels.append(1)
    for pt in negative_points:
        points.append(pt)
        labels.append(0)

    bboxes = np.array([zone_box], dtype=np.float32)
    points_arr = np.array([points], dtype=np.float32) if points else None
    labels_arr = np.array([labels], dtype=np.int64) if labels else None

    predictor.set_image(image)
    results = predictor(
        bboxes=bboxes, points=points_arr, labels=labels_arr, multimask_output=True,
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


def process_mask_to_polygon(mask: np.ndarray, global_occupancy: np.ndarray, morph_kernel: int, epsilon_rate: float):
    """
    マスクにモルフォロジー変換をかけ、内部の穴を埋めた後、
    すでに確定した領域(global_occupancy)を引き算してからポリゴン化する。
    """
    # サイズを確実に合わせる
    h, w = global_occupancy.shape
    if mask.shape != (h, w):
        mask = cv2.resize((mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # 1. モルフォロジー変換（アイロンがけ）
    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    # ---------------------------------------------------------
    # 1.5. 【追加】内部の「穴」を完全に塗りつぶしてソリッドなブロックにする
    # ---------------------------------------------------------
    contours_for_fill, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(mask_uint8)
    if contours_for_fill:
        # 一番大きい輪郭の中身をすべて白で塗りつぶす（SAMの塗り残しの穴が消滅する）
        largest_fill = max(contours_for_fill, key=cv2.contourArea)
        cv2.drawContours(solid_mask, [largest_fill], -1, 255, thickness=cv2.FILLED)
    else:
        solid_mask = mask_uint8

    # 2. 被り領域の除去（引き算）
    resolved_mask = cv2.bitwise_and(solid_mask, cv2.bitwise_not(global_occupancy))

    # 3. 輪郭抽出と平滑化（JSON用の外枠ポリゴンを取得するため RETR_EXTERNAL を使用）
    contours, _ = cv2.findContours(resolved_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], resolved_mask
        
    largest = max(contours, key=cv2.contourArea)
    if epsilon_rate > 0:
        perimeter = cv2.arcLength(largest, True)
        epsilon = epsilon_rate * perimeter
        approx_polygon = cv2.approxPolyDP(largest, epsilon, True)
    else:
        approx_polygon = largest

    polygon = [[int(p[0][0]), int(p[0][1])] for p in approx_polygon]
    return polygon, resolved_mask


def draw_overlay_from_mask(image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
    """
    引き算済みのマスク配列を使って、画像上に半透明の塗りと境界線を描画する。
    （穴が開いている部分はちゃんと避けて描画される）
    """
    # 塗りのレイヤーを作成
    colored_layer = np.zeros_like(image, dtype=np.uint8)
    colored_layer[mask > 0] = color
    
    # マスク領域のみ半透明で合成
    overlay = image.copy()
    idx = mask > 0
    overlay[idx] = (image[idx] * (1 - alpha) + colored_layer[idx] * alpha).astype(np.uint8)
    
    # 境界線の描画 (RETR_LISTを使って、切り抜かれた内側の境界線も描画する)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness=2)
    
    return overlay


def main():
    args = parse_args()
    weights = args.weights or get_default_weights()
    source = args.source or get_default_source()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] YOLO推論を実行中...")
    results = run_predict(
        weights=weights, source=source, save_dir=args.save_dir, imgsz=args.imgsz, conf=args.conf,
    )

    print(f"[INFO] SAMモデルを読み込み中: {args.sam_model}")
    overrides = dict(conf=CONFIG["sam_conf"], task="segment", mode="predict", imgsz=CONFIG["sam_imgsz"], model=args.sam_model)
    predictor = SAMPredictor(overrides=overrides)

    colors = CONFIG["colors"]

    for i, r in enumerate(results):
        orig_img = r.orig_img
        if orig_img is None:
            continue
        img = orig_img.copy()
        if len(orig_img.shape) == 2:
            img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

        zone_prompts = extract_prompts_from_result(r, zone_margin=args.zone_margin)
        
        # --- 1. すべてのゾーンの生マスクを取得してリストに保存 ---
        zone_masks_info = []
        for j, zp in enumerate(zone_prompts):
            mask = run_sam_for_zone(
                predictor, orig_img, zp["zone_box"], zp["positive_points"], zp["negative_points"],
            )
            if mask is None:
                continue
            
            area = np.sum(mask > 0)
            zone_masks_info.append({
                "zone_index": j,
                "raw_mask": mask,
                "area": area,
                "color": colors[j % len(colors)]
            })

        # --- 2. 面積が小さい順にソート ---
        zone_masks_info.sort(key=lambda x: x["area"])

        # --- 3. 面積が小さい順に処理し、排他マスクを更新していく ---
        h, w = orig_img.shape[:2]
        global_occupancy = np.zeros((h, w), dtype=np.uint8)
        
        polygons_all = []
        overlay_img = img.copy()

        for info in zone_masks_info:
            polygon, resolved_mask = process_mask_to_polygon(
                info["raw_mask"], 
                global_occupancy, 
                args.morph_kernel, 
                args.epsilon_rate
            )
            
            if not polygon:
                continue
                
            polygons_all.append({
                "zone_index": info["zone_index"], 
                "polygon": polygon
            })
            
            # 確定した領域をグローバルマスクに足し込む
            global_occupancy = cv2.bitwise_or(global_occupancy, resolved_mask)
            
            # 引き算済みのマスクを使って描画
            overlay_img = draw_overlay_from_mask(overlay_img, resolved_mask, info["color"], alpha=CONFIG["overlay_alpha"])

        # JSONの出力順を元のインデックス順に戻す
        polygons_all.sort(key=lambda x: x["zone_index"])

        stem = Path(r.path).stem if r.path else f"image_{i}"
        overlay_path = output_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_img)
        print(f"[INFO] オーバーレイ保存: {overlay_path}")

        json_path = output_dir / f"{stem}_polygons.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"source": str(r.path) if r.path else f"image_{i}", "zones": polygons_all},
                f, ensure_ascii=False, indent=2,
            )
        print(f"[INFO] ポリゴンJSON保存: {json_path}")

    print(f"[INFO] セグメント結果を保存しました: {output_dir.resolve()}")


if __name__ == "__main__":
    main()