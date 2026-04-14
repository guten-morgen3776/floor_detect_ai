"""
SAM3 (Segment Anything Model 3) を使ったプロンプトベースセグメンテーション

プロンプト種別:
  - text      : テキストで対象を指定（SAM3SemanticPredictor 使用）★メイン
  - points    : ポジティブ/ネガティブポイント指定
  - boxes     : バウンディングボックス指定
  - everything: プロンプトなしで全領域を自動セグメント

実行例:
  # テキストプロンプト（カンマ区切りで複数指定）
  python segment_sam3.py --source image.jpg --prompt-type text \
      --text "floor" "wall" "door"

  # 説明的フレーズも使用可能
  python segment_sam3.py --source image.jpg --prompt-type text \
      --text "wooden floor" "white wall"

  # ポイントプロンプト（座標はx,y形式、ラベル1=正, 0=負）
  python segment_sam3.py --source image.jpg --prompt-type points \
      --points "500,400" --labels 1

  # バウンディングボックスプロンプト
  python segment_sam3.py --source image.jpg --prompt-type boxes \
      --boxes "100,200,500,600"

  # 全領域自動セグメント
  python segment_sam3.py --source image.jpg --prompt-type everything
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────
# Kaggle 環境判別
# ─────────────────────────────────────────────
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
if IS_KAGGLE:
    _default_output_dir = "/kaggle/working/sam3_exp001"
else:
    _default_output_dir = "/content/drive/MyDrive/sam3_exp001"


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3でテキスト・ポイント・ボックスプロンプトによるセグメンテーションを実行"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="セグメント対象画像のパス（単一ファイルまたはディレクトリ）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sam3.pt",
        help="SAMモデルのパスまたは名前（デフォルト: sam3.pt）",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="text",
        choices=["text", "points", "boxes", "everything"],
        help="プロンプトの種類: text / points / boxes / everything（デフォルト: text）",
    )
    # ── テキストプロンプト ──
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        default=None,
        help=(
            "テキストプロンプト。複数指定可。"
            "例: --text 'floor' 'wall' 'door'"
            "    --text 'wooden floor' 'white wall'"
        ),
    )
    # ── ポイントプロンプト ──
    parser.add_argument(
        "--points",
        type=str,
        nargs="+",
        default=None,
        help="ポイントプロンプト。'x,y' 形式で複数指定可。例: --points 300,400 500,600",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        help="各ポイントのラベル（1=ポジティブ, 0=ネガティブ）。--points と対応。",
    )
    # ── ボックスプロンプト ──
    parser.add_argument(
        "--boxes",
        type=str,
        nargs="+",
        default=None,
        help="ボックスプロンプト。'x1,y1,x2,y2' 形式で複数指定可。例: --boxes 100,200,500,600",
    )
    # ── 共通オプション ──
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_default_output_dir,
        help=f"結果の保存先（デフォルト: {_default_output_dir}）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="セグメントマスクの信頼度閾値（デフォルト: 0.25）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMSのIoU閾値（デフォルト: 0.7）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="推論時の画像サイズ（デフォルト: 1024）",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="FP16（半精度）で推論してGPUメモリ・速度を改善（デフォルト: True）",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="ポリゴン座標をJSONに保存する（デフォルト: True）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="マスクオーバーレイの透明度（0.0〜1.0、デフォルト: 0.4）",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# プロンプト解析
# ─────────────────────────────────────────────
def parse_points(point_strs: list[str]) -> np.ndarray:
    """'x,y' 形式の文字列リストを (N,2) の ndarray に変換"""
    pts = []
    for s in point_strs:
        x, y = map(float, s.split(","))
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def parse_boxes(box_strs: list[str]) -> np.ndarray:
    """'x1,y1,x2,y2' 形式の文字列リストを (N,4) の ndarray に変換"""
    boxes = []
    for s in box_strs:
        vals = list(map(float, s.split(",")))
        if len(vals) != 4:
            raise ValueError(f"ボックス形式が不正です: '{s}' (x1,y1,x2,y2 が必要)")
        boxes.append(vals)
    return np.array(boxes, dtype=np.float32)


# ─────────────────────────────────────────────
# 可視化
# ─────────────────────────────────────────────
COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 255, 80),
    (255, 80, 255),
    (80, 255, 255),
    (200, 150, 80),
    (80, 200, 150),
    (150, 80, 200),
    (200, 200, 200),
]


def draw_masks_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """マスクを半透明オーバーレイとして描画。labelsを指定するとラベル名を表示。"""
    result = image.copy()
    for i, mask in enumerate(masks):
        color = COLORS[i % len(COLORS)]
        mask_bool = mask.astype(bool)
        overlay = result.copy()
        overlay[mask_bool] = color
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        # 輪郭線
        mask_uint8 = mask_bool.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2)
        # テキストラベルを輪郭の重心に表示
        if labels and i < len(labels) and contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    result,
                    labels[i],
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
    return result


def draw_point_box_prompts(
    image: np.ndarray,
    points: np.ndarray | None,
    point_labels: np.ndarray | None,
    boxes: np.ndarray | None,
) -> np.ndarray:
    """ポイント・ボックスプロンプトを画像上に描画"""
    result = image.copy()
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 255), 2)
    if points is not None and point_labels is not None:
        for pt, lbl in zip(points, point_labels):
            x, y = int(pt[0]), int(pt[1])
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(result, (x, y), 8, color, -1)
            cv2.circle(result, (x, y), 8, (255, 255, 255), 2)
    return result


# ─────────────────────────────────────────────
# マスク → ポリゴン変換
# ─────────────────────────────────────────────
def mask_to_polygon(mask: np.ndarray) -> list[list[int]]:
    """バイナリマスクから最大輪郭のポリゴン頂点リストを返す"""
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    return [[int(p[0][0]), int(p[0][1])] for p in largest]


# ─────────────────────────────────────────────
# 画像収集
# ─────────────────────────────────────────────
def collect_images(source: str) -> list[Path]:
    """source がファイルなら1枚、ディレクトリなら全画像を返す"""
    src = Path(source)
    if src.is_file():
        return [src]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return sorted(p for p in src.iterdir() if p.suffix.lower() in exts)


# ─────────────────────────────────────────────
# 結果の保存
# ─────────────────────────────────────────────
def save_results(
    output_dir: Path,
    img_path: Path,
    masks: np.ndarray,
    overlay_img: np.ndarray,
    prompt_type: str,
    text_prompts: list[str] | None,
    points_arr: np.ndarray | None,
    labels_arr: np.ndarray | None,
    boxes_arr: np.ndarray | None,
    model_name: str,
    save_json: bool,
) -> dict:
    stem = img_path.stem
    overlay_path = output_dir / f"{stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay_img)
    print(f"[INFO]   オーバーレイ保存: {overlay_path}")

    result_entry = {
        "image": str(img_path),
        "num_masks": len(masks),
        "overlay": str(overlay_path),
    }

    if save_json:
        polygons = []
        for i, mask in enumerate(masks):
            polygon = mask_to_polygon(mask)
            label = text_prompts[i] if text_prompts and i < len(text_prompts) else None
            polygons.append(
                {
                    "mask_index": i,
                    "label": label,
                    "polygon": polygon,
                    "area_px": int(np.sum(mask > 0)),
                }
            )

        meta = {
            "source": str(img_path),
            "model": model_name,
            "prompt_type": prompt_type,
            "prompts": {
                "text": text_prompts,
                "points": points_arr.tolist() if points_arr is not None else None,
                "labels": labels_arr.tolist() if labels_arr is not None else None,
                "boxes": boxes_arr.tolist() if boxes_arr is not None else None,
            },
            "num_masks": len(masks),
            "masks": polygons,
        }
        json_path = output_dir / f"{stem}_masks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[INFO]   JSON保存: {json_path}")
        result_entry["json"] = str(json_path)

    return result_entry


# ─────────────────────────────────────────────
# テキストプロンプトによる推論
# ─────────────────────────────────────────────
def run_text_segment(args, image_paths: list[Path], output_dir: Path) -> list[dict]:
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=args.model,
        half=args.half,
        save=False,  # 手動保存するため無効化
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    all_results = []

    for img_path in image_paths:
        print(f"[INFO] 処理中: {img_path.name}")
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] 読み込み失敗（スキップ）: {img_path}")
            continue

        # 画像をセット（同一画像に複数クエリを投げる場合も効率的）
        predictor.set_image(str(img_path))

        results = predictor(text=args.text)

        predictor.reset_image()

        if not results:
            print(f"[WARN] 結果なし: {img_path.name}")
            continue

        r = results[0]
        if r.masks is None or r.masks.data is None:
            print(f"[WARN] マスクなし: {img_path.name}")
            continue

        masks = r.masks.data.cpu().numpy()  # (N, H, W)
        print(f"[INFO]   検出マスク数: {len(masks)}")

        # テキストラベルをマスクに対応付け（1テキスト → 1マスクが基本）
        overlay_img = draw_masks_overlay(
            image_bgr, masks, labels=args.text, alpha=args.alpha
        )

        entry = save_results(
            output_dir=output_dir,
            img_path=img_path,
            masks=masks,
            overlay_img=overlay_img,
            prompt_type="text",
            text_prompts=args.text,
            points_arr=None,
            labels_arr=None,
            boxes_arr=None,
            model_name=args.model,
            save_json=args.save_json,
        )
        all_results.append(entry)

    return all_results


# ─────────────────────────────────────────────
# ポイント・ボックス・everything による推論
# ─────────────────────────────────────────────
def run_geometric_segment(args, image_paths: list[Path], output_dir: Path) -> list[dict]:
    from ultralytics import SAM

    print(f"[INFO] SAMモデルを読み込み中: {args.model}")
    model = SAM(args.model)

    points_arr = None
    labels_arr = None
    boxes_arr = None

    if args.prompt_type == "points":
        if not args.points:
            raise ValueError("--prompt-type points の場合は --points が必要です")
        points_arr = parse_points(args.points)
        if args.labels:
            if len(args.labels) != len(args.points):
                raise ValueError("--points と --labels の個数が一致しません")
            labels_arr = np.array(args.labels, dtype=np.int64)
        else:
            labels_arr = np.ones(len(points_arr), dtype=np.int64)

    elif args.prompt_type == "boxes":
        if not args.boxes:
            raise ValueError("--prompt-type boxes の場合は --boxes が必要です")
        boxes_arr = parse_boxes(args.boxes)

    all_results = []

    for img_path in image_paths:
        print(f"[INFO] 処理中: {img_path.name}")
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] 読み込み失敗（スキップ）: {img_path}")
            continue

        if args.prompt_type == "everything":
            results = model(
                str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
            )
        elif args.prompt_type == "points":
            results = model(
                str(img_path),
                points=points_arr,
                labels=labels_arr,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
            )
        elif args.prompt_type == "boxes":
            results = model(
                str(img_path),
                bboxes=boxes_arr,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
            )

        if not results:
            print(f"[WARN] 結果なし: {img_path.name}")
            continue

        r = results[0]
        if r.masks is None or r.masks.data is None:
            print(f"[WARN] マスクなし: {img_path.name}")
            continue

        masks = r.masks.data.cpu().numpy()
        print(f"[INFO]   検出マスク数: {len(masks)}")

        overlay_img = draw_masks_overlay(image_bgr, masks, alpha=args.alpha)
        overlay_img = draw_point_box_prompts(overlay_img, points_arr, labels_arr, boxes_arr)

        entry = save_results(
            output_dir=output_dir,
            img_path=img_path,
            masks=masks,
            overlay_img=overlay_img,
            prompt_type=args.prompt_type,
            text_prompts=None,
            points_arr=points_arr,
            labels_arr=labels_arr,
            boxes_arr=boxes_arr,
            model_name=args.model,
            save_json=args.save_json,
        )
        all_results.append(entry)

    return all_results


# ─────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    if not Path(args.source).exists():
        raise FileNotFoundError(f"ソースが見つかりません: {args.source}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(args.source)
    if not image_paths:
        raise FileNotFoundError(f"画像が見つかりません: {args.source}")
    print(f"[INFO] 処理対象: {len(image_paths)} 枚")
    print(f"[INFO] プロンプト種別: {args.prompt_type}")

    if args.prompt_type == "text":
        if not args.text:
            raise ValueError("--prompt-type text の場合は --text が必要です")
        print(f"[INFO] テキストプロンプト: {args.text}")
        all_results = run_text_segment(args, image_paths, output_dir)
    else:
        all_results = run_geometric_segment(args, image_paths, output_dir)

    # サマリ保存
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 完了: {len(all_results)} 枚を処理しました")
    print(f"[INFO] 結果保存先: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
