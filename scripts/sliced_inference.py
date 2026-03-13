"""
YOLOv11 分割推論（パッチベース推論）スクリプト（建物見取り図データセット用）

- predict.py と同じモデル・入力・出力形式を用い、推論部分のみ SAHI による分割推論に差し替え
- 画像を縦2×横2の4分割で処理し、20%オーバーラップ＋NMSで重複Boxを統合
- 今回は「YOLOのBox検出結果を取得する」ところまで（SAM連携は後回し）

パスについて:
  - デフォルトの weights / source は predict.py と同様（このスクリプトの位置からプロジェクトルートを推定）
  - 実行はプロジェクトルートから推奨: python scripts/sliced_inference.py
"""

import argparse
import math
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# -----------------------------------------------------------------------------
# CONFIG: パス・パラメータのまとめ（処理ロジックは変更しない）
# -----------------------------------------------------------------------------
CONFIG = {
    # 画像として扱う拡張子（predict.py の source に渡すフォルダと同様）
    "image_extensions": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    # zone クラスの面積フィルタ（ピクセル²）：この値以下の zone Box はノイズとして除外（door/window は対象外）
    "min_area_threshold": 40000,
    # デフォルトパス（プロジェクトルートからの相対。省略時は get_default_weights / get_default_source で解決）
    "default_weights_rel": "runs/detect/train/weights/best.pt",
    "default_source_rel": "kaggle/input/datasets/umairinayat/floor-plans-500-annotated-object-detection/archive-3/test/images",
    # 推論・保存のデフォルト
    "save_dir": "results",
    "imgsz": 640,
    "conf": 0.25,
    # SAHI 分割推論
    "overlap_ratio": 0.2,
    "postprocess_type": "NMS",
    "postprocess_match_threshold": 0.5,
    "postprocess_match_metric": "IOU",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv11 分割推論・可視化（見取り図・SAHI）"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="学習済み重み（best.pt）のパス。省略時は runs/detect/train/weights/best.pt を使用。",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="推論対象の画像フォルダ（test/images/）。省略時はデータセットの test/images を使用。",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=CONFIG["save_dir"],
        help="推論結果（描画済み画像）の保存先ディレクトリ（デフォルト: results）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=CONFIG["imgsz"],
        help="推論時の画像サイズ（学習時と同じ 640 を推奨）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIG["conf"],
        help="検出の信頼度閾値（デフォルト: 0.25）",
    )
    return parser.parse_args()


def get_default_weights():
    """学習スクリプトのデフォルト出力先に対応する best.pt のパス。"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / CONFIG["default_weights_rel"])


def get_default_source():
    """データセットの test/images のデフォルトパス。"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / CONFIG["default_source_rel"])


def collect_image_paths(source: str):
    """source がディレクトリの場合、その中の画像パスを列挙する。"""
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


def get_slice_dimensions(height: int, width: int, overlap_ratio: float = 0.2):
    """
    縦2×横2の4分割で 20% オーバーラップになる slice_height, slice_width を計算する。
    ステップ = slice * (1 - overlap_ratio) で、2枚で全体をカバーするには
    1.8 * slice >= size が必要なので slice = ceil(size / 1.8)。
    """
    denom = 1.0 + (1.0 - overlap_ratio)  # 1.8 for 0.2 overlap
    slice_height = int(math.ceil(height / denom))
    slice_width = int(math.ceil(width / denom))
    return slice_height, slice_width


def filter_small_zone_boxes(result, min_area_threshold: float):
    """
    NMS 結合後の Box リストから、zone クラスで面積が min_area_threshold 以下のものを除外する。
    door / window は面積に関係なく残す。
    """
    filtered = []
    for obj in result.object_prediction_list:
        name = getattr(obj.category, "name", "")
        if name != "zone":
            filtered.append(obj)
            continue
        bbox = obj.bbox
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny
        area = w * h
        if area > min_area_threshold:
            filtered.append(obj)
    result.object_prediction_list = filtered
    return result


def draw_boxes_and_save(image, result, save_path):
    """SAHI の PredictionResult から bbox を描画し、save_path に保存する。"""
    img = image.copy()
    for obj in result.object_prediction_list:
        bbox = obj.bbox
        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = getattr(obj.category, "name", "object")
        score = getattr(obj.score, "value", 0.0)
        cv2.putText(
            img,
            f"{label} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.imwrite(str(save_path), img)


def run_sliced_predict(
    weights,
    source,
    save_dir="results",
    imgsz=640,
    conf=0.25,
):
    """
    SAHI を用いた分割推論を実行し、画像ごとの PredictionResult のリストを返す。
    モデルロード・入力画像の扱い・保存先・信頼度などは predict.py の run_predict と揃える。
    """
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=conf,
        image_size=imgsz,
        # device は省略し、predict.py と同様にライブラリのデフォルト（cuda 利用可能時は cuda）に従う
    )

    image_paths = collect_image_paths(source)
    if not image_paths:
        raise FileNotFoundError(
            f"推論対象の画像がありません: {source}\n"
            f"対応拡張子: {', '.join(CONFIG['image_extensions'])}"
        )

    save_path_dir = Path(save_dir)
    save_path_dir.mkdir(parents=True, exist_ok=True)

    overlap_ratio = CONFIG["overlap_ratio"]
    postprocess_type = CONFIG["postprocess_type"]
    postprocess_match_threshold = CONFIG["postprocess_match_threshold"]
    postprocess_match_metric = CONFIG["postprocess_match_metric"]

    results_list = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        height, width = img.shape[:2]
        slice_height, slice_width = get_slice_dimensions(
            height, width, overlap_ratio
        )

        result = get_sliced_prediction(
            img,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type=postprocess_type,
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_match_metric=postprocess_match_metric,
        )
        # 面積によるノイズ除去：zone のみ、CONFIG の閾値以下の極小 Box を除外
        filter_small_zone_boxes(result, CONFIG["min_area_threshold"])
        results_list.append((path, result))

        out_path = save_path_dir / path.name
        draw_boxes_and_save(img, result, out_path)

    print(f"[INFO] 推論結果の画像を保存しました: {save_path_dir.resolve()}")
    return results_list


def main():
    args = parse_args()
    weights = args.weights
    if weights is None:
        weights = get_default_weights()
        print(f"[INFO] --weights 未指定のためデフォルトを使用: {weights}")
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"重みファイルが見つかりません: {weights}\n"
            "先に学習を実行するか、--weights で best.pt のパスを指定してください。"
        )

    source = args.source
    if source is None:
        source = get_default_source()
        print(f"[INFO] --source 未指定のためデフォルトを使用: {source}")
    if not Path(source).exists():
        raise FileNotFoundError(
            f"推論対象のパスが見つかりません: {source}\n"
            "データセットの test/images を用意するか、--source で画像フォルダを指定してください。"
        )

    save_dir = args.save_dir
    return run_sliced_predict(
        weights=weights,
        source=source,
        save_dir=save_dir,
        imgsz=args.imgsz,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
