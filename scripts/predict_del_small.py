"""
YOLOv11 推論・可視化スクリプト（建物見取り図データセット用）
- 学習で得た best.pt を読み込み、test/images/ に対して推論
- バウンディングボックス描画済み画像を results/ に保存
- "zone" クラスかつ面積が閾値以下のボックスを除去

パスについて:
  - デフォルトの weights / source は「このスクリプトの位置」からプロジェクトルートを推定しています。
  - 実行はプロジェクトルートから推奨: python scripts/predict_del_small.py
  - その場合、結果は ./results/ に保存され、test 画像は kaggle/input/.../archive-3/test/images を参照します。
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

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
    # --- 推論パラメータ ---
    "imgsz": 640,
    "conf": 0.25,
    # --- 小面積zoneボックス除去の閾値（ピクセル面積） ---
    "area_threshold": 10000,
}
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 推論・可視化（見取り図）")
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
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=CONFIG["area_threshold"],
        help="zoneクラスのボックス面積閾値（ピクセル単位）。これ以下のzoneボックスを除去（デフォルト: 10000）",
    )
    return parser.parse_args()


def get_default_weights():
    """学習スクリプトのデフォルト出力先に対応する best.pt のパス。"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / CONFIG["weights_relpath"])


def get_default_source():
    """データセットの test/images のデフォルトパス。"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / CONFIG["source_relpath"])


def filter_small_zone_boxes(results, area_threshold, save_dir):
    """zoneクラスかつ面積が閾値以下のボックスを除去し、該当画像を再保存する。"""
    total_removed = 0
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        names = result.names
        keep = []
        removed = 0
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = names[cls_id]
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            area = (x2 - x1) * (y2 - y1)

            if cls_name == "zone" and area <= area_threshold:
                removed += 1
                continue
            keep.append(i)

        if removed > 0:
            result.boxes = boxes[keep]
            img = result.plot()
            save_path = Path(save_dir) / Path(result.path).name
            cv2.imwrite(str(save_path), img)
            print(
                f"[INFO] {Path(result.path).name}: "
                f"zone小面積ボックス {removed}個除去 (閾値={area_threshold:.0f}px²)"
            )
            total_removed += removed

    print(f"[INFO] 合計 {total_removed}個の小面積zoneボックスを除去しました")
    return results


def main():
    args = parse_args()
    weights = args.weights
    if weights is None:
        weights = get_default_weights()
        print(f"[INFO] --weights 未指定のためデフォルトを使用: {weights}")
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"重みファイルが見つかりません: {weights}\n"
            "先に yolo_train01.py で学習を実行するか、--weights で best.pt のパスを指定してください。"
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
    return run_predict(
        weights=weights,
        source=source,
        save_dir=save_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        area_threshold=args.area_threshold,
    )


def run_predict(
    weights,
    source,
    save_dir="results",
    imgsz=640,
    conf=0.25,
    area_threshold=10000,
):
    """YOLO推論を実行し、小面積zoneボックスを除去した results オブジェクトを返す。"""
    model = YOLO(weights)
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        save=True,
        project=".",
        name=save_dir,
        exist_ok=True,
    )
    print(f"[INFO] 推論結果の画像を保存しました: {Path(save_dir).resolve()}")

    print(f"[INFO] 小面積zoneボックス除去を開始 (閾値={area_threshold:.0f}px²)")
    results = filter_small_zone_boxes(results, area_threshold, save_dir)

    return results


if __name__ == "__main__":
    main()
