"""
YOLOv11 推論・可視化スクリプト（建物見取り図データセット用）
- 学習で得た best.pt を読み込み、test/images/ に対して推論
- バウンディングボックス描画済み画像を results/ に保存

パスについて:
  - デフォルトの weights / source は「このスクリプトの位置」からプロジェクトルートを推定しています。
  - 実行はプロジェクトルートから推奨: python scripts/predict.py
  - その場合、結果は ./results/ に保存され、test 画像は kaggle/input/.../archive-3/test/images を参照します。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


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
        default="results",
        help="推論結果（描画済み画像）の保存先ディレクトリ（デフォルト: results）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推論時の画像サイズ（学習時と同じ 640 を推奨）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="検出の信頼度閾値（デフォルト: 0.25）",
    )
    return parser.parse_args()


def get_default_weights():
    """学習スクリプトのデフォルト出力先に対応する best.pt のパス。"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(project_root / "runs/detect/train/weights/best.pt")


def get_default_source():
    """データセットの test/images のデフォルトパス。"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return str(
        project_root
        / "kaggle/input/datasets/umairinayat/floor-plans-500-annotated-object-detection/archive-3/test/images"
    )


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
    )


def run_predict(
    weights,
    source,
    save_dir="results",
    imgsz=640,
    conf=0.25,
):
    """YOLO推論を実行し、results オブジェクトを返す（segment_SAM 等から呼び出し用）。"""
    # ultralytics は project/name に画像を保存するため、project='.', name=save_dir で指定
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
    return results


if __name__ == "__main__":
    main()
