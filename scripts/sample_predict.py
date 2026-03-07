"""
YOLOv11 推論・可視化スクリプト（自前画像1枚用）
- predict.py と同じ推論過程を使用
- 学習で得た best.pt を読み込み、指定した1枚の画像に対して推論
- バウンディングボックス描画済み画像を results/ に保存

パスについて:
  - デフォルトの weights は「このスクリプトの位置」からプロジェクトルートを推定しています。
  - 実行はプロジェクトルートから推奨: python scripts/sample_predict.py --source /path/to/your/image.png
  - 結果は ./results/ に保存されます。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 推論・可視化（自前画像1枚）")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="推論対象の画像1枚のパス（例: /path/to/your/image.png）",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="学習済み重み（best.pt）のパス。省略時は runs/detect/train/weights/best.pt を使用。",
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
    if not Path(source).exists():
        raise FileNotFoundError(
            f"推論対象の画像が見つかりません: {source}\n"
            "--source で正しい画像パスを指定してください。"
        )

    save_dir = args.save_dir

    # predict.py と同じ推論過程
    model = YOLO(weights)
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=".",
        name=save_dir,
        exist_ok=True,
    )

    print(f"[INFO] 推論結果の画像を保存しました: {Path(save_dir).resolve()}")
    return results


if __name__ == "__main__":
    main()
