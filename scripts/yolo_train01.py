"""
YOLOv11 学習スクリプト（建物見取り図データセット用）
- 初期重み: yolo11n.pt
- エポック数: 50（短めで結果を早く確認）
- 画像サイズ: 640

パスについて:
  - data.yaml 内の train/val/test は、YOLO により「data.yaml の置き場所」からの相対パスで解釈されます。
  - 本データセットの data.yaml は ../train/images, ../valid/images, ../test/images なので、
    archive-3/data.yaml を指定すれば、archive-3 を基準に正しく解決されます。
  - 実行はプロジェクトルートから推奨: python scripts/yolo_train01.py [--data /path/to/data.yaml]

Kaggle 実行時:
  - /kaggle/input/ は読み取り専用のため、data.yaml を /kaggle/working/data.yaml に動的生成します。
  - train/val/test は /kaggle/input/.../archive-3/ からの絶対パスで指定します。
  - 学習結果（runs/ 等）は /kaggle/working/ 配下に保存されます。
"""

import argparse
import yaml
from pathlib import Path

from ultralytics import YOLO

# Kaggle 上でのデータセット配置（input は読み取り専用）
KAGGLE_DATASET_BASE = Path(
    "/kaggle/input/datasets/umairinayat/floor-plans-500-annotated-object-detection/archive-3"
)
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_DATA_YAML = KAGGLE_WORKING / "data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 学習（見取り図: door / window / zone）")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="data.yaml へのパス。省略時はプロジェクトルート基準のデフォルトパスを使用。"
        " data.yaml 内の train/val/test は yaml ファイルの場所からの相対パスで解釈されます。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="学習エポック数（デフォルト: 50）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="入力画像サイズ（デフォルト: 640）",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="バッチサイズ（デフォルト: 16、メモリ不足時は 8 などに下げてください）",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="学習結果の保存先ディレクトリ（デフォルト: runs/detect）",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="実験名（project 配下のフォルダ名、デフォルト: train）",
    )
    return parser.parse_args()


def is_kaggle_env():
    """Kaggle カーネル上で実行されているか（/kaggle/working が存在するか）で判定。"""
    return KAGGLE_WORKING.is_dir()


def generate_kaggle_data_yaml() -> str:
    """
    Kaggle 用の data.yaml を /kaggle/working/data.yaml に動的生成する。
    train/val/test は /kaggle/input/... から始まる絶対パスを指定。
    クラスは nc: 3, names: ['door', 'window', 'zone']。
    生成した YAML のパスを返す。
    """
    data_config = {
        "train": str(KAGGLE_DATASET_BASE / "train" / "images"),
        "val": str(KAGGLE_DATASET_BASE / "valid" / "images"),
        "test": str(KAGGLE_DATASET_BASE / "test" / "images"),
        "nc": 3,
        "names": ["door", "window", "zone"],
    }
    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
    with open(KAGGLE_DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"[INFO] Kaggle 用 data.yaml を生成しました: {KAGGLE_DATA_YAML}")
    return str(KAGGLE_DATA_YAML)


def get_default_data_yaml():
    """プロジェクトルート基準の data.yaml のデフォルトパス。"""
    # このスクリプトは scripts/ にある想定。プロジェクトルートは 1 つ上。
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default = (
        project_root
        / "kaggle/input/datasets/umairinayat/floor-plans-500-annotated-object-detection/archive-3/data.yaml"
    )
    return str(default)


def main():
    args = parse_args()

    # Kaggle 環境: input は読み取り専用のため data.yaml を working に生成し、結果も working 配下に保存
    if is_kaggle_env():
        if args.data is None:
            data_yaml = generate_kaggle_data_yaml()
        else:
            data_yaml = args.data
        project = str(KAGGLE_WORKING / "runs" / "detect")
        print(f"[INFO] Kaggle 実行: 学習結果を {project} 配下に保存します")
    else:
        data_yaml = args.data
        if data_yaml is None:
            data_yaml = get_default_data_yaml()
            print(f"[INFO] --data 未指定のためデフォルトを使用: {data_yaml}")
        project = args.project

    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"data.yaml が見つかりません: {data_yaml}\n"
            "Kaggle からデータを取得している場合はパスを確認するか、"
            "--data で絶対パスを指定してください。"
        )

    # yolo11n.pt は ultralytics が自動ダウンロードする場合があります（初回のみ）
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=project,
        name=args.name,
    )

    # 学習後、best.pt は project/name/weights/best.pt に保存されます
    out_weights = Path(project) / args.name / "weights" / "best.pt"
    print(f"[INFO] ベスト重み: {out_weights}")
    return results


if __name__ == "__main__":
    main()
