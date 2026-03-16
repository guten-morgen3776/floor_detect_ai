"""
YOLO セグメンテーション学習スクリプト（Kaggle 環境用）
- データセット: my_cubicasa_dataset 構成（images/train, images/val, labels/train, labels/val）
- モデル: yolo26n-seg.pt（Ultralytics）
- Kaggle では input が読み取り専用のため、data.yaml を CONFIG のパスに書き換えて
  /kaggle/working/data.yaml に保存してから学習を実行する。
"""

import yaml
from pathlib import Path

from ultralytics import YOLO

# =============================================================================
# CONFIG: パス・ハイパーパラメータ（Kaggle 用に書き換えやすいよう冒頭で定義）
# =============================================================================
CONFIG = {
    # ---- Kaggle パス ----
    "kaggle_dataset_path": "/kaggle/input/your-dataset",  # データセットのルート（data.yaml, images/, labels/ がある場所）
    "kaggle_working": "/kaggle/working",
    "kaggle_output_yaml": "/kaggle/working/data.yaml",
    # ---- ローカル用（Kaggle でないときに読む data.yaml のパス） ----
    "local_data_yaml": None,  # None のときはスクリプト基準で my_cubicasa_dataset/data.yaml を参照
    # ---- モデル ----
    "model": "yolo26n-seg.pt",
    # ---- ハイパーパラメータ ----
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "project": "runs/segment",
    "name": "train",
}


def _script_dir():
    return Path(__file__).resolve().parent


def _default_local_data_yaml():
    """ローカル用 data.yaml のデフォルトパス（my_cubicasa_dataset 構成）。"""
    return _script_dir() / "my_cubicasa_dataset" / "data.yaml"


def is_kaggle_env():
    """Kaggle カーネル上かどうか（/kaggle/working の存在で判定）。"""
    return Path(CONFIG["kaggle_working"]).is_dir()


def build_kaggle_data_yaml():
    """
    データセット付属の data.yaml を読み、path を Kaggle 用に書き換えて
    CONFIG["kaggle_output_yaml"] に保存する。
    """
    source_yaml = Path(CONFIG["kaggle_dataset_path"]) / "data.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(
            f"Kaggle 用の data.yaml 元ファイルが見つかりません: {source_yaml}\n"
            "CONFIG['kaggle_dataset_path'] がデータセットのルート（data.yaml がある場所）を指しているか確認してください。"
        )
    with open(source_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # path を Kaggle のデータセットルートに統一（train/val は相対のまま）
    data["path"] = CONFIG["kaggle_dataset_path"]
    out_path = Path(CONFIG["kaggle_output_yaml"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"[INFO] Kaggle 用 data.yaml を保存しました: {out_path}")
    return str(out_path)


def get_data_yaml():
    """学習に使う data.yaml のパスを返す（Kaggle 時は書き換え済みの working 用）。"""
    if is_kaggle_env():
        return build_kaggle_data_yaml()
    local = CONFIG.get("local_data_yaml")
    path = Path(local) if local else _default_local_data_yaml()
    if not path.exists():
        raise FileNotFoundError(
            f"data.yaml が見つかりません: {path}\n"
            "CONFIG['local_data_yaml'] を設定するか、scripts/my_cubicasa_dataset/data.yaml を用意してください。"
        )
    return str(path)


def main():
    data_yaml = get_data_yaml()
    if is_kaggle_env():
        project = str(Path(CONFIG["kaggle_working"]) / CONFIG["project"])
        print(f"[INFO] Kaggle 実行: 学習結果を {project} 配下に保存します")
    else:
        project = CONFIG["project"]

    model = YOLO(CONFIG["model"])
    results = model.train(
        data=data_yaml,
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch"],
        imgsz=CONFIG["imgsz"],
        project=project,
        name=CONFIG["name"],
    )
    out_weights = Path(project) / CONFIG["name"] / "weights" / "best.pt"
    print(f"[INFO] ベスト重み: {out_weights}")
    return results


if __name__ == "__main__":
    main()
