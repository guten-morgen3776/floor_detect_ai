"""
YOLO セグメンテーション学習スクリプト（Kaggle 環境用 / 高度ファインチューニング版）
- データセット: my_cubicasa_dataset 構成（images/train, images/val, labels/train, labels/val）
- モデル: yolo26n-seg.pt（Ultralytics）
- Kaggle では input が読み取り専用のため、data.yaml を CONFIG のパスに書き換えて
  /kaggle/working/data.yaml に保存してから学習を実行する。
"""

import yaml
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import YOLO

# =============================================================================
# CONFIG: パス・ハイパーパラメータ（Kaggle 用に書き換えやすいよう冒頭で定義）
# =============================================================================
CONFIG = {
    # ---- Kaggle パス ----
    "kaggle_dataset_path": "/kaggle/input/your-dataset",  # データセットのルート（data.yaml, images/, labels/ がある場所）
    "kaggle_working": "/kaggle/working",
    "kaggle_output_yaml": "/kaggle/working/data.yaml",
    # ---- ローカル用（Kaggle でないときに読む data.yaml のパス）----
    "local_data_yaml": None,  # None のときはスクリプト基準で my_cubicasa_dataset/data.yaml を参照
    # ---- モデル ----
    "model": "yolo26n-seg.pt",
    # ---- ハイパーパラメータ ----
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "project": "runs/segment",
    "name": "train",
    # ---- 高度ファインチューニング用パラメータ ----
    # 特徴量抽出層の凍結（浅い層を固定）。Ultralytics の freeze 引数に渡す。
    # 0: 全層学習 / 1〜: 浅い層から順に凍結
    "freeze_layers": 3,
    # Early Stopping の patience（このエポック数だけ指標が改善しなければ打ち切り）
    "patience": 15,
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


def _init_head_module_weights(m: nn.Module):
    """
    Detect/Segment ヘッド内部の Conv/Linear/Norm 層を再初期化するためのヘルパー。
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(
        m,
        (
            nn.BatchNorm2d,
            nn.GroupNorm,
            nn.LayerNorm,
            nn.InstanceNorm2d,
        ),
    ):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


def reinitialize_model_head(model: YOLO):
    """
    事前学習済み YOLO セグメンテーションモデルの Head（Detect/Segment 層）のみ
    重みをランダム再初期化する。
    """
    core_model = getattr(model, "model", None)
    if core_model is None:
        print("[WARN] YOLO model から内部モデルを取得できませんでした。Head の再初期化をスキップします。")
        return

    modules = getattr(core_model, "model", None)
    if modules is None:
        print("[WARN] YOLO core model に model 属性が見つかりません。Head の再初期化をスキップします。")
        return

    target_class_names = {"Detect", "Segment", "Segment26"}
    reinit_count = 0
    for module in modules:
        if module.__class__.__name__ in target_class_names:
            module.apply(_init_head_module_weights)
            reinit_count += 1
            print(f"[INFO] Head モジュール {module.__class__.__name__} の重みを再初期化しました。")

    if reinit_count == 0:
        print("[WARN] Detect / Segment ヘッドが見つかりませんでした。再初期化は行われていません。")


def main():
    data_yaml = get_data_yaml()
    if is_kaggle_env():
        project = str(Path(CONFIG["kaggle_working"]) / CONFIG["project"])
        print(f"[INFO] Kaggle 実行: 学習結果を {project} 配下に保存します")
    else:
        project = CONFIG["project"]

    model = YOLO(CONFIG["model"])

    # ============================================================
    # Head（Detect/Segment）のみランダム再初期化してから学習
    # ============================================================
    reinitialize_model_head(model)

    results = model.train(
        data=data_yaml,
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch"],
        imgsz=CONFIG["imgsz"],
        project=project,
        name=CONFIG["name"],
        # 特徴量抽出層の凍結（浅い層の freeze）
        freeze=CONFIG["freeze_layers"],
        # Early Stopping（patience エポック改善なしで打ち切り）
        patience=CONFIG["patience"],
        device=[0, 1]
    )
    out_weights = Path(project) / CONFIG["name"] / "weights" / "best.pt"
    print(f"[INFO] ベスト重み: {out_weights}")
    return results


if __name__ == "__main__":
    main()

