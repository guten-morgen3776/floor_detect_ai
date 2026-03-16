"""
YOLO セグメンテーション推論スクリプト
- 学習済み重み（best.pt）を読み込み、画像またはディレクトリに対して推論を実行
- Ultralytics ライブラリを使用
- Kaggle / ローカル両対応（出力ディレクトリを CONFIG から制御）
"""

from pathlib import Path

from ultralytics import YOLO

# =============================================================================
# CONFIG: パス・パラメータ（Kaggle でも書き換えやすいよう冒頭で定義）
# =============================================================================
CONFIG = {
    # ---- 重みファイル ----
    # 学習スクリプト train_yoloseg.py で保存される best.pt を想定
    # 例: ローカル -> "runs/segment/train/weights/best.pt"
    #     Kaggle  -> "/kaggle/working/runs/segment/train/weights/best.pt"
    "weights": "runs/segment/train/weights/best.pt",

    # ---- 推論対象 ----
    # 画像ファイルパス or ディレクトリパス
    # 例: "path/to/image.png" または "path/to/images_dir"
    "source": "path/to/your/image_or_dir",

    # ---- 出力先設定 ----
    # ベースとなる出力ディレクトリ名（project）
    # Kaggle 環境では /kaggle/working 配下にこの project 名で保存される
    # ローカル環境ではカレントディレクトリからの相対パスとして使用
    "project": "runs/segment-predict",
    "name": "predict",  # サブディレクトリ名

    # Kaggle 判定用
    "kaggle_working": "/kaggle/working",

    # ---- 推論パラメータ ----
    "conf": 0.25,     # 信頼度しきい値
    "imgsz": 640,     # 推論時の画像サイズ
    "device": None,   # 例: "0"（GPU0）、"cpu"、None なら自動選択
}


def is_kaggle_env() -> bool:
    """Kaggle カーネル上かどうか（/kaggle/working の存在で判定）。"""
    return Path(CONFIG["kaggle_working"]).is_dir()


def resolve_project_dir() -> str:
    """
    推論結果保存用の project パスを解決して返す。
    - Kaggle: /kaggle/working/<project>
    - ローカル: CONFIG["project"] をそのまま使用
    """
    project = Path(CONFIG["project"])
    if is_kaggle_env():
        base = Path(CONFIG["kaggle_working"])
        project = base / project
        print(f"[INFO] Kaggle 実行: 推論結果を {project} 配下に保存します")
    else:
        print(f"[INFO] ローカル実行: 推論結果を {project} 配下に保存します")
    project.mkdir(parents=True, exist_ok=True)
    return str(project)


def resolve_weights_path() -> str:
    """
    重みファイル（best.pt）のパスを解決して返す。
    - 絶対パスの場合はそのまま使用
    - 相対パスの場合はスクリプト実行ディレクトリからの相対として解釈
    """
    w = Path(CONFIG["weights"])
    if not w.is_absolute():
        # スクリプトファイルの場所を基準に解決
        script_dir = Path(__file__).resolve().parent
        w = script_dir.parent / w  # scripts/ の 1 つ上をプロジェクトルートと想定
    if not w.exists():
        raise FileNotFoundError(
            f"重みファイルが見つかりません: {w}\n"
            "CONFIG['weights'] のパスを確認してください。"
        )
    return str(w)


def main():
    # パス解決
    weights_path = resolve_weights_path()
    project_dir = resolve_project_dir()
    source = CONFIG["source"]

    if not source:
        raise ValueError("CONFIG['source'] が空です。推論したい画像ファイルまたはディレクトリを指定してください。")

    print(f"[INFO] 使用重み: {weights_path}")
    print(f"[INFO] 推論対象: {source}")

    # モデル読み込み
    model = YOLO(weights_path)

    # 推論実行（セグメンテーションのマスク付き画像を保存）
    results = model.predict(
        source=source,
        conf=CONFIG["conf"],
        imgsz=CONFIG["imgsz"],
        project=project_dir,
        name=CONFIG["name"],
        save=True,            # マスク描画済み画像を保存
        device=CONFIG["device"],
    )

    print(f"[INFO] 推論完了。結果は {project_dir}/{CONFIG['name']} 配下に保存されています。")
    return results


if __name__ == "__main__":
    main()

