# Kaggle Input データ説明

`kaggle/input/` 内の学習・検証・テストデータの構成と件数です。

---

## ファイル構成

```
kaggle/input/
└── datasets/
    └── umairinayat/
        └── floor-plans-500-annotated-object-detection/
            └── archive-3/
                ├── data.yaml              # データセット設定（クラス名・パス等）
                ├── README.dataset.txt     # データセット概要
                ├── README.roboflow.txt    # Roboflow エクスポート情報
                ├── train/
                │   ├── images/            # 学習用画像 (.jpg)
                │   └── labels/            # 学習用アノテーション (.txt, YOLO形式)
                ├── valid/
                │   ├── images/            # 検証用画像
                │   └── labels/            # 検証用アノテーション
                └── test/
                    ├── images/            # テスト用画像
                    └── labels/            # テスト用アノテーション
```

---

## 件数

| 区分 | 画像数 | ラベル数 |
|------|--------|----------|
| **学習 (train)** | 837 | 837 |
| **検証 (valid)** | 80 | 80 |
| **テスト (test)** | 43 | 43 |
| **合計** | **960** | **960** |

- 画像とラベルは 1:1 対応（1画像につき1つの `.txt` ラベル）
- 画像形式: JPEG (`.jpg`)
- ラベル形式: YOLO（1行1オブジェクト、正規化座標）

---

## データセット概要

- **名称**: Floor Plans 500（Roboflow エクスポート）
- **出典**: [Roboflow Universe - floor-plans-500-xqdyk](https://universe.roboflow.com/umair-ab9ex/floor-plans-500-xqdyk)
- **ライセンス**: CC BY 4.0
- **用途**: 間取り図（フロアプラン）上の **物体検出**（YOLO 学習用）

### クラス（3クラス）

| ID | クラス名 | 説明 |
|----|----------|------|
| 0 | door | ドア |
| 1 | window | 窓 |
| 2 | zone | ゾーン（領域） |

### ラベル形式（YOLO）

各 `.txt` は1行1オブジェクトで、次の形式です。

```
<class_id> <x_center> <y_center> <width> <height>
```

- 座標は画像幅・高さで正規化された 0〜1 の相対値
- `x_center`, `y_center`: バウンディングボックス中心
- `width`, `height`: バウンディングボックスの幅・高さ

### 前処理・データ拡張（Roboflow）

- 元画像から 90° 回転（なし / 時計回り / 反時計回り）のいずれかが等確率で適用された 3 バージョンが含まれる構成です。

---

## 学習時の参照パス（data.yaml）

`data.yaml` では相対パスで以下が指定されています。

- `train`: `../train/images`
- `val`: `../valid/images`
- `test`: `../test/images`

Kaggle や別環境で使う場合は、`data.yaml` のパスを実環境に合わせて書き換えるか、学習スクリプトで絶対パスを指定してください。

例（archive-3 を基準にする場合）:

```yaml
# data.yaml を編集する例
train: /path/to/kaggle/input/datasets/umairinayat/floor-plans-500-annotated-object-detection/archive-3/train/images
val:   /path/to/.../archive-3/valid/images
test:  /path/to/.../archive-3/test/images
```
