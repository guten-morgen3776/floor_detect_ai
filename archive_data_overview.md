# archive(1) データ概要

`/Users/aokitenju/Downloads/archive (1)` 内のデータの調査結果をまとめます。

---

## 1. どんなデータか

### データセット名・用途
- **Cubicasa5k** に基づくデータセットです。
- **間取り図（フロアプラン）** の画像と、それに対応する **アノテーション（壁・部屋のバウンディングボックス）** が含まれます。
- 物体検出（Object Detection）やセグメンテーション用の教師データとして利用できる形式です。

### データの種類

| 種類 | 説明 |
|------|------|
| **画像** | 間取り図の PNG（`F1_original.png`, `F1_scaled.png` など）。1物件あたり1〜複数フロア（F1, F2, F3...）あり。 |
| **ベクター図** | 各物件の `model.svg`。壁・ドア・窓・部屋（Space）・寸法などが SVG で記述された元データ。 |
| **アノテーション** | COCO 形式の JSON。**壁（wall）** と **部屋（room）** の2クラスでバウンディングボックス（bbox: [x, y, width, height]）が付与されている。 |

### カテゴリ（クラス）
- **category_id: 1** → `wall`（壁）
- **category_id: 2** → `room`（部屋）

### ファイル数（目安）
- **総ファイル数**: 約 17,351
- **PNG 画像**: 約 12,342
- **SVG**: 約 5,000
- **JSON（COCO）**: 3（train / val / test 用）

### データ分割
- **train**: 4,199 件（パスリスト）
- **val**: 399 件
- **test**: 399 件  
※ `train.txt` / `val.txt` / `test.txt` は **フォルダパス** のリスト（1行1パス）。

---

## 2. フォルダ構成

```
archive (1)/
├── cubicasa5k/
│   └── cubicasa5k/
│       ├── train.txt          # 学習用フォルダパス一覧（4,199行）
│       ├── val.txt            # 検証用フォルダパス一覧（399行）
│       ├── test.txt           # テスト用フォルダパス一覧（399行）
│       ├── colorful/          # 約276物件（カラフルな間取り図）
│       │   └── <id>/
│       │       ├── model.svg
│       │       ├── F1_original.png, F1_scaled.png
│       │       └── （複数フロアの場合は F2_*, F3_* など）
│       ├── high_quality/      # 約992物件（高品質）
│       │   └── <id>/
│       │       ├── model.svg
│       │       ├── F1_original.png, F1_scaled.png
│       │       └── （同上）
│       └── high_quality_architectural/  # 約3,732物件（高品質・建築向け）
│           └── <id>/
│               ├── model.svg
│               ├── F1_original.png
│               └── F1_scaled.png
│
└── cubicasa5k_coco/          # COCO形式アノテーション
    ├── train_coco_pt.json    # 学習用（images + annotations + categories）
    ├── val_coco_pt.json      # 検証用
    └── test_coco_pt.json     # テスト用
```

### 各フォルダの役割

| パス | 役割 |
|------|------|
| `cubicasa5k/cubicasa5k/` | 間取り図の画像・SVG の実体。`train/val/test` はここからの相対パスで指定。 |
| `cubicasa5k/cubicasa5k/colorful` | カラフルなスタイルの間取り図。 |
| `cubicasa5k/cubicasa5k/high_quality` | 高品質サブセット。1物件で複数フロア（F2, F3 など）がある場合あり。 |
| `cubicasa5k/cubicasa5k/high_quality_architectural` | 高品質・建築向け。多くのサンプルが F1 のみ。 |
| `cubicasa5k_coco/` | 上記画像に対する **壁・部屋** の COCO 形式 bbox アノテーション。 |

### 1物件フォルダ内の典型的なファイル

- **model.svg**  
  間取りのベクター表現（壁・ドア・窓・部屋名・寸法など）。
- **F1_original.png**  
  1階の間取り図画像（元解像度）。
- **F1_scaled.png**  
  同じ1階をスケールした画像。  
※ 2階以上がある場合は `F2_original.png`, `F2_scaled.png` などが追加される。

### COCO JSON のパスについて
JSON 内の `file_name` は Kaggle 環境を想定した絶対パス（例: `/kaggle/input/cubicasa5k/...`）になっています。  
ローカルで使う場合は、このパスを `archive (1)` 内の実際のパスに置き換える必要があります。

---

## 3. まとめ

- **データの正体**: 間取り図画像＋壁/部屋の bbox アノテーション（Cubicasa5k 系）。
- **フォルダ構成**: `cubicasa5k/cubicasa5k/` に3種のサブセット（colorful / high_quality / high_quality_architectural）、各サブセット内は `<id>/` 単位で `model.svg` と PNG が格納。`cubicasa5k_coco/` に train/val/test 用の COCO JSON が3つ。
- **用途**: 間取り図における壁・部屋の検出や、フロアプラン解析の学習データとして利用可能。
