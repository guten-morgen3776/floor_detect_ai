# archive-4 データセット調査レポート

調査日: 2025年3月17日  
対象: `/Users/aokitenju/Downloads/archive-4`

---

## 1. フォルダ構成

```
archive-4/
├── mds_V2_5.372k.csv          # メタデータCSV（約383MB）
└── modified-swiss-dwellings-v2/
    ├── train/
    │   ├── struct_in/         # 入力画像（.npy）
    │   ├── graph_in/          # 入力グラフ（.pickle）
    │   ├── graph_out/         # 出力グラフ（.pickle）
    │   └── full_out/          # 出力画像（.npy）
    └── test/
        ├── struct_in/
        ├── graph_in/
        ├── graph_out/
        └── full_out/
```

- **トップレベル**: CSV 1ファイルと、`modified-swiss-dwellings-v2` ディレクトリ1つ。
- **modified-swiss-dwellings-v2**: 建物・住居系データセット（Swiss Dwellings 系）の改変版。`train` と `test` に分割。
- **各分割内**: 同一IDに対して `struct_in`（画像入力）、`graph_in`（グラフ入力）、`graph_out`（グラフ出力）、`full_out`（画像出力）の4種類が対応している。

---

## 2. データの数

| 種類 | 形式 | train | test | 合計 |
|------|------|-------|------|------|
| struct_in | .npy | 4,572 | 800 | 5,372 |
| graph_in | .pickle | 4,572 | 800 | 5,372 |
| graph_out | .pickle | 4,572 | 800 | 5,372 |
| full_out | .npy | 4,572 | 800 | 5,372 |
| **メタデータ** | **CSV** | — | — | **1,086,846行**（ヘッダー除く） |

- 画像・グラフは **1サンプルあたり4ファイル**（同一IDで `struct_in`, `graph_in`, `graph_out`, `full_out` が1組）。
- サンプルIDは数値（例: `10000`, `10009`, `15193`, `4877`）。train と test でIDの集合は異なる。
- CSV の行数（約108.7万行）は「フロアプラン内の領域・壁・ドア等のポリゴン単位」のレコード数で、画像サンプル数（5,372）とは一致しない。

---

## 3. データの格納形式

### 3.1 struct_in / full_out（.npy）

- **形式**: NumPy 配列（`.npy`）
- **形状**: `(512, 512, 3)`（高さ × 幅 × チャンネル）
- **型**: `float16`
- **内容**:
  - **struct_in**: 入力のフロアプラン画像。値域はおおよそ -19.5 ～ 255.0（画像＋何らかの正規化 or マスク）。
  - **full_out**: 対応する出力画像。値域はおおよそ -16.1 ～ 16.7（struct_in より狭い範囲で、推論・変換結果と考えられる）。

### 3.2 graph_in / graph_out（.pickle）

- **形式**: Python の pickle（`.pickle`）
- **内容**: グラフ構造（例: NetworkX のグラフオブジェクト）。読み込みには `networkx` が必要な場合あり。
- **役割**:
  - **graph_in**: 入力フロアプランに対応するグラフ表現（部屋・壁・ドア等のノード/エッジ）。
  - **graph_out**: それに対応する出力グラフ（推論・変換後のグラフと考えられる）。

### 3.3 メタデータ（mds_V2_5.372k.csv）

- **形式**: CSV（カンマ区切り）
- **サイズ**: 約 383MB
- **行数**: 1,086,847 行（1行目はヘッダー → データ行は 1,086,846）
- **文字コード**: UTF-8 想定（要確認）

---

## 4. 各データの内容

### 4.1 struct_in（入力画像）

- 512×512×3 の float16 画像。
- フロアプランや構造図の「入力」として使われるデータ。
- 値に 255 付近が含まれるため、RGB 画像またはマスクを float に変換したものと考えられる。

### 4.2 full_out（出力画像）

- 同じく 512×512×3、float16。
- struct_in と同じIDの「出力」で、値域が -16～16 程度に収まっており、正規化された予測結果や別表現の画像と推測される。

### 4.3 graph_in / graph_out（グラフ）

- フロアプランをグラフ（ノード＝部屋/壁/ドア等、エッジ＝接続関係）で表現したもの。
- 入力グラフ（graph_in）と出力グラフ（graph_out）のペアで、何らかのグラフ変換・予測タスク用と解釈できる。

### 4.4 mds_V2_5.372k.csv（メタデータ）

- **主な列**:
  - `Unnamed: 0`, `Unnamed: 0.1`: 行インデックス
  - `apartment_id`, `site_id`, `building_id`, `plan_id`, `floor_id`, `unit_id`, `area_id`: 建物・フロア・ユニット・領域のID
  - `unit_usage`: 用途（RESIDENTIAL, PUBLIC など）
  - `entity_type`, `entity_subtype`: エンティティ種別（area, WALL, DOOR, WINDOW など）
  - `geom`: ジオメトリ（WKT の POLYGON 文字列）
  - `elevation`, `height`: 高さ情報
  - `zoning`: ゾーン（Zone1～4, Structure, Door, Window 等）
  - `roomtype`: 部屋タイプ（Structure, Door, Window, Bedroom, Bathroom, Kitchen 等）

- **entity_subtype の例**（上位）:
  - WALL, DOOR, WINDOW, ENTRANCE_DOOR, SHAFT, BATHROOM, CORRIDOR, ROOM, BALCONY, KITCHEN, BEDROOM, LIVING_DINING, COLUMN, STOREROOM, STAIRCASE, ELEVATOR, LIVING_ROOM, VOID など。

- **roomtype の例**（上位）:
  - Structure, Door, Window, Bedroom, Entrance Door, Bathroom, Corridor, Balcony, Kitchen, Livingroom, Stairs, Storeroom, Dining など。

- **unit_usage**:
  - RESIDENTIAL（約84,486件）, PUBLIC（約15,514件）※10万行サンプル時点の傾向。

- **用途**: フロアプラン上の「領域・壁・ドア・窓」などをポリゴン単位で記述したメタデータ。画像・グラフのIDと紐付けて、建物・フロア・部屋種別などの属性を参照するために使える。

---

## 5. まとめ

| 項目 | 内容 |
|------|------|
| **データセット名** | archive-4（modified-swiss-dwellings-v2 ＋ mds_V2_5.372k.csv） |
| **タスク想定** | フロアプラン画像・グラフの入力から、画像/グラフの変換・予測（構造理解・生成など） |
| **画像** | 512×512×3, float16, .npy（struct_in = 入力, full_out = 出力） |
| **グラフ** | .pickle（graph_in = 入力, graph_out = 出力）、NetworkX 等を想定 |
| **メタデータ** | CSV 約108.7万行、建物・フロア・部屋・ポリゴン属性を記述 |
| **サンプル数** | 画像/グラフは 5,372 サンプル（train 4,572 + test 800） |

以上が archive-4 内データセットの調査結果です。
