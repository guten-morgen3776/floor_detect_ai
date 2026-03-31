# ２段目モデル（Boundary Patch Refinement）詳細設計書

> 参考論文: "Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation" (Tang et al., 2021)

---

## 1. システム全体像

```
CubiCasa5k 訓練画像
    │
    ├─ [RGB画像]
    │
    └─ Stage1: Mask2Former 推論
            ↓
        予測マスク群（部屋ごとのバイナリマスク）
            ↓
    ┌────────────────────────────────────────┐
    │    境界パッチ抽出（Boundary Patch Extraction）   │
    │  ① 境界ピクセル検出 → Dense sliding window    │
    │  ② パッチ候補生成（中心が境界上）              │
    │  ③ NMS でフィルタリング（閾値 0.25）           │
    └───────────────────┬────────────────────┘
                        ↓
            [image_patch: RGB 3ch]  +  [mask_patch: 1ch]
                        ↓
    ┌────────────────────────────────────────┐
    │    Stage2: 軽量 Refinement Network             │
    │  入力: 4ch (RGB 3ch + binary mask 1ch)        │
    │  出力: 2クラス バイナリセグメンテーション       │
    │  モデル: MobileNetV3-Small ベース U-Net        │
    └───────────────────┬────────────────────┘
                        ↓
            精緻化済み境界パッチ群
                        ↓
    ┌────────────────────────────────────────┐
    │    パッチ再統合（Reassemble）                  │
    │  ・重複領域: ロジットを平均 → 閾値 0.5         │
    │  ・非精緻化領域: 元の予測をそのまま維持         │
    └───────────────────┬────────────────────┘
                        ↓
                精緻化済み最終マスク
```

---

## 2. パッチ切り出し（Boundary Patch Extraction）

### 2.1 境界ピクセルの検出

Mask2Formerの予測マスクから境界ピクセルを抽出する。

```python
import cv2
import numpy as np

def extract_boundary_pixels(mask: np.ndarray, dilation_radius: int = 1) -> np.ndarray:
    """
    バイナリマスクから境界ピクセルを抽出する。
    mask: H x W, uint8 (0 or 1)
    returns: boundary_map (H x W, bool)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2*dilation_radius+1, 2*dilation_radius+1))
    dilated  = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded   = cv2.erode(mask.astype(np.uint8), kernel)
    boundary = (dilated - eroded).astype(bool)
    return boundary
```

### 2.2 Dense Sliding Window によるパッチ候補生成

論文の手法に従い、**境界ピクセルが中心領域に来るように** スライディングウィンドウでパッチ候補を密に生成する。

```
パッチサイズ: 64 × 64 ピクセル（パディングなし）
スライドストライド: patch_size / 4 = 16 px（密なカバレッジ）
条件: ボックス中心が境界ピクセル上にある候補のみ採用
```

```python
def generate_patch_candidates(boundary_map: np.ndarray,
                               patch_size: int = 64,
                               stride: int = 16) -> list[dict]:
    """
    境界に沿ってパッチ候補のバウンディングボックスを生成する。
    returns: list of {'x1', 'y1', 'x2', 'y2', 'score'}
    """
    H, W = boundary_map.shape
    half = patch_size // 2
    candidates = []

    # 境界ピクセルの座標を取得
    boundary_ys, boundary_xs = np.where(boundary_map)

    # stride ごとにグリッド化（重複を抑えながらも密に）
    for cy, cx in zip(boundary_ys[::stride//4], boundary_xs[::stride//4]):
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(W, x1 + patch_size)
        y2 = min(H, y1 + patch_size)

        # 境界ピクセルが中心付近にあるかチェック
        center_region = boundary_map[
            max(0, cy-stride):min(H, cy+stride),
            max(0, cx-stride):min(W, cx+stride)
        ]
        if center_region.sum() > 0:
            # スコア: パッチ内の境界ピクセル密度（NMS用）
            patch_boundary = boundary_map[y1:y2, x1:x2]
            score = patch_boundary.mean()
            candidates.append({'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'score':score})

    return candidates
```

### 2.3 NMS によるパッチフィルタリング

候補の重複を排除しつつ境界カバレッジを維持する。

**論文パラメータ:**
- 学習時 NMS 閾値: `0.25`（固定）
- 推論時 NMS 閾値: `0.25`〜`0.55`（精度/速度トレードオフ）
- 画像1枚あたり平均パッチ数: 閾値 0.25 で約 135 枚、0.55 で約 332 枚

```python
def nms_patches(candidates: list[dict], iou_threshold: float = 0.25) -> list[dict]:
    """
    パッチ候補に NMS を適用する。
    iou_threshold: この値より IoU が大きいパッチを除去
    """
    if not candidates:
        return []

    boxes = np.array([[c['x1'], c['y1'], c['x2'], c['y2']] for c in candidates])
    scores = np.array([c['score'] for c in candidates])

    # スコア降順でソート
    order = scores.argsort()[::-1]
    kept = []

    while order.size > 0:
        i = order[0]
        kept.append(candidates[i])
        if order.size == 1:
            break

        # IoU 計算
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_j = (boxes[order[1:],2]-boxes[order[1:],0]) * \
                 (boxes[order[1:],3]-boxes[order[1:],1])
        iou = inter / (area_i + area_j - inter + 1e-6)

        # IoU が閾値以下のものだけ残す
        order = order[1:][iou <= iou_threshold]

    return kept
```

### 2.4 パッチ・マスクの切り出しとリサイズ

```python
def extract_patches(image: np.ndarray,
                    pred_mask: np.ndarray,
                    patch_boxes: list[dict],
                    input_size: int = 128) -> list[dict]:
    """
    image    : H x W x 3 (RGB)
    pred_mask: H x W (binary, 0/1)
    patch_boxes: NMS後のパッチリスト
    input_size : refinement network への入力サイズ（128 or 256）
    """
    patches = []
    for box in patch_boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

        # RGB パッチ
        img_patch  = image[y1:y2, x1:x2]           # H x W x 3
        # マスクパッチ（binary: 0 or 1）
        mask_patch = pred_mask[y1:y2, x1:x2]        # H x W

        # リサイズ（モデル入力サイズに統一）
        img_patch_resized  = cv2.resize(img_patch,  (input_size, input_size),
                                         interpolation=cv2.INTER_LINEAR)
        mask_patch_resized = cv2.resize(mask_patch.astype(np.float32),
                                         (input_size, input_size),
                                         interpolation=cv2.INTER_NEAREST)

        patches.append({
            'box': (x1, y1, x2, y2),
            'image': img_patch_resized,    # input_size x input_size x 3
            'mask':  mask_patch_resized,   # input_size x input_size (float32)
        })
    return patches
```

**正規化:**
- RGB パッチ: ImageNet mean/std で正規化
- マスクパッチ: `mean=0.5, std=0.5` で正規化（論文準拠）

```python
# マスクの正規化
mask_normalized = (mask_patch - 0.5) / 0.5  # [-1, 1] に変換

# 4ch 入力の結合
input_4ch = np.concatenate([
    img_normalized,                          # 3 x H x W
    mask_normalized[np.newaxis, :]           # 1 x H x W
], axis=0)                                   # 4 x H x W
```

---

## 3. ２段目モデル設計

### 3.1 アーキテクチャ: MobileNetV3-Small ベース U-Net

```
入力: 4ch (RGB 3ch + mask 1ch), 128×128 or 256×256

Encoder (MobileNetV3-Small)
  ├─ Layer0: Conv 3→16, stride=2  → 64×64×16
  ├─ Layer1: Bottleneck           → 32×32×24   [skip1]
  ├─ Layer2: Bottleneck           → 16×16×40   [skip2]
  ├─ Layer3: Bottleneck           → 8×8×96     [skip3]
  └─ Layer4: Bottleneck           → 4×4×576    [bottleneck]

Decoder (軽量)
  ├─ Up×2 + skip3 → Conv → 8×8×96
  ├─ Up×2 + skip2 → Conv → 16×16×40
  ├─ Up×2 + skip1 → Conv → 32×32×24
  ├─ Up×2         → Conv → 64×64×16
  └─ Up×2         → Conv → 128×128×2

出力: 2クラス（foreground / background）logits
```

```python
import torch
import torch.nn as nn
import torchvision.models as models

class LightweightRefineNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()

        # MobileNetV3-Small を encoder として使用
        backbone = models.mobilenet_v3_small(pretrained=True)

        # 最初の conv を 4ch 対応に変更
        orig_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(in_channels, orig_conv.out_channels,
                              kernel_size=orig_conv.kernel_size,
                              stride=orig_conv.stride,
                              padding=orig_conv.padding, bias=False)
        # 最初の3chの重みを引き継ぎ、mask ch は平均初期化
        with torch.no_grad():
            new_conv.weight[:, :3] = orig_conv.weight
            new_conv.weight[:, 3:] = orig_conv.weight.mean(dim=1, keepdim=True)
        backbone.features[0][0] = new_conv

        # Encoder の各ステージを分離
        self.enc0 = backbone.features[0]       # stride 2
        self.enc1 = backbone.features[1:3]     # stride 2
        self.enc2 = backbone.features[3:6]     # stride 2
        self.enc3 = backbone.features[6:10]    # stride 2
        self.enc4 = backbone.features[10:]

        # Decoder
        self.dec3 = self._decoder_block(576 + 96, 96)
        self.dec2 = self._decoder_block(96  + 40, 40)
        self.dec1 = self._decoder_block(40  + 24, 24)
        self.dec0 = self._decoder_block(24  + 16, 16)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, num_classes, 1)
        )

    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        s0 = self.enc0(x)     # /2
        s1 = self.enc1(s0)    # /4
        s2 = self.enc2(s1)    # /8
        s3 = self.enc3(s2)    # /16
        bt = self.enc4(s3)    # /32

        d3 = self.dec3(torch.cat([bt, s3], dim=1))   # /16
        d2 = self.dec2(torch.cat([d3, s2], dim=1))   # /8
        d1 = self.dec1(torch.cat([d2, s1], dim=1))   # /4
        d0 = self.dec0(torch.cat([d1, s0], dim=1))   # /2
        return self.head(d0)                           # 元サイズ
```

**パラメータ数目安:** ~1.5M（MobileNetV3-Small ベース）

---

## 4. 学習方法

### 4.1 学習データの準備

```
前処理フロー（CubiCasa5k 4,000枚に対して実施）

1. Mask2Former で全訓練画像を推論 → 予測マスク群を保存
2. 予測マスクと GT マスクを IoU でマッチング
3. IoU > 0.5 の予測マスクのみを学習対象とする（論文準拠）
   ※ IoU が低すぎると境界が GT とかけ離れており学習が不安定になる
4. 採用された予測マスクの境界に沿ってパッチを切り出す
5. 同じ bbox で GT マスクパッチも切り出す
```

```python
def prepare_training_patches(image, pred_masks, gt_masks,
                               iou_threshold=0.5, patch_size=64):
    """
    1枚の画像から学習用パッチペアを生成する。
    Returns: list of (image_patch, mask_patch, gt_patch)
    """
    samples = []

    for pred_mask in pred_masks:
        # GT との IoU を計算して最良マッチを探す
        best_iou, best_gt = 0, None
        for gt_mask in gt_masks:
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            iou = intersection / (union + 1e-6)
            if iou > best_iou:
                best_iou, best_gt = iou, gt_mask

        if best_iou < iou_threshold:
            continue  # スキップ

        # 境界パッチ抽出
        boundary = extract_boundary_pixels(pred_mask)
        candidates = generate_patch_candidates(boundary, patch_size)
        kept_boxes = nms_patches(candidates, iou_threshold=0.25)

        for box in kept_boxes:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            img_p  = image[y1:y2, x1:x2]
            mask_p = pred_mask[y1:y2, x1:x2]
            gt_p   = best_gt[y1:y2, x1:x2]
            samples.append((img_p, mask_p, gt_p))

    return samples
```

**期待パッチ数:**
- 1枚の画像から ~100 パッチ（部屋数 × 境界パッチ数）
- 4,000枚 × 100 = **約 40万パッチ**（学習データ）

### 4.2 学習パラメータ（論文準拠＋軽量化調整）

| パラメータ | 論文 (HRNetV2) | 本実装 (MobileNetV3-UNet) |
|-----------|--------------|--------------------------|
| Optimizer | SGD | SGD |
| 初期学習率 | 0.01 | 0.01 |
| Momentum | 0.9 | 0.9 |
| Weight Decay | 0.0005 | 0.0005 |
| LR スケジューラ | Poly (power=0.9) | Poly (power=0.9) |
| 学習イテレーション | 160K | 80K〜120K（要調整） |
| バッチサイズ | 32 (4 GPU) | 64〜128（A100 1枚） |
| 入力サイズ | 128×128 | 128×128 |
| Loss | Binary Cross Entropy | Binary Cross Entropy |

```python
# 学習設定
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01, momentum=0.9, weight_decay=0.0005
)

# Poly LR スケジューラ
def poly_lr(step, total_steps, power=0.9):
    return (1 - step / total_steps) ** power

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: poly_lr(step, total_steps=120000)
)

# Loss（2クラスなので CE で OK）
criterion = nn.CrossEntropyLoss()
```

### 4.3 データ拡張

```python
import albumentations as A

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    # マスクには適用しない
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),  # RGB のみ
], additional_targets={'mask': 'mask', 'gt': 'mask'})

# マスクの正規化は別途実施
# mask_normalized = (mask - 0.5) / 0.5
```

**注意:** 幾何学的変換（回転・スケール等）は適用しない。境界パッチの切り出し時点で向き情報が重要なため。

---

## 5. パッチ再統合（Reassemble）

精緻化された境界パッチを元のマスクに統合する。

### 5.1 アルゴリズム

```python
def reassemble_patches(original_mask: np.ndarray,
                        refined_patches: list[dict],
                        input_size: int = 128) -> np.ndarray:
    """
    精緻化済みパッチを元のマスクサイズに統合する。

    refined_patches: list of {
        'box': (x1, y1, x2, y2),
        'logits': np.ndarray (2 x input_size x input_size)  # モデル出力
    }
    """
    H, W = original_mask.shape

    # ロジットと重みの累積マップ（重複領域の平均化用）
    logit_sum = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    for patch in refined_patches:
        x1, y1, x2, y2 = patch['box']
        ph, pw = y2 - y1, x2 - x1

        # logits を元のパッチサイズにリサイズ
        fg_logit = patch['logits'][1]  # class 1 (foreground) のロジット
        fg_logit_resized = cv2.resize(fg_logit, (pw, ph),
                                       interpolation=cv2.INTER_LINEAR)

        logit_sum[y1:y2, x1:x2] += fg_logit_resized
        weight_map[y1:y2, x1:x2] += 1.0

    # パッチが存在する領域のみ更新
    patch_region = weight_map > 0
    averaged_logits = np.where(patch_region,
                                logit_sum / (weight_map + 1e-6),
                                -1e9)  # パッチ外は確実に元マスクを維持

    # 閾値 0.5 でバイナリ化（sigmoid 後に 0.5、つまり logit > 0）
    refined_binary = (averaged_logits > 0).astype(np.uint8)

    # 最終マスク: パッチ領域は精緻化結果、それ以外は元の予測
    final_mask = original_mask.copy()
    final_mask[patch_region] = refined_binary[patch_region]

    return final_mask
```

### 5.2 重複パッチの処理（詳細）

```
パッチAとパッチBが重複している領域:

  パッチA の logit: +1.2  (foreground の確信度が高い)
  パッチB の logit: -0.3  (backgroundに傾いている)
  平均ロジット: (1.2 + (-0.3)) / 2 = +0.45

  → 閾値 0 (= sigmoid 0.5) なので: +0.45 > 0 → foreground

NMS 閾値を上げるほど重複が増え、平均化による補正が効く
（論文では 0.55 で AP/AF が飽和）
```

---

## 6. やるべきことリスト（実装ロードマップ）

### Phase 1: 学習データ生成
- [ ] Mask2Former で全訓練画像（4,000枚）を推論し、予測マスクを `.npy` で保存
- [ ] GT マスクとの IoU マッチング処理の実装・テスト
- [ ] パッチ切り出し処理のユニットテスト（境界検出・NMS・リサイズ）
- [ ] `Dataset` クラスの実装（オンデマンドでパッチ生成 or 全パッチを事前保存）

### Phase 2: ２段目モデル実装
- [ ] `LightweightRefineNet` の実装と動作確認
- [ ] 4ch 入力の前処理パイプライン実装（正規化含む）
- [ ] 学習ループ実装（Poly LR・バッチ処理・GPU利用）

### Phase 3: 学習
- [ ] 学習データに対して Mask2Former で推論（A100 使用）
- [ ] ２段目モデルを学習（バッチサイズ 64〜128、80K〜120K iterations）
- [ ] Validation: パッチ単位 IoU と、マスク単位 IoU の両方を監視

### Phase 4: 推論パイプライン統合
- [ ] パッチ再統合処理の実装
- [ ] End-to-end の推論スクリプト作成
- [ ] 定性評価（境界の視覚的改善確認）

---

## 7. 論文との差分・注意点

| 項目 | 論文 (BPR) | 本実装 |
|------|-----------|--------|
| ベースモデル | Mask R-CNN | Mask2Former |
| タスク | Instance seg (自然画像) | Room seg (建築図面) |
| Refinement Net | HRNetV2-W18-Small | MobileNetV3-Small UNet |
| データセット | Cityscapes | CubiCasa5k |
| GT品質 | 高品質ポリゴン | SVG由来ラスタ（比較的クリーン） |
| 画像特性 | 自然画像・色彩豊富 | 建築図面・線画・低テクスチャ |

**建築図面特有の考慮点:**
- 壁の線が細く、パッチサイズ 64px でも複数の部屋境界を含む可能性がある → mask patch による識別が特に重要
- 部屋の境界は直線が多いため、論文より精緻化が容易な可能性がある
- 色情報が少ないため、RGB パッチよりも mask patch への依存度が高くなる可能性がある → mask patch 正規化のチューニングが重要

---

## 8. ハイパーパラメータ感度（論文の Ablation より）

| パラメータ | 推奨値 | 根拠 |
|-----------|--------|------|
| パッチサイズ | 64×64 | パディングなし 64 が最良（論文 Table 3） |
| リサイズ後入力 | 128×128 | 速度と精度のバランス（256 でわずかに改善） |
| NMS 閾値（学習時） | 0.25 | 固定（論文準拠） |
| NMS 閾値（推論時） | 0.25〜0.55 | 高いほど精度↑、速度↓（0.55 付近で飽和） |
| IoU フィルタ | 0.5 | これ以下は境界がずれすぎて学習に有害 |
| mask patch 正規化 | mean=0.5, std=0.5 | 論文 Appendix A 準拠 |
| mask patch あり/なし | **あり必須** | なし → AP が 39.8 から 20.1 に激減（論文 Table 2） |
