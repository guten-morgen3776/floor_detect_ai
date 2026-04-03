"""
model.py  ―  LightweightRefineNet

MobileNetV3-Small を Encoder に使った軽量 U-Net。
入力: 4ch (RGB 3ch + pred_mask 1ch), shape (B, 4, H, W)
出力: 2-class logits,               shape (B, 2, H, W)

MobileNetV3-Small の各ステージ出力チャネル (128×128 入力時):
  enc0 = features[0]     → 64×64,  16ch
  enc1 = features[1]     → 32×32,  16ch
  enc2 = features[2:4]   → 16×16,  24ch
  enc3 = features[4:9]   → 8×8,    48ch
  enc4 = features[9:]    → 4×4,   576ch  (最後の Conv 96→576 を含む)

Decoder:
  dec3: up×2 + cat(576+48=624) → 96ch  (4→8)
  dec2: up×2 + cat(96+24=120)  → 40ch  (8→16)
  dec1: up×2 + cat(40+16=56)   → 24ch  (16→32)
  dec0: up×2 + cat(24+16=40)   → 16ch  (32→64)
  head: up×2 + Conv(16→2)             (64→128)
"""

import torch
import torch.nn as nn
import torchvision.models as M

from config import IN_CHANNELS, NUM_CLASSES, INPUT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# チャネル定数 (MobileNetV3-Small, 検証済み)
# ─────────────────────────────────────────────────────────────────────────────
_ENC0_CH = 16
_ENC1_CH = 16
_ENC2_CH = 24
_ENC3_CH = 48
_BT_CH   = 576


# ─────────────────────────────────────────────────────────────────────────────
# デバッグ用: 実際のチャネル数を表示
# ─────────────────────────────────────────────────────────────────────────────

def check_encoder_channels(input_size: int = INPUT_SIZE) -> None:
    """
    MobileNetV3-Small の各 Encoder ステージ出力形状を表示する。
    モデル定義前に呼んでチャネル数を確認するためのユーティリティ。
    """
    backbone = M.mobilenet_v3_small(
        weights=M.MobileNet_V3_Small_Weights.DEFAULT
    )
    x  = torch.zeros(1, 3, input_size, input_size)
    s0 = backbone.features[0](x)
    s1 = backbone.features[1](s0)
    s2 = backbone.features[2:4](s1)
    s3 = backbone.features[4:9](s2)
    bt = backbone.features[9:](s3)
    print(f"enc0 (features[0])    : {tuple(s0.shape)}  ch={s0.shape[1]}")
    print(f"enc1 (features[1])    : {tuple(s1.shape)}  ch={s1.shape[1]}")
    print(f"enc2 (features[2:4])  : {tuple(s2.shape)}  ch={s2.shape[1]}")
    print(f"enc3 (features[4:9])  : {tuple(s3.shape)}  ch={s3.shape[1]}")
    print(f"enc4 (features[9:])   : {tuple(bt.shape)}  ch={bt.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# モデル本体
# ─────────────────────────────────────────────────────────────────────────────

class LightweightRefineNet(nn.Module):
    """
    MobileNetV3-Small ベースの軽量 U-Net。
    約 1.5M パラメータ。
    """

    def __init__(self,
                 in_channels: int = IN_CHANNELS,
                 num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Encoder (MobileNetV3-Small) ───────────────────────────────────────
        backbone = M.mobilenet_v3_small(
            weights=M.MobileNet_V3_Small_Weights.DEFAULT
        )

        # 最初の Conv を in_channels ch 対応に差し替える
        orig_conv = backbone.features[0][0]   # Conv2d(3, 16, 3, stride=2, ...)
        new_conv  = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = orig_conv.weight
            # mask ch は既存重みの平均で初期化
            new_conv.weight[:, 3:] = orig_conv.weight.mean(dim=1, keepdim=True)
        backbone.features[0][0] = new_conv

        self.enc0 = backbone.features[0]       # /2  → 16ch
        self.enc1 = backbone.features[1]       # /2  → 16ch
        self.enc2 = backbone.features[2:4]     # /2  → 24ch
        self.enc3 = backbone.features[4:9]     # /2  → 48ch
        self.enc4 = backbone.features[9:]      # /2 + more → 576ch

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec3 = self._dec_block(_BT_CH   + _ENC3_CH, 96)   # 624 → 96
        self.dec2 = self._dec_block(96        + _ENC2_CH, 40)   # 120 → 40
        self.dec1 = self._dec_block(40        + _ENC1_CH, 24)   #  56 → 24
        self.dec0 = self._dec_block(24        + _ENC0_CH, 16)   #  40 → 16
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    # ── ブロック定義 ──────────────────────────────────────────────────────────

    @staticmethod
    def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 4, H, W) float32

        Returns
        -------
        logits : (B, 2, H, W) float32
        """
        s0 = self.enc0(x)               # (B, 16, H/2,  W/2)
        s1 = self.enc1(s0)              # (B, 16, H/4,  W/4)
        s2 = self.enc2(s1)              # (B, 24, H/8,  W/8)
        s3 = self.enc3(s2)              # (B, 48, H/16, W/16)
        bt = self.enc4(s3)              # (B,576, H/32, W/32)

        d3 = self.dec3(torch.cat([bt, s3], dim=1))  # (B, 96, H/16, W/16)
        d2 = self.dec2(torch.cat([d3, s2], dim=1))  # (B, 40, H/8,  W/8)
        d1 = self.dec1(torch.cat([d2, s1], dim=1))  # (B, 24, H/4,  W/4)
        d0 = self.dec0(torch.cat([d1, s0], dim=1))  # (B, 16, H/2,  W/2)
        return self.head(d0)                          # (B,  2, H,    W)
