"""
config.py  ―  exp01-bpr 全設定の一元管理

全スクリプトから `from config import *` で参照する。
パス設定は CLAUDE.md に従い Kaggle / Google Colab を自動判別する。
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 環境判別
# ─────────────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# ─────────────────────────────────────────────────────────────────────────────
# パス設定
# ─────────────────────────────────────────────────────────────────────────────
if IS_KAGGLE:
    # Kaggle 環境 ─────────────────────────────────────────────────────────────
    IMG_DIR_TRAIN   = Path("/kaggle/input/cubicasa-mask2former/images/train")
    IMG_DIR_VAL     = Path("/kaggle/input/cubicasa-mask2former/images/val")
    GT_DIR_TRAIN    = Path("/kaggle/input/cubicasa-instance/train")
    GT_DIR_VAL      = Path("/kaggle/input/cubicasa-instance/val")
    STAGE1_WEIGHTS  = Path("/kaggle/input/mask2former-exp04/best_model.pth")
    PRED_DIR_TRAIN  = Path("/kaggle/working/bpr_pred_masks/train")
    PRED_DIR_VAL    = Path("/kaggle/working/bpr_pred_masks/val")
    OUTPUT_DIR      = Path("/kaggle/working/bpr_exp01")
else:
    # Google Colab 環境 ────────────────────────────────────────────────────────
    IMG_DIR_TRAIN   = Path("/content/drive/MyDrive/cubicasa_mask2former/images/train")
    IMG_DIR_VAL     = Path("/content/drive/MyDrive/cubicasa_mask2former/images/val")
    GT_DIR_TRAIN    = Path("/content/drive/MyDrive/cubicasa_instance/train")
    GT_DIR_VAL      = Path("/content/drive/MyDrive/cubicasa_instance/val")
    STAGE1_WEIGHTS  = Path("/content/drive/MyDrive/mask2former_exp04/best_model.pth")
    PRED_DIR_TRAIN  = Path("/content/drive/MyDrive/bpr_pred_masks/train")
    PRED_DIR_VAL    = Path("/content/drive/MyDrive/bpr_pred_masks/val")
    OUTPUT_DIR      = Path("/content/drive/MyDrive/bpr_exp01")

BEST_MODEL = OUTPUT_DIR / "best_model.pth"
LAST_MODEL = OUTPUT_DIR / "last_model.pth"
LOSS_PNG   = OUTPUT_DIR / "loss.png"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 (Mask2Former)
# ─────────────────────────────────────────────────────────────────────────────
STAGE1_MODEL_ID  = "facebook/mask2former-swin-small-coco-instance"
STAGE1_IMG_SIZE  = 1024   # Letterbox サイズ (train04 と同一)
STAGE1_THRESHOLD = 0.5    # confidence 閾値

# ─────────────────────────────────────────────────────────────────────────────
# Boundary Patch Extraction
# ─────────────────────────────────────────────────────────────────────────────
DILATION_RADIUS  = 1      # 境界ピクセル検出の膨張半径
PATCH_SIZE       = 64     # 切り出しパッチサイズ (原寸 px)
PATCH_STRIDE     = 16     # スライドストライド (px)
NMS_IOU_TRAIN    = 0.25   # 学習時 NMS 閾値 (論文固定値)
NMS_IOU_INFER    = 0.35   # 推論時 NMS 閾値 (精度優先なら 0.55 まで上げる)
IOT_FILTER_THR   = 0.5    # GT との IoU フィルタ閾値 (これ未満の予測は学習対象外)

# ─────────────────────────────────────────────────────────────────────────────
# Refinement Network
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SIZE   = 128    # refinement network への入力サイズ (px)
IN_CHANNELS  = 4     # RGB 3ch + pred_mask 1ch
NUM_CLASSES  = 2     # foreground / background

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
TOTAL_ITERS     = 100_000
LR              = 0.01
MOMENTUM        = 0.9
WEIGHT_DECAY    = 0.0005
POLY_POWER      = 0.9
VAL_EVERY_ITERS = 5_000    # N iterations ごとに検証
NUM_WORKERS     = 4
IMG_CACHE_SIZE  = 64       # per-worker 画像 LRU キャッシュサイズ
