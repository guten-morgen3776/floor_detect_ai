"""
OneFormer instance segmentation fine-tuning on CubiCasa room dataset.
Dataset: cubicasa_mask2former/ (COCO format, single class: room)

Mask2Former からの主な変更点
-------------------------------
* モデル : OneFormerForUniversalSegmentation (shi-labs/oneformer_coco_swin_large)
* プロセッサ : OneFormerProcessor
    - OneFormerImageProcessor（リサイズ・正規化・アノテーション変換）
    - CLIPTokenizer（タスク文字列 "instance" をトークン ID に変換）
* タスク条件付き入力 : task_inputs=["instance"] を processor / model 双方に渡す
* text_inputs : クラス名から生成されるクエリ用テキスト記述子
    processor が自動生成し、model(**batch) でそのまま渡す
* 正規化 : albumentations から削除 → OneFormerProcessor が内部で実施
* Dataset の返り値 :
    - image          : uint8 (H, W, 3) — プロセッサに渡す生画像
    - segmentation_map : (H, W) uint8 — ピクセル値 = inst_id (1-indexed; 0=bg)
    - instance_id_to_semantic_id : Dict[int, int] — inst_id → class label
* CollateFn : callable クラスとして実装（multiprocessing DataLoader との互換性）

Instance mask strategy (unchanged)
------------------------------------
uint8 の「index mask」（ピクセル値 = inst_id）でメモリを節約。
albumentations で画像と同期してクロップ・フリップ後、
processor が inst_id ごとの binary mask を生成する。

Usage (single GPU):
  python train01.py --batch_size 2 --epochs 50 --fp16

Usage (multi-GPU, e.g. T4×2 on Kaggle):
  accelerate launch train01.py --batch_size 2 --epochs 50 --fp16
"""

from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None   # suppress DecompressionBomb for large floor plans

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from accelerate import Accelerator

from transformers import (
    OneFormerForUniversalSegmentation,
    OneFormerConfig,
    OneFormerProcessor,
)
import albumentations as A

# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch: linear_sum_assignment に NaN/Inf が渡された場合に ValueError が
# 発生するのを防ぐ。fp16 学習やコストマトリクス計算の数値不安定性が原因で
# 混入することがある。無効値を大きな有限値に置換して安全に処理する。
# ─────────────────────────────────────────────────────────────────────────────
import numpy as _np
import scipy.optimize as _scipy_opt
from transformers.models.oneformer import modeling_oneformer as _oneformer_mod

_orig_lsa = _scipy_opt.linear_sum_assignment

def _safe_linear_sum_assignment(cost_matrix):
    if hasattr(cost_matrix, "numpy"):
        arr = cost_matrix.numpy()
    elif not isinstance(cost_matrix, _np.ndarray):
        arr = _np.array(cost_matrix)
    else:
        arr = cost_matrix
    if not _np.isfinite(arr).all():
        arr = _np.nan_to_num(arr, nan=1e9, posinf=1e9, neginf=-1e9)
    return _orig_lsa(arr)

_oneformer_mod.linear_sum_assignment = _safe_linear_sum_assignment

# ─────────────────────────────────────────────────────────────────────────────
# Kaggle 環境判別
# ─────────────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# OneFormer COCO 学習済みモデル（Swin-Large バックボーン）
# 代替: "shi-labs/oneformer_coco_dinat_large" (DiNAT バックボーン)
DEFAULT_MODEL_ID = "shi-labs/oneformer_coco_swin_large"

if IS_KAGGLE:
    DATA_ROOT  = Path("/kaggle/input/datasets/taka3776toshi/cubicasa-mask2former/cubicasa_mask2former")
    OUTPUT_DIR = Path("/kaggle/working/exp001")
else:
    DATA_ROOT  = Path(os.environ.get("KAGGLE_DATA_ROOT")) / "cubicasa_mask2former"
    OUTPUT_DIR = Path("/content/drive/MyDrive/oneformer_exp001")

ANNO_TRAIN    = DATA_ROOT / "annotations" / "instances_train.json"
ANNO_VAL      = DATA_ROOT / "annotations" / "instances_val.json"
IMG_DIR_TRAIN = DATA_ROOT / "images" / "train"
IMG_DIR_VAL   = DATA_ROOT / "images" / "val"

BEST_MODEL   = OUTPUT_DIR / "best_model.pth"
LAST_MODEL   = OUTPUT_DIR / "last_model.pth"
LOSS_PNG     = OUTPUT_DIR / "loss.png"
PROCESSOR_DIR = OUTPUT_DIR / "processor"   # processor.save_pretrained() 保存先

# Image size limits
MAX_W = 5000
MAX_H = 4000

# Letterbox resize target（長辺をこのサイズに縮小し、短辺は黒パディングで揃える）
# OneFormerProcessor はその後さらに shortest_edge=800 でリサイズする
IMG_SIZE = 1024

# Minimum surviving mask pixels to keep an instance after crop
MIN_MASK_PX = 64

# Single class: room
NUM_LABELS      = 1
CAT_ID_TO_LABEL = {1: 0}    # COCO category_id=1 → 0-indexed model label
ID2LABEL        = {0: "room"}
LABEL2ID        = {"room": 0}

# OneFormer タスクトークン（インスタンスセグメンテーションに固定）
TASK_INPUT = "instance"


# ─────────────────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OneFormer room instance segmentation")
    p.add_argument("--model_id",      type=str,   default=DEFAULT_MODEL_ID)
    p.add_argument("--batch_size",    type=int,   default=1,
                   help="Per-GPU batch size (keep small: 1024×1024 + OneFormer Swin-L)")
    p.add_argument("--grad_accum",    type=int,   default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--early_stop",    type=int,   default=10,
                   help="Patience for early stopping (epochs without improvement)")
    p.add_argument("--fp16",          action="store_true",
                   help="Mixed-precision training (fp16) via accelerate")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────
def load_coco_items(json_path: Path) -> list[dict]:
    """Load COCO JSON and return list of {image: info, anns: [...]}."""
    with open(json_path) as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}
    ann_map: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        ann_map[ann["image_id"]].append(ann)

    return [
        {"image": img_info, "anns": ann_map[img_id]}
        for img_id, img_info in img_map.items()
    ]


def filter_huge_images(items: list[dict]) -> tuple[list[dict], int]:
    """Drop items whose image exceeds MAX_W × MAX_H."""
    kept, skipped = [], 0
    for item in items:
        info = item["image"]
        w, h = info["width"], info["height"]
        if w <= MAX_W and h <= MAX_H:
            kept.append(item)
        else:
            print(f"  Skipped huge image: {info['file_name']} ({w}×{h})")
            skipped += 1
    return kept, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Albumentations transforms
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(split: str) -> A.Compose:
    """
    Letterbox resize 戦略（exp03-GC と同じ）。

    LongestMaxSize で長辺を IMG_SIZE(=1024) に縮小（アスペクト比保持）し、
    PadIfNeeded で短辺を黒パディングして 1024×1024 に揃える。
    RandomCrop は使わないため、**全部屋が画像に収まる**。

    Mask2Former 版との差分: A.Normalize を削除。
    正規化は OneFormerProcessor が内部で行うため uint8 のまま渡す。

    Train : LongestMaxSize(1024) → PadIfNeeded(1024×1024) → 幾何拡張 → ColorJitter
    Val   : LongestMaxSize(1024) → PadIfNeeded(1024×1024)
    """
    letterbox = [
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=0,   # cv2.BORDER_CONSTANT
            value=0,
            mask_value=0,
        ),
    ]
    bbox_params = A.BboxParams(
        format="coco",
        min_area=MIN_MASK_PX,
        min_visibility=0.1,
        label_fields=["category_ids", "ann_ids"],
    )

    if split == "train":
        return A.Compose([
            *letterbox,
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
            # NOTE: A.Normalize は使わない — OneFormerProcessor が担う
        ], bbox_params=bbox_params)
    else:
        return A.Compose([
            *letterbox,
        ], bbox_params=bbox_params)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class CubiCasaInstanceDataset(Dataset):
    """
    COCO-format instance segmentation dataset for CubiCasa floor plans.

    Mask2Former 版との差分
    ----------------------
    返り値が (pixel_values, mask_labels, class_labels) テンソルから
    以下の 3 キーの dict に変わった:

        image                    : (H, W, 3) uint8 numpy array
                                   クロップ済みだが正規化前（processor に渡す生画像）
        segmentation_map         : (H, W) uint8 numpy array
                                   ピクセル値 = inst_id (1-indexed; 0 = background)
                                   生き残ったインスタンスのみ含む
        instance_id_to_semantic_id : Dict[int, int]
                                   {inst_id → 0-indexed class label}
                                   OneFormerProcessor がこのマッピングを使って
                                   binary mask と class_label を生成する

    Instance mask strategy (memory-efficient)
    ------------------------------------------
    1 枚の uint8 "index mask" にインスタンス ID を書き込み、albumentations で
    画像と同期してクロップ・フリップ。crop 後に inst_id ごとの binary mask を
    「processor が」生成するため、Dataset 側では binary mask を作らない。
    """

    def __init__(self, items: list[dict], img_dir: Path, transform: A.Compose):
        self.items     = items
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item     = self.items[idx]
        img_info = item["image"]
        anns     = item["anns"]
        H, W     = img_info["height"], img_info["width"]

        img = np.array(
            Image.open(self.img_dir / img_info["file_name"]).convert("RGB")
        )  # uint8 (H, W, 3)

        # ── Index mask + bbox list ────────────────────────────────────────────
        index_mask  = np.zeros((H, W), dtype=np.uint8)
        bboxes_coco = []
        cat_ids     = []
        ann_ids     = []

        for ann in anns:
            if ann["iscrowd"] == 1:
                continue
            seg = ann["segmentation"]
            if not isinstance(seg, list) or len(seg) == 0:
                continue

            inst_id = len(ann_ids) + 1
            poly    = seg[0]
            pts     = list(zip(poly[::2], poly[1::2]))
            if len(pts) < 3:
                continue

            pil_m = Image.new("L", (W, H), 0)
            ImageDraw.Draw(pil_m).polygon(pts, fill=1)
            m_arr = np.array(pil_m, dtype=np.uint8)
            if m_arr.sum() == 0:
                continue

            index_mask[m_arr > 0] = inst_id

            x, y, bw, bh = ann["bbox"]
            x  = max(0.0, float(x));  y  = max(0.0, float(y))
            bw = min(float(bw), W - x); bh = min(float(bh), H - y)
            if bw <= 0 or bh <= 0:
                continue

            ann_ids.append(len(ann_ids))
            bboxes_coco.append([x, y, bw, bh])
            cat_ids.append(ann["category_id"])

        # ── Albumentations (no normalize) ────────────────────────────────────
        result = self.transform(
            image=img,
            mask=index_mask,
            bboxes=bboxes_coco,
            category_ids=cat_ids,
            ann_ids=ann_ids,
        )

        cropped_img    = result["image"]           # (1024, 1024, 3) — dtype はほぼ uint8
        cropped_index  = result["mask"]            # (1024, 1024) uint8
        surviving_orig = result["ann_ids"]
        surviving_cats = result["category_ids"]

        # ColorJitter などで dtype が変わる場合に uint8 を保証する
        if cropped_img.dtype != np.uint8:
            cropped_img = cropped_img.clip(0, 255).astype(np.uint8)

        # ── Build segmentation_map + instance_id_to_semantic_id ──────────────
        # OneFormerProcessor が segmentation_map から binary mask を生成するため
        # 生き残ったインスタンスのみを含むクリーンな index mask を作る。
        clean_segmap               = np.zeros_like(cropped_index)
        instance_id_to_semantic_id: dict[int, int] = {}

        for orig_i, c in zip(surviving_orig, surviving_cats):
            inst_id = orig_i + 1
            if int((cropped_index == inst_id).sum()) >= MIN_MASK_PX:
                clean_segmap[cropped_index == inst_id] = inst_id
                instance_id_to_semantic_id[int(inst_id)] = int(CAT_ID_TO_LABEL[c])

        return {
            "image":                    cropped_img,          # (H, W, 3) uint8
            "segmentation_map":         clean_segmap,         # (H, W) uint8
            "instance_id_to_semantic_id": instance_id_to_semantic_id,  # {int: int}
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────
class CollateFn:
    """OneFormerProcessor を内包した DataLoader 用 collate_fn。

    callable クラスとして実装しているのは、multiprocessing DataLoader の
    ワーカープロセスへのシリアライズ（fork/pickle）に対応するため。

    processor が行うこと
    --------------------
    1. 画像のリサイズ・正規化 → pixel_values (B, 3, H, W)
    2. タスク文字列 "instance" のトークン化 → task_inputs (B, seq_len)
    3. クラス名 "room" から各クエリのテキスト記述子生成 → text_inputs (B, Q, seq_len)
    4. segmentation_map + instance_id_to_semantic_id →
           mask_labels  : List[Tensor(N_i, H, W)]
           class_labels : List[Tensor(N_i,)]
    """

    def __init__(self, processor: OneFormerProcessor, task: str = TASK_INPUT):
        self.processor = processor
        self.task      = task

    def _make_text_inputs(self, batch_size: int) -> torch.Tensor:
        """
        processor が segmentation_maps 付き呼び出し時に text_inputs を生成しない
        場合のフォールバック。OneFormerModel は text_inputs=None だと
        text_queries.flatten(1) で AttributeError になるため必須。

        "a photo of a <class>" を num_text 個生成し
        (batch_size, num_text, seq_len) の input_ids テンソルを返す。
        """
        num_text    = self.processor.image_processor.num_text
        class_names = list(self.processor.image_processor.id2label.values())
        texts = [
            f"a photo of a {class_names[i % len(class_names)]}"
            for i in range(num_text)
        ]
        tok = self.processor.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # (num_text, seq_len) → (batch_size, num_text, seq_len)
        return tok["input_ids"].unsqueeze(0).expand(batch_size, -1, -1).clone()

    def __call__(self, batch: list[dict]) -> dict:
        # インスタンスが0件のサンプルはコストマトリクスが空になり NaN を生むため除外する
        valid = [b for b in batch if len(b["instance_id_to_semantic_id"]) > 0]
        if len(valid) == 0:
            # バッチ全体がインスタンス無しの場合は元のバッチをそのまま使う
            # (processor は空マップを処理できるため、matcher 側の patch で吸収)
            valid = batch
        inputs = self.processor(
            images=[b["image"] for b in valid],
            # task_inputs はバッチ内の全画像に同じタスクを指定する
            task_inputs=[self.task] * len(valid),
            segmentation_maps=[b["segmentation_map"] for b in valid],
            # instance_id_to_semantic_id は画像ごとの dict のリスト
            instance_id_to_semantic_id=[b["instance_id_to_semantic_id"] for b in valid],
            return_tensors="pt",
        )
        result = dict(inputs)

        # processor が segmentation_maps 付きで呼ばれる学習パスでは
        # text_inputs を生成しないことがある → 手動で補完する
        if result.get("text_inputs") is None:
            result["text_inputs"] = self._make_text_inputs(len(valid))

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Model + Processor
# ─────────────────────────────────────────────────────────────────────────────
def build_model_and_processor(
    model_id: str,
) -> tuple[OneFormerForUniversalSegmentation, OneFormerProcessor]:
    """
    OneFormerForUniversalSegmentation と OneFormerProcessor をロードし
    ヘッドをカスタムクラス数（num_labels=1）に差し替えて返す。

    processor の id2label を上書きする理由
    ---------------------------------------
    OneFormerProcessor（内部の OneFormerImageProcessor）は id2label を参照して
    text_inputs のクエリテキスト（例: "a room"）を生成する。
    元の COCO 学習済みモデルは 133 クラスを持つため、そのままでは
    "a room" ではなく COCO のクラス名でテキストが生成されてしまう。
    """
    # ── Processor ──────────────────────────────────────────────────────────
    processor = OneFormerProcessor.from_pretrained(model_id)
    # カスタムラベルセットを上書き
    processor.image_processor.id2label  = {int(k): v for k, v in ID2LABEL.items()}
    processor.image_processor.num_labels = NUM_LABELS
    # ignore_index=0 を明示する。
    # デフォルトは None のため、segmentation_map の背景ピクセル(値=0)が
    # instance_id_to_semantic_id のキーとして参照されてしまい KeyError が発生する。
    processor.image_processor.ignore_index = 0

    # metadata をカスタムクラス用に上書きする。
    # processor はデフォルトで COCO の metadata（133クラス）を保持しているため、
    # get_instance_annotations が metadata["thing_ids"] と metadata["class_names"] を
    # 参照したとき、class_id=0 ("room") が COCO クラスとして誤って処理される。
    # その結果コストマトリクスに NaN/Inf が混入し
    # linear_sum_assignment で ValueError が発生する。
    #
    # 必要なフィールド:
    #   "thing_ids"   : list で定義（.index() が呼ばれるため set 不可）
    #   "class_names" : num_class_obj dict のキー生成と texts 生成で使用
    #   "0"           : str(class_id) をキーとするクラス名ルックアップで使用
    #
    # post_process_instance_segmentation でも "thing_ids" の .index() が呼ばれる
    # ため、predict 側でも同じ metadata を設定する必要がある（predict01.py 参照）。
    processor.image_processor.metadata = {
        "thing_ids":   list(ID2LABEL.keys()),          # [0]
        "class_names": list(ID2LABEL.values()),        # ["room"]
        **{str(k): v for k, v in ID2LABEL.items()},   # {"0": "room"}
    }

    # ── Model ───────────────────────────────────────────────────────────────
    config = OneFormerConfig.from_pretrained(
        model_id,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        is_training=True,   # text_mapper は config.is_training=True のときのみ呼ばれる
                            # デフォルト False のままだと text_queries=None になり
                            # contrastive loss で AttributeError が発生する
    )
    # num_text: processor が生成するテキストクエリの本数。
    # デフォルトは None のため "an instance photo" * None で TypeError になる。
    # モデルの num_queries（クエリスロット数）と揃える必要がある。
    processor.image_processor.num_text = config.num_queries

    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,   # head: 133 → 1 クラス
    )
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val loops
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    model,
    loader: DataLoader,
    optimizer: AdamW,
    accelerator: Accelerator,
    is_train: bool,
) -> float:
    """
    OneFormer 学習/検証 1 エポック。

    batch キー（processor が生成）
    --------------------------------
    pixel_values  : (B, 3, H, W) tensor  — accelerate が自動でデバイス転送
    task_inputs   : (B, seq_len)  tensor  — 同上
    text_inputs   : (B, Q, seq_len) tensor — 同上（推論時は None のこともある）
    pixel_mask    : (B, H, W) tensor      — 同上（存在しない場合もある）
    mask_labels   : List[Tensor(N_i, H, W)] — list なので手動デバイス転送が必要
    class_labels  : List[Tensor(N_i,)]      — 同上
    """
    model.train(is_train)
    total_loss = 0.0
    n_steps    = 0

    phase = "Train" if is_train else "Val  "
    pbar = tqdm(loader, desc=phase, leave=False, unit="batch",
                disable=not accelerator.is_main_process)

    for batch in pbar:
        # list-of-tensors は accelerate が自動転送しないため明示的に転送する
        mask_labels  = [m.to(accelerator.device) for m in batch["mask_labels"]]
        class_labels = [c.to(accelerator.device) for c in batch["class_labels"]]

        if is_train:
            with accelerator.accumulate(model):
                outputs = model(
                    pixel_values  = batch["pixel_values"],
                    task_inputs   = batch["task_inputs"],
                    # text_inputs: クエリのテキスト記述子（processor が生成）
                    # None の場合は model がデフォルト動作にフォールバックする
                    text_inputs   = batch.get("text_inputs"),
                    pixel_mask    = batch.get("pixel_mask"),
                    mask_labels   = mask_labels,
                    class_labels  = class_labels,
                )
                loss = outputs.loss   # Hungarian-matching loss（CE + mask BCE+Dice）

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                outputs = model(
                    pixel_values  = batch["pixel_values"],
                    task_inputs   = batch["task_inputs"],
                    text_inputs   = batch.get("text_inputs"),
                    pixel_mask    = batch.get("pixel_mask"),
                    mask_labels   = mask_labels,
                    class_labels  = class_labels,
                )
                loss = outputs.loss

        total_loss += loss.item()
        n_steps    += 1
        if accelerator.is_main_process:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{total_loss / n_steps:.4f}",
            )

    return total_loss / max(n_steps, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Curve saving
# ─────────────────────────────────────────────────────────────────────────────
def save_curves(train_losses: list[float], val_losses: list[float]) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("OneFormer Loss Curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(LOSS_PNG, dpi=150); plt.close()
    print(f"Saved loss curve → {LOSS_PNG}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="fp16" if args.fp16 else "no",
    )

    if accelerator.is_main_process:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Device           : {accelerator.device}")
        print(f"Num processes    : {accelerator.num_processes}")
        print(f"Model            : {args.model_id}")
        print(f"Batch size/GPU   : {args.batch_size}")
        print(f"Grad accum steps : {args.grad_accum}")
        print(f"Effective batch  : {args.batch_size * accelerator.num_processes * args.grad_accum}")
        print(f"Mixed precision  : {args.fp16}")
        print(f"Epochs           : {args.epochs}  (early stop patience={args.early_stop})")
        print()

    # ── Model + Processor ─────────────────────────────────────────────────────
    # build_model_and_processor は全プロセスで実行するが、モデルのデバイス転送は
    # accelerator.prepare() に委ねる（ここでは .to(device) しない）
    if accelerator.is_main_process:
        print("Loading model and processor …")
    model, processor = build_model_and_processor(args.model_id)

    # Processor はステートレスなので main process のみ保存すれば十分
    if accelerator.is_main_process:
        PROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(PROCESSOR_DIR)
        print(f"Processor saved → {PROCESSOR_DIR}")

    # ── Load & filter annotations ─────────────────────────────────────────────
    if accelerator.is_main_process:
        print("Loading annotations …")
    train_items = load_coco_items(ANNO_TRAIN)
    val_items   = load_coco_items(ANNO_VAL)

    if accelerator.is_main_process:
        print("Filtering huge images …")
    train_items, tr_skip = filter_huge_images(train_items)
    val_items,   va_skip = filter_huge_images(val_items)

    if accelerator.is_main_process:
        print(f"Train : skipped {tr_skip:3d}, kept {len(train_items)}")
        print(f"Val   : skipped {va_skip:3d}, kept {len(val_items)}")
        print()

    # ── Datasets & DataLoaders ─────────────────────────────────────────────────
    train_ds = CubiCasaInstanceDataset(train_items, IMG_DIR_TRAIN, get_transforms("train"))
    val_ds   = CubiCasaInstanceDataset(val_items,   IMG_DIR_VAL,   get_transforms("val"))

    if accelerator.is_main_process:
        print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    collate = CollateFn(processor, task=TASK_INPUT)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    # ── Optimizer, Scheduler ──────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # accelerator.prepare() がデバイス転送・DDP ラップ・DistributedSampler をすべて担う
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []

    if accelerator.is_main_process:
        print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>9}  {'Time':>7}")
        print("-" * 42)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss  = run_epoch(model, train_loader, optimizer, accelerator, is_train=True)
        val_loss = run_epoch(model, val_loader,   optimizer, accelerator, is_train=False)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        elapsed = time.time() - t0

        if accelerator.is_main_process:
            print(f"{epoch:>6}  {tr_loss:>10.4f}  {val_loss:>9.4f}  {elapsed:>6.1f}s")

            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), LAST_MODEL)

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save(unwrapped.state_dict(), BEST_MODEL)
                print(f"         *** New best val loss={best_val_loss:.4f}  → {BEST_MODEL}")
            else:
                patience_count += 1
                if patience_count >= args.early_stop:
                    print(f"\nEarly stopping at epoch {epoch} (patience={args.early_stop})")
                    break

    if accelerator.is_main_process:
        save_curves(train_losses, val_losses)
        print(f"\nDone.  Best val loss: {best_val_loss:.4f}")
        print(f"Best model  : {BEST_MODEL}")
        print(f"Last model  : {LAST_MODEL}")
        print(f"Processor   : {PROCESSOR_DIR}")


if __name__ == "__main__":
    main()
