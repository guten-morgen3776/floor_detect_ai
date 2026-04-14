"""
predict01.py  -  OneFormer instance segmentation inference

Strategy
--------
全画像を OneFormerProcessor でリサイズ（内部で shortest_edge=800 に調整）して
1 回推論し、processor.post_process_instance_segmentation() でマスクを
元解像度に復元してオーバーレイ画像を保存する。

Mask2Former 版との主な変更点
-----------------------------
* 前処理 : albumentations + 手動正規化 → OneFormerProcessor に一本化
* モデル入力 : pixel_values のみ → pixel_values + task_inputs (+text_inputs)
* 後処理 : 手動 sigmoid + interpolate → processor.post_process_instance_segmentation()
    - 戻り値: {"segmentation": (H,W) LongTensor, "segments_info": [...]}
    - segments_info の各要素: {"id": int, "label_id": int, "score": float}

Outputs (per image)
-------------------
  <stem>_mask.png    : インスタンス ID マスク (uint8, 0=background, 1..N=instance)
  <stem>_overlay.png : 元画像 + 半透明インスタンスカラーオーバーレイ

Usage:
    python predict01.py <input_dir> [--weights <path>] [--output <dir>]
                        [--model_id <str>] [--threshold <float>]
                        [--processor_dir <path>]
"""

from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

import os
import argparse
import numpy as np
from pathlib import Path

import torch
from transformers import (
    OneFormerForUniversalSegmentation,
    OneFormerConfig,
    OneFormerProcessor,
)

# ─────────────────────────────────────────────────────────────────────────────
# Kaggle 環境判別
# ─────────────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "shi-labs/oneformer_coco_swin_large"

NUM_LABELS = 1
ID2LABEL   = {0: "room"}
LABEL2ID   = {"room": 0}

THRESHOLD  = 0.5    # post_process_instance_segmentation に渡す confidence 閾値

# OneFormer タスクトークン（インスタンスセグメンテーションに固定）
TASK_INPUT = "instance"

if IS_KAGGLE:
    DEFAULT_WEIGHTS       = Path("/kaggle/working/exp001/best_model.pth")
    DEFAULT_OUTPUT_DIR    = Path("/kaggle/working/predict01")
    DEFAULT_PROCESSOR_DIR = Path("/kaggle/working/exp001/processor")
else:
    DEFAULT_WEIGHTS       = Path("/content/drive/MyDrive/oneformer_exp001/best_model.pth")
    DEFAULT_OUTPUT_DIR    = Path("/content/drive/MyDrive/oneformer_predict01")
    DEFAULT_PROCESSOR_DIR = Path("/content/drive/MyDrive/oneformer_exp001/processor")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# インスタンス描画用カラーパレット (RGB)
PALETTE = [
    (220,  60,  60), ( 60, 200,  60), ( 60,  60, 220),
    (220, 200,  60), (200,  60, 220), ( 60, 220, 220),
    (220, 130,  60), (130, 220,  60), ( 60, 130, 220),
    (220,  60, 130), (130,  60, 220), ( 60, 220, 130),
    (160,  60,  60), ( 60, 160,  60), ( 60,  60, 160),
    (160, 160,  60), (160,  60, 160), ( 60, 160, 160),
    (200, 100,  60), (100, 200,  60), ( 60, 100, 200),
    (200,  60, 100), (100,  60, 200), ( 60, 200, 100),
]


# ─────────────────────────────────────────────────────────────────────────────
# Model + Processor
# ─────────────────────────────────────────────────────────────────────────────
def build_model_and_processor(
    model_id: str,
    weights_path: Path,
    processor_dir: Path | None,
) -> tuple[OneFormerForUniversalSegmentation, OneFormerProcessor]:
    """
    学習済み OneFormer モデルと OneFormerProcessor をロードして返す。

    Processor のロード順序
    -----------------------
    1. processor_dir が存在する場合 → そこからロード（学習時の設定を引き継ぐ）
    2. 存在しない場合 → model_id からロードし id2label を上書き

    Model のロード
    -------------
    学習時と同じ config (num_labels=1) でモデルを構築し、
    保存済みの state_dict をロードする。
    """
    # ── Processor ──────────────────────────────────────────────────────────
    if processor_dir is not None and processor_dir.exists():
        processor = OneFormerProcessor.from_pretrained(str(processor_dir))
        print(f"Processor loaded from: {processor_dir}")
    else:
        processor = OneFormerProcessor.from_pretrained(model_id)
        processor.image_processor.id2label  = {int(k): v for k, v in ID2LABEL.items()}
        processor.image_processor.num_labels = NUM_LABELS
        processor.image_processor.ignore_index = 0
        processor.image_processor.metadata = {
            "thing_ids":   list(ID2LABEL.keys()),
            "class_names": list(ID2LABEL.values()),
            **{str(k): v for k, v in ID2LABEL.items()},
        }
        print(f"Processor loaded from: {model_id} (id2label overridden)")

    # ── Model ───────────────────────────────────────────────────────────────
    config = OneFormerConfig.from_pretrained(
        model_id,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    # processor_dir からロードした場合でも num_text が None のままのことがある。
    # config.num_queries（=150）と揃えることで TypeError を防ぐ。
    if processor.image_processor.num_text is None:
        processor.image_processor.num_text = config.num_queries

    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict_instances(
    processor: OneFormerProcessor,
    model: OneFormerForUniversalSegmentation,
    image_rgb: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[float], list[int]]:
    """
    元画像に対してインスタンスセグメンテーションを実行する。

    Flow
    ----
    1. OneFormerProcessor で前処理（リサイズ・正規化・task_inputs トークン化）
    2. model(**inputs) でフォワード（task_inputs="instance" 条件付き）
    3. processor.post_process_instance_segmentation() でマスクを元解像度に復元

    OneFormer 特有の引数
    --------------------
    * task_inputs=["instance"] : プロセッサとモデル双方に渡すタスクトークン
    * text_inputs              : プロセッサが推論時に生成する場合は model に渡す
                                 生成されない場合（None）は model がデフォルト動作

    post_process_instance_segmentation の主な引数
    ----------------------------------------------
    * threshold               : クエリの confidence 閾値（default 0.5）
    * mask_threshold          : binary mask の閾値（default 0.5）
    * overlap_mask_area_threshold : 重複マスクのフィルタ率（default 0.8）
    * target_sizes            : 出力マスクを元解像度にリサイズするための (H, W) リスト

    戻り値
    ------
    masks  : (N, orig_H, orig_W) bool ndarray
    scores : [N] float list
    labels : [N] int list  (0-indexed class label)
    """
    orig_H, orig_W = image_rgb.shape[:2]

    # ── 前処理 ───────────────────────────────────────────────────────────────
    # processor がリサイズ・正規化・タスクトークン生成を一括処理する
    inputs = processor(
        images=[image_rgb],          # (H, W, 3) uint8 numpy array
        task_inputs=[TASK_INPUT],    # タスク条件 = "instance" に固定
        return_tensors="pt",
    )
    # すべてのテンソルをデバイスに転送（list などの非テンソル値は除外）
    inputs_on_device = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # ── Forward ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(**inputs_on_device)

    # ── 後処理 ───────────────────────────────────────────────────────────────
    # post_process_instance_segmentation は内部で threshold を適用したうえで
    # マスクを target_sizes に指定した元解像度にリサイズして返す。
    #
    # results[0] の構造:
    #   "segmentation"   : (orig_H, orig_W) LongTensor
    #                      ピクセル値 = segments_info の "id"（0 = no instance）
    #   "segments_info"  : list of dict
    #       {"id": int, "label_id": int, "score": float}
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,               # query confidence 閾値
        mask_threshold=0.5,                # binary mask 閾値
        overlap_mask_area_threshold=0.8,   # 重複フィルタ率
        target_sizes=[(orig_H, orig_W)],   # 元解像度に復元
    )[0]  # バッチの 0 番目（今回は 1 枚ずつ処理）

    segmentation = results["segmentation"].cpu().numpy()   # (H, W) int64
    segments     = results["segments_info"]                # list of dict

    masks:  list[np.ndarray] = []
    scores: list[float]      = []
    labels: list[int]        = []

    for seg in segments:
        binary_mask = (segmentation == seg["id"])
        if binary_mask.sum() == 0:
            continue
        masks.append(binary_mask)
        scores.append(float(seg["score"]))
        labels.append(int(seg["label_id"]))

    if masks:
        return np.stack(masks), scores, labels
    else:
        return np.zeros((0, orig_H, orig_W), dtype=bool), [], []


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers  (Mask2Former 版と同一)
# ─────────────────────────────────────────────────────────────────────────────
def make_instance_id_mask(masks: np.ndarray, orig_H: int, orig_W: int) -> np.ndarray:
    """masks (N, H, W) bool → インスタンス ID マスク (H, W) uint8"""
    id_mask = np.zeros((orig_H, orig_W), dtype=np.uint8)
    for i, m in enumerate(masks):
        id_mask[m] = i + 1
    return id_mask


def make_overlay(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    scores: list[float],
    alpha: float = 0.45,
) -> np.ndarray:
    """元画像にインスタンスマスクを半透明で重ねたオーバーレイ画像を返す。"""
    overlay = image_rgb.astype(np.float32).copy()

    for i, m in enumerate(masks):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        overlay[m, 0] = overlay[m, 0] * (1 - alpha) + color[0] * alpha
        overlay[m, 1] = overlay[m, 1] * (1 - alpha) + color[1] * alpha
        overlay[m, 2] = overlay[m, 2] * (1 - alpha) + color[2] * alpha

    pil  = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for i, (m, score) in enumerate(zip(masks, scores)):
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        label_text = f"#{i + 1} {score:.2f}"
        draw.text((cx - 1, cy - 1), label_text, fill=(0, 0, 0),       font=font)
        draw.text((cx,     cy    ), label_text, fill=(255, 255, 255),  font=font)

    return np.array(pil)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
def save_results(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    scores: list[float],
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_H, orig_W = image_rgb.shape[:2]

    id_mask   = make_instance_id_mask(masks, orig_H, orig_W)
    mask_path = output_dir / f"{stem}_mask.png"
    Image.fromarray(id_mask).save(mask_path)
    print(f"  Saved mask    → {mask_path}")

    overlay      = make_overlay(image_rgb, masks, scores) if len(masks) > 0 else image_rgb.copy()
    overlay_path = output_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    print(f"  Saved overlay → {overlay_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "OneFormer instance segmentation inference  "
            "(processor resize → infer → restore to original resolution)"
        )
    )
    p.add_argument("input_dir",        type=str,
                   help="Directory containing input floor plan images")
    p.add_argument("--weights",        type=str, default=str(DEFAULT_WEIGHTS),
                   help=f"Path to best_model.pth  (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--output",         type=str, default=str(DEFAULT_OUTPUT_DIR),
                   help=f"Output directory  (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--model_id",       type=str, default=DEFAULT_MODEL_ID,
                   help=f"HuggingFace model ID used during training  (default: {DEFAULT_MODEL_ID})")
    p.add_argument("--threshold",      type=float, default=THRESHOLD,
                   help=f"Instance confidence threshold  (default: {THRESHOLD})")
    p.add_argument("--processor_dir",  type=str, default=str(DEFAULT_PROCESSOR_DIR),
                   help=f"Saved processor directory (default: {DEFAULT_PROCESSOR_DIR})")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    input_dir     = Path(args.input_dir)
    output_dir    = Path(args.output)
    weights       = Path(args.weights)
    processor_dir = Path(args.processor_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    print(f"Device        : {DEVICE}")
    print(f"Model ID      : {args.model_id}")
    print(f"Weights       : {weights}")
    print(f"Processor dir : {processor_dir}")
    print(f"Input dir     : {input_dir}  ({len(image_paths)} images)")
    print(f"Output dir    : {output_dir}")
    print(f"Task input    : {TASK_INPUT}  (fixed for instance segmentation)")
    print(f"Threshold     : {args.threshold}")

    print("\nLoading model …")
    model, processor = build_model_and_processor(
        args.model_id,
        weights,
        processor_dir if processor_dir.exists() else None,
    )
    print("Model loaded.\n")

    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name}")
        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        H, W = image_rgb.shape[:2]
        print(f"  Image size : {W}×{H}")

        masks, scores, labels = predict_instances(
            processor, model, image_rgb, threshold=args.threshold
        )
        print(f"  Instances detected : {len(masks)}")
        for j, (s, lb) in enumerate(zip(scores, labels)):
            print(f"    #{j + 1:3d}  label={ID2LABEL.get(int(lb), lb)}  score={s:.3f}")

        save_results(image_rgb, masks, scores, output_dir, stem=img_path.stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
