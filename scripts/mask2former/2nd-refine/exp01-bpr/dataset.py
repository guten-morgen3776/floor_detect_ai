"""
dataset.py  ―  BoundaryPatchDataset

学習・検証用パッチのオンデマンド生成。
ディスクには予測マスク NPZ だけを保存し、パッチ化はメモリ上で実施する。

インデックス構築 (_build_index) は初回のみ実行し、
pkl ファイルにキャッシュして2回目以降は高速ロードする。
"""

import pickle
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import albumentations as A
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import config
from boundary_patch import (
    extract_boundary_pixels,
    generate_patch_candidates,
    nms_patches,
    compute_iou_batch,
)


# ─────────────────────────────────────────────────────────────────────────────
# データ拡張
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(is_train: bool) -> A.Compose:
    """
    学習 / 検証用の albumentations Transform を返す。

    RGB のみ PhotoMetric 拡張を適用し、mask / gt には適用しない。
    幾何学変換は HorizontalFlip のみ (建築図面の向き情報を保持するため)。
    """
    common = [A.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])]
    if is_train:
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
        ] + common
    else:
        transforms = common

    return A.Compose(transforms,
                     additional_targets={"pred_mask": "mask", "gt_mask": "mask"})


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryPatchDataset(Dataset):
    """
    1サンプル = 1境界パッチ (4ch 入力 + GT ラベル)

    Parameters
    ----------
    img_dir      : RGB 画像ディレクトリ (colorful_XXXXX_F1_scaled.png)
    pred_dir     : step1 出力ディレクトリ (colorful_XXXXX_F1_scaled_pred.npz)
    gt_dir       : GT インスタンスマスクディレクトリ (colorful_XXXXX_F1_scaled.npz)
    split        : "train" or "val" (キャッシュファイル名に使用)
    use_cache    : True ならインデックスを pkl に保存 / ロードする
    """

    def __init__(self,
                 img_dir: Path,
                 pred_dir: Path,
                 gt_dir: Path,
                 split: str = "train",
                 use_cache: bool = True):
        self.img_dir   = Path(img_dir)
        self.pred_dir  = Path(pred_dir)
        self.gt_dir    = Path(gt_dir)
        self.split     = split
        self.transform = get_transforms(is_train=(split == "train"))

        # per-worker 画像キャッシュ (各 DataLoader worker がコピーを持つ)
        self._img_cache  = {}
        self._pred_cache = {}
        self._gt_cache   = {}

        # インデックス構築 / ロード
        cache_path = config.OUTPUT_DIR / f"patch_index_{split}.pkl"
        if use_cache and cache_path.exists():
            print(f"[Dataset] Loading patch index from {cache_path}")
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
        else:
            print(f"[Dataset] Building patch index for {split} ...")
            self.samples = self._build_index()
            if use_cache:
                config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(self.samples, f)

        n_imgs = len(set(s[0] for s in self.samples))
        print(f"[Dataset] {split}: {len(self.samples):,} patches from {n_imgs:,} images")

    # ─────────────────────────────────────────────────────────────────────────
    # インデックス構築
    # ─────────────────────────────────────────────────────────────────────────

    def _build_index(self) -> list[tuple]:
        """
        各画像について pred マスクと GT マスクをマッチングし、
        境界パッチの (stem, pred_idx, box, gt_idx) リストを返す。

        box = {'x1', 'y1', 'x2', 'y2', 'score'}
        """
        # pred ファイルが存在するステムのみ対象
        stems = sorted(
            p.stem.replace("_pred", "")
            for p in self.pred_dir.glob("*_pred.npz")
        )
        if not stems:
            raise FileNotFoundError(
                f"pred マスクが見つかりません: {self.pred_dir}\n"
                "step1_generate_pred_masks.py を先に実行してください。"
            )

        samples = []

        for stem in tqdm(stems, desc=f"  indexing {self.split}"):
            gt_path   = self.gt_dir  / f"{stem}.npz"
            pred_path = self.pred_dir / f"{stem}_pred.npz"

            if not gt_path.exists():
                continue

            pred_data  = np.load(pred_path)
            pred_masks = pred_data["masks"]   # (N_pred, H, W) bool
            gt_data    = np.load(gt_path)
            gt_masks   = gt_data["masks"]     # (N_gt,   H, W) bool

            if len(pred_masks) == 0 or len(gt_masks) == 0:
                continue

            for pred_idx, pred_mask in enumerate(pred_masks):
                # GT とのマッチング
                ious         = compute_iou_batch(pred_mask, gt_masks)
                best_gt_idx  = int(ious.argmax())
                best_iou     = float(ious[best_gt_idx])

                if best_iou < config.IOT_FILTER_THR:
                    continue  # 低品質な予測は学習対象外

                # 境界パッチ候補生成
                boundary   = extract_boundary_pixels(pred_mask, config.DILATION_RADIUS)
                candidates = generate_patch_candidates(boundary,
                                                       config.PATCH_SIZE,
                                                       config.PATCH_STRIDE)
                kept_boxes = nms_patches(candidates, config.NMS_IOU_TRAIN)

                for box in kept_boxes:
                    samples.append((stem, pred_idx, box, best_gt_idx))

        return samples

    # ─────────────────────────────────────────────────────────────────────────
    # 画像 / マスク のキャッシュ付きロード
    # ─────────────────────────────────────────────────────────────────────────

    def _get_image(self, stem: str) -> np.ndarray:
        if stem not in self._img_cache:
            if len(self._img_cache) >= config.IMG_CACHE_SIZE:
                self._img_cache.pop(next(iter(self._img_cache)))
            path = self.img_dir / f"{stem}.png"
            self._img_cache[stem] = np.array(Image.open(path).convert("RGB"))
        return self._img_cache[stem]

    def _get_pred_masks(self, stem: str) -> np.ndarray:
        if stem not in self._pred_cache:
            if len(self._pred_cache) >= config.IMG_CACHE_SIZE:
                self._pred_cache.pop(next(iter(self._pred_cache)))
            path = self.pred_dir / f"{stem}_pred.npz"
            self._pred_cache[stem] = np.load(path)["masks"]
        return self._pred_cache[stem]

    def _get_gt_masks(self, stem: str) -> np.ndarray:
        if stem not in self._gt_cache:
            if len(self._gt_cache) >= config.IMG_CACHE_SIZE:
                self._gt_cache.pop(next(iter(self._gt_cache)))
            path = self.gt_dir / f"{stem}.npz"
            self._gt_cache[stem] = np.load(path)["masks"]
        return self._gt_cache[stem]

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset interface
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem, pred_idx, box, gt_idx = self.samples[idx]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        # ── データロード ──────────────────────────────────────────────────────
        image      = self._get_image(stem)            # (H, W, 3) uint8
        pred_masks = self._get_pred_masks(stem)       # (N_pred, H, W) bool
        gt_masks   = self._get_gt_masks(stem)         # (N_gt,   H, W) bool

        pred_mask = pred_masks[pred_idx]              # (H, W) bool
        gt_mask   = gt_masks[gt_idx]                  # (H, W) bool

        # ── パッチ切り出し & リサイズ ─────────────────────────────────────────
        img_patch  = image[y1:y2, x1:x2]                              # (ph, pw, 3)
        mask_patch = pred_mask[y1:y2, x1:x2].astype(np.float32)       # (ph, pw)
        gt_patch   = gt_mask[y1:y2, x1:x2].astype(np.uint8)           # (ph, pw) 0or1

        sz = config.INPUT_SIZE
        img_patch  = cv2.resize(img_patch,  (sz, sz), interpolation=cv2.INTER_LINEAR)
        mask_patch = cv2.resize(mask_patch, (sz, sz), interpolation=cv2.INTER_NEAREST)
        gt_patch   = cv2.resize(gt_patch,   (sz, sz), interpolation=cv2.INTER_NEAREST)

        # ── データ拡張 (幾何変換 + RGB 色変換 + Normalize) ────────────────────
        aug = self.transform(
            image=img_patch,
            pred_mask=mask_patch,
            gt_mask=gt_patch,
        )
        img_aug  = aug["image"]       # (sz, sz, 3) float32, ImageNet normalized
        mask_aug = aug["pred_mask"]   # (sz, sz) float32, [0, 1]
        gt_aug   = aug["gt_mask"]     # (sz, sz) uint8,   0 or 1

        # ── mask 正規化: [0,1] → [-1,1] ──────────────────────────────────────
        mask_norm = (mask_aug.astype(np.float32) - 0.5) / 0.5  # (sz, sz)

        # ── 4ch テンソル組み立て ──────────────────────────────────────────────
        img_t  = torch.from_numpy(img_aug.transpose(2, 0, 1).copy())    # (3, sz, sz)
        mask_t = torch.from_numpy(mask_norm[np.newaxis].copy())         # (1, sz, sz)
        input_4ch = torch.cat([img_t, mask_t], dim=0)                   # (4, sz, sz)

        label = torch.from_numpy(gt_aug.astype(np.int64))               # (sz, sz)

        return input_4ch, label
