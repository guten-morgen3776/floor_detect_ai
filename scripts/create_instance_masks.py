"""
Create per-image instance masks from COCO-format annotations.

Output: cubicasa_instance/{train,val}/{image_stem}.npy
  Each .npy file is a (M, H, W) bool array where M = number of rooms in the image.
"""

import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


SRC_ROOT = Path("/Users/aokitenju/floor_detect_ai-1/cubicasa_mask2former")
OUT_ROOT = Path("/Users/aokitenju/floor_detect_ai-1/cubicasa_instance")

SPLITS = ["train", "val"]


def build_instance_masks(ann_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(ann_path))
    img_ids = coco.getImgIds()
    total = len(img_ids)

    print(f"Processing {total} images -> {out_dir}")

    for i, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        H, W = img_info["height"], img_info["width"]
        file_name = img_info["file_name"]          # e.g. "colorful_10052_F1_scaled.png"
        stem = Path(file_name).stem                # e.g. "colorful_10052_F1_scaled"

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            # No rooms: save empty (0, H, W) array so the file still exists
            masks = np.zeros((0, H, W), dtype=bool)
        else:
            masks = np.zeros((len(anns), H, W), dtype=bool)
            for j, ann in enumerate(anns):
                # segmentation can be polygon list or RLE
                seg = ann["segmentation"]
                if isinstance(seg, list):
                    # polygon format: convert to RLE first
                    rle = mask_utils.frPyObjects(seg, H, W)
                    rle = mask_utils.merge(rle)
                else:
                    # already RLE
                    rle = seg
                masks[j] = mask_utils.decode(rle).astype(bool)

        out_path = out_dir / f"{stem}.npz"
        np.savez_compressed(str(out_path), masks=masks)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] saved {stem}.npy  shape={masks.shape}")

    print(f"Done. {total} files written to {out_dir}\n")


def main():
    for split in SPLITS:
        ann_path = SRC_ROOT / "annotations" / f"instances_{split}.json"
        out_dir  = OUT_ROOT / split
        build_instance_masks(ann_path, out_dir)

    print("All splits complete.")


if __name__ == "__main__":
    main()
