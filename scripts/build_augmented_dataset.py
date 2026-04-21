"""
Build mask2former_augmented_data/ by merging three floor-plan datasets:
  1. cubicasa_mask2former      (already in target format)
  2. Floor Plans               (COCO, filter room category_id=1)
  3. floor-plan-layout-detection (COCO, category_id=1 = Room Segmentation)

Output format (same as cubicasa_mask2former):
  mask2former_augmented_data/
    images/
      train/   *.png / *.jpg
      val/     *.png / *.jpg
    annotations/
      instances_train.json
      instances_val.json

Image filenames are prefixed with dataset source to avoid collisions:
  cubicasa__<original>
  floorplans__<original>
  layout__<original>
"""

import json
import shutil
from pathlib import Path

ROOT = Path("/Users/aokitenju/floor_detect_ai-1")
OUT  = ROOT / "mask2former_augmented_data"

# ── helpers ──────────────────────────────────────────────────────────────────

def new_coco_skeleton():
    return {
        "info": {"description": "Merged floor-plan room segmentation dataset"},
        "licenses": [],
        "categories": [{"id": 1, "name": "room", "supercategory": "floor"}],
        "images": [],
        "annotations": [],
    }


class Merger:
    def __init__(self):
        self.train = new_coco_skeleton()
        self.val   = new_coco_skeleton()
        self._img_id   = 0
        self._ann_id   = 0

    def _next_img_id(self):
        self._img_id += 1
        return self._img_id

    def _next_ann_id(self):
        self._ann_id += 1
        return self._ann_id

    def add_split(self, *, split: str,
                  src_img_dir: Path,
                  coco: dict,
                  prefix: str,
                  room_cat_ids: set):
        """
        split: 'train' or 'val'
        src_img_dir: directory that contains the source images
        coco: loaded JSON dict
        prefix: short string prepended to filenames (e.g. 'cubicasa')
        room_cat_ids: set of category_ids that represent rooms
        """
        target = self.train if split == "train" else self.val
        out_img_dir = OUT / "images" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)

        # Build old_image_id → new info map
        old_to_new = {}
        for img in coco["images"]:
            orig_name = img["file_name"]
            new_name  = f"{prefix}__{orig_name}"
            new_id    = self._next_img_id()
            old_to_new[img["id"]] = (new_id, new_name)

            # copy image
            src = src_img_dir / orig_name
            dst = out_img_dir / new_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            elif not src.exists():
                print(f"  [WARN] missing: {src}")

            target["images"].append({
                "id": new_id,
                "file_name": new_name,
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            })

        # Filter & remap annotations
        for ann in coco["annotations"]:
            if ann["category_id"] not in room_cat_ids:
                continue
            img_id = ann["image_id"]
            if img_id not in old_to_new:
                continue
            new_img_id, _ = old_to_new[img_id]
            new_ann = {
                "id": self._next_ann_id(),
                "image_id": new_img_id,
                "category_id": 1,
                "segmentation": ann.get("segmentation", []),
                "area": ann.get("area", 0),
                "bbox": ann.get("bbox", []),
                "iscrowd": ann.get("iscrowd", 0),
            }
            target["annotations"].append(new_ann)

    def write(self):
        ann_dir = OUT / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        for split, data in [("train", self.train), ("val", self.val)]:
            out_path = ann_dir / f"instances_{split}.json"
            with open(out_path, "w") as f:
                json.dump(data, f)
            print(f"Wrote {out_path}  "
                  f"({len(data['images'])} images, {len(data['annotations'])} annotations)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(exist_ok=True)
    merger = Merger()

    # 1. cubicasa_mask2former ──────────────────────────────────────────────
    print("=== cubicasa_mask2former ===")
    cub_root = ROOT / "cubicasa_mask2former"
    for split, json_name in [("train", "instances_train.json"),
                              ("val",   "instances_val.json")]:
        with open(cub_root / "annotations" / json_name) as f:
            coco = json.load(f)
        merger.add_split(
            split=split,
            src_img_dir=cub_root / "images" / split,
            coco=coco,
            prefix="cubicasa",
            room_cat_ids={1},
        )
        print(f"  {split}: {len(coco['images'])} images")

    # 2. Floor Plans ──────────────────────────────────────────────────────
    print("=== Floor Plans ===")
    fp_root = ROOT / "Floor Plans"
    # train → train,  valid+test → val
    for src_split, dst_split in [("train", "train"), ("valid", "val"), ("test", "val")]:
        with open(fp_root / src_split / "_annotations.coco.json") as f:
            coco = json.load(f)
        merger.add_split(
            split=dst_split,
            src_img_dir=fp_root / src_split,
            coco=coco,
            prefix=f"floorplans_{src_split}",
            room_cat_ids={1},          # category_id 1 = 'room'
        )
        print(f"  {src_split}→{dst_split}: {len(coco['images'])} images")

    # 3. floor-plan-layout-detection ──────────────────────────────────────
    print("=== floor-plan-layout-detection ===")
    ld_root = ROOT / "floor-plan-layout-detection"
    # train → train,  valid+test → val
    for src_split, dst_split in [("train", "train"), ("valid", "val"), ("test", "val")]:
        with open(ld_root / src_split / "_annotations.coco.json") as f:
            coco = json.load(f)
        merger.add_split(
            split=dst_split,
            src_img_dir=ld_root / src_split,
            coco=coco,
            prefix=f"layout_{src_split}",
            room_cat_ids={1},          # category_id 1 = 'Room Segmentation'
        )
        print(f"  {src_split}→{dst_split}: {len(coco['images'])} images")

    # Write JSON
    merger.write()
    print("\nDone.")


if __name__ == "__main__":
    main()
