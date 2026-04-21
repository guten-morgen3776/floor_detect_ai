"""
mask2former_augmented_data の下位50%を除外したデータセットを生成する

- スコアは全データ(train+val)の分布で正規化
- 閾値は合計の50パーセンタイル
- train/val それぞれで上位50%を残す
- 画像ファイルは symlink で参照
- 出力先: /Users/aokitenju/floor_detect_ai-1/mask2former_augmented_data_50/
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

SRC_ROOT = Path("/Users/aokitenju/floor_detect_ai-1/mask2former_augmented_data")
DST_ROOT = Path("/Users/aokitenju/floor_detect_ai-1/mask2former_augmented_data_50")
SPLITS   = ["train", "val"]

# ─────────────────────────────────────────────
# 指標計算 (annotation非依存の5指標)
# ─────────────────────────────────────────────

def bbox_centroid(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2

def are_adjacent(a, b, tol=5.0):
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1 + bw, by1 + bh
    return max(ax1, bx1) - min(ax2, bx2) <= tol and max(ay1, by1) - min(ay2, by2) <= tol

def compute_metrics(coco_data):
    img_map = {img["id"]: img for img in coco_data["images"]}
    ann_by_img = defaultdict(list)
    for ann in coco_data["annotations"]:
        if not ann.get("iscrowd", False):
            ann_by_img[ann["image_id"]].append(ann)

    records = []
    for img_id, anns in ann_by_img.items():
        img = img_map[img_id]
        W, H = img["width"], img["height"]
        n = len(anns)
        areas     = [a["area"] for a in anns]
        bboxes    = [a["bbox"] for a in anns]
        centroids = [bbox_centroid(a["bbox"]) for a in anns]

        space_util = sum(areas) / (W * H)
        cv         = float(np.std(areas) / np.mean(areas)) if n > 1 else 0.0
        adj        = sum(1 for i in range(n) for j in range(i+1, n)
                         if are_adjacent(bboxes[i], bboxes[j]))
        spread     = (float(np.std([c[0]/W for c in centroids])**2
                          + np.std([c[1]/H for c in centroids])**2)
                      if n > 1 else 0.0)

        records.append({
            "image_id":       img_id,
            "n_rooms":        n,
            "space_util":     space_util,
            "area_cv":        cv,
            "adj_count":      adj,
            "spatial_spread": spread,
        })
    return records

def compute_scores(records, ref):
    def _norm(arr, ref_arr):
        mn, mx = ref_arr.min(), ref_arr.max()
        return np.zeros_like(arr) if mx == mn else np.clip((arr - mn) / (mx - mn), 0, 1)
    def _a(k, r): return np.array([x[k] for x in r])
    return (0.35 * _norm(_a("n_rooms",        records), _a("n_rooms",        ref))
          + 0.25 * _norm(_a("adj_count",      records), _a("adj_count",      ref))
          + 0.25 * _norm(_a("area_cv",        records), _a("area_cv",        ref))
          + 0.10 * _norm(_a("space_util",     records), _a("space_util",     ref))
          + 0.05 * _norm(_a("spatial_spread", records), _a("spatial_spread", ref)))

# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    print("=== mask2former_augmented_data_50 生成 ===\n")

    # ── 全splitのアノテーション読み込み ──
    split_data = {}
    all_recs   = []
    for split in SPLITS:
        path = SRC_ROOT / "annotations" / f"instances_{split}.json"
        with open(path) as f:
            split_data[split] = json.load(f)
        recs = compute_metrics(split_data[split])
        all_recs.extend(recs)
        print(f"  [{split}] 図面数: {len(recs):,}")

    # ── 全データ基準でスコア計算 → 50パーセンタイル閾値 ──
    all_scores = compute_scores(all_recs, all_recs)
    threshold  = float(np.percentile(all_scores, 50))
    print(f"\n  全体 50パーセンタイル スコア閾値: {threshold:.4f}")

    # image_id → score マッピング
    score_map = {r["image_id"]: float(s) for r, s in zip(all_recs, all_scores)}

    # ── 出力ディレクトリ作成 ──
    (DST_ROOT / "annotations").mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_orig = 0

    for split in SPLITS:
        coco = split_data[split]

        # 閾値より大きい image_id を抽出 (上位50%)
        kept_ids    = {img["id"] for img in coco["images"]
                       if score_map.get(img["id"], 0.0) > threshold}
        kept_images = [img for img in coco["images"] if img["id"] in kept_ids]
        kept_anns   = [ann for ann in coco["annotations"] if ann["image_id"] in kept_ids]

        n_orig = len(coco["images"])
        n_kept = len(kept_images)
        total_orig += n_orig
        total_kept += n_kept

        print(f"\n  [{split}]")
        print(f"    元の図面数:           {n_orig:,}")
        print(f"    残す図面数:           {n_kept:,}  ({n_kept/n_orig*100:.1f}%)")
        print(f"    残すアノテーション数: {len(kept_anns):,}")

        new_coco = {
            "info":        coco.get("info", {}),
            "licenses":    coco.get("licenses", []),
            "categories":  coco["categories"],
            "images":      kept_images,
            "annotations": kept_anns,
        }
        out_path = DST_ROOT / "annotations" / f"instances_{split}.json"
        with open(out_path, "w") as f:
            json.dump(new_coco, f)
        print(f"    → {out_path}")

    print(f"\n  合計: {total_orig:,} 件 → {total_kept:,} 件 ({total_kept/total_orig*100:.1f}% 残存)")

    # ── images は symlink ──
    img_link = DST_ROOT / "images"
    if not img_link.exists():
        img_link.symlink_to((SRC_ROOT / "images").resolve())
        print(f"\n  images/ symlink 作成")
        print(f"    {img_link} -> {(SRC_ROOT / 'images').resolve()}")
    else:
        print(f"\n  images/ symlink は既存")

    print(f"\n出力先: {DST_ROOT}")
    print("=== 完了 ===")

if __name__ == "__main__":
    main()
