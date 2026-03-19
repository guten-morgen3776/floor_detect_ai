"""
Cubicasa5k の model.svg から「壁 (Wall)」のポリゴンだけを抽出し、
対応する F*_original.(jpg|png) と同じキャンバス上に壁マスクを生成する。

要点（extract_room_polygons.py と同じ方針）:
- SVG は BeautifulSoup で解析する（XMLパーサーは使わない）
# viewBox を読み取り、画像サイズとの比率からスケール係数を計算して座標に適用する
# transform / matrix 等のネストした変換は対象外
- 対象は id または class に "Wall" を含む <g> の「直下 (recursive=False)」の <polygon> のみ
"""

import glob
import warnings
from pathlib import Path
from typing import List, Tuple

from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
from PIL import Image, ImageDraw
from tqdm import tqdm

# 巨大画像（建築図面）のDOS警告を無効化
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def parse_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    if not points_str or not points_str.strip():
        return []
    result: List[Tuple[float, float]] = []
    tokens = points_str.strip().split()
    for token in tokens:
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                result.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return result


def _class_to_str(class_attr) -> str:
    if class_attr is None:
        return ""
    if isinstance(class_attr, list):
        return " ".join(str(x) for x in class_attr)
    return str(class_attr)

def create_wall_mask(svg_path: Path, reference_img_path: Path, output_mask_path: Path) -> bool:
    with Image.open(reference_img_path) as img:
        img_w, img_h = img.size

    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    text = svg_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")

    # ── viewBox からスケール係数を計算 ──
    svg_tag = soup.find("svg")
    scale_x, scale_y = 1.0, 1.0
    if svg_tag:
        vb = (svg_tag.get("viewBox") or "").split()
        if len(vb) == 4:
            vb_w, vb_h = float(vb[2]), float(vb[3])
            if vb_w > 0 and vb_h > 0:
                scale_x = img_w / vb_w
                scale_y = img_h / vb_h

    polygons_drawn = 0

    for g in soup.find_all("g"):
        g_id = str(g.get("id", ""))
        g_class_str = _class_to_str(g.get("class", ""))

        if "Wall" not in g_id and "Wall" not in g_class_str:
            continue

        for poly in g.find_all("polygon", recursive=False):
            points_attr = poly.get("points")
            coords = parse_polygon_points(points_attr or "")
            if len(coords) >= 3:
                # スケール適用
                scaled = [(x * scale_x, y * scale_y) for x, y in coords]
                draw.polygon(scaled, fill=255)
                polygons_drawn += 1

    if polygons_drawn == 0:
        print(f"Warning: No walls drawn for {reference_img_path.name}")
        return False

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_mask_path, format="PNG")
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_dir = Path(args.input_root)
    output_dir = Path(args.output_root)
    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    folders = glob.glob(f"{base_dir}/*/*/")
    success_count = 0

    for folder_path in tqdm(folders, desc="Processing"):
        folder = Path(folder_path)
        svg_path = folder / "model.svg"
        if not svg_path.exists():
            continue

        # 対象画像は original のみ（scaled は座標が一致しないため除外）
        img_paths = (
            list(folder.glob("F*_original.jpg"))
            + list(folder.glob("F*_original.jpeg"))
            + list(folder.glob("F*_original.png"))
        )
        if not img_paths:
            continue

        for img_path in img_paths:
            file_id = f"{folder.parent.name}_{folder.name}_{img_path.stem}"
            mask_out = out_masks / f"{file_id}.png"
            img_out = out_images / f"{file_id}.png"

            if create_wall_mask(svg_path, img_path, mask_out):
                Image.open(img_path).convert("RGB").save(img_out)
                success_count += 1

    print(f"Done. Successfully created {success_count} pairs.")


if __name__ == "__main__":
    main()

