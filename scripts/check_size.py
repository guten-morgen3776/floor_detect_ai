"""
1サンプルに対して以下を実行する:
- SVG の viewBox / width / height を表示
- 画像のピクセルサイズを表示
- svg 直下の g の transform を表示
- viewBox と画像サイズから scale を算出し、polygon をスケールしてマスク描画
"""
from pathlib import Path
from typing import List, Tuple

import warnings
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# 対象: Cubicasa5k の 1 サンプル
SAMPLE_DIR = Path("/Users/aokitenju/Downloads/archive (1)/cubicasa5k/cubicasa5k/colorful/34")
SVG_PATH = SAMPLE_DIR / "model.svg"
IMG_PATH = SAMPLE_DIR / "F1_original.png"


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


def main() -> None:
    text = SVG_PATH.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    svg_tag = soup.find("svg")

    # SVG の座標系サイズ
    print("=== SVG ===")
    vb_attr = svg_tag.get("viewbox") or svg_tag.get("viewBox")
    print("viewBox:", vb_attr)
    print("width:", svg_tag.get("width"))
    print("height:", svg_tag.get("height"))

    # 画像のピクセルサイズ
    with Image.open(IMG_PATH) as img:
        img_w, img_h = img.size
    print("\n=== Image ===")
    print("size (width, height):", (img_w, img_h))

    # svg 直下の g の transform
    print("\n=== Direct child <g> of <svg> (transform) ===")
    for g in soup.find("svg").find_all("g", recursive=False):
        print(g.get("transform"))

    # viewBox からスケール算出（html.parser は属性を小文字にするので viewbox で取得）
    vb_str = svg_tag.get("viewbox") or svg_tag.get("viewBox") or ""
    vb = vb_str.split()
    if len(vb) == 4:
        vb_w, vb_h = float(vb[2]), float(vb[3])
        scale_x = img_w / vb_w
        scale_y = img_h / vb_h
    else:
        scale_x = scale_y = 1.0
    print("\n=== Scale (image / viewBox) ===")
    print("scale_x:", scale_x, "scale_y:", scale_y)

    # 壁ポリゴンをスケール適用してマスクに描画
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    for g in soup.find_all("g"):
        g_id = str(g.get("id", ""))
        g_class_str = _class_to_str(g.get("class", ""))
        if "Wall" not in g_id and "Wall" not in g_class_str:
            continue
        for poly in g.find_all("polygon", recursive=False):
            points_attr = poly.get("points")
            coords = parse_polygon_points(points_attr or "")
            if len(coords) >= 3:
                scaled_coords = [(x * scale_x, y * scale_y) for x, y in coords]
                draw.polygon(scaled_coords, fill=255)

    out_path = SAMPLE_DIR / "wall_mask_scaled.png"
    mask.save(out_path, format="PNG")
    print("\n=== Output ===")
    print("Saved wall mask (scaled):", out_path)


if __name__ == "__main__":
    main()
