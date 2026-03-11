#!/usr/bin/env python3
"""
Cubicasa5k の model.svg から壁（Wall）のみを抽出し、
白黒のマスク画像（PNG）を生成するスクリプト。

依存:
  - Python: lxml, cairosvg, opencv-python, numpy
  - システム: cairo ライブラリ（cairosvg 用）
    macOS: brew install cairo
    Ubuntu: sudo apt install libcairo2
"""

import argparse
import copy
import os
import re
import sys

try:
    import cairosvg
except OSError as e:
    if "cairo" in str(e).lower():
        sys.exit(
            "cairosvg はシステムの cairo ライブラリが必要です。\n"
            "  macOS: brew install cairo\n"
            "  Ubuntu: sudo apt install libcairo2\n"
            f"詳細: {e}"
        )
    raise

import cv2
import numpy as np
from lxml import etree

SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {None: SVG_NS}
WHITE = "#FFFFFF"

# 壁の直接の子として許容するプリミティブ要素（これ以外の子はコピーしない）
PRIMITIVE_SHAPE_LOCAL_NAMES = frozenset({"polygon", "polyline", "path", "line", "rect", "circle", "ellipse"})

# 抽出後のツリーから削除する id/class に含まれるキーワード（Door, Window 等）
EXCLUDED_ID_CLASS_KEYWORDS = (
    "Door",
    "Window",
    "Furniture",
    "Space",
    "Panel",
    "Threshold",
    "Glass",
    "Railing",
    "Stairs",
    "Dimension",
    "FixedFurniture",
    "TextLabel",
    "BoundaryPolygon",
    "Steps",
    "Flight",
    "Winding",
    "WalkinLine",
    "NameLabel",
    "DimensionMeasureLabel",
    "DimensionMark",
    "SpaceDimensionsLabel",
    "Visual",
    "Marker",
    "InnerPolygon",
    "Hanger",
    "Direction",
    "Name",
    "Doors",
    "electricitySign",
    "Bench",
    "OuterDrain",
    "InnerDrain",
    "Faucet",
    "PanelArea",
)


def _local_name(elem: etree._Element) -> str:
    """要素のローカル名（名前空間を除いたタグ名）を返す。"""
    return etree.QName(elem).localname if elem.tag and elem.tag.startswith("{") else (elem.tag or "")


def _is_wall_element(elem: etree._Element) -> bool:
    """要素が id="Wall" または class に "Wall" を含むか判定する。"""
    elem_id = elem.get("id")
    if elem_id is not None and elem_id == "Wall":
        return True
    cls = elem.get("class")
    if cls is not None and "Wall" in cls.split():
        return True
    return False


def _collect_wall_elements(root: etree._Element) -> list[etree._Element]:
    """
    ルート以下から、id="Wall" または class="Wall" を持つ要素を収集する。
    親がすでに Wall の場合は追加しない（重複を避ける）。
    """
    walls = []
    for elem in root.iter():
        if not _is_wall_element(elem):
            continue
        # 祖先に Wall がいる場合はスキップ（親 Wall に含まれるため）
        parent = elem.getparent()
        while parent is not None and parent is not root:
            if _is_wall_element(parent):
                break
            parent = parent.getparent()
        else:
            walls.append(elem)
    return walls


def _copy_wall_strict(wall_el: etree._Element) -> etree._Element:
    """
    Wall 要素をコピーするが、直接の子は polygon / polyline / path / line / rect 等の
    プリミティブのみとし、Door/Window 等の <g> 子は含めない。
    """
    tag = wall_el.tag
    # 同じタグ・属性で新要素を作成
    new_el = etree.Element(tag, attrib=dict(wall_el.attrib))
    for child in wall_el:
        if _local_name(child) in PRIMITIVE_SHAPE_LOCAL_NAMES:
            new_el.append(copy.deepcopy(child))
    return new_el


def _force_white_fill_stroke(elem: etree._Element) -> None:
    """要素とその子孫の fill / stroke を #FFFFFF にし、style 内も上書きする。"""
    # 属性で fill / stroke を上書き
    elem.set("fill", WHITE)
    elem.set("stroke", WHITE)
    # style 内の fill / stroke を置換（存在する場合）
    style = elem.get("style")
    if style:
        style = re.sub(r"fill:\s*[^;]+", "fill: #FFFFFF", style)
        style = re.sub(r"stroke:\s*[^;]+", "stroke: #FFFFFF", style)
        elem.set("style", style)
    for child in elem:
        _force_white_fill_stroke(child)


def _id_or_class_contains_excluded(elem: etree._Element) -> bool:
    """id または class に EXCLUDED_ID_CLASS_KEYWORDS のいずれかが含まれるか。"""
    for attr in ("id", "class"):
        val = elem.get(attr)
        if val is None:
            continue
        # class は空白区切りトークン、id は単一文字列
        tokens = val.split() if attr == "class" else [val]
        for token in tokens:
            for keyword in EXCLUDED_ID_CLASS_KEYWORDS:
                if keyword in token or token in keyword:
                    return True
    return False


def _remove_excluded_groups(root: etree._Element) -> None:
    """
    ツリー全体をスキャンし、id または class に Door, Window, Furniture, Space 等を
    含むグループ（<g>）を明示的に remove() で削除する。
    """
    to_remove = []
    for elem in root.iter():
        if _local_name(elem) != "g":
            continue
        if _id_or_class_contains_excluded(elem):
            to_remove.append(elem)
    # 深い順に削除（子を先に削除してから親を削除）
    to_remove.sort(key=lambda e: (-len(list(e.iter())),))
    for elem in to_remove:
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)


def _build_wall_svg_string(original_root: etree._Element, wall_elements: list[etree._Element]) -> str:
    """
    元の SVG の viewBox/width/height を引き継いだ新しい SVG ルートを作り、
    壁要素のみを追加した SVG 文字列を返す。
    背景は黒 (#000000) の rect で埋める。
    """
    # 新しいルートを作成（元の svg の属性をコピー）
    tag = f"{{{SVG_NS}}}svg"
    new_root = etree.Element(tag, nsmap=NSMAP)
    for attr in ("width", "height", "viewBox", "version", "id"):
        val = original_root.get(attr)
        if val is not None:
            new_root.set(attr, val)
    if original_root.get("xmlns"):
        new_root.set("xmlns", SVG_NS)
    if original_root.get("xmlns:xlink"):
        new_root.set("xmlns:xlink", "http://www.w3.org/1999/xlink")

    # 背景用の黒い rect（viewBox の範囲を覆う）
    vb = original_root.get("viewBox")
    if vb:
        parts = vb.strip().split()
        if len(parts) == 4:
            x, y, w, h = parts[0], parts[1], parts[2], parts[3]
            rect = etree.SubElement(new_root, f"{{{SVG_NS}}}rect")
            rect.set("x", x)
            rect.set("y", y)
            rect.set("width", w)
            rect.set("height", h)
            rect.set("fill", "#000000")

    # 壁要素を厳格にコピー（Wall とその直接の子の polygon/polyline/path 等のみ）
    for wall_el in wall_elements:
        el_copy = _copy_wall_strict(wall_el)
        _force_white_fill_stroke(el_copy)
        new_root.append(el_copy)

    # 万全を期すため、抽出後のツリー全体をスキャンし Door/Window/Furniture/Space 等を削除
    _remove_excluded_groups(new_root)

    return etree.tostring(
        new_root,
        encoding="unicode",
        method="xml",
    )


def _render_svg_to_png(svg_string: str) -> bytes:
    """SVG 文字列を cairosvg で PNG バイト列にレンダリングする（背景は黒）。"""
    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        background_color="#000000",
    )
    return png_bytes


def _resize_to_reference(png_bytes: bytes, ref_height: int, ref_width: int) -> bytes:
    """PNG を (ref_width, ref_height) にリサイズして PNG バイト列で返す。"""
    nparr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode PNG from rendered SVG.")
    # 参照サイズは (height, width)、cv2.resize は (width, height)
    resized = cv2.resize(img, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode(".png", resized)
    return buf.tobytes()


def _get_f1_scaled_size(model_svg_path: str) -> tuple[int, int]:
    """model.svg と同じディレクトリの F1_scaled.png の (height, width) を返す。"""
    dir_path = os.path.dirname(os.path.abspath(model_svg_path))
    f1_path = os.path.join(dir_path, "F1_scaled.png")
    if not os.path.isfile(f1_path):
        raise FileNotFoundError(f"F1_scaled.png not found in same directory as model.svg: {f1_path}")
    img = cv2.imread(f1_path)
    if img is None:
        raise ValueError(f"Failed to read F1_scaled.png: {f1_path}")
    h, w = img.shape[:2]
    return h, w


def _get_image_size(image_path: str) -> tuple[int, int]:
    """任意の画像ファイルの (height, width) を返す。"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    return h, w


def generate_wall_mask(
    model_svg_path: str,
    output_png_path: str,
    reference_image_path: str | None = None,
) -> None:
    """
    model.svg をパースして壁のみの白黒マスク PNG を生成し、
    指定画像（または F1_scaled.png）と同じサイズにリサイズして保存する。

    Parameters
    ----------
    model_svg_path : str
        入力の model.svg のパス
    output_png_path : str
        出力するマスク PNG のパス
    reference_image_path : str, optional
        リサイズ基準とする画像のパス。指定時は F1_scaled.png の代わりにこの画像のサイズを使用する。
    """
    with open(model_svg_path, "rb") as f:
        tree = etree.parse(f)
    root = tree.getroot()

    wall_elements = _collect_wall_elements(root)
    if not wall_elements:
        raise ValueError(f"No Wall elements found in {model_svg_path}")

    svg_string = _build_wall_svg_string(root, wall_elements)
    png_bytes = _render_svg_to_png(svg_string)

    if reference_image_path is not None:
        ref_height, ref_width = _get_image_size(reference_image_path)
    else:
        ref_height, ref_width = _get_f1_scaled_size(model_svg_path)
    resized_bytes = _resize_to_reference(png_bytes, ref_height, ref_width)

    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)) or ".", exist_ok=True)
    with open(output_png_path, "wb") as f:
        f.write(resized_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cubicasa5k model.svg から壁のみの白黒マスク PNG を生成する"
    )
    parser.add_argument(
        "model_svg",
        type=str,
        help="入力の model.svg のパス",
    )
    parser.add_argument(
        "output_png",
        type=str,
        help="出力するマスク PNG のパス",
    )
    args = parser.parse_args()

    try:
        generate_wall_mask(args.model_svg, args.output_png)
        print(f"Saved wall mask to {args.output_png}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
