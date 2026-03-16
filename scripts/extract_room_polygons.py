"""
Cubicasa5k の model.svg から部屋（Space）のポリゴン座標を抽出するスクリプト。
BeautifulSoup を使用して SVG を解析する。
"""

from pathlib import Path
from typing import List, Tuple
import warnings

from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def parse_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    """
    polygon の points 属性文字列を [(X1, Y1), (X2, Y2), ...] に変換する。
    数値はカンマとスペースで区切られている形式を想定（例: "169.10,208.83 169.10,107.71"）。
    """
    if not points_str or not points_str.strip():
        return []
    result: List[Tuple[float, float]] = []
    # スペースで区切って "x,y" のトークン列を得る
    tokens = points_str.strip().split()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                result.append((x, y))
            except ValueError:
                continue
    return result


def extract_room_polygons_from_svg(svg_path: Path) -> List[List[Tuple[float, float]]]:
    """
    1つの model.svg ファイルを読み込み、class に "Space" を含む <g> 内の
    <polygon> の points を取得し、ポリゴン座標リストのリストで返す。

    Returns:
        各要素が 1 つの部屋ポリゴン（(x,y) のリスト）であるリスト。
        例: [ [(x1,y1), (x2,y2), ...], [(x1,y1), ...], ... ]
    """
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    # html.parser は標準ライブラリのみで動作（xml パーサーは lxml が必要）
    soup = BeautifulSoup(text, "html.parser")

    polygons_list: List[List[Tuple[float, float]]] = []

    # class 属性に "Space" を含む <g> をすべて取得
    for g in soup.find_all("g", class_=True):
        class_attr = g.get("class", [])
        # BeautifulSoup は class をリストで返すことがある
        if isinstance(class_attr, list):
            class_str = " ".join(class_attr)
        else:
            class_str = str(class_attr)
        if "Space" not in class_str:
            continue

        # この <g> の直下の <polygon> のみ取得（子孫の DimensionMark 等は除く）
        for polygon in g.find_all("polygon", recursive=False):
            points_attr = polygon.get("points")
            if points_attr is None:
                continue
            coords = parse_polygon_points(points_attr)
            if coords:
                polygons_list.append(coords)

    return polygons_list


def extract_room_polygons_from_directory(
    root_dir: Path,
) -> List[Tuple[Path, List[List[Tuple[float, float]]]]]:
    """
    指定ディレクトリ以下を再帰的に走査し、すべての model.svg に対して
    部屋ポリゴンを抽出する。

    Args:
        root_dir: 走査のルートディレクトリ（pathlib.Path）

    Returns:
        (model.svg の Path, そのファイルのポリゴンリスト) のリスト。
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        return []

    results: List[Tuple[Path, List[List[Tuple[float, float]]]]] = []
    for svg_path in sorted(root.rglob("model.svg")):
        if not svg_path.is_file():
            continue
        try:
            polygons = extract_room_polygons_from_svg(svg_path)
            results.append((svg_path, polygons))
        except Exception as e:
            # エラー時はログしてスキップ（必要なら logging に変更可）
            print(f"Warning: failed to process {svg_path}: {e}", flush=True)
    return results


def main() -> None:
    """例: cubicasa5k ディレクトリを指定して実行"""
    import sys

    default_root = Path(
        "/Users/aokitenju/Downloads/archive (1)/cubicasa5k/cubicasa5k"
    )
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_root

    results = extract_room_polygons_from_directory(root_dir)
    print(f"Found {len(results)} model.svg files under {root_dir}")

    for svg_path, polygons in results:
        print(f"\n{svg_path.relative_to(root_dir)}: {len(polygons)} room polygon(s)")
        for i, poly in enumerate(polygons):
            print(f"  polygon[{i}] points: {len(poly)} vertices")
            if poly:
                print(f"    first 3: {poly[:3]}")


if __name__ == "__main__":
    main()
