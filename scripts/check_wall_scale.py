from pathlib import Path
from bs4 import BeautifulSoup
from PIL import Image

folder = Path("/Users/aokitenju/Downloads/archive (1)/cubicasa5k/cubicasa5k/colorful/34")

try:
    soup = BeautifulSoup((folder / "model.svg").read_text(), "xml")
except Exception:
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup((folder / "model.svg").read_text(), "html.parser")

# 画像サイズ
with Image.open(next(folder.glob("F*_original.png"))) as img:
    img_w, img_h = img.size

# viewBoxスケール
root = soup.find("svg")
viewbox = (root.get("viewBox") or root.get("viewbox")) if root else None
if not viewbox:
    raise SystemExit("viewBox not found in SVG")
vb = viewbox.split()
vb_w, vb_h = float(vb[2]), float(vb[3])
scale_x = img_w / vb_w
scale_y = img_h / vb_h
print(f"scale_x={scale_x:.4f} scale_y={scale_y:.4f}")

# 最初のWallポリゴンの座標を変換前後で表示
for g in soup.find_all("g"):
    if "Wall" not in str(g.get("id", "")):
        continue
    poly = g.find("polygon", recursive=False)
    if poly is None:
        continue
    points = poly.get("points", "").strip().split()[:4]  # 最初の4点だけ
    print("変換前:", points)
    for token in points:
        x, y = map(float, token.split(","))
        print(f"  ({x:.1f}, {y:.1f}) → ({x*scale_x:.1f}, {y*scale_y:.1f})")
    break
