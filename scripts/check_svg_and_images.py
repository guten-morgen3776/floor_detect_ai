from pathlib import Path
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from PIL import Image

folder = Path("/Users/aokitenju/Downloads/archive (1)/cubicasa5k/cubicasa5k/colorful/34")

svg_path = folder / "model.svg"
try:
    svg = BeautifulSoup(svg_path.read_text(), "xml")
except Exception:
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    svg = BeautifulSoup(svg_path.read_text(), "html.parser")

root = svg.find("svg")
viewbox = (root.get("viewBox") or root.get("viewbox")) if root else None
if viewbox:
    vb = viewbox.split()
    print(f"viewBox: {vb[2]} x {vb[3]}")
else:
    print("viewBox not found")

for img in folder.glob("F*"):
    with Image.open(img) as i:
        print(f"{img.name}: {i.size}")
