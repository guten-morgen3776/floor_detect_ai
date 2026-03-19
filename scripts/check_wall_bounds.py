from pathlib import Path
from bs4 import BeautifulSoup

folder = Path("/Users/aokitenju/Downloads/archive (1)/cubicasa5k/cubicasa5k/colorful/34")

try:
    soup = BeautifulSoup((folder / "model.svg").read_text(), "xml")
except Exception:
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup((folder / "model.svg").read_text(), "html.parser")

xs, ys = [], []
for g in soup.find_all("g"):
    g_id = str(g.get("id", ""))
    g_class = " ".join(g.get("class", []) if isinstance(g.get("class"), list) else [str(g.get("class", ""))])
    if "Wall" not in g_id and "Wall" not in g_class:
        continue
    for poly in g.find_all("polygon", recursive=False):
        points = poly.get("points", "")
        for token in points.strip().split():
            parts = token.split(",")
            if len(parts) >= 2:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))

if xs and ys:
    print(f"x: min={min(xs):.1f} max={max(xs):.1f}")
    print(f"y: min={min(ys):.1f} max={max(ys):.1f}")
else:
    print("No wall polygon points found")

print(f"viewBox: 1387.57 x 1520.93")
print(f"scaled:  1481 x 1558")
print(f"original: 482 x 507")
