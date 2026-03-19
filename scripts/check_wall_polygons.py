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

for g in soup.find_all("g"):
    g_id = str(g.get("id", ""))
    g_class = g.get("class", "")
    g_class_str = " ".join(g_class) if isinstance(g_class, list) else str(g_class)

    if "Wall" in g_id or "Wall" in g_class_str:
        polys = g.find_all("polygon", recursive=False)
        print(f"id={g_id!r} class={g_class_str!r} → polygon直下: {len(polys)}個")
