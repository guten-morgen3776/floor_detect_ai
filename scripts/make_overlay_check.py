#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

from PIL import Image


def _list_common_filenames(images_dir: Path, masks_dir: Path) -> list[str]:
    img_names = {p.name for p in images_dir.iterdir() if p.is_file()}
    mask_names = {p.name for p in masks_dir.iterdir() if p.is_file()}
    common = sorted(img_names & mask_names)
    return common


def _color_rgba(color: str) -> tuple[int, int, int, int]:
    c = color.lower().strip()
    if c in ("red", "r"):
        return (255, 0, 0, 255)
    if c in ("blue", "b"):
        return (0, 128, 255, 255)
    raise ValueError("color must be 'red' or 'blue'")


def _overlay_one(
    img_path: Path,
    mask_path: Path,
    out_path: Path,
    color: str,
    alpha: int,
) -> bool:
    with Image.open(img_path) as base:
        base_rgba = base.convert("RGBA")

    with Image.open(mask_path) as m:
        mask_l = m.convert("L")

    if base_rgba.size != mask_l.size:
        print(f"[SKIP] size mismatch: {img_path.name} img={base_rgba.size} mask={mask_l.size}")
        return False

    # mask: 0..255 -> alpha map (0..alpha)
    a = mask_l.point(lambda v: 0 if v == 0 else alpha)

    r, g, b, _ = _color_rgba(color)
    overlay = Image.new("RGBA", base_rgba.size, (r, g, b, 0))
    overlay.putalpha(a)

    out = Image.alpha_composite(base_rgba, overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, format="PNG")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay-check images vs wall masks (pixel alignment)")
    ap.add_argument("--root", type=Path, default=None, help="root containing images/ and masks/")
    ap.add_argument("--images-dir", type=Path, default=None, help="directory containing original images")
    ap.add_argument("--masks-dir", type=Path, default=None, help="directory containing mask images")
    ap.add_argument("--out-dir", type=Path, default=None, help="output directory (default: <root>/overlay_check)")
    ap.add_argument("--num", type=int, default=8, help="number of pairs to sample (default: 8)")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("--color", type=str, default="red", help="overlay color: red|blue (default: red)")
    ap.add_argument("--alpha", type=int, default=120, help="overlay alpha for wall pixels (0-255)")
    args = ap.parse_args()

    if args.root is None and (args.images_dir is None or args.masks_dir is None):
        raise SystemExit("Provide --root or both --images-dir and --masks-dir.")

    if args.root is not None:
        root = args.root
        images_dir = root / "images"
        masks_dir = root / "masks"
        out_dir = args.out_dir or (root / "overlay_check")
    else:
        images_dir = args.images_dir
        masks_dir = args.masks_dir
        out_dir = args.out_dir or (Path.cwd() / "overlay_check")

    if not images_dir.exists():
        raise SystemExit(f"images-dir not found: {images_dir}")
    if not masks_dir.exists():
        raise SystemExit(f"masks-dir not found: {masks_dir}")

    common = _list_common_filenames(images_dir, masks_dir)
    if not common:
        raise SystemExit(f"No common filenames between {images_dir} and {masks_dir}")

    if args.seed is not None:
        random.seed(args.seed)

    # 5〜10枚程度：指定がなければ 8、指定が小さすぎる/大きすぎる場合は調整
    n = args.num
    if n <= 0:
        n = 8
    n = max(5, min(10, n))
    n = min(n, len(common))

    chosen = random.sample(common, k=n)

    ok = 0
    for name in chosen:
        img_path = images_dir / name
        mask_path = masks_dir / name
        out_path = out_dir / name
        if _overlay_one(img_path, mask_path, out_path, color=args.color, alpha=int(args.alpha)):
            ok += 1

    print(f"done. saved={ok}/{len(chosen)} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

