"""
Prepare CubiCasa5k-style dataset for pix2pix.

Input (default):
  cubicasa5k_processed/{train,val,test}/
    images/<name>.<ext>
    masks/<name>.<ext>

Output (created next to input root):
  cubicasa5k_pix2pix/{train,val,test}/<name>.<ext>

For each paired file name:
  - resize both to 256x256
    - images: BILINEAR (or BICUBIC)
    - masks : NEAREST (avoid gray pixels)
  - concatenate horizontally: [image | mask] -> 512x256
  - save with original file name
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


LOGGER = logging.getLogger("prepare_cubicasa5k_pix2pix")


def _iter_image_files(dir_path: Path) -> Iterable[Path]:
    # Allow common raster formats. (Add more if needed.)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _resolve_mask_path(masks_dir: Path, img_name: str) -> Optional[Path]:
    """
    Requirement is "same file name", so we try that first.
    If missing, fall back to common CubiCasa-like naming patterns:
      - *_image.ext -> *_mask.ext
      - <stem>_mask.ext
      - replace occurrences of 'image' with 'mask' in stem
    """
    direct = masks_dir / img_name
    if direct.exists():
        return direct

    img_path = Path(img_name)
    stem = img_path.stem
    suffix = img_path.suffix

    candidates = []
    if stem.endswith("_image"):
        candidates.append(masks_dir / f"{stem[:-6]}_mask{suffix}")  # remove "_image"
    candidates.append(masks_dir / f"{stem}_mask{suffix}")
    if "image" in stem:
        candidates.append(masks_dir / f"{stem.replace('image', 'mask')}{suffix}")

    for c in candidates:
        if c.exists():
            return c
    return None


def _open_image(path: Path) -> Optional[Image.Image]:
    try:
        # Ensure the file handle closes quickly (Pillow does lazy loading).
        with Image.open(path) as im:
            return im.copy()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        LOGGER.warning("Failed to open image. Skipping: %s (%s)", path, e)
        return None


def _resize_pair(
    img: Image.Image,
    mask: Image.Image,
    size: Tuple[int, int],
    *,
    img_resample: Image.Resampling,
    mask_resample: Image.Resampling,
) -> Tuple[Image.Image, Image.Image]:
    # pix2pix commonly uses RGB; ensure consistent mode to avoid surprises.
    img_rgb = img.convert("RGB")
    # Masks can be L/P/RGB; convert to RGB for concatenation output.
    # NEAREST resampling is what prevents boundary blur.
    mask_rgb = mask.convert("RGB")

    img_rs = img_rgb.resize(size, resample=img_resample)
    mask_rs = mask_rgb.resize(size, resample=mask_resample)
    return img_rs, mask_rs


def _concat_horiz(left: Image.Image, right: Image.Image) -> Image.Image:
    if left.size != right.size:
        raise ValueError(f"Size mismatch: left={left.size} right={right.size}")
    w, h = left.size
    out = Image.new("RGB", (w * 2, h))
    out.paste(left, (0, 0))
    out.paste(right, (w, 0))
    return out


def process_split(
    src_split_dir: Path,
    dst_split_dir: Path,
    *,
    size: Tuple[int, int],
    img_resample: Image.Resampling,
    mask_resample: Image.Resampling,
) -> None:
    images_dir = src_split_dir / "images"
    masks_dir = src_split_dir / "masks"

    if not images_dir.is_dir():
        LOGGER.warning("Missing images dir. Skipping split: %s", images_dir)
        return
    if not masks_dir.is_dir():
        LOGGER.warning("Missing masks dir. Skipping split: %s", masks_dir)
        return

    dst_split_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(_iter_image_files(images_dir))
    if not image_files:
        LOGGER.warning("No image files found in: %s", images_dir)
        return

    for img_path in tqdm(image_files, desc=f"{src_split_dir.name}", unit="img"):
        mask_path = _resolve_mask_path(masks_dir, img_path.name)
        if mask_path is None:
            LOGGER.warning("Mask missing. Skipping: %s (looked for same-name and common variants)", img_path.name)
            continue

        img = _open_image(img_path)
        if img is None:
            continue
        mask = _open_image(mask_path)
        if mask is None:
            continue

        try:
            img_rs, mask_rs = _resize_pair(
                img,
                mask,
                size,
                img_resample=img_resample,
                mask_resample=mask_resample,
            )
            merged = _concat_horiz(img_rs, mask_rs)
        except Exception as e:
            LOGGER.warning("Failed processing pair. Skipping: %s (%s)", img_path.name, e)
            continue

        out_path = dst_split_dir / img_path.name
        try:
            merged.save(out_path)
        except Exception as e:
            LOGGER.warning("Failed to save output. Skipping: %s (%s)", out_path, e)
            continue


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[1]
    src_root_default = repo_root_default / "cubicasa5k_processed"
    dst_root_default = repo_root_default / "cubicasa5k_pix2pix"

    p = argparse.ArgumentParser(description="Prepare cubicasa5k_processed for pix2pix (paired concat).")
    p.add_argument("--src-root", type=Path, default=src_root_default, help="Source root (cubicasa5k_processed).")
    p.add_argument("--dst-root", type=Path, default=dst_root_default, help="Destination root (cubicasa5k_pix2pix).")
    p.add_argument("--size", type=int, nargs=2, default=(256, 256), metavar=("W", "H"), help="Resize size.")
    p.add_argument(
        "--image-resample",
        choices=("bilinear", "bicubic"),
        default="bilinear",
        help="Resampling method for input images.",
    )
    p.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console log level.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    src_root: Path = args.src_root
    dst_root: Path = args.dst_root
    size = (int(args.size[0]), int(args.size[1]))

    if args.image_resample == "bicubic":
        img_resample = Image.Resampling.BICUBIC
    else:
        img_resample = Image.Resampling.BILINEAR
    mask_resample = Image.Resampling.NEAREST

    if not src_root.is_dir():
        LOGGER.error("Source root not found: %s", src_root)
        return 2

    # Requirement: create destination with train/val/test
    for split in ("train", "val", "test"):
        (dst_root / split).mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        src_split_dir = src_root / split
        dst_split_dir = dst_root / split
        if not src_split_dir.is_dir():
            LOGGER.warning("Split dir missing. Skipping: %s", src_split_dir)
            continue
        process_split(
            src_split_dir,
            dst_split_dir,
            size=size,
            img_resample=img_resample,
            mask_resample=mask_resample,
        )

    LOGGER.info("Done. Output: %s", dst_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
