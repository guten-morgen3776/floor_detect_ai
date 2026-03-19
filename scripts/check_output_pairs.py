from pathlib import Path

images_dir = Path("/Users/aokitenju/Downloads/wall_mask_out/images")
masks_dir = Path("/Users/aokitenju/Downloads/wall_mask_out/masks")

img_names = {p.name for p in images_dir.iterdir()}
mask_names = {p.name for p in masks_dir.iterdir()}

common = img_names & mask_names
print(f"共通ファイル数: {len(common)}")
print("サンプル:", list(common)[:3])

# imagesにあってmasksにないもの
only_img = img_names - mask_names
print(f"imagesのみ: {len(only_img)}", list(only_img)[:3])
