#!/usr/bin/env python3
"""
Cubicasa5k データセット全体を一括処理し、
images/ と masks/（壁マスク）を cubicasa5k_processed/ に出力する。
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# 同一プロセス内で import するため、process_one_item 内では遅延 import を使う
# （cairosvg 等の読み込みをワーカーで行う）


def _rel_path_to_unique_name(rel_path: str) -> str:
    """相対パスを一意のファイル名ベースに変換（例: high_quality/0001/ -> high_quality_0001）。"""
    return rel_path.strip().strip("/").replace("/", "_")


def _choose_image_path(src_dir: str) -> str | None:
    """F1_scaled.png を優先し、なければ F1_original.png のパスを返す。存在しなければ None。"""
    for name in ("F1_scaled.png", "F1_original.png"):
        p = os.path.join(src_dir, name)
        if os.path.isfile(p):
            return p
    return None


def process_one_item(args: tuple) -> tuple[str, str | None]:
    """
    1件分の処理を行う（ワーカー用）。コピー元は base_dir、出力先は output_dir。

    Returns
    -------
    (unique_name, error_message)
        成功時は error_message は None。失敗時はエラーメッセージを返す。
    """
    base_dir, output_dir, split_name, rel_path, repo_root = args
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # ワーカープロセス内で import（cairo 等の初期化を子プロセスで行う）
    from scripts.svg_wall_mask import generate_wall_mask

    rel_path = rel_path.strip().strip("/")
    if not rel_path:
        return (_rel_path_to_unique_name(rel_path) or "empty", "Empty relative path")
    src_dir = os.path.join(base_dir, rel_path)
    unique_name = _rel_path_to_unique_name(rel_path)
    if not unique_name:
        return ("unknown", "Could not derive unique name")

    images_out = os.path.join(output_dir, split_name, "images")
    masks_out = os.path.join(output_dir, split_name, "masks")
    image_filename = f"{unique_name}_image.png"
    mask_filename = f"{unique_name}_mask.png"
    image_dst = os.path.join(images_out, image_filename)
    mask_dst = os.path.join(masks_out, mask_filename)

    try:
        image_src = _choose_image_path(src_dir)
        if image_src is None:
            return (unique_name, f"Neither F1_scaled.png nor F1_original.png in {src_dir}")
        model_svg = os.path.join(src_dir, "model.svg")
        if not os.path.isfile(model_svg):
            return (unique_name, f"model.svg not found: {model_svg}")

        os.makedirs(images_out, exist_ok=True)
        os.makedirs(masks_out, exist_ok=True)
        shutil.copy2(image_src, image_dst)

        # マスクは「コピーした画像」と同じサイズで出力する
        generate_wall_mask(
            model_svg,
            mask_dst,
            reference_image_path=image_dst,
        )
        return (unique_name, None)
    except Exception as e:
        return (unique_name, str(e))


def _load_split_paths(base_dir: str, split_name: str) -> list[str]:
    """train.txt / val.txt / test.txt を読み、相対パスのリストを返す。"""
    txt_path = os.path.join(base_dir, f"{split_name}.txt")
    if not os.path.isfile(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cubicasa5k を一括処理して images と壁マスクを出力する"
    )
    parser.add_argument(
        "base_dir",
        type=str,
        nargs="?",
        default=None,
        help="cubicasa5k のルート（train.txt があるディレクトリ）。未指定時は archive (1)/cubicasa5k/cubicasa5k を想定。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cubicasa5k_processed",
        help="出力先ルートディレクトリ（デフォルト: cubicasa5k_processed）",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default=None,
        help="処理するスプリット（未指定時は train / val / test すべて）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="各スプリットの先頭 N 件だけ処理（テスト用。例: --limit 10）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="並列ワーカー数（未指定時は os.cpu_count()）",
    )
    args = parser.parse_args()

    if args.base_dir is None:
        # スクリプト基準で archive (1) を探すか、カレントの cubicasa5k を使う
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        candidates = [
            os.path.join(repo_root, "archive (1)", "cubicasa5k", "cubicasa5k"),
            os.path.join(os.getcwd(), "cubicasa5k"),
            "archive (1)/cubicasa5k/cubicasa5k",
        ]
        base_dir = None
        for c in candidates:
            if os.path.isdir(c) and os.path.isfile(os.path.join(c, "val.txt")):
                base_dir = c
                break
        if base_dir is None:
            print("Error: base_dir を特定できません。base_dir を引数で指定してください。", file=sys.stderr)
            return 1
    else:
        base_dir = os.path.abspath(args.base_dir)

    if not os.path.isdir(base_dir):
        print(f"Error: base_dir が存在しません: {base_dir}", file=sys.stderr)
        return 1
    for name in ("train.txt", "val.txt", "test.txt"):
        if not os.path.isfile(os.path.join(base_dir, name)):
            print(f"Error: {name} がありません: {base_dir}", file=sys.stderr)
            return 1

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits = [args.split] if args.split else ["train", "val", "test"]
    all_tasks: list[tuple[str, str, str, str, str]] = []  # base_dir, output_dir, split, rel_path, repo_root
    output_dir = os.path.abspath(args.output_dir)
    for split_name in splits:
        paths = _load_split_paths(base_dir, split_name)
        if args.limit is not None:
            paths = paths[: args.limit]
        for rel_path in paths:
            all_tasks.append((base_dir, output_dir, split_name, rel_path, repo_root))

    if not all_tasks:
        print("処理対象が 0 件です。")
        return 0

    for split_name in splits:
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(output_dir, split_name, sub), exist_ok=True)

    workers = args.workers or os.cpu_count() or 1
    failed: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_item, t): t for t in all_tasks}
        with tqdm(total=len(futures), desc="batch_process", unit="item") as pbar:
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    unique_name, err = fut.result()
                    if err is not None:
                        failed.append((unique_name, err))
                except Exception as e:
                    _, _, _, rel, _ = task
                    failed.append((_rel_path_to_unique_name(rel), str(e)))
                pbar.update(1)

    if failed:
        print(f"\n失敗: {len(failed)} 件")
        for name, msg in failed[:20]:
            print(f"  {name}: {msg}")
        if len(failed) > 20:
            print(f"  ... 他 {len(failed) - 20} 件")
    else:
        print(f"\n完了: 全 {len(all_tasks)} 件を出力しました。")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
