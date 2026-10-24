import argparse
import json
import os
import random
import shutil
import time
import zipfile
from pathlib import Path


RAW_TRAIN_INPUT = "ISIC2018_Task1-2_Training_Input"
RAW_TRAIN_MASK = "ISIC2018_Task1_Training_GroundTruth"
RAW_VALID_INPUT = "ISIC2018_Task1-2_Validation_Input"
RAW_VALID_MASK = "ISIC2018_Task1_Validation_GroundTruth"
RAW_TEST_INPUT = "ISIC2018_Task1-2_Test_Input"
RAW_TEST_MASK = "ISIC2018_Task1_Test_GroundTruth"

OFFICIAL_SPLIT_FOLDERS = {
    "train": (RAW_TRAIN_INPUT, RAW_TRAIN_MASK),
    "valid": (RAW_VALID_INPUT, RAW_VALID_MASK),
    "test": (RAW_TEST_INPUT, RAW_TEST_MASK),
}

OFFICIAL_EXPECTED_COUNTS = {
    RAW_TRAIN_INPUT: 2594,
    RAW_TRAIN_MASK: 2594,
    RAW_VALID_INPUT: 100,
    RAW_VALID_MASK: 100,
    RAW_TEST_INPUT: 1000,
    RAW_TEST_MASK: 1000,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ISIC18 splits for EMCAD.")
    parser.add_argument(
        "--source",
        type=str,
        default="official",
        choices=["official", "hf"],
        help="Use extracted official folders or a Hugging Face mirror dataset.",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="./data/ISIC2018/raw",
        help="Root containing the extracted official ISIC18 folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./data/ISIC2018",
        help="Root where train/valid/test folders will be created.",
    )
    parser.add_argument(
        "--hf-dataset-id",
        type=str,
        default="surajbijjahalli/ISIC2018",
        help="HF dataset id used when --source=hf.",
    )
    parser.add_argument(
        "--hf-public-splits",
        type=str,
        default="train,validation",
        help="Comma-separated labeled HF splits to merge before 80/10/10 resplitting.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train split ratio."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation split ratio."
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="How to materialize split files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing split folders before recreating them.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def count_media_files(path: Path):
    if not path.is_dir():
        return 0
    return sum(
        1
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def find_archive_for_folder(folder_name: str, search_roots):
    archive_name = f"{folder_name}.zip"
    for root in search_roots:
        candidate = root / archive_name
        if candidate.is_file():
            return candidate
    return None


def ensure_extracted_folder(raw_root: Path, folder_name: str, search_roots):
    target_dir = raw_root / folder_name
    expected_count = OFFICIAL_EXPECTED_COUNTS.get(folder_name)
    if target_dir.is_dir():
        actual_count = count_media_files(target_dir)
        if expected_count is None or actual_count == expected_count:
            return None
        print(
            f"Re-extracting incomplete folder {target_dir}: "
            f"found {actual_count}, expected {expected_count}"
        )
        shutil.rmtree(target_dir)

    archive_path = find_archive_for_folder(folder_name, search_roots)
    if archive_path is None:
        search_desc = ", ".join(str(root) for root in search_roots)
        raise FileNotFoundError(
            f"Missing extracted folder {target_dir} and archive {folder_name}.zip "
            f"under: {search_desc}"
        )

    print(f"Extracting {archive_path} -> {raw_root}")
    ensure_dir(raw_root)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(raw_root)
    if not target_dir.is_dir():
        raise FileNotFoundError(
            f"Archive extracted but folder not found: {target_dir}"
        )
    if expected_count is not None:
        actual_count = count_media_files(target_dir)
        if actual_count != expected_count:
            raise RuntimeError(
                f"Unexpected file count after extracting {archive_path}: "
                f"found {actual_count}, expected {expected_count}"
            )
    return str(archive_path.resolve())


def collect_pairs(image_dir: Path, mask_dir: Path):
    images = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = image_path.stem
        candidates = [
            mask_dir / f"{stem}_segmentation.png",
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}_segmentation.jpg",
            mask_dir / f"{stem}.jpg",
        ]
        mask_path = next((item for item in candidates if item.exists()), None)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for image: {image_path}")
        images.append(
            {
                "id": stem,
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
            }
        )
    return images


def validate_raw_root(raw_root: Path):
    expected = []
    for image_folder, mask_folder in OFFICIAL_SPLIT_FOLDERS.values():
        expected.extend([image_folder, mask_folder])
    missing = [name for name in expected if not (raw_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing extracted ISIC18 folders under raw root: " + ", ".join(missing)
        )


def prepare_official_raw_root(raw_root: Path, output_root: Path):
    extracted_archives = []
    search_roots = []
    for root in (raw_root, output_root):
        if root not in search_roots:
            search_roots.append(root)

    for image_folder, mask_folder in OFFICIAL_SPLIT_FOLDERS.values():
        for folder_name in (image_folder, mask_folder):
            archive_path = ensure_extracted_folder(raw_root, folder_name, search_roots)
            if archive_path is not None:
                extracted_archives.append(archive_path)

    validate_raw_root(raw_root)
    return sorted(set(extracted_archives))


def load_official_split_map(raw_root: Path):
    split_map = {}
    for split_name, (image_folder, mask_folder) in OFFICIAL_SPLIT_FOLDERS.items():
        split_map[split_name] = collect_pairs(
            raw_root / image_folder, raw_root / mask_folder
        )
    return split_map


def export_hf_image(image_obj, path: Path):
    # HF image feature returns a PIL image object.
    path.parent.mkdir(parents=True, exist_ok=True)
    image_obj.save(path)


def load_hf_pairs(output_root: Path, dataset_id: str, hf_public_splits: str):
    from datasets import concatenate_datasets
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files

    cache_root = output_root / "raw" / "hf_cache"
    export_root = output_root / "raw" / "hf_export"
    ensure_dir(cache_root)
    ensure_dir(export_root)

    split_names = [item.strip() for item in hf_public_splits.split(",") if item.strip()]
    if not split_names:
        raise ValueError("No HF public splits specified.")
    repo_files = list_repo_files(dataset_id, repo_type="dataset")
    data_files = {}
    allow_patterns = ["README.md", ".gitattributes"]
    for split_name in split_names:
        prefix = f"data/{split_name}-"
        parquet_files = sorted(item for item in repo_files if item.startswith(prefix))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in HF dataset {dataset_id} for split {split_name}"
            )
        data_files[split_name] = parquet_files
        allow_patterns.extend(parquet_files)

    local_paths = {}
    total_files = sum(len(items) for items in data_files.values())
    downloaded = 0
    for split_name in split_names:
        local_paths[split_name] = []
        for rel_path in data_files[split_name]:
            downloaded += 1
            success = False
            last_error = None
            for attempt in range(1, 9):
                try:
                    local_file = hf_hub_download(
                        repo_id=dataset_id,
                        filename=rel_path,
                        repo_type="dataset",
                        cache_dir=str(cache_root),
                        resume_download=True,
                    )
                    local_paths[split_name].append(local_file)
                    print(
                        f"[HF] {downloaded}/{total_files} ready: {rel_path} "
                        f"(attempt {attempt})"
                    )
                    success = True
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[HF] retry {attempt} failed for {rel_path}: {exc}",
                        flush=True,
                    )
                    time.sleep(min(10, attempt * 2))
            if not success:
                raise RuntimeError(
                    f"Failed to download HF file after retries: {rel_path}"
                ) from last_error

    datasets_to_merge = []
    for split_name in split_names:
        local_files = local_paths[split_name]
        ds = load_dataset("parquet", data_files={split_name: local_files}, split=split_name)
        datasets_to_merge.append(ds)
    dataset = (
        datasets_to_merge[0]
        if len(datasets_to_merge) == 1
        else concatenate_datasets(datasets_to_merge)
    )
    pairs = []
    image_dir = export_root / "images"
    mask_dir = export_root / "masks"
    ensure_dir(image_dir)
    ensure_dir(mask_dir)
    for idx, sample in enumerate(dataset):
        stem = f"isic18_{idx:05d}"
        image_path = image_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}_segmentation.png"
        if not image_path.exists():
            export_hf_image(sample["image"], image_path)
        if not mask_path.exists():
            export_hf_image(sample["label"], mask_path)
        pairs.append(
            {
                "id": stem,
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
            }
        )
    return pairs


def split_items(items, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError(
            f"Invalid split sizes from total={total}: "
            f"train={train_count}, valid={val_count}, test={test_count}"
        )
    train_items = shuffled[:train_count]
    valid_items = shuffled[train_count : train_count + val_count]
    test_items = shuffled[train_count + val_count :]
    return {
        "train": train_items,
        "valid": valid_items,
        "test": test_items,
    }


def materialize_link(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def write_split(split_name: str, items, output_root: Path, link_mode: str):
    split_root = output_root / split_name
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    clear_dir(images_dir)
    clear_dir(masks_dir)
    for item in items:
        image_src = Path(item["image"])
        mask_src = Path(item["mask"])
        image_dst = images_dir / image_src.name
        mask_dst = masks_dir / mask_src.name
        materialize_link(image_src, image_dst, link_mode)
        materialize_link(mask_src, mask_dst, link_mode)


def write_manifest(split_map, output_root: Path, meta):
    manifest_path = output_root / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "counts": {key: len(value) for key, value in split_map.items()},
                "splits": split_map,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    raw_root = Path(args.raw_root).resolve()

    if args.source == "official":
        extracted_archives = prepare_official_raw_root(raw_root, output_root)
        split_map = load_official_split_map(raw_root)
        raw_root_for_meta = str(raw_root)
    else:
        merged_pairs = load_hf_pairs(
            output_root=output_root,
            dataset_id=args.hf_dataset_id,
            hf_public_splits=args.hf_public_splits,
        )
        extracted_archives = []
        raw_root_for_meta = f"hf://{args.hf_dataset_id}/{args.hf_public_splits}"

    ensure_dir(output_root)
    if args.force:
        for split_name in ("train", "valid", "test"):
            clear_dir(output_root / split_name)

    if args.source == "hf":
        split_map = split_items(
            merged_pairs,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    for split_name, items in split_map.items():
        write_split(split_name, items, output_root, args.link_mode)

    total_items = sum(len(items) for items in split_map.values())
    write_manifest(
        split_map,
        output_root,
        meta={
            "source": args.source,
            "raw_root": raw_root_for_meta,
            "protocol": (
                "official_isic18_task1"
                if args.source == "official"
                else "hf_public_resplit"
            ),
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "link_mode": args.link_mode,
            "total_items": total_items,
            "extracted_archives": extracted_archives,
        },
    )
    print(
        "Prepared ISIC18 splits:",
        {split_name: len(items) for split_name, items in split_map.items()},
    )


if __name__ == "__main__":
    main()
