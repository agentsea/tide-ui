import argparse
import hashlib
from collections import defaultdict
from typing import Dict

import datasets
from datasets import load_dataset
from tqdm import tqdm


def bb2points(example, idx):
    """
    Transform a bounding box in [x1, y1, x2, y2] format to a point in the center of the box.
    """
    bb = example["bbox"]
    example["point"] = [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]
    return example


def is_english(example):
    """
    Filter function to check if example is in English.
    """
    return example["language"] == "English"


def deduplicate_by_image_name(dataset, num_proc):
    """
    Remove examples where a name appears multiple times within the same image.
    Uses parallel processing while preserving order.
    """
    print("First pass: collecting image hashes and counting names...")
    image_name_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for example in tqdm(dataset, desc="Collecting name counts per image"):
        image = example["image"]
        image_bytes = image.tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        name = example["name"]
        image_name_counts[image_hash][name] += 1

    duplicate_name_count = sum(
        1
        for image_counts in image_name_counts.values()
        for count in image_counts.values()
        if count > 1
    )

    print(
        f"\nNumber of names that appear multiple times in their images: {duplicate_name_count}"
    )

    print("\nSecond pass: filtering duplicates...")

    keep_mask = []
    original_indices = []

    for idx, example in enumerate(tqdm(dataset, desc="Filtering duplicates")):
        image = example["image"]
        image_bytes = example["image"].tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        name = example["name"]
        if image_name_counts[image_hash][name] == 1:
            keep_mask.append(True)
            original_indices.append(idx)
        else:
            keep_mask.append(False)

    num_duplicates = len(dataset) - sum(keep_mask)
    print(f"\nDuplication statistics:")
    print(f"Total examples: {len(dataset)}")
    print(
        f"Duplicate examples removed: {num_duplicates} ({num_duplicates/len(dataset)*100:.2f}%)"
    )
    print(f"Examples kept: {sum(keep_mask)} ({sum(keep_mask)/len(dataset)*100:.2f}%)")

    if duplicate_name_count > 0:
        print("\nExample duplicates (up to 5 cases):")
        count = 0
        for image_hash, name_counts in image_name_counts.items():
            for name, freq in name_counts.items():
                if freq > 1:
                    print(
                        f"Image hash: {image_hash[:8]}..., Name: {name}, Occurrences: {freq}"
                    )
                    count += 1
                    if count >= 5:
                        break
            if count >= 5:
                break

    indices_to_keep = [idx for idx, keep in enumerate(keep_mask) if keep]

    return dataset.select(indices_to_keep)


def main(args):
    print(f"Running with {args.num_proc} processes")

    print("Loading dataset...")
    ds = load_dataset("agentsea/wave-ui")
    filtered_ds = {}

    for split in ds.keys():
        print(f"\nProcessing {split} split...")
        print("Filtering English examples...")
        filtered_ds[split] = ds[split].filter(
            is_english, num_proc=args.num_proc, desc="Filtering English examples"
        )

        print("Converting bounding boxes to points...")
        filtered_ds[split] = filtered_ds[split].map(
            bb2points,
            with_indices=True,
            desc="Converting bboxes to points",
            num_proc=args.num_proc,
        )

        print("Deduplicating names within images...")
        filtered_ds[split] = deduplicate_by_image_name(
            filtered_ds[split], args.num_proc
        )

    print("\nCalculating retention rates...")
    original_sizes = {split: len(ds[split]) for split in ds.keys()}
    filtered_sizes = {split: len(filtered_ds[split]) for split in filtered_ds.keys()}
    retention_rates = {
        split: filtered_sizes[split] / original_sizes[split] for split in ds.keys()
    }
    worst_split = min(retention_rates, key=retention_rates.get)

    print("\nRetention rates:")
    for split, rate in retention_rates.items():
        print(f"{split}: {rate*100:.2f}%")

    split_proportions = {"train": 0.8, "test": 0.1, "validation": 0.1}
    total_size = filtered_sizes[worst_split] / split_proportions[worst_split]
    new_sizes = {
        split: int(total_size * split_proportions[split]) for split in split_proportions
    }

    print("\nRebalancing splits...")
    for split in filtered_ds.keys():
        if len(filtered_ds[split]) > new_sizes[split]:
            filtered_ds[split] = filtered_ds[split].select(range(new_sizes[split]))

    worst_split_size = total_size - sum(
        len(filtered_ds[split]) for split in filtered_ds.keys() if split != worst_split
    )
    filtered_ds[worst_split] = filtered_ds[worst_split].select(
        range(int(worst_split_size))
    )

    print("\nFinal dataset sizes:")
    for split in filtered_ds.keys():
        print(f"{split}: {len(filtered_ds[split])} examples")

    print("\nPreparing for upload...")
    for split in filtered_ds.keys():
        filtered_ds[split] = filtered_ds[split].select_columns(
            ["image", "resolution", "name", "point"]
        )

    filtered_ds = datasets.DatasetDict(filtered_ds)
    print("\nPushing to hub...")
    filtered_ds.push_to_hub("agentsea/wave-ui-points", private=True)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform WaveUI from bounding boxes to points for English examples, "
        "deduplicate names within images, and rebalance the dataset."
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes to use for parallel processing (default: 4)",
    )
    args = parser.parse_args()
    main(args)
