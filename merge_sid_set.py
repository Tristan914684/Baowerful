"""
Merge SID_Set into your EXISTING data/train and data/test folders
(data/train/real, data/train/fake, data/test/real, data/test/fake),
without touching or overwriting any files already there.

label 0 -> real
label 1 (full_synthetic) -> fake
label 2 (tampered)       -> fake

HF `train` split -> your data/train/*
HF `val` split   -> your data/test/*   (kept disjoint from train, same as
                                          your original test folder)

Usage:
    pip install datasets pillow huggingface_hub --break-system-packages
    python merge_sid_set.py --repo_id <namespace/SID_Set> --data_root data

Run a small test first:
    python merge_sid_set.py --repo_id <namespace/SID_Set> --data_root data --limit_train 200 --limit_test 50
"""

import argparse
import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image


LABEL_TO_CLASS = {0: "real", 1: "fake", 2: "fake"}


def merge_split(ds_iterable, out_dir: Path, img_format: str, limit, prefix: str):
    out_dir_real = out_dir / "real"
    out_dir_fake = out_dir / "fake"
    out_dir_real.mkdir(parents=True, exist_ok=True)
    out_dir_fake.mkdir(parents=True, exist_ok=True)

    ext = "png" if img_format.upper() == "PNG" else "jpg"
    count = 0
    skipped = 0

    for i, example in enumerate(ds_iterable):
        if limit is not None and count >= limit:
            break

        label = example["label"]
        cls = LABEL_TO_CLASS.get(label)
        if cls is None:
            continue

        img = example["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        if img.mode != "RGB" and img_format.upper() == "JPEG":
            img = img.convert("RGB")

        img_id = str(example.get("img_id", f"idx{i}")).replace("/", "_")
        target_dir = out_dir_real if cls == "real" else out_dir_fake
        target_path = target_dir / f"{prefix}{img_id}.{ext}"

        # Never overwrite -- guards against re-running the script twice
        # or a filename collision with your existing data.
        if target_path.exists():
            skipped += 1
            continue

        img.save(target_path, img_format.upper())
        count += 1

        if count % 500 == 0:
            print(f"  ...saved {count} images to {out_dir.name} (skipped {skipped} existing)")

    print(f"Done: added {count} new images to {out_dir} (skipped {skipped} already-existing filenames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="HF dataset repo id, e.g. 'namespace/SID_Set'")
    parser.add_argument("--data_root", default="data", help="Root folder containing train/ and test/ (default: data)")
    parser.add_argument("--format", default="jpg", choices=["png", "jpg"],
                         help="Match your existing files -- yours are .jpg, so default is jpg")
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--prefix", default="sid_", help="Filename prefix to avoid collisions with existing files")
    args = parser.parse_args()

    img_format = "PNG" if args.format == "png" else "JPEG"
    data_root = Path(args.data_root)

    print("Streaming HF 'train' split -> merging into data/train ...")
    train_ds = load_dataset(args.repo_id, split="train", streaming=True)
    merge_split(train_ds, data_root / "train", img_format, args.limit_train, args.prefix)

    print("Streaming HF 'validation' split -> merging into data/test ...")
    val_ds = load_dataset(args.repo_id, split="validation", streaming=True)
    merge_split(val_ds, data_root / "test", img_format, args.limit_test, args.prefix)

    print("\nMerge complete. New folder counts:")
    for split in ["train", "test"]:
        for cls in ["real", "fake"]:
            p = data_root / split / cls
            n = len(list(p.glob("*"))) if p.exists() else 0
            print(f"  {split}/{cls}: {n} files")


if __name__ == "__main__":
    main()