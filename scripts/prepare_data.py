"""
prepare_data.py - Downloads and structures the solar panel soiling dataset.

Creates the expected layout with train/val/test splits:

    data/
      train/clean/*.jpg
      train/dirty/*.jpg
      val/clean/*.jpg
      val/dirty/*.jpg
      test/clean/*.jpg
      test/dirty/*.jpg

Usage:
    python scripts/prepare_data.py             # auto download via kagglehub
    python scripts/prepare_data.py --manual    # interactive folder guidance
    python scripts/prepare_data.py --resplit   # re-split existing data into train/val/test
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

DEFAULT_ARCHIVE = "safwanshamsir99/solar-photovoltaics-panell-for-dust-dectection"
ALTERNATES = [
    "hemanthsai7/solar-panel-dust-detection",
    "pythonafroz/solar-panel-images",
]

KNOWN_LABELS_OK = {"clean", "dirty", "dusty"}


def normalize_label(raw: str) -> str | None:
    r = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if r.endswith("_png") or r.endswith("_jpg"):
        r = r.rsplit("_", 1)[0]
    if r in ("clean", "cleaned"):
        return "clean"
    if r in ("dirty", "dust", "dusty", "dust_detection", "dusty_panel", "soiled"):
        return "dirty"
    return None


def collect(source_root: str, mapping: dict) -> None:
    """Walk a downloaded dataset folder and copy images into train/test split."""
    split = 0.8
    rng = np.random.RandomState(42)
    for cls, (dirn, dst) in mapping.items():
        folder = os.path.join(source_root, dirn)
        if not os.path.isdir(folder):
            continue
        imgs = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        rng.shuffle(imgs)
        n_tr = int(split * len(imgs))
        for dst_split, bucket in (("train", imgs[:n_tr]), ("test", imgs[n_tr:])):
            for img in bucket:
                out = os.path.join(DATA_DIR, dst_split, cls, img)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(os.path.join(folder, img), out)
        print(f"  {cls}: {len(imgs)} images")


def resplit_existing_data(val_ratio: float = 0.10) -> None:
    """Re-split existing train/test data into train/val/test.

    Moves val_ratio of training images from train/ into val/.
    """
    rng = np.random.RandomState(42)
    for cls in ("clean", "dirty"):
        train_dir = os.path.join(DATA_DIR, "train", cls)
        val_dir = os.path.join(DATA_DIR, "val", cls)
        os.makedirs(val_dir, exist_ok=True)

        if not os.path.isdir(train_dir):
            print(f"  Warning: {train_dir} not found, skipping")
            continue

        imgs = sorted(
            f for f in os.listdir(train_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        rng.shuffle(imgs)
        n_val = max(1, int(val_ratio * len(imgs)))
        val_imgs = imgs[:n_val]

        for img in val_imgs:
            src = os.path.join(train_dir, img)
            dst = os.path.join(val_dir, img)
            shutil.move(src, dst)

        n_remaining = len([f for f in os.listdir(train_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  {cls}: moved {n_val} to val, {n_remaining} remain in train")


def auto_download(archive: str) -> None:
    import kagglehub

    print(f"Downloading {archive} ...")
    path = kagglehub.dataset_download(archive)
    print(f"Downloaded to {path}")
    ds = path if os.path.isdir(path) else os.path.dirname(path)
    subdirs = sorted(
        d for d in os.listdir(ds)
        if os.path.isdir(os.path.join(ds, d))
    )
    mapping = {}
    for d in subdirs:
        lab = normalize_label(d)
        if lab:
            mapping.setdefault(lab, (d, lab))
    if len(mapping) < 2:
        raise SystemExit(
            "Could not infer clean/dirty folders - run with --manual and pass --source "
            "pointing at a folder that contains clean/ and dirty/ subfolders."
        )
    collect(ds, mapping)


def manual_setup(source: str) -> None:
    mapping = {}
    for d in sorted(os.listdir(source)):
        lab = normalize_label(d)
        if lab:
            mapping.setdefault(lab, (d, lab))
    if len(mapping) < 2:
        raise SystemExit(
            f"No clean/dirty folders found under {source}. Expected folders named "
            "'clean' and 'dirty'."
        )
    collect(source, mapping)


def main():
    ap = argparse.ArgumentParser(description="Prepare solar panel dust dataset")
    ap.add_argument("--source", help="path to an already-downloaded dataset folder")
    ap.add_argument("--manual", action="store_true",
                    help="do not auto-download; only structure --source")
    ap.add_argument("--resplit", action="store_true",
                    help="re-split existing train data into train/val/test")
    ap.add_argument("--val-ratio", type=float, default=0.10,
                    help="fraction of training data to move to val (default: 0.10)")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.resplit:
        print("Re-splitting existing data into train/val/test...")
        resplit_existing_data(args.val_ratio)
    elif args.source:
        manual_setup(args.source)
        print("\nNow re-splitting into train/val/test...")
        resplit_existing_data(args.val_ratio)
    elif args.manual:
        raise SystemExit("--manual requires --source /path/to/dataset")
    else:
        try:
            auto_download(DEFAULT_ARCHIVE)
            print("\nNow re-splitting into train/val/test...")
            resplit_existing_data(args.val_ratio)
        except ImportError:
            raise SystemExit(
                "kagglehub not installed - run: pip install kagglehub\n"
                "or download the dataset manually and re-run with --source."
            )

    total = 0
    for split in ("train", "val", "test"):
        for cls in ("clean", "dirty"):
            folder = os.path.join(DATA_DIR, split, cls)
            n = len([f for f in os.listdir(folder)] if os.path.isdir(folder) else [])
            total += n
            print(f"  data/{split}/{cls}: {n} images")
    print(f"Total images: {total}")


if __name__ == "__main__":
    main()
