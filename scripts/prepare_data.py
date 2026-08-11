"""
prepare_data.py - Downloads and structures the solar panel soiling dataset.

Automatically downloads the Kaggle "Solar Photovoltaics Panel for Dust Detection"
dataset via kagglehub when Kaggle credentials are configured, and copies the
images into the expected layout:

    data/
      train/clean/*.jpg
      train/dirty/*.jpg
      test/clean/*.jpg
      test/dirty/*.jpg

Usage:
    python scripts/prepare_data.py             # auto download via kagglehub
    python scripts/prepare_data.py --manual    # interactive folder guidance

If kagglehub is unavailable or unauthenticated, download the dataset manually
from the URLs printed below and re-run this script pointed at the archive
folder with --source /path/to/downloaded/folder.
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


def auto_download(archive: str) -> None:
    import kagglehub

    print(f"Downloading {archive} ...")
    path = kagglehub.dataset_download(archive)
    print(f"Downloaded to {path}")
    ds = path if os.path.isdir(path) else os.path.dirname(path)
    # infer label folders
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
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    if args.source:
        manual_setup(args.source)
    elif args.manual:
        raise SystemExit("--manual requires --source /path/to/dataset")
    else:
        try:
            auto_download(DEFAULT_ARCHIVE)
        except ImportError:
            raise SystemExit(
                "kagglehub not installed - run: pip install kagglehub\n"
                "or download the dataset manually and re-run with --source."
            )

    total = 0
    for split in ("train", "test"):
        for cls in ("clean", "dirty"):
            folder = os.path.join(DATA_DIR, split, cls)
            n = len([f for f in os.listdir(folder)] if os.path.isdir(folder) else [])
            total += n
            print(f"  data/{split}/{cls}: {n} images")
    print(f"Total images: {total}")


if __name__ == "__main__":
    main()