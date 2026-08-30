"""
merge_dataset.py - merge an external clean/dirty dataset folder into data/ using
an 80/20 stratified split, consistent with prepare_data.py.

Usage:
    python scripts/merge_dataset.py --source ~/Desktop/Downloads/safwan_ds/dataset
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def split(source_dir: str) -> None:
    rng = np.random.RandomState(7)
    total = 0
    for raw_dir, cls in (("clean", "clean"), ("dirty", "dirty")):
        folder = os.path.join(source_dir, raw_dir)
        if not os.path.isdir(folder):
            print(f"  skip {raw_dir} (not found)")
            continue
        imgs = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        rng.shuffle(imgs)
        n_tr = int(0.8 * len(imgs))
        for split_name, bucket in (("train", imgs[:n_tr]), ("test", imgs[n_tr:])):
            out = os.path.join(DATA_DIR, split_name, cls)
            os.makedirs(out, exist_ok=True)
            for img in bucket:
                shutil.copy2(os.path.join(folder, img), os.path.join(out, img))
        total += len(imgs)
        print(f"  {cls}: +{len(imgs)} images")
    print(f"Merged {total} images from {source_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder containing clean/ + dirty/")
    args = ap.parse_args()
    split(args.source)

    for split_name in ("train", "test"):
        for cls in ("clean", "dirty"):
            print(f"  data/{split_name}/{cls}: {len(os.listdir(os.path.join(DATA_DIR, split_name, cls)))}")
    print("Now re-extract features (delete Models/cache_features.npz first) and retrain.")


if __name__ == "__main__":
    main()