"""Pool all binary clean/dirty sources into one stratified 80/10/10 split.

Sources:
  - data/{train,val,test}/{clean,dirty}            (842 images, original)
  - datasets_raw/dataset3/Detect_solar_dust/{Clean,Dusty}   (2562 images)
  - datasets_raw/dataset2/Faulty_solar_panel/{Clean,Dusty}  (subset, binary only)

Output: data_combined/{train,val,test}/{clean,dirty}/
"""
import os
import shutil
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [
    ("data", None),  # already split into train/val/test with clean/dirty
    (os.path.join("datasets_raw", "dataset3", "Detect_solar_dust"),
     {"Clean": "clean", "Dusty": "dirty"}),
    (os.path.join("datasets_raw", "dataset2", "Faulty_solar_panel"),
     {"Clean": "clean", "Dusty": "dirty"}),
]
OUT = os.path.join(ROOT, "data_combined")
SEED = 42
SPLITS = (0.8, 0.1, 0.1)


def gather():
    items = []
    base = os.path.join(ROOT, SOURCES[0][0])
    for split in ("train", "val", "test"):
        for cls in ("clean", "dirty"):
            d = os.path.join(base, split, cls)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        items.append((os.path.join(d, f), cls))
    for src, mapping in SOURCES[1:]:
        for raw, label in mapping.items():
            d = os.path.join(ROOT, src, raw)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        items.append((os.path.join(d, f), label))
    return items


def main():
    random.seed(SEED)
    items = gather()
    bylabel = defaultdict(list)
    for p, l in items:
        bylabel[l].append(p)

    if os.path.exists(OUT):
        shutil.rmtree(OUT)

    counts = {}
    for label, paths in bylabel.items():
        random.shuffle(paths)
        n = len(paths)
        ntr = int(n * SPLITS[0])
        nval = int(n * SPLITS[1])
        split_for = (["train"] * ntr) + (["val"] * nval) + (["test"] * (n - ntr - nval))
        for p, sp in zip(paths, split_for):
            dst = os.path.join(OUT, sp, label, os.path.basename(p))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(p, dst)
            counts[(sp, label)] = counts.get((sp, label), 0) + 1

    print("Created", OUT)
    for k in sorted(counts):
        print("  ", k, counts[k])
    print("TOTAL", sum(counts.values()))


if __name__ == "__main__":
    main()
