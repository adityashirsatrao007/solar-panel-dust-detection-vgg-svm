import os
import sys
import json
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_solar_dust import extract_features, IMG_SIZE
import joblib

MODELS = "Models"
SVM = joblib.load(os.path.join(MODELS, "svm_classifier.pkl"))
SCALER = joblib.load(os.path.join(MODELS, "scaler.pkl"))


def class_subdirs(root):
    out = {}
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d):
            out[name.lower()] = d
    return out


def normalize_label(name):
    n = name.lower()
    if n == "clean":
        return "clean"
    if n in ("dusty", "dirty", "dust"):
        return "dirty"
    return None


def evaluate(root, batch=32):
    sub = class_subdirs(root)
    paths, gts = [], []
    skipped = []
    for cls_name, d in sub.items():
        lab = normalize_label(cls_name)
        if lab is None:
            skipped.append(cls_name)
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(d, f))
                gts.append(lab)
    if not paths:
        print(f"  ! no clean/dirty classes under {root}")
        return None
    if skipped:
        print(f"  (skipped non-binary subdirs: {skipped})")

    feats, _, _ = extract_features(paths, IMG_SIZE, batch, backbone="efficientnetb2")
    X = SCALER.transform(feats)
    pred = SVM.predict(X)
    proba = SVM.predict_proba(X)
    conf = proba.max(axis=1)
    label_to_int = {"clean": 0, "dirty": 1}
    y_true = np.array([label_to_int[g] for g in gts])
    correct = int(np.sum(y_true == pred))
    acc = correct / len(gts)

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    rep = classification_report(y_true, pred, labels=[0, 1],
                               target_names=["clean", "dirty"],
                               output_dict=True, zero_division=0)
    mean_conf = float(np.mean(conf))
    print(f"  images={len(gts)}  accuracy={acc*100:.2f}%  mean_conf={mean_conf:.3f}")
    print("  confusion rows=truth [clean, dirty]:")
    print(f"    clean -> pred clean:{cm[0][0]}, dirty:{cm[0][1]}")
    print(f"    dirty -> pred clean:{cm[1][0]}, dirty:{cm[1][1]}")
    wrong = int(cm[0][1] + cm[1][0])
    print(f"  misclassified: {wrong}")
    return {
        "dataset": root,
        "n": len(gts),
        "accuracy": round(acc, 4),
        "mean_confidence": round(mean_conf, 4),
        "misclassified": wrong,
        "confusion": cm.tolist(),
        "report": rep,
    }


if __name__ == "__main__":
    roots = sys.argv[1:] or [
        "datasets_raw/dataset3/Detect_solar_dust",
        "datasets_raw/dataset2/Faulty_solar_panel",
    ]
    results = []
    for r in roots:
        print(f"\n=== {r} ===")
        res = evaluate(r)
        if res:
            results.append(res)
    with open("external_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved external_eval_results.json")
