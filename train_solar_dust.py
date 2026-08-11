"""
train_solar_dust.py - End-to-end training for the VGG16 + SVM dust detection model.

Pipeline:
  1. Load a directory dataset (train/ and test/ subfolders per class).
  2. VGG16 (ImageNet, frozen) Global-Average-Pooled feature extraction.
  3. StandardScaler + Support Vector Machine with a coarse-to-fine, early-pruned
     search over (C, gamma).
  4. Optional small calibration head (Dense softmax on the standardized vector)
     trained with early stopping to produce validation loss/accuracy curves and
     robust confidence estimates.
  5. Saves Models/svm_classifier.pkl, Models/scaler.pkl, Models/class_names.json
     and generates paper figures (confusion matrix, ROC/AUC, metrics, confidence).

Usage:
    python train_solar_dust.py --data /path/to/dataset
    python train_solar_dust.py --data ./dataset --kernel rbf --train-head
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

IMG_SIZE = 128
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Train VGG16+SVM dust classifier")
    p.add_argument("--data", required=True, help="root with train/ and test/ folders")
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--kernel", default="linear", choices=["linear", "rbf"])
    p.add_argument("--svm-c", default="0.1,1,10,100")
    p.add_argument("--gamma", default=None, help="comma list for RBF; default scale")
    p.add_argument("--train-head", action="store_true",
                   help="train a small Dense calibration head and save curves figure")
    p.add_argument("--head-epochs", type=int, default=60)
    p.add_argument("--cache", default="cache_features.npz", help="dump/load features")
    p.add_argument("--out", default="Models")
    p.add_argument("--figures", default=".")
    return p.parse_args()


def extract_features(data_dir, args):
    """Extract GAP features + labels for a split."""
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    )
    if not class_names:
        raise RuntimeError(f"No class folders found under {data_dir}")

    base = VGG16(weights="imagenet", include_top=False, input_shape=(args.img_size, args.img_size, 3))
    fe = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))

    features, labels = [], []
    for cls_idx, name in enumerate(class_names):
        folder = os.path.join(data_dir, name)
        paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not paths:
            continue
        print(f"  [{name}] loading {len(paths)} images")
        for i in range(0, len(paths), args.batch_size):
            batch_paths = paths[i:i + args.batch_size]
            batch = np.zeros((len(batch_paths), args.img_size, args.img_size, 3), dtype=np.float32)
            for j, p in enumerate(batch_paths):
                batch[j] = img_to_array(load_img(p, target_size=(args.img_size, args.img_size))) / 255.0
            feats = fe.predict(batch, verbose=0)
            features.append(feats)
            labels.append(np.full(len(batch_paths), cls_idx, dtype=int))
        _ = cls_idx
    return np.vstack(features), np.concatenate(labels), class_names


def make_figures(y_test, y_pred, y_proba, class_names, scores, out_dir,
                 fsx=None, fsy=None, head=None):
    """Generate the paper figures (Fig 5, 6, 7, 3, and optionally 8)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (auc, confusion_matrix, roc_curve)

    os.makedirs(out_dir, exist_ok=True)

    # Fig. 5 - performance metrics bar chart
    names = list(scores.keys())
    vals = list(scores.values())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(names, vals, color=["#4c78a8", "#e45756", "#54a24b", "#f2a900"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Fig. 5 - Performance Metrics for the Hybrid VGG16-SVM Model")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig5_metrics.png"), dpi=200); plt.close(fig)

    # Fig. 6 - confusion matrix
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Fig. 6 - Dust Classification Confusion Matrix")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig6_confusion_matrix.png"), dpi=200); plt.close(fig)

    # Fig. 3 - ROC / AUC
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if class_names.__len__() == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        ax.plot(fpr, tpr, color="#e45756", lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        from sklearn.preprocessing import label_binarize
        yb = label_binarize(y_test, classes=np.arange(len(class_names)))
        for k in range(len(class_names)):
            fpr, tpr, _ = roc_curve(yb[:, k], y_proba[:, k])
            ax.plot(fpr, tpr, lw=1.8, label=f"{class_names[k]} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Fig. 3 - ROC & AUC Curves"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig3_roc_auc.png"), dpi=200); plt.close(fig)

    # Fig. 7 - confidence distribution
    conf = y_proba.max(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(0.4, 1.0, 8)
    ax.hist(conf, bins=bins, color="#54a24b", edgecolor="white")
    ax.axvline(0.85, color="#e45756", ls="--", label="review threshold")
    ax.set_xlabel("Max decision probability"); ax.set_ylabel("Samples")
    ax.set_title("Fig. 7 - Model Prediction Confidence Distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig7_confidence.png"), dpi=200); plt.close(fig)

    # Fig. 8 - optional head curves
    if head is not None:
        _h = head
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(_h["accuracy"], color="#4c78a8", label="train acc")
        ax1.plot(_h["val_accuracy"], color="#e45756", label="val acc")
        ax1.set_ylabel("Accuracy")
        ax2 = ax1.twinx()
        ax2.plot(_h["loss"], color="#54a24b", ls="--", label="train loss")
        ax2.plot(_h["val_loss"], color="#f2a900", ls="--", label="val loss")
        ax2.set_ylabel("Loss")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right")
        ax1.set_xlabel("Epoch"); ax1.set_title("Fig. 8 - Validation Loss and Accuracy Trends")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig8_loss_accuracy.png"), dpi=200); plt.close(fig)


def main():
    args = parse_args()
    t0 = time.time()
    model_dir = args.out
    os.makedirs(model_dir, exist_ok=True)

    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    np.random.seed(SEED)

    train_dir = os.path.join(args.data, "train")
    test_dir = os.path.join(args.data, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise SystemExit("--data must contain train/ and test/ directories")

    # 0) cache features if present
    cache_path = os.path.join(model_dir, args.cache)
    if os.path.exists(cache_path):
        print("Loading cached features...")
        d = np.load(cache_path)
        Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
        class_names = json.loads(str(d["class_names"])) if "class_names" in d else sorted(set(ytr))
    else:
        print("Extracting VGG16 features...")
        Xtr, ytr, class_names = extract_features(train_dir, args)
        Xte, yte, _ = extract_features(test_dir, args)
        np.savez(cache_path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                 class_names=json.dumps(class_names))

    n_cls = len(class_names)
    print(f"train {Xtr.shape} | test {Xte.shape} | classes {class_names}")

    # 1) standardize
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    # 2) SVM coarse-to-fine early-pruned search
    C_grid = [float(c) for c in args.svm_c.split(",")]
    gamma_grid = list(map(float, args.gamma.split(","))) if args.gamma else None
    if args.kernel == "rbf" and not gamma_grid:
        gamma_grid = ["scale", "auto"]

    def _score(C, gamma):
        svm = SVC(kernel=args.kernel, C=C, gamma=gamma or "scale", probability=True,
                  class_weight="balanced", random_state=SEED)
        cvs = []
        for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xtr_s, ytr):
            svm.fit(Xtr_s[tr], ytr[tr])
            cvs.append(accuracy_score(ytr[va], svm.predict(Xtr_s[va])))
        return float(np.mean(cvs)), svm

    print("Coarse-to-fine early-pruned search...")
    candidates = [{"C": c, "gamma": g} for c in C_grid for g in (gamma_grid or [None])]
    best, best_score = None, -1.0
    patience, misses = 3, 0
    for cand in candidates:
        cv_score, _ = _score(cand["C"], cand["gamma"])
        print(f"  {cand} -> cv_acc {cv_score:.4f}")
        if cv_score > best_score:
            best_score, best = cv_score, cand
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                print("  early-pruned (plateau)")
                break
    if args.kernel == "rbf":
        base_g = best["gamma"]
        if base_g in (None, "auto", "scale"):
            base_g = 1.0 / Xtr_s.shape[1]
        for scale in [0.5, 2.0]:
            cand = {"C": best["C"], "gamma": float(base_g) * scale}
            cv_score, _ = _score(cand["C"], cand["gamma"])
            if cv_score > best_score:
                best, best_score = cand, cv_score
    print(f"Best hyperparams: {best} (cv {best_score:.4f})")

    # 3) final fit
    svm = SVC(kernel=args.kernel, C=best["C"], gamma=best["gamma"] or "scale",
              probability=True, class_weight="balanced", random_state=SEED)
    svm.fit(Xtr_s, ytr)
    y_pred = svm.predict(Xte_s)
    y_proba = svm.predict_proba(Xte_s)

    scores = {
        "accuracy": accuracy_score(yte, y_pred),
        "precision": precision_score(yte, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(yte, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(yte, y_pred, average="weighted", zero_division=0),
    }
    print("\nTest metrics:", {k: round(v, 4) for k, v in scores.items()})
    print("\nClassification report:\n", classification_report(yte, y_pred, zero_division=0))

    # 4) optional calibration head for curves + confidence
    head = None
    if args.train_head and n_cls >= 2:
        import tensorflow as tf
        from tensorflow.keras import layers
        xtr, xva = Xtr_s[: int(0.8 * len(Xtr_s))].astype(np.float32), \
        Xtr_s[int(0.8 * len(Xtr_s)):].astype(np.float32)
        ytrh = np.asarray(ytr[: len(xtr)], dtype=np.int32)
        yvah = np.asarray(ytr[len(xtr):], dtype=np.int32)
        m = tf.keras.Sequential([
            layers.Input(shape=(Xtr_s.shape[1],), dtype=tf.float32),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_cls, activation="softmax"),
        ])
        m.compile("adam", "sparse_categorical_crossentropy")
        class _AccHistory(tf.keras.callbacks.Callback):
            """Record train/val accuracy per epoch (works around the Keras 3.15
            'accuracy' metric dtype bug on some builds)."""

            def __init__(self, xtr_, ytr_, xva_, yva_):
                super().__init__()
                self.ep_acc, self.ep_val_acc = [], []
                self._x, self._y = xtr_, ytr_
                self._xv, self._yv = xva_, yva_

            def on_epoch_end(self, epoch, logs=None):
                tr_pred = np.argmax(self.model.predict(self._x, verbose=0), axis=1)
                self.ep_acc.append(float(np.mean(tr_pred == self._y)))
                va_pred = np.argmax(self.model.predict(self._xv, verbose=0), axis=1)
                self.ep_val_acc.append(float(np.mean(va_pred == self._yv)))

            def history(self):
                return {"accuracy": self.ep_acc, "val_accuracy": self.ep_val_acc}

        acc_hook = _AccHistory(xtr, ytrh, xva, yvah)
        head = m.fit(xtr, ytrh, epochs=args.head_epochs, batch_size=32,
                     validation_data=(xva, yvah), verbose=0,
                     callbacks=[acc_hook,
                                tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)]).history
        head.update(acc_hook.history())
        m.save(os.path.join(model_dir, "calibration_head.h5"))
        m.save(os.path.join(model_dir, "calibration_head.h5"))
        print("Calibration head saved to", os.path.join(model_dir, "calibration_head.h5"))

    # 5) persist artifacts
    joblib.dump(svm, os.path.join(model_dir, "svm_classifier.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    with open(os.path.join(model_dir, "class_names.json"), "w", encoding="utf-8") as fh:
        json.dump(class_names, fh)
    meta = {
        "img_size": args.img_size,
        "kernel": args.kernel,
        "svm": best,
        "classes": class_names,
        "train_samples": int(len(ytr)),
        "test_samples": int(len(yte)),
        "scores": scores,
    }
    with open(os.path.join(model_dir, "pipeline_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("Artifacts saved under", model_dir)

    # 6) figures
    make_figures(yte, y_pred, y_proba, class_names, scores, args.figures, head=head)
    print(f"Figures saved under {args.figures}   [{(time.time() - t0) / 60:.1f} min total]")


if __name__ == "__main__":
    main()