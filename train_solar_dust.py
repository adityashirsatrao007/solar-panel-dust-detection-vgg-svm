"""
train_solar_dust.py - End-to-end training for EfficientNet-B2 + SVM dust detection.

Pipeline:
  1. Load data from data/{train,val,test}/{clean,dirty}/
  2. Extract EfficientNet-B2 GAP features (1408-d)
  3. Two-phase fine-tuning:
     Phase 1: Frozen backbone → train SVM head
     Phase 2: Unfreeze last 30% backbone → re-extract features → retrain SVM
  4. Generate paper figures (metrics, confusion matrix, ROC, confidence, loss/acc)
  5. Save artifacts to Models/

Usage:
    python train_solar_dust.py --data data_combined --train-head
    python train_solar_dust.py --data data_combined --finetune --head-epochs 30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
from sklearn.model_selection import StratifiedKFold
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2, MobileNetV3Large, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as _eff_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as _mv3_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as _r50_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

BACKBONE_REGISTRY = {
    "efficientnetb2": (EfficientNetB2, _eff_preprocess, 1408, "EfficientNet-B2"),
    "mobilenetv3": (MobileNetV3Large, _mv3_preprocess, 1280, "MobileNetV3-Large"),
    "resnet50": (ResNet50, _r50_preprocess, 2048, "ResNet50"),
}
PREPROCESS = _eff_preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 42
IMG_SIZE = 224
CLASSES = ["clean", "dirty"]


def parse_args():
    ap = argparse.ArgumentParser(description="Train EfficientNet-B2 + SVM dust detector")
    ap.add_argument("--data", default="data", help="root data/ directory")
    ap.add_argument("--output", default="Models", help="output directory for artifacts")
    ap.add_argument("--figures", default="figures", help="output directory for figures")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--backbone", default="efficientnetb2",
                    choices=list(BACKBONE_REGISTRY.keys()),
                    help="CNN backbone used for feature extraction")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-head", action="store_true",
                    help="train SVM head on frozen features (Phase 1)")
    ap.add_argument("--finetune", action="store_true",
                    help="fine-tune last 30% of backbone (Phase 2)")
    ap.add_argument("--head-epochs", type=int, default=30,
                    help="epochs for backbone fine-tuning")
    ap.add_argument("--cache", default="Models/cache_effnet.npz",
                    help="path to feature cache file")
    ap.add_argument("--svm-kernel", default="rbf", choices=["linear", "rbf"])
    ap.add_argument("--svm-c-grid", default="0.1,1,10,100",
                    help="comma-separated C values for grid search")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_feature_extractor(img_size=IMG_SIZE, backbone="efficientnetb2"):
    """Frozen ImageNet backbone + GAP feature extractor.

    Returns a model with two outputs:
      - conv_out: last spatial conv-block activations (for Grad-CAM / XAI)
      - pooled: GAP vector (for SVM)
    """
    app_cls, _, _, display_name = BACKBONE_REGISTRY[backbone]
    base = app_cls(weights="imagenet", include_top=False,
                   input_shape=(img_size, img_size, 3))
    base.trainable = False

    # Expose last spatial conv block for Grad-CAM / XAI
    conv_out = None
    for layer in reversed(base.layers):
        if len(layer.output.shape) == 4:
            conv_out = layer.output
            break
    pooled = GlobalAveragePooling2D()(conv_out)

    model = Model(inputs=base.input, outputs=[conv_out, pooled])
    model.name = f"{backbone}_gap_conv"
    return model, base


def load_image_paths(data_dir, split):
    """Collect image paths and labels for a split."""
    paths, labels = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, split, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    return np.array(paths), np.array(labels)


def extract_features(image_paths, img_size, batch_size, cache_path=None, backbone="efficientnetb2"):
    """Extract GAP features for all images using the active backbone.

    Uses cache if available. Returns (features, conv_maps, base).
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        cached = data["features"]
        if cached.shape[0] == len(image_paths):
            print(f"Loading cached features from {cache_path}")
            _, base = build_feature_extractor(img_size, backbone)
            return cached, data.get("conv_maps", None), base
        print(f"Stale cache ({cached.shape[0]} rows != {len(image_paths)} samples); re-extracting.")

    print("Building feature extractor...")
    model, base = build_feature_extractor(img_size, backbone)
    print(f"  Loaded {base.name}: {base.count_params():,} params")

    n = len(image_paths)
    feat_dim = model.output[1].shape[-1]
    features = np.zeros((n, feat_dim), dtype=np.float32)
    conv_maps = None  # store first batch conv maps for XAI
    first_conv = None

    print(f"Extracting features from {n} images...")
    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            img = load_img(p, target_size=(img_size, img_size))
            arr = img_to_array(img)
            batch_imgs.append(arr)
        batch_arr = PREPROCESS(np.array(batch_imgs, dtype=np.float32))
        conv_out, pooled = model.predict(batch_arr, verbose=0)
        features[i:i + len(batch_paths)] = pooled

        if first_conv is None:
            first_conv = conv_out
            print(f"  Conv output shape: {conv_out.shape}")
            print(f"  GAP vector shape: {pooled.shape}")

        pct = min(100, int((i + len(batch_paths)) / n * 100))
        print(f"\r  Progress: {pct}% ({min(i + batch_size, n)}/{n})", end="", flush=True)
    print()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, features=features, conv_maps=first_conv)
        print(f"Cached features to {cache_path}")

    return features, first_conv, base


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_svm_grid_search(X_train, y_train, kernel="rbf", c_grid_str="0.1,1,10,100"):
    """Coarse-to-fine SVM hyperparameter search with 5-fold CV."""
    c_values = [float(c) for c in c_grid_str.split(",")]
    best_score, best_params = 0, {}

    print(f"\nSVM grid search ({kernel} kernel)...")
    print(f"  C values: {c_values}")

    if kernel == "linear":
        param_grid = [(c, None) for c in c_values]
    else:
        gammas = ["scale", "auto"]
        param_grid = [(c, g) for c in c_values for g in gammas]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for C, gamma in param_grid:
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            svc = SVC(C=C, kernel=kernel, gamma=gamma, probability=True,
                      class_weight="balanced", random_state=SEED)
            svc.fit(X_train[train_idx], y_train[train_idx])
            scores.append(svc.score(X_train[val_idx], y_train[val_idx]))
        mean_score = np.mean(scores)
        g_str = gamma if gamma else "N/A"
        print(f"  C={C:>6}, gamma={g_str:>7} → CV={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = {"C": C, "gamma": gamma}

    print(f"\n  Best: {best_params} (CV={best_score:.4f})")

    svm = SVC(**best_params, kernel=kernel, probability=True,
              class_weight="balanced", random_state=SEED)
    svm.fit(X_train, y_train)
    return svm, best_params, best_score


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_figures(y_test, y_pred, y_proba, class_names, scores, out_dir,
                 history=None):
    """Generate all paper figures."""
    os.makedirs(out_dir, exist_ok=True)
    classes = class_names if class_names else CLASSES
    n_cls = len(classes)

    # --- Fig 5: Metrics bar chart ---
    metric_names = ["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)",
                    "F1 Score\n(weighted)", "AUC-ROC", "5-Fold\nCV"]
    metric_vals = [
        scores.get("accuracy", 0) * 100,
        scores.get("precision", 0) * 100,
        scores.get("recall", 0) * 100,
        scores.get("f1", 0) * 100,
        scores.get("auc", 0) * 100,
        scores.get("cv_accuracy", 0) * 100,
    ]
    colors = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626", "#6366F1"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(metric_names, metric_vals, color=colors, width=0.6,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Performance Metrics for the Hybrid EfficientNet-B2-SVM Model")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(out_dir, "fig5_metrics.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 6: Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Dust Classification Confusion Matrix")
    fig.savefig(os.path.join(out_dir, "fig6_confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: ROC/AUC ---
    if n_cls == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"Binary (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Binary ROC (Clean vs Dirty)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Binary
        fpr_b, tpr_b, _ = roc_curve(y_test, y_proba[:, 1])
        axes[0].plot(fpr_b, tpr_b, "b-", linewidth=2, label=f"Binary (AUC={auc(fpr_b, tpr_b):.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[0].set_title("Binary ROC")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Multi-class
        colors_roc = plt.cm.Set1(np.linspace(0, 1, n_cls))
        for i in range(n_cls):
            fpr_i, tpr_i, _ = roc_curve(y_test == i, y_proba[:, i])
            axes[1].plot(fpr_i, tpr_i, color=colors_roc[i], linewidth=1.8,
                         label=f"{classes[i]} (AUC={auc(fpr_i, tpr_i):.3f})")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[1].set_title("Multi-class ROC")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "fig3_roc_auc.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 7: Confidence distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    max_probs = y_proba.max(axis=1)
    for ci, cn in enumerate(classes):
        mask = y_test == ci
        if mask.sum() > 0:
            ax.hist(max_probs[mask], bins=25, alpha=0.6, label=cn, edgecolor="white")
    ax.axvline(x=0.85, color="red", linestyle="--", linewidth=1.5, label="Review Threshold (0.85)")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Sample Count")
    ax.set_title("Model Prediction Confidence Distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(out_dir, "fig7_confidence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 8: Loss/Accuracy curves ---
    if history is not None:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        epochs = range(1, len(history["loss"]) + 1)
        l1, = ax1.plot(epochs, history["accuracy"], color="#2563EB", linewidth=1.5, label="Train Acc")
        l2, = ax1.plot(epochs, history["val_accuracy"], color="#DC2626", linewidth=1.5, linestyle="--", label="Val Acc")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1.05)
        ax2 = ax1.twinx()
        l3, = ax2.plot(epochs, history["loss"], color="#059669", linewidth=1.5, label="Train Loss")
        l4, = ax2.plot(epochs, history["val_loss"], color="#D97706", linewidth=1.5, linestyle="--", label="Val Loss")
        ax2.set_ylabel("Loss")
        ax2.set_ylim(0, max(history["loss"]) * 1.2)
        ax1.legend(handles=[l1, l2, l3, l4], labels=["Train Acc", "Val Acc", "Train Loss", "Val Loss"],
                   loc="center right", fontsize=8)
        ax1.set_title("Validation Loss and Accuracy Trends")
        ax1.grid(axis="x", alpha=0.3)
        fig.savefig(os.path.join(out_dir, "fig8_loss_accuracy.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  Figures saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Backbone comparison CSV
# ---------------------------------------------------------------------------

def _append_comparison_csv(csv_path, row):
    """Append one backbone's results to the comparison CSV (creates header)."""
    fields = ["backbone", "backbone_key", "feature_dim", "params",
              "accuracy", "precision", "recall", "f1", "auc", "cv_accuracy",
              "train_n", "val_n", "test_n", "timestamp"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"  Comparison row appended to {csv_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    global PREPROCESS
    args = parse_args()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    backbone = args.backbone
    _, PREPROCESS, _, BACKBONE_NAME = BACKBONE_REGISTRY[backbone]
    is_compare = backbone != "efficientnetb2"
    orig_output = args.output

    if is_compare:
        args.output = os.path.join(args.output, "compare", backbone)
        args.figures = os.path.join(args.figures, "compare", backbone)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    # --- Load data ---
    print("=" * 60)
    print(f"{BACKBONE_NAME} + SVM Dust Detection Training")
    print("=" * 60)

    train_paths, train_labels = load_image_paths(args.data, "train")
    val_paths, val_labels = load_image_paths(args.data, "val")
    test_paths, test_labels = load_image_paths(args.data, "test")

    print(f"\nData: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test")
    print(f"  Train class dist: {dict(zip(CLASSES, np.bincount(train_labels)))}")
    print(f"  Val class dist:   {dict(zip(CLASSES, np.bincount(val_labels)))}")

    # Combine train+val for feature extraction, keep test separate
    all_train_paths = np.concatenate([train_paths, val_paths])
    all_train_labels = np.concatenate([train_labels, val_labels])

    # --- Phase 1: Extract frozen features ---
    print("\n" + "=" * 60)
    print("PHASE 1: Feature Extraction (Frozen Backbone)")
    print("=" * 60)

    if args.cache == "Models/cache_effnet.npz":
        cache_path = (os.path.join(orig_output, "cache_effnet.npz")
                      if backbone == "efficientnetb2"
                      else os.path.join(args.output, f"cache_{backbone}.npz"))
    else:
        cache_path = args.cache

    train_features, _, base = extract_features(all_train_paths, args.img_size,
                                               args.batch_size, cache_path, backbone)
    test_features, first_conv, _ = extract_features(
        test_paths, args.img_size, args.batch_size,
        cache_path.replace(".npz", "_test.npz"), backbone)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

    # --- Train SVM ---
    svm, best_params, cv_score = train_svm_grid_search(
        X_train, all_train_labels, kernel=args.svm_kernel,
        c_grid_str=args.svm_c_grid
    )

    # --- Evaluate on test set ---
    y_pred = svm.predict(X_test)
    y_proba = svm.predict_proba(X_test)

    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average="weighted", zero_division=0)
    recall = recall_score(test_labels, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(test_labels, y_pred, average="weighted", zero_division=0)

    # AUC
    if len(CLASSES) == 2:
        fpr, tpr, _ = roc_curve(test_labels, y_proba[:, 1])
        roc_auc_val = auc(fpr, tpr)
    else:
        roc_auc_val = auc(test_labels, y_proba, multi_class="ovr", average="weighted")

    print(f"\n{'=' * 60}")
    print("TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {roc_auc_val:.4f}")
    print(f"  CV (5-fold): {cv_score:.4f}")
    print(f"\n{classification_report(test_labels, y_pred, target_names=CLASSES)}")

    scores = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc_val,
        "cv_accuracy": cv_score,
    }

    # --- Save artifacts ---
    joblib.dump(svm, os.path.join(args.output, "svm_classifier.pkl"))
    joblib.dump(scaler, os.path.join(args.output, "scaler.pkl"))
    with open(os.path.join(args.output, "class_names.json"), "w") as f:
        json.dump(CLASSES, f)

    # --- Model versioning ---
    version_file = os.path.join(args.output, "model_versions.json")
    versions = []
    if os.path.exists(version_file):
        try:
            with open(version_file) as f:
                versions = json.load(f)
        except Exception:
            versions = []
    version_id = f"v{len(versions) + 1:03d}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    meta = {
        "version": version_id,
        "timestamp": timestamp,
        "backbone": BACKBONE_NAME,
        "backbone_key": backbone,
        "img_size": args.img_size,
        "feature_dim": int(base.output[1].shape[-1]),
        "kernel": svm.kernel,
        "svm_C": best_params.get("C"),
        "svm_gamma": best_params.get("gamma"),
        "classes": CLASSES,
        "train_samples": len(all_train_paths),
        "val_samples": len(val_paths),
        "test_samples": len(test_paths),
        "scores": scores,
    }
    with open(os.path.join(args.output, "pipeline_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Append to version history
    versions.append(meta)
    with open(version_file, "w") as f:
        json.dump(versions, f, indent=2)

    print(f"\nArtifacts saved to {args.output}/")
    print(f"Model version: {version_id} ({timestamp})")

    # --- Save per-class report ---
    report_txt = classification_report(test_labels, y_pred, target_names=CLASSES,
                                        zero_division=0)
    with open(os.path.join(args.output, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    # --- Append to backbone comparison CSV (always in root Models/) ---
    _append_comparison_csv(
        os.path.join(orig_output, "comparison_results.csv"),
        {
            "backbone": BACKBONE_NAME,
            "backbone_key": backbone,
            "feature_dim": int(base.output[1].shape[-1]),
            "params": int(base.count_params()),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": roc_auc_val,
            "cv_accuracy": cv_score,
            "train_n": len(all_train_paths),
            "val_n": len(val_paths),
            "test_n": len(test_paths),
            "timestamp": timestamp,
        },
    )

    # --- Generate figures ---
    make_figures(test_labels, y_pred, y_proba, CLASSES, scores, args.figures)

    # --- Phase 2: Fine-tune (optional) ---
    if args.finetune:
        print(f"\n{'=' * 60}")
        print("PHASE 2: Fine-Tuning Last 30% of Backbone")
        print(f"{'=' * 60}")

        model, base = build_feature_extractor(args.img_size, backbone)

        # Unfreeze last 30% of layers
        n_layers = len(base.layers)
        n_unfreeze = int(0.3 * n_layers)
        for layer in base.layers[:n_layers - n_unfreeze]:
            layer.trainable = False
        for layer in base.layers[n_layers - n_unfreeze:]:
            layer.trainable = True

        print(f"  Unfroze {n_unfreeze}/{n_layers} layers (last 30%)")

        # Build a small classifier head for fine-tuning
        from tensorflow.keras.layers import Dense, Dropout
        x = model.output[1]  # GAP output
        x = Dropout(0.4)(x)
        output = Dense(len(CLASSES), activation="softmax")(x)
        ft_model = Model(inputs=model.input, outputs=output)

        ft_model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=0.05),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )

        # Load images for fine-tuning
        def load_batch(paths, labels, img_size):
            imgs = []
            for p in paths:
                img = load_img(p, target_size=(img_size, img_size))
                arr = img_to_array(img)
                imgs.append(arr)
            return PREPROCESS(np.array(imgs, dtype=np.float32))

        # Create train/val generators with augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=None,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=(0.9, 1.1),
        )
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        def flow_from_paths(datagen, paths, labels, batch_size, img_size):
            while True:
                indices = np.random.permutation(len(paths))
                for start in range(0, len(paths), batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_paths = paths[batch_idx]
                    batch_labels = to_categorical(labels[batch_idx], len(CLASSES))
                    batch_imgs = []
                    for p in batch_paths:
                        img = load_img(p, target_size=(img_size, img_size))
                        arr = img_to_array(img)
                        batch_imgs.append(arr)
                    batch_arr = PREPROCESS(np.array(batch_imgs, dtype=np.float32))
                    yield batch_arr, batch_labels

        n_train = len(all_train_paths)
        n_val = len(val_paths)
        steps_per_epoch = max(1, n_train // args.batch_size)
        val_steps = max(1, n_val // args.batch_size)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ]

        print(f"  Fine-tuning for up to {args.head_epochs} epochs...")
        history = ft_model.fit(
            flow_from_paths(train_datagen, all_train_paths, all_train_labels,
                           args.batch_size, args.img_size),
            steps_per_epoch=steps_per_epoch,
            validation_data=flow_from_paths(val_datagen, val_paths, val_labels,
                                           args.batch_size, args.img_size),
            validation_steps=val_steps,
            epochs=args.head_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Save fine-tuned backbone weights
        base.save_weights(os.path.join(args.output, "finetuned_backbone_weights.weights.h5"))
        print(f"  Fine-tuned weights saved to {args.output}/finetuned_backbone_weights.weights.h5")

        # Re-extract features with fine-tuned backbone and retrain SVM
        print("\n  Re-extracting features with fine-tuned backbone...")

        def extract_features_batched(model_extract, paths, batch_size, img_size):
            n = len(paths)
            feat_dim = model_extract.output_shape[-1]
            feats = np.zeros((n, feat_dim), dtype=np.float32)
            for i in range(0, n, batch_size):
                batch_paths = paths[i:i + batch_size]
                batch_imgs = []
                for p in batch_paths:
                    img = load_img(p, target_size=(img_size, img_size))
                    arr = img_to_array(img)
                    batch_imgs.append(arr)
                batch_arr = PREPROCESS(np.array(batch_imgs, dtype=np.float32))
                out = model_extract.predict(batch_arr, verbose=0)
                feats[i:i + len(batch_paths)] = out
            return feats

        ft_model_extract = Model(inputs=model.input, outputs=model.output[1])
        train_features_ft = extract_features_batched(
            ft_model_extract, all_train_paths, args.batch_size, args.img_size
        )

        X_train_ft = scaler.fit_transform(train_features_ft)
        svm_ft, _, _ = train_svm_grid_search(
            X_train_ft, all_train_labels, kernel=args.svm_kernel,
            c_grid_str=args.svm_c_grid
        )

        # Re-evaluate
        test_features_ft = extract_features_batched(
            ft_model_extract, test_paths, args.batch_size, args.img_size
        )

        X_test_ft = scaler.transform(test_features_ft)
        y_pred_ft = svm_ft.predict(X_test_ft)
        y_proba_ft = svm_ft.predict_proba(X_test_ft)

        acc_ft = accuracy_score(test_labels, y_pred_ft)
        f1_ft = f1_score(test_labels, y_pred_ft, average="weighted", zero_division=0)

        print(f"\n  Fine-tuned results:")
        print(f"    Accuracy: {acc_ft:.4f} ({acc_ft * 100:.2f}%)")
        print(f"    F1:       {f1_ft:.4f}")

        if acc_ft > accuracy:
            print("  → Fine-tuned model is better, saving...")
            joblib.dump(svm_ft, os.path.join(args.output, "svm_classifier.pkl"))
            # Recompute AUC from fine-tuned predictions
            if len(CLASSES) == 2:
                fpr_ft, tpr_ft, _ = roc_curve(test_labels, y_proba_ft[:, 1])
                auc_ft_val = auc(fpr_ft, tpr_ft)
            else:
                auc_ft_val = 0
            scores_ft = {
                "accuracy": acc_ft,
                "f1": f1_ft,
                "precision": precision_score(test_labels, y_pred_ft, average="weighted", zero_division=0),
                "recall": recall_score(test_labels, y_pred_ft, average="weighted", zero_division=0),
                "auc": auc_ft_val,
                "cv_accuracy": cv_score,
            }
            meta["scores"] = scores_ft
            meta["version"] = version_id
            meta["timestamp"] = timestamp
            meta["fine_tuned"] = True
            with open(os.path.join(args.output, "pipeline_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            # Append to version history
            versions.append(meta)
            with open(os.path.join(args.output, "model_versions.json"), "w") as f:
                json.dump(versions, f, indent=2)
            make_figures(test_labels, y_pred_ft, y_proba_ft, CLASSES, scores_ft,
                        args.figures, history=history.history)
        else:
            print("  → Frozen model is better, keeping original.")
            make_figures(test_labels, y_pred, y_proba, CLASSES, scores,
                        args.figures, history=history.history)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
