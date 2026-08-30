"""
make_xai_figures.py - Generates the explainable-AI paper figures from the
bundled sample images, so the repository ships reproducible XAI outputs:

  figures/fig9_gradcam.png      - Grad-CAM overlays (clean vs dirty)
  figures/fig10_shap.png        - linear-model SHAP channel importance

Run:  python scripts/make_xai_figures.py
"""
from __future__ import annotations

import glob
import os
import sys

import joblib
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import explanations

IMG_SIZE = 128
FIG_DIR = os.path.join(ROOT, "figures")


def sample_images():
    clean = sorted(glob.glob(os.path.join(ROOT, "static", "uploads", "Imgclean_*.jpg")))
    dirty = sorted(glob.glob(os.path.join(ROOT, "static", "uploads", "Imgdirty_*.jpg")))
    return clean, dirty


def fig9_gradcam(clean_paths, dirty_paths):
    fe = explanations.get_extractor()
    pl = explanations.load_pipeline()
    svm, scaler = pl["svm"], pl["scaler"]

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(11, 7))
    picks = [clean_paths[0], clean_paths[1], clean_paths[2], dirty_paths[0], dirty_paths[1], dirty_paths[2]]
    titles = ["Clean", "Clean", "Clean", "Dirty", "Dirty", "Dirty"]
    for ax, path, title in zip(axes.ravel(), picks, titles):
        out = explanations.predict(path, fe)
        probs = svm.predict_proba(scaler.transform(out["pooled"][None, :]))[0]
        target = int(np.argmax(probs))
        hm = explanations.gradcam(svm, scaler, out["conv"], out["pooled"], target)
        overlay = explanations.overlay_heatmap(out["x"][0], hm)
        ax.imshow(np.asarray(overlay))
        ax.set_title(f"{title} | P(dirty)={probs[1]:.2f}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Fig. 9 - Grad-CAM localization heatmaps (EfficientNet-B2 last conv block -> SVM)")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "fig9_gradcam.png"), dpi=200)
    plt.close(fig)
    print("wrote figures/fig9_gradcam.png")


def fig10_shap(pairs):
    fe = explanations.get_extractor()
    pl = explanations.load_pipeline()
    feat, labels = [], []
    for path, lab in pairs:
        pooled = explanations.predict(path, fe)["pooled"]
        feat.append(pooled)
        labels.append(lab)
    X = np.vstack(feat)
    fig, _ = explanations.shap_top_channels(X, pl["svm"], pl["scaler"], max_display=15)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "fig10_shap.png"), dpi=200)
    plt.close(fig)
    print("wrote figures/fig10_shap.png")


def main():
    clean, dirty = sample_images()
    if not clean or not dirty:
        raise SystemExit("No bundled Imgclean_*/Imgdirty_* samples found under static/uploads/")
    fig9_gradcam(clean, dirty)
    fig10_shap([(p, 0) for p in clean[:12]] + [(p, 1) for p in dirty[:11]])


if __name__ == "__main__":
    main()