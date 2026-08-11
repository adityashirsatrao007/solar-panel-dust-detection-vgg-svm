"""
explanations.py - Explainable AI module for the VGG16-SVM dust detection pipeline.

Provides three behaviours described in the paper:
  1. Grad-CAM localization heatmaps over the last convolutional block (block5_conv3).
     For a linear SVM the gradient of the decision score wrt the pooled features is
     simply coef_ / scale, which makes the localization solvable in closed form.
     For an RBF SVM we compute the analytic Jacobian of the RBF decision function.
  2. SHAP feature attributions over the 512-dimensional GAP vector that feeds the
     SVM head (offline/paper figure generation).
  3. Confidence-gated human review flag when the maximum decision probability drops
     below a threshold.

Usage:
    python explanations.py --image static/uploads/Imgclean_12_0.jpg
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = 128
CONF_THRESHOLD = 0.85
DEFAULT_LABELS = ["clean", "dirty"]


def _to_list(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def class_labels() -> list[str]:
    """Load the class names persisted by the training script, else infer count."""
    path = os.path.join("Models", "class_names.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return DEFAULT_LABELS


@tf.keras.utils.register_keras_serializable(package="dustxai")
def _identity(x):
    return tf.identity(x)


def load_pipeline() -> dict:
    """Return a dict with svm, scaler and class names."""
    svm = joblib.load(os.path.join("Models", "svm_classifier.pkl"))
    scaler = joblib.load(os.path.join("Models", "scaler.pkl"))
    labels = class_labels()
    if len(labels) != svm.classes_.size:
        labels = [f"class_{i}" for i in range(svm.classes_.size)]
    return {"svm": svm, "scaler": scaler, "labels": labels}


def build_feature_extractor():
    """VGG16 (imagenet, frozen) returning block5_conv3 feature maps + GAP vector."""
    base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    conv = base.get_layer("block5_conv3").output
    pooled = GlobalAveragePooling2D()(conv)
    model = Model(inputs=base.input, outputs=[conv, pooled])
    model.name = "vgg16_gap_conv"
    return model


_model_cache: dict = {}


def get_extractor():
    if "fe" not in _model_cache:
        _model_cache["fe"] = build_feature_extractor()
    return _model_cache["fe"]


def preprocess(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    img = load_img(image_path, target_size=target_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr.astype(np.float32), axis=0)


def predict(image_path, extractor=None):
    """Return normalized sample + conv maps + pooled features."""
    fe = extractor or get_extractor()
    x = preprocess(image_path)
    conv_out, pooled = fe.predict(x, verbose=0)
    return {"x": x, "conv": conv_out[0], "pooled": pooled[0]}


def svm_gradient(svm, scaler, pooled, target_idx):
    """
    Closed-form gradient of the SVM decision score for `target_idx` with respect
    to the pooled (pre-scaler) feature vector.
    - linear kernel: d(decision)/dz = coef_ / scale (Hardtanh 0-gradient guard).
    - rbf kernel:    analytic Jacobian via the support vectors.
    """
    z = np.asarray(pooled, dtype=np.float64).reshape(1, -1)
    zs = scaler.transform(z).ravel()

    if svm.kernel == "linear":
        # binary: coef_ row is class[1] vs class[0]; sign flips for class[0]
        coef = svm.coef_[0]
        sign = 1.0 if _to_list(svm.classes_)[target_idx] == 1 else -1.0
        grad = sign * coef / np.maximum(scaler.scale_, 1e-12)
        return grad

    if svm.kernel == "rbf":
        gamma = float(getattr(svm, "_gamma", svm.gamma) or (1.0 / zs.shape[0]))
        sv = svm.support_vectors_
        dual_lo = np.asarray(svm.dual_coef_)
        alpha_y = dual_lo[0] if dual_lo.ndim == 1 else dual_lo.sum(axis=0)
        k = np.exp(-gamma * ((zs[None, :] - sv) ** 2).sum(axis=1))
        const = np.asarray(alpha_y) * k
        grad = -2.0 * gamma * (const[:, None] * (zs[None, :] - sv)).sum(axis=0)
        return grad / np.maximum(scaler.scale_, 1e-12)

    raise NotImplementedError(f"Grad-CAM support for kernel '{svm.kernel}' is not available.")


def gradcam(svm, scaler, conv, pooled, target_idx):
    """Return a (H, W) heatmap in [0, 1] localising the dust evidence."""
    grad = svm_gradient(svm, scaler, pooled, target_idx)
    heatmap = np.maximum(np.tensordot(grad, conv, axes=(0, 2)), 0.0)
    if (heatmap <= 0).all() and svm.classes_.size == 2:
        # "absence" class: invert to the complementary class so the overlay
        # still highlights where surface texture evidence lives.
        heatmap = np.maximum(np.tensordot(
            svm_gradient(svm, scaler, pooled, 1 - target_idx), conv, axes=(0, 2)), 0.0)
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax
    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.55, cmap_name="jet"):
    """Blend `heatmap` over `image` (0-1 float array) and return an RGB PIL image."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        _cmap = getattr(matplotlib.colormaps, "get_cmap", None)
        if callable(_cmap):
            colormap = _cmap(cmap_name)
        else:
            colormap = matplotlib.cm.get_cmap(cmap_name) if hasattr(matplotlib.cm, "get_cmap") else None
    except Exception:
        colormap = None

    h_resized = np.asarray(Image.fromarray(np.uint8(255 * np.clip(heatmap, 0, 1))).resize(
        (image.shape[1], image.shape[0]), Image.BILINEAR)) / 255.0

    if colormap is not None:
        rgb = colormap(h_resized)[:, :, :3]
    else:
        rgb = np.stack([h_resized, h_resized * 0.3, 1 - h_resized], axis=-1)

    img_rgb = np.clip(image / 255.0, 0, 1) if image.max() > 1.0 else np.clip(image, 0, 1)
    overlay = (1 - alpha) * img_rgb + alpha * rgb
    return Image.fromarray(np.uint8(255 * np.clip(overlay, 0, 1)))


def activation_ratio(heatmap, threshold=0.5):
    """Spatial concentration of the localization mass.

    Returns the fraction of the total heatmap mass contained inside the single
    most-activated quadrant. A concentrated (high) value indicates dust evidence
    is localised; a low spread value indicates a broadly clean surface.
    """
    h = np.asarray(heatmap, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return 0.0
    hh = h.shape[0] // 2
    hw = h.shape[1] // 2
    quadrants = [h[:hh, :hw], h[:hh, hw:], h[hh:, :hw], h[hh:, hw:]]
    return float(max(q.sum() for q in quadrants) / total)


def explain_image(image_path, extractor=None, threshold=CONF_THRESHOLD):
    """
    Full explainable audit for one image:
    probs, label, Grad-CAM overlay ndarray, activation ratio, review flag.
    """
    pipeline = load_pipeline()
    svm, scaler, labels = pipeline["svm"], pipeline["scaler"], pipeline["labels"]
    out = predict(image_path, extractor)
    probs = svm.predict_proba(scaler.transform(out["pooled"][None, :]))[0]
    target = int(np.argmax(probs))

    heatmap = gradcam(svm, scaler, out["conv"], out["pooled"], target)
    review = bool(probs[target] < threshold)

    return {
        "image_path": image_path,
        "probabilities": {
            str(labels[i]): float(probs[i]) for i in range(len(labels))
        },
        "predicted_class": labels[target],
        "confidence": float(probs[target]),
        "requires_review": review,
        "activation_ratio": activation_ratio(heatmap),
        "heatmap": heatmap,
        "overlay": overlay_heatmap(out["x"][0], heatmap),
    }


def shap_top_channels(features, svm, scaler, nsamples=64, max_display=15, seed=7):
    """
    Offline SHAP attribution over the GAP vector feeding the SVM head.
    For the deployed linear model the attributions are computed exactly and in
    closed form (phi_j = coef_j * (z_j - E[z_j]) in scaled space). For RBF kernels
    we fall back to the KernelExplainer. Designed for offline paper-figure use.

    Returns: (matplotlib Figure, shap_values | None)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = np.asarray(features, dtype=np.float64)
    zs = scaler.transform(feats)
    bkg_mean = zs[: min(len(zs), 50)].mean(axis=0)

    if getattr(svm, "kernel", None) == "linear":
        coef = svm.coef_[0]
        phi = (zs - bkg_mean) * coef
    else:
        import shap

        explainer = shap.KernelExplainer(svm.predict_proba, zs[: min(len(zs), 50)])
        phi = np.asarray(explainer.shap_values(zs)[: min(len(zs), nsamples)])
        if phi.ndim != 2:  # multiclass -> argmax class slice
            cl = int(np.argmax(svm.predict(zs[: min(len(zs), nsamples)]), axis=1))
            phi = np.asarray(phi)[cl]

    mean_abs = np.abs(phi).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-max_display:][::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(np.arange(len(top_idx))[::-1], mean_abs[top_idx], color="#4c78a8")
    ax.set_yticks(np.arange(len(top_idx))[::-1])
    ax.set_yticklabels([f"ch-{i}" for i in top_idx])
    ax.set_xlabel("Mean |SHAP| (channel contribution to dust decision)")
    ax.set_title("Fig. 10 - SHAP feature importance - SVM dust decision")
    plt.tight_layout()
    return fig, phi


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Explainable dust audit")
    parser.add_argument("--image", required=True, help="Path to a panel image")
    parser.add_argument("--out", default="static/explain", help="Output folder")
    parser.add_argument("--shap", action="store_true",
                        help="Also run offline SHAP on Models feature dump (expert)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    result = explain_image(args.image)
    base = os.path.splitext(os.path.basename(args.image))[0]
    overlay_path = os.path.join(args.out, f"{base}_gradcam.png")
    result["overlay"].save(overlay_path)
    result.pop("overlay")
    result.pop("heatmap")
    result["saved_overlay"] = overlay_path
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _main()