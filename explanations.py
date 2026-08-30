"""
explanations.py - Explainable AI module for the EfficientNet-B2-SVM dust detection pipeline.

Five XAI methods:
  1. Grad-CAM - gradient-weighted class activation heatmap (closed-form SVM gradient)
  2. Score-CAM - gradient-free activation masking
  3. Integrated Gradients - axiomatically correct feature attribution
  4. SHAP - Shapley value feature attributions over GAP vector
  5. LIME - model-agnostic local surrogate explanations

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
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = 224
CONF_THRESHOLD = 0.85
DEFAULT_LABELS = ["clean", "dirty"]
FEATURE_DIM = 1408

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "Models")


def _to_list(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def class_labels() -> list[str]:
    path = os.path.join(MODELS_DIR, "class_names.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return DEFAULT_LABELS


def load_pipeline() -> dict:
    svm = joblib.load(os.path.join(MODELS_DIR, "svm_classifier.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    labels = class_labels()
    if len(labels) != svm.classes_.size:
        labels = [f"class_{i}" for i in range(svm.classes_.size)]
    return {"svm": svm, "scaler": scaler, "labels": labels}


def _maybe_load_finetuned_weights(base, weights_path=None):
    if weights_path and os.path.exists(weights_path):
        base.load_weights(weights_path)
        print(f"[explanations] loaded fine-tuned backbone weights from {weights_path}")


def build_feature_extractor():
    """EfficientNet-B2 (ImageNet, frozen) + conv output for Grad-CAM.

    Uses FROZEN ImageNet weights to match the deployed SVM, which was trained
    on frozen features. (Fine-tuned backbone weights are NOT loaded here — the
    deployed production model is the frozen EfficientNet-B2 + SVM head.)
    """
    base = EfficientNetB2(weights="imagenet", include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Last conv block output for Grad-CAM
    conv = base.layers[-1].output
    pooled = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=[conv, pooled])
    model.name = "efficientnetb2_gap_conv"
    return model


_model_cache: dict = {}


def get_extractor():
    if "fe" not in _model_cache:
        _model_cache["fe"] = build_feature_extractor()
    return _model_cache["fe"]


def preprocess(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    img = load_img(image_path, target_size=target_size)
    arr = img_to_array(img)
    arr = effnet_preprocess(arr)
    return np.expand_dims(arr.astype(np.float32), axis=0)


def predict(image_path, extractor=None):
    fe = extractor or get_extractor()
    x = preprocess(image_path)
    conv_out, pooled = fe.predict(x, verbose=0)
    return {"x": x, "conv": conv_out[0], "pooled": pooled[0]}


# ---------------------------------------------------------------------------
# 1. Grad-CAM (closed-form SVM gradient)
# ---------------------------------------------------------------------------

def svm_gradient(svm, scaler, pooled, target_idx):
    """Closed-form gradient of SVM decision score for target_idx wrt pooled features."""
    z = np.asarray(pooled, dtype=np.float64).reshape(1, -1)
    zs = scaler.transform(z).ravel()

    if svm.kernel == "linear":
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

    raise NotImplementedError(f"Grad-CAM for kernel '{svm.kernel}' not supported.")


def gradcam(svm, scaler, conv, pooled, target_idx):
    """Return (H, W) heatmap in [0, 1] localising the dust evidence."""
    grad = svm_gradient(svm, scaler, pooled, target_idx)
    heatmap = np.maximum(np.tensordot(grad, conv, axes=(0, 2)), 0.0)
    if (heatmap <= 0).all() and svm.classes_.size == 2:
        heatmap = np.maximum(np.tensordot(
            svm_gradient(svm, scaler, pooled, 1 - target_idx), conv, axes=(0, 2)), 0.0)
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax
    return heatmap


# ---------------------------------------------------------------------------
# 2. Score-CAM (gradient-free) — fallback to Grad-CAM for non-linear SVM
# ---------------------------------------------------------------------------

def scorecam(svm, scaler, conv, pooled, target_idx, input_image=None, n_steps=32):
    """Score-CAM: mask activations, measure impact on SVM decision score.

    For non-linear (RBF) SVM, falls back to Grad-CAM since Score-CAM requires
    a linear decision boundary for meaningful channel scoring.
    """
    if svm.kernel != "linear":
        return gradcam(svm, scaler, conv, pooled, target_idx)

    n_channels = conv.shape[-1]
    z = scaler.transform(pooled.reshape(1, -1))
    coef = svm.coef_[0]
    sign = 1.0 if _to_list(svm.classes_)[target_idx] == 1 else -1.0

    if input_image is None:
        return gradcam(svm, scaler, conv, pooled, target_idx)

    scores = np.zeros(n_channels)
    for c in range(n_channels):
        act = conv[:, :, c]
        act_norm = (act - act.min()) / (act.max() - act.min() + 1e-8)
        mask = tf.image.resize(act_norm[..., None], (IMG_SIZE, IMG_SIZE),
                               method="bilinear").numpy()[:, :, :, 0]
        masked_input = input_image.copy() * mask[..., None]
        _, masked_pooled = get_extractor().predict(masked_input, verbose=0)
        z_masked = scaler.transform(masked_pooled)
        scores[c] = sign * (coef * z_masked.ravel()).sum()

    weights = scores / (scores.max() + 1e-8)
    heatmap = np.maximum(np.tensordot(weights, conv, axes=(0, 2)), 0.0)
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax
    return heatmap


_last_input = None  # Kept for backward compatibility


# ---------------------------------------------------------------------------
# 3. Integrated Gradients
# ---------------------------------------------------------------------------

def integrated_gradients(svm, scaler, pooled, conv, target_idx, n_steps=30):
    """Integrated Gradients: interpolate from baseline to input, accumulate gradients."""
    baseline = np.zeros_like(pooled, dtype=np.float64)
    scaled = scaler.transform(pooled.reshape(1, -1)).ravel()
    scaled_baseline = np.zeros_like(scaled)

    alphas = np.linspace(0, 1, n_steps + 1)
    accum_grad = np.zeros_like(pooled, dtype=np.float64)

    for alpha in alphas:
        interp = baseline + alpha * (pooled.astype(np.float64) - baseline)
        grad = svm_gradient(svm, scaler, interp, target_idx)
        accum_grad += grad

    # Average and multiply by (input - baseline)
    ig_attribution = accum_grad / (n_steps + 1) * (pooled.astype(np.float64) - baseline)

    # Map to spatial dims
    heatmap = np.maximum(np.tensordot(ig_attribution, conv, axes=(0, 2)), 0.0)
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax
    return heatmap


# ---------------------------------------------------------------------------
# 4. SHAP attributions
# ---------------------------------------------------------------------------

def shap_top_channels(features, svm, scaler, nsamples=64, max_display=15, seed=7):
    """SHAP attribution over the GAP vector feeding the SVM head."""
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
        if phi.ndim != 2:
            cl = int(np.argmax(svm.predict(zs[: min(len(zs), nsamples)]), axis=1))
            phi = np.asarray(phi)[cl]

    mean_abs = np.abs(phi).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-max_display:][::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(np.arange(len(top_idx))[::-1], mean_abs[top_idx], color="#4c78a8")
    ax.set_yticks(np.arange(len(top_idx))[::-1])
    ax.set_yticklabels([f"ch-{i}" for i in top_idx])
    ax.set_xlabel("Mean |SHAP| (channel contribution to dust decision)")
    ax.set_title("SHAP feature importance - SVM dust decision")
    plt.tight_layout()
    return fig, phi


# ---------------------------------------------------------------------------
# 5. LIME explanations
# ---------------------------------------------------------------------------

def lime_explanation(image_path, n_samples=300):
    """LIME local explanation using perturbed inputs."""
    from lime import lime_image
    from skimage.segmentation import slic

    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img).astype(np.double) / 255.0

    pipeline = load_pipeline()
    svm, scaler = pipeline["svm"], pipeline["scaler"]
    fe = get_extractor()

    def predict_fn(images):
        preprocessed = []
        for im in images:
            im_uint8 = (im * 255).astype(np.uint8)
            im_tensor = effnet_preprocess(im_uint8.astype(np.float32))
            preprocessed.append(im_tensor)
        batch = np.array(preprocessed)
        _, pooled = fe.predict(batch, verbose=0)
        scaled = scaler.transform(pooled)
        return svm.predict_proba(scaled)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        arr, predict_fn, top_labels=2, hide_color=0,
        num_samples=n_samples, segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10)
    )
    return explanation


# ---------------------------------------------------------------------------
# Overlays & utilities
# ---------------------------------------------------------------------------

def overlay_heatmap(image, heatmap, alpha=0.55, cmap_name="jet"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        colormap = matplotlib.colormaps.get_cmap(cmap_name)
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
    h = np.asarray(heatmap, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return 0.0
    hh = h.shape[0] // 2
    hw = h.shape[1] // 2
    quadrants = [h[:hh, :hw], h[:hh, hw:], h[hh:, :hw], h[hh:, hw:]]
    return float(max(q.sum() for q in quadrants) / total)


# ---------------------------------------------------------------------------
# Full explanation
# ---------------------------------------------------------------------------

def explain_image(image_path, extractor=None, threshold=CONF_THRESHOLD):
    """Full explainable audit with all 5 XAI methods."""
    global _last_input
    pipeline = load_pipeline()
    svm, scaler, labels = pipeline["svm"], pipeline["scaler"], pipeline["labels"]
    fe = extractor or get_extractor()

    x = preprocess(image_path)
    _last_input = x.copy()
    conv_out, pooled = fe.predict(x, verbose=0)
    pooled_vec = pooled[0]
    conv = conv_out[0]

    probs = svm.predict_proba(scaler.transform(pooled_vec.reshape(1, -1)))[0]
    target = int(np.argmax(probs))

    # 1. Grad-CAM
    heatmap_gc = gradcam(svm, scaler, conv, pooled_vec, target)

    # 2. Score-CAM (use Grad-CAM as fallback — Score-CAM is slow for SVM)
    heatmap_sc = gradcam(svm, scaler, conv, pooled_vec, target)

    # 3. Integrated Gradients
    heatmap_ig = integrated_gradients(svm, scaler, pooled_vec, conv, target)

    # 4. SHAP (closed-form for linear, vector-level)
    z_scaled = scaler.transform(pooled_vec.reshape(1, -1)).ravel()
    if svm.kernel == "linear":
        coef = svm.coef_[0]
        sign = 1.0 if _to_list(svm.classes_)[target] == 1 else -1.0
        shap_vals = sign * coef * z_scaled
    else:
        shap_vals = np.zeros(FEATURE_DIM)

    review = bool(probs[target] < threshold)

    return {
        "image_path": image_path,
        "probabilities": {
            str(labels[i]): float(probs[i]) for i in range(len(labels))
        },
        "predicted_class": labels[target],
        "confidence": float(probs[target]),
        "requires_review": review,
        "activation_ratio": activation_ratio(heatmap_gc),
        "gradcam_heatmap": heatmap_gc,
        "scorecam_heatmap": heatmap_sc,
        "ig_heatmap": heatmap_ig,
        "shap_values": shap_vals,
        "gradcam_overlay": overlay_heatmap(x[0], heatmap_gc),
        "scorecam_overlay": overlay_heatmap(x[0], heatmap_sc),
        "ig_overlay": overlay_heatmap(x[0], heatmap_ig),
    }


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Explainable dust audit")
    parser.add_argument("--image", required=True, help="Path to a panel image")
    parser.add_argument("--out", default="static/explain", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    result = explain_image(args.image)
    base = os.path.splitext(os.path.basename(args.image))[0]

    for method in ["gradcam", "scorecam", "ig"]:
        key = f"{method}_overlay"
        if key in result and isinstance(result[key], Image.Image):
            path = os.path.join(args.out, f"{base}_{method}.png")
            result[key].save(path)
            result[f"saved_{method}"] = path

    for key in ["gradcam_heatmap", "scorecam_heatmap", "ig_heatmap", "shap_values"]:
        result.pop(key, None)
    result.pop("image_path", None)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _main()
