"""
Solar Panel Dust Detection — Streamlit App
===========================================
Hybrid EfficientNet-B2 + SVM pipeline with Grad-CAM explainability.
Deployed on Streamlit Community Cloud.
"""

import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import streamlit as st
import joblib
from PIL import Image

st.set_page_config(page_title="Solar Panel Dust Detection", page_icon="☀️", layout="centered")

# ── Model loading (single cached model for both predict + Grad-CAM) ─────────
@st.cache_resource(show_spinner="Loading model from Hugging Face Hub...")
def load_pipeline():
    from huggingface_hub import hf_hub_download
    svm_p = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "svm_classifier.pkl")
    sc_p  = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "scaler.pkl")
    cn_p  = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "class_names.json")
    meta_p = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "pipeline_meta.json")
    svm = joblib.load(svm_p)
    scaler = joblib.load(sc_p)
    with open(cn_p) as f:
        labels = json.load(f)
    with open(meta_p) as f:
        meta = json.load(f)
    return svm, scaler, labels, meta


@st.cache_resource(show_spinner="Loading EfficientNet-B2 backbone...")
def load_models():
    """Load ONE EfficientNet-B2 with two outputs: conv (for Grad-CAM) + pooled (for SVM)."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    conv_out = base.layers[-1].output          # (B, 7, 7, 1408)
    pooled_out = GlobalAveragePooling2D()(base.output)  # (B, 1408)
    full_model = Model(inputs=base.input, outputs=[conv_out, pooled_out])
    return full_model, preprocess_input


# ── Predict + Grad-CAM (single forward pass) ────────────────────────────────
def analyze(image: Image.Image):
    svm, scaler, labels, meta = load_pipeline()
    model, preprocess = load_models()

    img = image.convert("RGB").resize((224, 224))
    arr = preprocess(np.expand_dims(np.array(img, dtype=np.float32), 0))

    # Single forward pass → conv + pooled
    conv, pooled = model.predict(arr, verbose=0)
    pooled, conv = pooled[0], conv[0]

    # SVM prediction
    z = scaler.transform(pooled.reshape(1, -1))
    probs = svm.predict_proba(z)[0]
    idx = int(np.argmax(probs))
    label = labels[idx] if idx < len(labels) else str(idx)
    conf = float(probs.max())
    dirty_idx = labels.index("dirty") if "dirty" in labels else len(probs) - 1
    dustiness = float(probs[dirty_idx]) * 100

    # Grad-CAM via SVM gradient
    zs = scaler.transform(pooled.reshape(1, -1)).ravel()
    gamma = float(getattr(svm, "_gamma", svm.gamma) or (1.0 / zs.shape[0]))
    sv = svm.support_vectors_
    dual = np.asarray(svm.dual_coef_)
    alpha_y = dual[0] if dual.ndim == 1 else dual.sum(axis=0)
    k = np.exp(-gamma * ((zs[None, :] - sv) ** 2).sum(axis=1))
    grad = -2.0 * gamma * (alpha_y * k)[:, None] * (zs[None, :] - sv)
    grad = grad.sum(axis=0) / np.maximum(scaler.scale_, 1e-12)

    hm = np.maximum(np.tensordot(grad, conv, axes=(0, 2)), 0.0)
    hmax = hm.max()
    if hmax > 0:
        hm = hm / hmax

    # Build heatmap overlay
    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps.get_cmap("jet")
    hm_pil = Image.fromarray(np.uint8(255 * np.clip(hm, 0, 1))).resize(img.size, Image.BILINEAR)
    hm_arr = np.array(hm_pil).astype(np.float32) / 255.0
    rgb = cmap(hm_arr)[:, :, :3]
    orig = np.array(img).astype(np.float32) / 255.0
    overlay = (1 - 0.45) * orig + 0.45 * rgb
    overlay_img = Image.fromarray(np.uint8(255 * np.clip(overlay, 0, 1)))

    return label, conf, dustiness, probs, overlay_img, hm_pil


# ── UI ──────────────────────────────────────────────────────────────────────
st.title("☀️ Solar Panel Dust Detection")
st.caption("Hybrid EfficientNet-B2 + SVM · Explainable AI · "
           "[GitHub](https://github.com/adityashirsatrao007/solar-panel-dust-detection-vgg-svm)")

with st.sidebar:
    st.header("About")
    st.markdown("Upload a solar panel photo. The model classifies **clean** vs **dirty** and shows a Grad-CAM heatmap highlighting the dusty regions.")
    st.divider()
    st.markdown("**Architecture**")
    st.code("EfficientNet-B2 (frozen, ImageNet)\n  → GAP 1408-d\n  → RBF-SVM", language=None)
    st.divider()
    _, _, _, meta = load_pipeline()
    st.markdown("**Model metrics**")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{meta['scores']['accuracy']*100:.1f}%")
    c2.metric("AUC-ROC", f"{meta['scores']['auc']:.4f}")
    c1.metric("CV Acc", f"{meta['scores']['cv_accuracy']*100:.1f}%")
    c2.metric("Train", f"{meta['train_samples']:,}")
    st.divider()
    st.caption("v007 · 3,787-image merged corpus")

uploaded = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)

    with st.spinner("Analyzing image..."):
        label, conf, dustiness, probs, overlay_img, hm_img = analyze(img)

    # Result header
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if dustiness > 50:
            st.error(f"### 🔴 {label.upper()} — {dustiness:.1f}% dustiness")
        elif dustiness > 20:
            st.warning(f"### 🟡 {label.upper()} — {dustiness:.1f}% dustiness")
        else:
            st.success(f"### 🟢 {label.upper()} — {dustiness:.1f}% dustiness")
    with col_b:
        st.metric("Confidence", f"{conf*100:.1f}%")

    # Class probabilities
    st.markdown("**Class probabilities**")
    prob_cols = st.columns(2)
    for i, lbl in enumerate(["clean", "dirty"]):
        prob_cols[i].progress(probs[i], text=f"{lbl}: {probs[i]*100:.1f}%")

    st.divider()

    # Grad-CAM section
    st.subheader("🔍 Explainable AI — Grad-CAM")
    st.caption("Red/yellow regions are where the model looks to decide 'dirty'. Blue regions are considered clean.")

    cam_col1, cam_col2 = st.columns(2)
    with cam_col1:
        st.image(overlay_img, caption="Grad-CAM overlay on original", use_container_width=True)
    with cam_col2:
        st.image(hm_img, caption="Raw heatmap (red = dusty evidence)", use_container_width=True)

else:
    st.info("Upload a solar panel image to get started.")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
    1. **Feature extraction** — Frozen EfficientNet-B2 (ImageNet) → 1,408-d GAP vector
    2. **Classification** — RBF-SVM classifies clean vs dirty
    3. **Explainability** — Grad-CAM highlights the regions driving the decision
    """)
    st.markdown("**External validation:** 98.24% (2,562 imgs) / 99.74% (383 imgs) on two independent datasets.")
