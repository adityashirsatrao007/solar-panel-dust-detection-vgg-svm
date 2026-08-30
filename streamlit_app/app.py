"""
Solar Panel Dust Detection — Streamlit App
===========================================
Hybrid EfficientNet-B2 + SVM pipeline with Grad-CAM explainability.
Deployed on Streamlit Community Cloud.
"""

import os, json, tempfile, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import streamlit as st
import joblib
from PIL import Image

st.set_page_config(page_title="Solar Panel Dust Detection", page_icon="☀️", layout="centered")

# ── Model loading ───────────────────────────────────────────────────────────
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
def load_extractor():
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    base = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    pooled = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=pooled)
    return model, preprocess_input


# ── Prediction ──────────────────────────────────────────────────────────────
def predict(image: Image.Image):
    svm, scaler, labels, meta = load_pipeline()
    extractor, preprocess = load_extractor()
    img = image.convert("RGB").resize((224, 224))
    arr = preprocess(np.expand_dims(np.array(img, dtype=np.float32), 0))
    pooled = extractor.predict(arr, verbose=0)
    z = scaler.transform(pooled)
    probs = svm.predict_proba(z)[0]
    idx = int(np.argmax(probs))
    label = labels[idx] if idx < len(labels) else str(idx)
    conf = float(probs.max())
    dirty_idx = labels.index("dirty") if "dirty" in labels else len(probs) - 1
    dustiness = float(probs[dirty_idx]) * 100
    return label, conf, dustiness, probs


def gradcam(image: Image.Image):
    svm, scaler, labels, _ = load_pipeline()
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    base = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    conv_out = base.layers[-1].output
    pooled_out = GlobalAveragePooling2D()(base.output)
    full_model = Model(inputs=base.input, outputs=[conv_out, pooled_out])
    img = image.convert("RGB").resize((224, 224))
    arr = preprocess_input(np.expand_dims(np.array(img, dtype=np.float32), 0))
    conv, pooled = full_model.predict(arr, verbose=0)
    pooled, conv = pooled[0], conv[0]
    z = scaler.transform(pooled.reshape(1, -1)).ravel()
    gamma = float(getattr(svm, "_gamma", svm.gamma) or (1.0 / z.shape[0]))
    sv = svm.support_vectors_
    dual = np.asarray(svm.dual_coef_)
    alpha_y = dual[0] if dual.ndim == 1 else dual.sum(axis=0)
    k = np.exp(-gamma * ((z[None, :] - sv) ** 2).sum(axis=1))
    grad = -2.0 * gamma * (alpha_y * k)[:, None] * (z[None, :] - sv)
    grad = grad.sum(axis=0) / np.maximum(scaler.scale_, 1e-12)
    hm = np.maximum(np.tensordot(grad, conv, axes=(0, 2)), 0.0)
    hmax = hm.max()
    if hmax > 0:
        hm = hm / hmax
    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps.get_cmap("jet")
    hm_resized = np.array(Image.fromarray(np.uint8(255 * np.clip(hm, 0, 1))).resize(
        img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    rgb = cmap(hm_resized)[:, :, :3]
    orig = np.array(img).astype(np.float32) / 255.0
    overlay = (1 - 0.45) * orig + 0.45 * rgb
    return Image.fromarray(np.uint8(255 * np.clip(overlay, 0, 1)))


# ── UI ──────────────────────────────────────────────────────────────────────
st.title("☀️ Solar Panel Dust Detection")
st.caption("Hybrid EfficientNet-B2 + SVM · Explainable AI · [GitHub](https://github.com/adityashirsatrao007/Hybrid-VGG16-SVM-Framework-for-Automated-Dust-Detection-on-Solar-Panels-Advancing-Energy-Efficiency)")

with st.sidebar:
    st.header("About")
    st.markdown("Upload a solar panel photo. The model classifies **clean** vs **dirty** with Grad-CAM explainability.")
    st.divider()
    st.markdown("**Architecture**")
    st.code("EfficientNet-B2 (frozen, ImageNet)\n  → GAP 1408-d\n  → RBF-SVM", language=None)
    st.divider()
    svm, _, _, meta = load_pipeline()
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
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)
    with st.spinner("Analyzing..."):
        label, conf, dustiness, probs = predict(img)
    with col2:
        if dustiness > 50:
            st.error(f"**{label.upper()}** — {dustiness:.1f}% dustiness")
        elif dustiness > 20:
            st.warning(f"**{label.upper()}** — {dustiness:.1f}% dustiness")
        else:
            st.success(f"**{label.upper()}** — {dustiness:.1f}% dustiness")
        st.metric("Confidence", f"{conf*100:.1f}%")
        st.markdown("**Class probabilities**")
        for i, lbl in enumerate(["clean", "dirty"]):
            st.progress(probs[i], text=f"{lbl}: {probs[i]*100:.1f}%")
    st.divider()
    with st.spinner("Generating Grad-CAM heatmap..."):
        heatmap = gradcam(img)
    st.subheader("Grad-CAM Localization")
    st.caption("Highlights regions most influential to the SVM decision.")
    c3, c4 = st.columns(2)
    with c3:
        st.image(heatmap, caption="Heatmap overlay", use_container_width=True)
    with c4:
        st.image(img, caption="Original", use_container_width=True)
else:
    st.info("Upload a solar panel image to get started.")
    st.markdown("---")
    st.markdown("**How it works:** Frozen EfficientNet-B2 → 1,408-d GAP features → RBF-SVM → Grad-CAM explainability.")
    st.markdown("**External validation:** 98.24% (2,562 imgs) / 99.74% (383 imgs) on two independent datasets.")
