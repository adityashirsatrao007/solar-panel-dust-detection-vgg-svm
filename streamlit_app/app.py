"""
Solar Panel Dust Detection — Streamlit App
===========================================
EfficientNet-B2 + SVM classification + Grad-CAM heatmap + Grounding DINO bounding boxes.
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import streamlit as st
import joblib
from PIL import Image, ImageDraw, ImageFont
import torch

st.set_page_config(page_title="Solar Panel Dust Detection", page_icon="☀️", layout="centered")

# ── Cached model loaders ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SVM pipeline from Hugging Face...")
def load_pipeline():
    from huggingface_hub import hf_hub_download
    svm_p = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "svm_classifier.pkl")
    sc_p  = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "scaler.pkl")
    cn_p  = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "class_names.json")
    meta_p = hf_hub_download("adityashirsatrao007/solar-panel-dust-xai", "pipeline_meta.json")
    import json
    svm = joblib.load(svm_p)
    scaler = joblib.load(sc_p)
    with open(cn_p) as f:
        labels = json.load(f)
    with open(meta_p) as f:
        meta = json.load(f)
    return svm, scaler, labels, meta


@st.cache_resource(show_spinner="Loading EfficientNet-B2 backbone...")
def load_efficientnet():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
    return full_model, preprocess_input


@st.cache_resource(show_spinner="Loading Grounding DINO (zero-shot detector)...")
def load_grounding_dino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    mdl = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    return proc, mdl


# ── Grad-CAM via SVM gradient ──────────────────────────────────────────────
def gradcam_overlay(conv, grad, img_size):
    hm = np.maximum(np.tensordot(grad, conv, axes=(0, 2)), 0.0)
    hmax = hm.max()
    if hmax > 0:
        hm = hm / hmax
    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps.get_cmap("jet")
    hm_pil = Image.fromarray(np.uint8(255 * np.clip(hm, 0, 1))).resize(img_size, Image.BILINEAR)
    hm_arr = np.array(hm_pil).astype(np.float32) / 255.0
    rgb = cmap(hm_arr)[:, :, :3]
    orig = np.array(Image.new("RGB", img_size, (128, 128, 128))).astype(np.float32) / 255.0
    overlay = (1 - 0.45) * orig + 0.45 * rgb
    return Image.fromarray(np.uint8(255 * np.clip(overlay, 0, 1))), hm_pil


# ── Grounding DINO detection ───────────────────────────────────────────────
def detect_dust(image, threshold=0.25):
    proc, mdl = load_grounding_dino()
    text = "dust. dirt. debris. soiling. dirty area."
    inputs = proc(images=image, text=text, return_tensors="pt")
    with torch.no_grad():
        outputs = mdl(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = proc.image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    scores = results["scores"].numpy()
    boxes = results["boxes"].numpy()
    labels_raw = results["labels"].tolist()

    # Map label IDs to phrases
    tok = proc.tokenizer
    encoded = tok(text, return_tensors="pt")
    tokens = tok.convert_ids_to_tokens(encoded.input_ids[0].tolist())
    phrases = [p.strip() for p in text.split(".") if p.strip()]

    phrase_map = {}
    phrase_idx = 0
    i = 1
    while i < len(tokens) - 1 and phrase_idx < len(phrases):
        if tokens[i] == ".":
            i += 1
            continue
        start = i
        while i < len(tokens) - 1 and tokens[i] != ".":
            i += 1
        phrase_map[start] = phrases[phrase_idx]
        phrase_idx += 1
        i += 1

    labels = []
    for lid in labels_raw:
        best = min(phrase_map.keys(), key=lambda k: abs(k - lid), default=None)
        labels.append(phrase_map.get(best, f"class_{lid}"))

    return scores, boxes, labels


def draw_boxes(image, scores, boxes, labels, color="red", width=3):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for score, box, label in zip(scores, boxes, labels):
        x1, y1, x2, y2 = box.astype(int)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = f"{label} {score:.0%}"
        bbox = draw.textbbox((x1, y1 - 25), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 25), text, fill="white", font=font)

    return draw_img


# ── Full analysis pipeline ─────────────────────────────────────────────────
def analyze(image: Image.Image):
    svm, scaler, labels, meta = load_pipeline()
    model, preprocess = load_efficientnet()

    img_resized = image.convert("RGB").resize((224, 224))
    arr = preprocess(np.expand_dims(np.array(img_resized, dtype=np.float32), 0))
    conv, pooled = model.predict(arr, verbose=0)
    pooled, conv = pooled[0], conv[0]

    z = scaler.transform(pooled.reshape(1, -1))
    probs = svm.predict_proba(z)[0]
    idx = int(np.argmax(probs))
    label = labels[idx] if idx < len(labels) else str(idx)
    conf = float(probs.max())
    dirty_idx = labels.index("dirty") if "dirty" in labels else len(probs) - 1
    dustiness = float(probs[dirty_idx]) * 100

    # SVM gradient for Grad-CAM
    zs = scaler.transform(pooled.reshape(1, -1)).ravel()
    gamma = float(getattr(svm, "_gamma", svm.gamma) or (1.0 / zs.shape[0]))
    sv = svm.support_vectors_
    dual = np.asarray(svm.dual_coef_)
    alpha_y = dual[0] if dual.ndim == 1 else dual.sum(axis=0)
    k = np.exp(-gamma * ((zs[None, :] - sv) ** 2).sum(axis=1))
    grad = -2.0 * gamma * (alpha_y * k)[:, None] * (zs[None, :] - sv)
    grad = grad.sum(axis=0) / np.maximum(scaler.scale_, 1e-12)

    grad_overlay, grad_heatmap = gradcam_overlay(conv, grad, image.convert("RGB").size)

    # Grounding DINO bounding boxes
    det_scores, det_boxes, det_labels = detect_dust(image.convert("RGB"))
    boxed_img = draw_boxes(image.convert("RGB"), det_scores, det_boxes, det_labels)

    return label, conf, dustiness, probs, grad_overlay, grad_heatmap, boxed_img, len(det_scores)


# ── UI ──────────────────────────────────────────────────────────────────────
st.title("☀️ Solar Panel Dust Detection")
st.caption("EfficientNet-B2 + SVM · Grad-CAM · Grounding DINO Bounding Boxes · "
           "[GitHub](https://github.com/adityashirsatrao007/solar-panel-dust-detection-vgg-svm)")

with st.sidebar:
    st.header("About")
    st.markdown(
        "Upload a solar panel photo. The system performs **three tasks**:\n"
        "1. **Classification** — Clean vs Dirty (SVM)\n"
        "2. **Grad-CAM** — Heatmap of evidence regions\n"
        "3. **Bounding Boxes** — Zero-shot dust localization (Grounding DINO)"
    )
    st.divider()
    st.markdown("**Architecture**")
    st.code(
        "EfficientNet-B2 (frozen) → GAP 1408-d\n"
        "  → RBF-SVM (classification)\n"
        "  → SVM-gradient Grad-CAM (heatmap)\n"
        "Grounding DINO-tiny (bounding boxes)",
        language=None,
    )
    st.divider()
    _, _, _, meta = load_pipeline()
    st.markdown("**Model metrics**")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{meta['scores']['accuracy']*100:.1f}%")
    c2.metric("AUC-ROC", f"{meta['scores']['auc']:.4f}")
    c1.metric("CV Acc", f"{meta['scores']['cv_accuracy']*100:.1f}%")
    c2.metric("Train", f"{meta['train_samples']:,}")
    st.divider()
    st.caption("v007 · 3,787 merged corpus · External: 98.2% / 99.7%")

uploaded = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)

    with st.spinner("Running EfficientNet-B2 + SVM classification..."):
        label, conf, dustiness, probs, grad_overlay, grad_heatmap, boxed_img, n_dets = analyze(img)

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

    st.markdown("**Class probabilities**")
    prob_cols = st.columns(2)
    for i, lbl in enumerate(["clean", "dirty"]):
        prob_cols[i].progress(probs[i], text=f"{lbl}: {probs[i]*100:.1f}%")

    st.divider()

    # Grad-CAM
    st.subheader("🔍 Grad-CAM — Where the model looks")
    st.caption("Red/yellow = regions driving the 'dirty' decision. Blue = considered clean.")
    g1, g2 = st.columns(2)
    with g1:
        st.image(grad_overlay, caption="Grad-CAM overlay", use_container_width=True)
    with g2:
        st.image(grad_heatmap, caption="Raw heatmap", use_container_width=True)

    st.divider()

    # Bounding boxes
    st.subheader("📦 Bounding Boxes — Zero-shot dust localization")
    st.caption(f"Grounding DINO detected **{n_dets}** dusty region(s) via text prompts (dust, dirt, debris, soiling).")
    st.image(boxed_img, caption="Grounding DINO bounding boxes", use_container_width=True)

else:
    st.info("Upload a solar panel image to get started.")
    st.markdown("---")
    st.markdown("**Three AI systems work together:**")
    st.markdown("""
    | System | Purpose | Method |
    |--------|---------|--------|
    | **Classification** | Clean or Dirty? | EfficientNet-B2 + RBF-SVM |
    | **Grad-CAM** | Where is the evidence? | SVM-gradient heatmap |
    | **Bounding Boxes** | Where is the dust? | Grounding DINO (zero-shot) |
    """)
    st.markdown("**External validation:** 98.24% / 99.74% on two independent datasets.")
