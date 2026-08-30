"""
test_pipeline.py - Tests for the EfficientNet-B2-SVM dust detection pipeline.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS = os.path.join(ROOT, "Models")

docker_marker = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS, "svm_classifier.pkl")),
    reason="Models/ not shipped",
)


class _FakeScaler:
    def transform(self, X):
        return np.asarray(X, dtype=np.float64)

    @property
    def scale_(self):
        return np.ones(1408)


class _FakeLinearSVM:
    kernel = "linear"
    classes_ = np.array([0, 1])

    def __init__(self):
        self.coef_ = np.random.randn(1, 1408)
        self._gamma = None
        self.dual_coef_ = None
        self.support_vectors_ = None

    def predict_proba(self, X):
        p = np.random.dirichlet([1, 1], size=len(X))
        return p

    def predict(self, X):
        return np.array([1] * len(X))


# --- Unit tests (no models needed) ---

def test_metadata_files_valid():
    cn = os.path.join(MODELS, "class_names.json")
    pm = os.path.join(MODELS, "pipeline_meta.json")
    if os.path.exists(cn):
        labels = json.load(open(cn))
        assert isinstance(labels, list)
        assert len(labels) >= 2
    if os.path.exists(pm):
        meta = json.load(open(pm))
        assert "scores" in meta or "svm_C" in meta


def test_class_labels_fallback():
    from explanations import class_labels
    labels = class_labels()
    assert len(labels) >= 2


def test_svm_gradient_linear_closed_form():
    from explanations import svm_gradient
    rng = np.random.RandomState(42)
    fake_scaler = _FakeScaler()
    fake_svm = _FakeLinearSVM()
    pooled = rng.randn(1408)
    grad_1 = svm_gradient(fake_svm, fake_scaler, pooled, 1)
    grad_0 = svm_gradient(fake_svm, fake_scaler, pooled, 0)
    expected_1 = fake_svm.coef_[0] / fake_scaler.scale_
    assert np.allclose(grad_1, expected_1)
    assert np.allclose(grad_0, -expected_1)


def test_gradcam_shape_and_range():
    from explanations import gradcam
    rng = np.random.RandomState(42)
    fake_scaler = _FakeScaler()
    fake_svm = _FakeLinearSVM()
    conv = rng.randn(7, 7, 1408).astype(np.float64)
    pooled = rng.randn(1408)
    hm = gradcam(fake_svm, fake_scaler, conv, pooled, 1)
    assert hm.shape == (7, 7)
    assert hm.min() >= 0.0
    assert hm.max() <= 1.0 + 1e-6


def test_overlay_heatmap_pil():
    from explanations import overlay_heatmap
    rng = np.random.RandomState(42)
    img = rng.rand(224, 224, 3).astype(np.float64)
    hm = rng.rand(7, 7).astype(np.float64)
    result = overlay_heatmap(img, hm)
    from PIL import Image
    assert isinstance(result, Image.Image)
    assert result.size == (224, 224)


def test_activation_ratio():
    from explanations import activation_ratio
    hm = np.zeros((10, 10))
    hm[:5, :5] = 1.0
    ratio = activation_ratio(hm)
    assert abs(ratio - 1.0) < 0.01

    hm2 = np.ones((10, 10))
    ratio2 = activation_ratio(hm2)
    assert abs(ratio2 - 0.25) < 0.01


@docker_marker
def test_deployed_pipeline_predicts():
    from explanations import load_pipeline, get_extractor, preprocess
    import joblib

    pipeline = load_pipeline()
    fe = get_extractor()

    uploads = os.path.join(ROOT, "static", "uploads")
    sample_imgs = [f for f in os.listdir(uploads)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))][:2]
    assert len(sample_imgs) > 0, "No sample images found in static/uploads/"

    for img_name in sample_imgs:
        path = os.path.join(uploads, img_name)
        x = preprocess(path)
        conv, pooled = fe.predict(x, verbose=0)
        scaled = pipeline["scaler"].transform(pooled)
        proba = pipeline["svm"].predict_proba(scaled)[0]
        assert proba.shape[0] >= 2
        assert 0.99 <= proba.sum() <= 1.01


@docker_marker
def test_explain_endpoint_via_test_client():
    from app import app
    client = app.test_client()
    uploads = os.path.join(ROOT, "static", "uploads")
    sample_imgs = [f for f in os.listdir(uploads)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))][:1]
    if not sample_imgs:
        pytest.skip("No sample images")

    with open(os.path.join(uploads, sample_imgs[0]), "rb") as f:
        resp = client.post("/explain", data={"file": f}, content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "gradcam_base64" in data
    assert "ig_base64" in data


@docker_marker
def test_analyze_endpoint_via_test_client():
    from app import app
    client = app.test_client()
    uploads = os.path.join(ROOT, "static", "uploads")
    sample_imgs = [f for f in os.listdir(uploads)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))][:1]
    if not sample_imgs:
        pytest.skip("No sample images")

    with open(os.path.join(uploads, sample_imgs[0]), "rb") as f:
        resp = client.post("/analyze", data={"file": f}, content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "dustiness" in data
    assert "confidence" in data
