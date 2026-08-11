"""Pipeline tests for the VGG16-SVM dust detector with explainable AI."""
import json
import os
import sys

import joblib
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import explanations  # noqa: E402

MODELS = os.path.join(ROOT, "Models")

docker_marker = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS, "svm_classifier.pkl")),
    reason="deployed models not present",
)


# ---------------------------------------------------------------- pure-python


def test_metadata_files_valid():
    with open(os.path.join(MODELS, "class_names.json")) as fh:
        labels = json.load(fh)
    with open(os.path.join(MODELS, "pipeline_meta.json")) as fh:
        meta = json.load(fh)
    assert labels == meta["classes"]
    assert len(labels) >= 2


def test_class_labels_fallback():
    assert explanations.class_labels()


class _FakeScaler:
    def __init__(self, n=512):
        self.mean_ = np.zeros(n)
        self.scale_ = np.ones(n)
        self.n_features_in_ = n

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


class _FakeLinearSVM:
    kernel = "linear"
    classes_ = np.array([0, 1])

    def __init__(self, n=512):
        self.coef_ = np.random.RandomState(0).randn(1, n)
        self.intercept_ = np.array([0.0])


def test_svm_gradient_linear_closed_form():
    scaler = _FakeScaler(8)
    svm = _FakeLinearSVM(8)
    pooled = np.random.RandomState(1).randn(8)
    grad = explanations.svm_gradient(svm, scaler, pooled, 1)
    np.testing.assert_allclose(grad, svm.coef_[0], rtol=1e-6)
    # class 0 flips the sign
    grad0 = explanations.svm_gradient(svm, scaler, pooled, 0)
    np.testing.assert_allclose(grad0, -svm.coef_[0], rtol=1e-6)


def test_gradcam_shape_and_range():
    scaler = _FakeScaler(4)
    svm = _FakeLinearSVM(4)
    conv = np.random.RandomState(2).rand(4, 4, 4)
    pooled = np.random.RandomState(3).randn(4)
    hm = explanations.gradcam(svm, scaler, conv, pooled, 1)
    assert hm.shape == (4, 4)
    assert hm.min() >= 0.0 and hm.max() <= 1.0
    assert 0.0 <= explanations.activation_ratio(hm) <= 1.0


def test_overlay_heatmap_pil():
    np.random.seed(4)
    img = np.random.rand(16, 16, 3).astype(np.float32)
    hm = np.random.rand(4, 4)
    out = explanations.overlay_heatmap(img, hm)
    assert out.mode == "RGB" and out.size == (16, 16)


# ---------------------------------------------------------------- deployed models


@docker_marker
def test_deployed_pipeline_predicts():
    pl = explanations.load_pipeline()
    fe = explanations.get_extractor()
    sample = os.path.join(ROOT, "static", "uploads")
    imgs = [
        os.path.join(sample, f)
        for f in os.listdir(sample)
        if f.startswith(("Imgclean", "Imgdirty")) and f.lower().endswith(".jpg")
    ]
    if not imgs:
        pytest.skip("no bundled sample images")
    out = explanations.predict(imgs[0], fe)
    assert out["conv"].shape[0] > 0 and out["pooled"].shape[0] == 512
    probs = pl["svm"].predict_proba(pl["scaler"].transform(out["pooled"][None, :]))[0]
    assert abs(float(probs.sum()) - 1.0) < 1e-6


@docker_marker
def test_explain_endpoint_via_test_client():
    import app as ap

    client = ap.app.test_client()
    sample = os.path.join(ROOT, "static", "uploads")
    img = next(
        (os.path.join(sample, f) for f in os.listdir(sample)
         if f.startswith("Imgclean")),
        None,
    )
    if not img:
        pytest.skip("no bundled clean sample")
    with open(img, "rb") as fh:
        r = client.post("/explain", data={"file": (fh, os.path.basename(img))},
                        content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    assert "gradcam_base64" in body and "requires_review" in body
    assert "predicted_class" in body and "probabilities" in body


@docker_marker
def test_analyze_endpoint_via_test_client():
    import app as ap

    client = ap.app.test_client()
    sample = os.path.join(ROOT, "static", "uploads")
    img = next(
        (os.path.join(sample, f) for f in os.listdir(sample)
         if f.startswith("Imgclean")),
        None,
    )
    if not img:
        pytest.skip("no bundled clean sample")
    with open(img, "rb") as fh:
        r = client.post("/analyze", data={"file": (fh, os.path.basename(img))},
                        content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True and "dustiness" in body