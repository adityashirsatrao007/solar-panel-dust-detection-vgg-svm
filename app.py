"""
app.py - Flask dashboard + /analyze + /explain APIs for EfficientNet-B2-SVM pipeline.
"""
from __future__ import annotations

import os
import time
import io
import base64
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# File logging is best-effort: if we lack write access to the log directory
# (e.g. tests run by a non-root user while the service writes logs as root),
# fall back to stderr only so the app/tests still start.
_handlers = [logging.StreamHandler()]
try:
    _handlers.append(logging.FileHandler(LOG_FILE))
except (PermissionError, OSError):
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=_handlers,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dust-detection-2025")
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "uploads")
app.config["EXPLAIN_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "explain")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB


@app.before_request
def log_request_info():
    """Log incoming request details."""
    if request.path in ("/analyze", "/explain"):
        logger.info(f"REQUEST {request.method} {request.path} from {request.remote_addr}")


@app.after_request
def log_response_info(response):
    """Log response status for API endpoints."""
    if request.path in ("/analyze", "/explain", "/health"):
        logger.info(f"RESPONSE {request.method} {request.path} -> {response.status_code}")
    return response

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Lazy-loaded globals — loaded on first request, not at startup
_pipeline = None
_extractor = None
_models_loaded = False


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def verify_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def ensure_models_loaded():
    """Load models on first request (lazy) to avoid GPU OOM at startup."""
    global _pipeline, _extractor, _models_loaded
    if not _models_loaded:
        from explanations import load_pipeline, get_extractor
        _pipeline = load_pipeline()
        _extractor = get_extractor()
        _models_loaded = True
        print("[app] Models loaded (lazy).")


def _b64_png(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_dustiness(image_path):
    import numpy as np
    from explanations import preprocess
    ensure_models_loaded()  # lazy-load on first request
    x = preprocess(image_path)
    conv_out, pooled = _extractor.predict(x, verbose=0)
    scaled = _pipeline["scaler"].transform(pooled)
    proba = _pipeline["svm"].predict_proba(scaled)[0]
    dirty_idx = list(_pipeline["svm"].classes_).index(1) if 1 in _pipeline["svm"].classes_ else -1
    if dirty_idx >= 0:
        dustiness = float(proba[dirty_idx] * 100)
    else:
        dustiness = float(proba[1] * 100) if len(proba) > 1 else 0.0
    confidence = float(proba.max())
    return {"dustiness": dustiness, "confidence": confidence, "proba": proba}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/explain_page")
def explain_page():
    return render_template("index.html", explain_mode=True)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    if not verify_image(path):
        os.remove(path)
        return jsonify({"error": "Invalid image file"}), 400

    t0 = time.time()
    result = predict_dustiness(path)
    elapsed = time.time() - t0

    os.remove(path)
    return jsonify({
        "dustiness": round(result["dustiness"], 2),
        "confidence": round(result["confidence"], 4),
        "processing_time": round(elapsed, 3),
    })


@app.route("/explain", methods=["POST"])
def explain():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["EXPLAIN_FOLDER"], exist_ok=True)
    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    if not verify_image(path):
        os.remove(path)
        return jsonify({"error": "Invalid image file"}), 400

    from explanations import explain_image
    ensure_models_loaded()  # lazy-load before using extractor
    t0 = time.time()
    result = explain_image(path, extractor=_extractor)
    elapsed = time.time() - t0

    os.remove(path)

    response = {
        "probabilities": result["probabilities"],
        "predicted_class": result["predicted_class"],
        "confidence": round(result["confidence"], 4),
        "requires_review": result["requires_review"],
        "activation_ratio": round(result["activation_ratio"], 4),
        "processing_time": round(elapsed, 3),
    }

    for method in ["gradcam", "scorecam", "ig"]:
        key = f"{method}_overlay"
        if key in result and hasattr(result[key], "save"):
            response[f"{method}_base64"] = _b64_png(result[key])

    return jsonify(response)


@app.route("/health")
def health():
    """Simple health check endpoint."""
    try:
        ensure_models_loaded()
        return jsonify({"status": "ok", "models_loaded": _models_loaded})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Models loaded lazily on first request (avoids GPU OOM at startup).


@app.route("/cleanup", methods=["POST"])
def cleanup():
    import shutil
    for folder in [app.config["UPLOAD_FOLDER"], app.config["EXPLAIN_FOLDER"]]:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 5001)))