from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import joblib
import numpy as np
import os
import logging
import base64
import io
import time
import shutil

from PIL import Image as PilImage

import explanations
from explanations import load_pipeline, get_extractor, gradcam, activation_ratio

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['EXPLAIN_FOLDER'] = 'static/explain'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB limit
os.makedirs(app.config['EXPLAIN_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rate limiter
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(app=app, key_func=get_remote_address,
                      default_limits=["100 per day", "10 per minute"])
except Exception as e:  # optional feature
    logging.warning(f"flask-limiter unavailable: {e}")
    limiter = None


def load_models():
    """Load the VGG16 feature extractor, scaler and SVM classifier."""
    global feature_extractor, svm_classifier, scaler, class_names
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    feature_extractor = Model(inputs=base_model.input,
                              outputs=GlobalAveragePooling2D()(base_model.output))
    svm_path = os.path.join('Models', 'svm_classifier.pkl')
    scaler_path = os.path.join('Models', 'scaler.pkl')
    if not os.path.exists(svm_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model files not found")
    svm_classifier = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)
    class_names = explanations.class_labels()
    if len(class_names) != svm_classifier.classes_.size:
        class_names = [f"class_{i}" for i in range(svm_classifier.classes_.size)]
    logging.info(f"Models loaded (classes={class_names})")


load_models()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def verify_image(image_path):
    try:
        with PilImage.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


@app.route('/analyze', methods=['POST'])
def analyze():
    """Original endpoint: per-image dustiness percentage (unchanged behaviour)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Allowed file types: png, jpg, jpeg'}), 400

    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        start = time.time()
        file.save(save_path)
        if not verify_image(save_path):
            os.remove(save_path)
            return jsonify({'error': 'Invalid or corrupted image'}), 400
        dustiness = predict_dustiness(save_path)
        elapsed = round(time.time() - start, 2)
        logging.info(f"Processed {filename} in {elapsed}s")
        if dustiness is not None:
            return jsonify({'success': True,
                            'image_path': f"uploads/{filename}",
                            'dustiness': dustiness,
                            'confidence': round(dustiness / 100, 2),
                            'processing_time': elapsed})
        os.remove(save_path)
        return jsonify({'error': 'Error processing image'}), 500
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({'error': 'Server error'}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route('/explain', methods=['POST'])
def explain():
    """
    Explainable AI endpoint: returns dust probabilities, a Grad-CAM localization
    overlay (base64 PNG), the activation ratio (localization metric), the predicted
    class and a confidence-gated human-review flag.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Allowed file types: png, jpg, jpeg'}), 400

    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    exp_path = os.path.join(app.config['EXPLAIN_FOLDER'], filename)
    try:
        start = time.time()
        file.save(save_path)
        if not verify_image(save_path):
            os.remove(save_path)
            return jsonify({'error': 'Invalid or corrupted image'}), 400

        fe = get_extractor()
        pipeline = load_pipeline()
        out = explanations.predict(save_path, fe)
        probs = pipeline['svm'].predict_proba(pipeline['scaler'].transform(out['pooled'][None, :]))[0]
        target = int(np.argmax(probs))

        heatmap = gradcam(pipeline['svm'], pipeline['scaler'],
                          out['conv'], out['pooled'], target)
        overlay = explanations.overlay_heatmap(out['x'][0], heatmap)
        overlay.save(exp_path)

        png_bytes = io.BytesIO()
        overlay.save(png_bytes, format='PNG')
        b64 = base64.b64encode(png_bytes.getvalue()).decode('ascii')

        result = {
            'success': True,
            'image_path': f"uploads/{filename}",
            'explain_image_path': f"explain/{filename}",
            'gradcam_base64': b64,
            'class_names': class_names,
            'probabilities': {class_names[i]: float(probs[i]) for i in range(len(class_names))},
            'predicted_class': class_names[target],
            'confidence': float(probs[target]),
            'requires_review': bool(probs[target] < explanations.CONF_THRESHOLD),
            'activation_ratio': activation_ratio(heatmap),
            'processing_time': round(time.time() - start, 2),
        }
        # still expose dustiness for dashboard compatibility (binary case)
        if len(class_names) == 2:
            result['dustiness'] = round(probs[target] * 100, 2)
            result['confidence'] = round(probs[target], 4)
        logging.info(f"Explained {filename} -> {result['predicted_class']}")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Explain error: {str(e)}")
        return jsonify({'error': 'Server error', 'detail': str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


def predict_dustiness(image_path):
    try:
        image = load_img(image_path, target_size=(128, 128))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        features = feature_extractor.predict(image_array)
        features = scaler.transform(features)
        dust_prob = svm_classifier.predict_proba(features)[:, 1][0]
        return round(dust_prob * 100, 2)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/explain_page')
def explain_page():
    return render_template('index.html', explain_mode=True)


@app.route('/cleanup', methods=['POST'])
def cleanup_uploads():
    try:
        for folder in (app.config['UPLOAD_FOLDER'], app.config['EXPLAIN_FOLDER']):
            if os.path.exists(folder):
                shutil.rmtree(folder)
                os.makedirs(folder, exist_ok=True)
        logging.info("Upload/explain folders cleaned")
        return jsonify({'success': True, 'message': 'Cleaned'}), 200
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)