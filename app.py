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
from PIL import Image
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import time
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB limit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

# Load models
def load_models():
    global feature_extractor, svm_classifier, scaler
    try:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
        svm_classifier_path = os.path.join('Models', 'svm_classifier.pkl')
        scaler_path = os.path.join('Models', 'scaler.pkl')
        if not os.path.exists(svm_classifier_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model files not found")
        svm_classifier = joblib.load(svm_classifier_path)
        scaler = joblib.load(scaler_path)
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

try:
    load_models()
except Exception as e:
    logging.critical("Failed to start application due to model loading error")
    raise SystemExit("Application cannot start. Check logs for details.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def verify_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Check if image is valid
        return True
    except Exception as e:
        logging.error(f"Invalid image file: {str(e)}")
        return False

@app.route('/analyze', methods=['POST'])
@limiter.limit("5 per minute")  # Limit to 5 requests per minute per IP
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            start_time = time.time()
            file.save(save_path)
            
            # Verify image integrity
            if not verify_image(save_path):
                os.remove(save_path)
                return jsonify({'error': 'Invalid or corrupted image'}), 400

            dustiness = predict_dustiness(save_path)
            if dustiness is not None:
                processing_time = time.time() - start_time
                logging.info(f"Processed file {filename} in {processing_time:.2f} seconds")
                return jsonify({
                    'success': True,
                    'image_path': f"uploads/{filename}",
                    'dustiness': dustiness,
                    'confidence': round(dustiness / 100, 2)  # Optional: for future use
                })
            os.remove(save_path)
            return jsonify({'error': 'Error processing image'}), 500
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return jsonify({'error': 'Server error'}), 500
        finally:
            # Cleanup: Remove file after processing (optional, can be toggled)
            if os.path.exists(save_path):
                os.remove(save_path)
    return jsonify({'error': 'Allowed file types: png, jpg, jpeg'}), 400

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

# Periodic cleanup of upload folder (optional)
@app.route('/cleanup', methods=['POST'])
def cleanup_uploads():
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
            os.makedirs(upload_folder, exist_ok=True)
            logging.info("Upload folder cleaned up")
            return jsonify({'success': True, 'message': 'Upload folder cleaned'}), 200
        return jsonify({'error': 'Upload folder does not exist'}), 404
    except Exception as e:
        logging.error(f"Error cleaning upload folder: {str(e)}")
        return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)