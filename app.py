import json
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.image_processor import preprocess_image, is_likely_plant_image

load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_final_model.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')
MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', 5 * 1024 * 1024))
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

model = None
labels = []


def load_model_and_labels() -> None:
    global model, labels

    if not os.path.exists(MODEL_PATH):
        logger.error('Model file not found: %s', MODEL_PATH)
        return

    if not os.path.exists(LABELS_PATH):
        logger.error('Labels file not found: %s', LABELS_PATH)
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH, 'r', encoding='utf-8') as file:
            labels = json.load(file)

        if not isinstance(labels, list):
            raise ValueError('Labels file must contain a JSON array of class names')

        logger.info('Loaded AI model from %s', MODEL_PATH)
        logger.info('Loaded %d labels from %s', len(labels), LABELS_PATH)
    except Exception:
        model = None
        labels = []
        logger.exception('Failed to load AI model or labels')


def get_uploaded_file():
    file = request.files.get('image') or request.files.get('file')
    if file is None:
        raise BadRequest('Image file is required')

    if not file.mimetype or file.mimetype.lower() not in ALLOWED_MIME_TYPES:
        raise BadRequest('Only JPEG, PNG, and WEBP images are supported')

    return file


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'success': True,
        'status': 'ok',
        'modelLoaded': model is not None,
        'labelsCount': len(labels),
        'maxUploadBytes': app.config['MAX_CONTENT_LENGTH']
    })


@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    if model is None:
        return jsonify({'success': False, 'message': 'AI model is not available'}), 503

    file = get_uploaded_file()
    image_bytes = file.read()
    if not image_bytes:
        raise BadRequest('Uploaded image file is empty')

    # --- OOD Guard: reject non-plant images before running the model ---
    try:
        import io as _io
        from PIL import Image as _Image
        _pil = _Image.open(_io.BytesIO(image_bytes))
        _pil = _pil.convert('RGB')
        if not is_likely_plant_image(_pil):
            return jsonify({
                'success': False,
                'message': 'No plant or leaf detected in the uploaded image. '
                           'Please upload a clear photo of a plant leaf.'
            }), 422
    except Exception as _e:
        logger.warning('Plant image pre-check failed (proceeding): %s', _e)
    # --- End OOD Guard ---

    processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        raise BadRequest('Failed to preprocess the uploaded image')

    predictions = model.predict(processed_image, verbose=0)
    raw_output = np.asarray(predictions[0], dtype=np.float32).reshape(-1)

    if raw_output.size == 0:
        return jsonify({'success': False, 'message': 'Model returned no predictions'}), 500

    if np.all(raw_output >= 0) and np.isclose(raw_output.sum(), 1.0, atol=1e-3):
        probabilities = raw_output
    else:
        shifted = raw_output - np.max(raw_output)
        exp_values = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values)

    class_idx = int(np.argmax(probabilities))
    confidence = float(np.clip(probabilities[class_idx], 0.0, 1.0))
    disease_name = labels[class_idx] if class_idx < len(labels) else 'Unknown'

    # --- Confidence threshold: treat very low-confidence results as undetected ---
    MIN_CONFIDENCE = float(os.getenv('MIN_PREDICTION_CONFIDENCE', '0.50'))
    if confidence < MIN_CONFIDENCE:
        return jsonify({
            'success': False,
            'message': 'Could not confidently identify the plant disease. '
                       'Please upload a clearer or closer photo of the affected leaf.'
        }), 422

    return jsonify({
        'success': True,
        'disease': disease_name,
        'confidence': round(confidence, 4)
    })


@app.errorhandler(RequestEntityTooLarge)
def handle_large_payload(error):
    logger.warning('Payload too large: %s', error)
    return jsonify({
        'success': False,
        'message': f'Request payload too large. Maximum size is {MAX_UPLOAD_SIZE} bytes.'
    }), 413


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    logger.warning('Bad request: %s', error)
    return jsonify({
        'success': False,
        'message': error.description if hasattr(error, 'description') else str(error)
    }), 400


@app.errorhandler(Exception)
def handle_internal_error(error):
    logger.exception('Unexpected error while handling request')
    return jsonify({
        'success': False,
        'message': 'Internal server error'
    }), 500


load_model_and_labels()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() in {'1', 'true', 'yes'}
    app.run(host='0.0.0.0', port=port, debug=debug)
