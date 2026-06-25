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

from utils.image_processor import preprocess_image

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

    processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        raise BadRequest('Failed to preprocess the uploaded image')

    predictions = model.predict(processed_image, verbose=0)
    logits = np.asarray(predictions[0], dtype=np.float32)
    class_idx = int(np.argmax(logits))
    confidence = float(logits[class_idx])
    disease_name = labels[class_idx] if class_idx < len(labels) else 'Unknown'

    return jsonify({
        'success': True,
        'disease': disease_name,
        'confidence': round(confidence * 100, 2)
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
