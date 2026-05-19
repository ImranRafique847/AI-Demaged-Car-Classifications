"""
AWS Lambda Handler
==================
Wraps the Flask app using Mangum (ASGI/WSGI adapter for Lambda)
This allows the Flask app to run on AWS Lambda + API Gateway
"""

import os
import json
import base64
import numpy as np

# Set TF to use minimal memory on Lambda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from io import BytesIO
import tempfile

# ── Load model once (Lambda reuses container between calls) ──
MODEL_PATH     = os.path.join(os.path.dirname(__file__), 'model', 'car_damage_model.h5')
CLASS_INFO_PATH= os.path.join(os.path.dirname(__file__), 'model', 'class_info.json')

print('Loading model...')
model = load_model(MODEL_PATH)

with open(CLASS_INFO_PATH) as f:
    class_info = json.load(f)

CLASS_INDICES = class_info['class_indices']
CLASS_LABELS  = class_info['class_labels']
IDX_TO_CLASS  = {str(v): k for k, v in CLASS_INDICES.items()}
IMG_SIZE      = tuple(class_info['img_size'])

DAMAGE_COLORS = {
    '04-whole':    '#28a745',
    '01-minor':    '#ffc107',
    '02-moderate': '#fd7e14',
    '03-severe':   '#dc3545',
}
DAMAGE_ICONS = {
    '04-whole':    '✅',
    '01-minor':    '⚠️',
    '02-moderate': '🔶',
    '03-severe':   '🚨',
}

print('Model loaded successfully!')


def predict_from_bytes(image_bytes):
    """Run prediction on image bytes."""
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds      = model.predict(img_array, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    pred_class = IDX_TO_CLASS[str(pred_idx)]
    confidence = float(preds[pred_idx]) * 100

    all_probs = [
        {
            'class':       k,
            'label':       CLASS_LABELS[k],
            'probability': round(float(preds[v]) * 100, 1),
            'color':       DAMAGE_COLORS[k],
        }
        for k, v in CLASS_INDICES.items()
    ]
    all_probs.sort(key=lambda x: x['probability'], reverse=True)

    return {
        'predicted_class': pred_class,
        'label':           CLASS_LABELS[pred_class],
        'confidence':      round(confidence, 1),
        'color':           DAMAGE_COLORS[pred_class],
        'icon':            DAMAGE_ICONS[pred_class],
        'all_probs':       all_probs,
    }


def read_html():
    """Read the HTML template."""
    html_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
    with open(html_path, 'r') as f:
        return f.read()


def handler(event, context):
    """
    AWS Lambda handler function.
    Handles both GET (serve HTML) and POST (predict) requests.
    """
    http_method = event.get('httpMethod', 'GET')
    path        = event.get('path', '/')

    # ── GET / → serve the HTML page ──────────────────────────
    if http_method == 'GET' and path == '/':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': read_html()
        }

    # ── POST /predict → run prediction ───────────────────────
    if http_method == 'POST' and path == '/predict':
        try:
            body = event.get('body', '')
            is_base64 = event.get('isBase64Encoded', False)

            # Parse multipart form data to extract image
            if is_base64:
                body = base64.b64decode(body)
            elif isinstance(body, str):
                body = body.encode()

            # Extract image bytes from multipart body
            content_type = event.get('headers', {}).get(
                'content-type', event.get('headers', {}).get('Content-Type', ''))

            boundary = None
            for part in content_type.split(';'):
                part = part.strip()
                if part.startswith('boundary='):
                    boundary = part[9:].strip()
                    break

            if not boundary:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({'error': 'Invalid request format'})
                }

            # Parse multipart to get image bytes
            boundary_bytes = ('--' + boundary).encode()
            parts = body.split(boundary_bytes)
            image_bytes = None

            for part in parts:
                if b'filename=' in part and b'Content-Type: image' in part:
                    # Split headers from body
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        image_bytes = part[header_end + 4:].rstrip(b'\r\n--')
                        break

            if not image_bytes:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({'error': 'No image found in request'})
                }

            result = predict_from_bytes(image_bytes)
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(result)
            }

        except Exception as e:
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': str(e)})
            }

    # ── 404 for unknown routes ────────────────────────────────
    return {
        'statusCode': 404,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'error': 'Not found'})
    }
