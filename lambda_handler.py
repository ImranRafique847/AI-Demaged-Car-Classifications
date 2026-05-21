"""
AWS Lambda Handler for Car Damage Detection
============================================
Accepts POST /predict with JSON body: {"image": "<base64>", "filename": "x.jpg"}
Model loaded lazily on first request to avoid Lambda init timeout.
"""

import os
import json
import base64
import numpy as np
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, 'model', 'car_damage_model.h5')
CLASS_INFO_PATH = os.path.join(BASE_DIR, 'model', 'class_info.json')
HTML_PATH       = os.path.join(BASE_DIR, 'templates', 'index.html')

_model       = None
_class_info  = None
_initialized = False

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


def _load_model():
    global _model, _class_info, _initialized
    if _initialized:
        return
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print('Loading model...')
    _model = load_model(MODEL_PATH)
    with open(CLASS_INFO_PATH) as f:
        _class_info = json.load(f)
    _initialized = True
    print('Model loaded.')


def predict_from_bytes(image_bytes: bytes) -> dict:
    import tensorflow as tf
    from PIL import Image

    _load_model()

    class_indices = _class_info['class_indices']
    class_labels  = _class_info['class_labels']
    idx_to_class  = {str(v): k for k, v in class_indices.items()}
    img_size      = tuple(_class_info['img_size'])

    def _preprocess(img):
        img = img.resize(img_size)
        arr = np.array(img, dtype=np.float32)
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    base_img = Image.open(BytesIO(image_bytes)).convert('RGB')

    # Test-Time Augmentation — 5 passes averaged
    augmented = [
        base_img,
        base_img.transpose(Image.FLIP_LEFT_RIGHT),
        base_img.rotate(10),
        base_img.rotate(-10),
        base_img.resize((int(img_size[0]*1.1), int(img_size[1]*1.1))),
    ]

    preds_sum = None
    for img in augmented:
        p = _model.predict(_preprocess(img), verbose=0)[0]
        preds_sum = p if preds_sum is None else preds_sum + p

    preds      = preds_sum / len(augmented)
    pred_idx   = int(np.argmax(preds))
    pred_class = idx_to_class[str(pred_idx)]
    confidence = float(preds[pred_idx]) * 100

    return {
        'predicted_class': pred_class,
        'label':           class_labels[pred_class],
        'confidence':      round(confidence, 1),
        'color':           DAMAGE_COLORS[pred_class],
        'icon':            DAMAGE_ICONS[pred_class],
    }


def read_html() -> str:
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def ok(body: dict):
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
        'body': json.dumps(body),
    }


def err(status: int, msg: str):
    return {
        'statusCode': status,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
        'body': json.dumps({'error': msg}),
    }


def handler(event, context):
    # Normalise method + path across API GW v1 and v2
    http_method = (event.get('httpMethod')
                   or event.get('requestContext', {})
                              .get('http', {}).get('method', 'GET'))
    path = (event.get('path') or event.get('rawPath', '/'))

    # ── GET / → HTML ─────────────────────────────────────────────────────────
    if http_method == 'GET' and path in ('/', ''):
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html; charset=utf-8'},
            'body': read_html(),
        }

    # ── CORS preflight ────────────────────────────────────────────────────────
    if http_method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin':  '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            },
            'body': '',
        }

    # ── POST /predict ─────────────────────────────────────────────────────────
    if http_method == 'POST' and path == '/predict':
        try:
            raw_body = event.get('body', '') or ''

            # API Gateway may base64-encode the body
            if event.get('isBase64Encoded', False):
                raw_body = base64.b64decode(raw_body).decode('utf-8')

            payload = json.loads(raw_body)
            b64_image = payload.get('image')
            if not b64_image:
                return err(400, 'Missing "image" field in JSON body')

            # Decode base64 → raw image bytes
            image_bytes = base64.b64decode(b64_image)
            result = predict_from_bytes(image_bytes)
            return ok(result)

        except json.JSONDecodeError:
            return err(400, 'Invalid JSON body')
        except Exception as e:
            print(f'Prediction error: {e}')
            return err(500, str(e))

    return err(404, 'Not found')
