"""
Car Damage Detection System - Flask Web Application
Run: python app.py
Then open: http://localhost:5000
"""

import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, 'model', 'car_damage_model.h5')
CLASS_INFO_PATH= os.path.join(BASE_DIR, 'model', 'class_info.json')
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXT    = {'png', 'jpg', 'jpeg', 'webp'}
IMG_SIZE       = (300, 300)  # EfficientNetB3 optimal size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ── Load model & class info ────────────────────────────────────────────────
print('Loading model...')
model = load_model(MODEL_PATH)

with open(CLASS_INFO_PATH) as f:
    class_info = json.load(f)

CLASS_INDICES = class_info['class_indices']
CLASS_LABELS  = class_info['class_labels']
IDX_TO_CLASS  = {str(v): k for k, v in CLASS_INDICES.items()}

# Damage level colors for UI
DAMAGE_COLORS = {
    '04-whole':    '#28a745',  # green
    '01-minor':    '#ffc107',  # yellow
    '02-moderate': '#fd7e14',  # orange
    '03-severe':   '#dc3545',  # red
}

DAMAGE_ICONS = {
    '04-whole':    '✅',
    '01-minor':    '⚠️',
    '02-moderate': '🔶',
    '03-severe':   '🚨',
}

print('Model loaded successfully!')

# ── Helper functions ───────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def predict_image(image_path):
    """Run prediction on an image file with Test-Time Augmentation."""
    IMG_SIZE = tuple(class_info.get('img_size', [300, 300]))

    # TTA — run 5 augmented passes and average
    augment_fns = [
        lambda img: img,                                          # original
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),        # flip
        lambda img: img.rotate(10),                              # rotate +10
        lambda img: img.rotate(-10),                             # rotate -10
        lambda img: img.resize((int(IMG_SIZE[0]*1.1), int(IMG_SIZE[1]*1.1)),
                               Image.LANCZOS).crop(
                               (int(IMG_SIZE[0]*0.05), int(IMG_SIZE[1]*0.05),
                                int(IMG_SIZE[0]*1.05), int(IMG_SIZE[1]*1.05))),  # zoom
    ]

    preds_sum = None
    for fn in augment_fns:
        img = load_img(image_path, target_size=IMG_SIZE)
        img = fn(img)
        img = img.resize(IMG_SIZE)
        img_array = img_to_array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        p = model.predict(img_array, verbose=0)[0]
        preds_sum = p if preds_sum is None else preds_sum + p

    preds     = preds_sum / len(augment_fns)
    pred_idx  = int(np.argmax(preds))
    pred_class= IDX_TO_CLASS[str(pred_idx)]
    confidence= float(preds[pred_idx]) * 100

    return {
        'predicted_class': pred_class,
        'label':           CLASS_LABELS[pred_class],
        'confidence':      round(confidence, 1),
        'color':           DAMAGE_COLORS[pred_class],
        'icon':            DAMAGE_ICONS[pred_class],
    }


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Accept JSON base64 payload (used by frontend on both local and Lambda)
    if request.is_json:
        data = request.get_json()
        b64_image = data.get('image')
        if not b64_image:
            return jsonify({'error': 'Missing image field'}), 400
        import base64
        from io import BytesIO
        image_bytes = base64.b64decode(b64_image)
        # Save to temp file for predict_image
        import tempfile
        suffix = '.' + (data.get('filename', 'img.jpg').rsplit('.', 1)[-1] or 'jpg')
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            result = predict_image(tmp_path)
        finally:
            os.unlink(tmp_path)
        return jsonify(result)

    # Legacy multipart fallback (direct curl / Postman testing)
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, PNG, or WEBP'}), 400
    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    result = predict_image(save_path)
    return jsonify(result)


if __name__ == '__main__':
    print('\n' + '='*50)
    print('  Car Damage Detection System')
    print('  Open: http://localhost:5000')
    print('='*50 + '\n')
    app.run(debug=True, host='0.0.0.0', port=5000)
