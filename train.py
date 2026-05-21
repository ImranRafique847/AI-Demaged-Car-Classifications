"""
AI Car Damage Detection System - Training Script
================================================
EfficientNetB0 — best for small datasets (1600 images)
Smaller model = less overfitting = better accuracy
Run: python train.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, 'training')
VAL_DIR    = os.path.join(BASE_DIR, 'validation')
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'car_damage_model.h5')

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE    = (224, 224)   # EfficientNetB0 optimal
BATCH_SIZE  = 32
EPOCHS_P1   = 40
EPOCHS_P2   = 30
NUM_CLASSES = 4

CLASS_NAMES = ['01-minor', '02-moderate', '03-severe', '04-whole']
CLASS_LABELS = {
    '01-minor':    'Minor Damage',
    '02-moderate': 'Moderate Damage',
    '03-severe':   'Severe Damage',
    '04-whole':    'No Damage (Whole)',
}


def print_dataset_summary():
    print('\n' + '='*55)
    print('  DATASET SUMMARY')
    print('='*55)
    total_train = total_val = 0
    for cls in CLASS_NAMES:
        t = len([f for f in os.listdir(os.path.join(TRAIN_DIR, cls))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        v = len([f for f in os.listdir(os.path.join(VAL_DIR, cls))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_train += t
        total_val   += v
        print(f'  {CLASS_LABELS[cls]:25s} | Train: {t:4d} | Val: {v:4d}')
    print('='*55)
    print(f'  {"TOTAL":25s} | Train: {total_train:4d} | Val: {total_val:4d}')
    print('='*55 + '\n')


def create_generators():
    print('Creating data generators...')

    preprocess = tf.keras.applications.efficientnet.preprocess_input

    # Moderate augmentation — not too aggressive for small dataset
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print(f'  Class indices : {train_gen.class_indices}')
    print(f'  Train batches : {len(train_gen)}')
    print(f'  Val batches   : {len(val_gen)}\n')
    return train_gen, val_gen


def get_class_weights(train_gen):
    print('Computing class weights...')
    labels  = train_gen.classes
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    cw      = dict(enumerate(weights))
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    for i, w in cw.items():
        print(f'  {CLASS_LABELS[idx_to_class[i]]:25s}: {w:.3f}')
    print()
    return cw


def build_model():
    print('Building EfficientNetB0 model (best for small datasets)...')

    base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base.trainable = False

    x   = base.output
    x   = GlobalAveragePooling2D()(x)
    x   = BatchNormalization()(x)
    x   = Dense(256, activation='relu')(x)
    x   = Dropout(0.5)(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f'  Total params     : {model.count_params():,}')
    print(f'  Trainable layers : {trainable}')
    return model, base


def get_callbacks():
    return [
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
    ]


def plot_history(h1, h2):
    acc     = h1.history['accuracy']     + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss    = h1.history['loss']         + h2.history['loss']
    val_loss= h1.history['val_loss']     + h2.history['val_loss']
    split   = len(h1.history['accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — EfficientNetB0', fontsize=13, fontweight='bold')

    for ax, tv, vv, title in [
        (axes[0], acc,  val_acc,  'Accuracy'),
        (axes[1], loss, val_loss, 'Loss'),
    ]:
        ax.plot(tv, label=f'Train {title}', color='royalblue')
        ax.plot(vv, label=f'Val {title}',   color='darkorange')
        ax.axvline(x=split-1, color='red', linestyle='--',
                   alpha=0.5, label='Fine-tune start')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, 'training_history.png')
    plt.savefig(out, dpi=100)
    plt.close()
    print(f'  Training history saved → {out}')


def evaluate(model, val_gen):
    print('\nEvaluating model...')
    val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    print(f'\n  ✅ Validation Accuracy : {val_acc*100:.2f}%')
    print(f'  ✅ Validation Loss     : {val_loss:.4f}')

    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen, verbose=1), axis=1)
    y_true = val_gen.classes

    idx_to_class  = {v: k for k, v in val_gen.class_indices.items()}
    display_names = [CLASS_LABELS[idx_to_class[i]] for i in range(NUM_CLASSES)]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(out, dpi=100)
    plt.close()
    print(f'  Confusion matrix saved → {out}')

    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=display_names))


def save_class_info(train_gen):
    info = {
        'class_indices': train_gen.class_indices,
        'class_labels':  CLASS_LABELS,
        'img_size':      list(IMG_SIZE),
        'model':         'EfficientNetB0',
    }
    out = os.path.join(MODEL_DIR, 'class_info.json')
    with open(out, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'  Class info saved → {out}')


def main():
    print('\n' + '='*55)
    print('  AI CAR DAMAGE DETECTION')
    print('  Model: EfficientNetB0 (small, fast, less overfitting)')
    print(f'  TensorFlow: {tf.__version__}')
    print(f'  GPU: {len(tf.config.list_physical_devices("GPU")) > 0}')
    print('='*55)

    print_dataset_summary()
    train_gen, val_gen = create_generators()
    class_weights = get_class_weights(train_gen)
    model, base_model = build_model()
    callbacks = get_callbacks()

    # Phase 1 — Train head only (base frozen)
    print(f'\nPhase 1: Training head ({EPOCHS_P1} epochs max)...')
    h1 = model.fit(
        train_gen,
        epochs=EPOCHS_P1,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    print(f'\nPhase 1 best: {max(h1.history["val_accuracy"])*100:.2f}%')

    # Phase 2 — Fine-tune top 20 layers only
    print(f'\nPhase 2: Fine-tuning top 20 layers...')
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    h2 = model.fit(
        train_gen,
        epochs=EPOCHS_P2,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    print(f'\nPhase 2 best: {max(h2.history["val_accuracy"])*100:.2f}%')

    plot_history(h1, h2)

    best_model = load_model(MODEL_PATH)
    evaluate(best_model, val_gen)
    save_class_info(train_gen)

    print('\n' + '='*55)
    print('  TRAINING COMPLETE!')
    print(f'  Model saved → {MODEL_PATH}')
    print('  Run: python app.py')
    print('  Open: http://localhost:5000')
    print('='*55 + '\n')


if __name__ == '__main__':
    main()
