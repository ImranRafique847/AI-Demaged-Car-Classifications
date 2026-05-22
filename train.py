"""
AI Car Damage Detection - 3-Class Severity Model (PyTorch + GPU)
=================================================================
Classes: Minor, Moderate, Severe (no "whole" class)
Best accuracy achieved: 72.2%
Run: python train.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TRAIN_DIR  = BASE_DIR / 'training'
VAL_DIR    = BASE_DIR / 'validation'
MODEL_DIR  = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'car_damage_model.pth'

# Only use 3 damage classes (exclude 04-whole)
CLASSES_TO_USE = ['01-minor', '02-moderate', '03-severe']

MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS_P1  = 30
EPOCHS_P2  = 20
NUM_CLASSES = 3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_LABELS = {
    '01-minor':    'Minor Damage',
    '02-moderate': 'Moderate Damage',
    '03-severe':   'Severe Damage',
}


def print_info():
    print('\n' + '='*60)
    print('  AI CAR DAMAGE SEVERITY DETECTION')
    print(f'  Classes: Minor / Moderate / Severe (3-class)')
    print(f'  Device : {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU    : {torch.cuda.get_device_name(0)}')
    print(f'  PyTorch: {torch.__version__}')
    print('='*60)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def is_valid_class(path):
    """Filter to only include our 3 classes."""
    return any(cls in str(path) for cls in CLASSES_TO_USE)


class FilteredImageFolder(datasets.ImageFolder):
    """ImageFolder that only loads specific classes."""
    def __init__(self, root, classes_to_use, **kwargs):
        super().__init__(root, **kwargs)
        # Filter samples to only include desired classes
        class_to_idx = {cls: i for i, cls in enumerate(classes_to_use)}
        filtered_samples = []
        for path, _ in self.samples:
            folder = Path(path).parent.name
            if folder in classes_to_use:
                filtered_samples.append((path, class_to_idx[folder]))
        self.samples = filtered_samples
        self.targets = [s[1] for s in filtered_samples]
        self.classes = classes_to_use
        self.class_to_idx = class_to_idx


def build_model(freeze_base=True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )

    return model.to(DEVICE)


def train_model(model, train_loader, val_loader, epochs, lr, phase_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = correct / total

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_acc = correct / total

        scheduler.step()
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'  Epoch {epoch+1:2d}/{epochs} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    print(f'\n  {phase_name} BEST: {best_val_acc*100:.2f}%')
    return history, best_val_acc


def evaluate_model(model, val_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    display_names = [CLASS_LABELS[c] for c in class_names]

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names)
    plt.title('Confusion Matrix — Damage Severity', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=100)
    plt.close()

    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=display_names))

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'  Final Accuracy: {acc*100:.2f}%')
    return acc


def main():
    print_info()

    train_transform, val_transform = get_transforms()

    # Load only 3 classes (exclude 04-whole)
    train_dataset = FilteredImageFolder(TRAIN_DIR, CLASSES_TO_USE, transform=train_transform)
    val_dataset   = FilteredImageFolder(VAL_DIR, CLASSES_TO_USE, transform=val_transform)

    print(f'\n  Classes: {train_dataset.classes}')
    print(f'  Train: {len(train_dataset)} | Val: {len(val_dataset)}')

    # Oversampling for balance
    targets = train_dataset.targets
    class_counts = np.bincount(targets)
    max_count = class_counts.max()
    sample_weights = [max_count / class_counts[t] for t in targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    print('\n  Class distribution:')
    for i, cls in enumerate(CLASSES_TO_USE):
        print(f'    {CLASS_LABELS[cls]:25s}: {class_counts[i]} images')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Build model
    model = build_model(freeze_base=True)

    # Phase 1: Train head
    print(f'\n{"="*60}')
    print(f'  Phase 1: Training head ({EPOCHS_P1} epochs)')
    print(f'{"="*60}')
    h1, _ = train_model(model, train_loader, val_loader, EPOCHS_P1, lr=1e-3, phase_name='Phase 1')

    # Phase 2: Fine-tune last 3 blocks
    print(f'\n{"="*60}')
    print(f'  Phase 2: Fine-tuning ({EPOCHS_P2} epochs)')
    print(f'{"="*60}')
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    h2, best_acc = train_model(model, train_loader, val_loader, EPOCHS_P2, lr=1e-4, phase_name='Phase 2')

    # Evaluate
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    final_acc = evaluate_model(model, val_loader, CLASSES_TO_USE)

    # Save class info
    info = {
        'class_indices': {name: i for i, name in enumerate(CLASSES_TO_USE)},
        'class_labels': CLASS_LABELS,
        'img_size': [IMG_SIZE, IMG_SIZE],
        'model': 'EfficientNet-B0 (PyTorch)',
        'accuracy': round(final_acc * 100, 2),
    }
    with open(MODEL_DIR / 'class_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print('\n' + '='*60)
    print(f'  FINAL ACCURACY: {final_acc*100:.1f}%')
    print(f'  Model saved → {MODEL_PATH}')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
