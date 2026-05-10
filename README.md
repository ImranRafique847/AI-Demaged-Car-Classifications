# 🚗 AI Car Damage Detection System

An end-to-end deep learning system that classifies car damage severity from images using Transfer Learning (MobileNetV2) and a Flask web application.

## 📊 Classes
| Class | Label | Description |
|-------|-------|-------------|
| 04-whole | No Damage | Undamaged car |
| 01-minor | Minor Damage | Small scratches or dents |
| 02-moderate | Moderate Damage | Visible damage |
| 03-severe | Severe Damage | Major structural damage |

## 📁 Project Structure
```
AI-Demaged-Car-Classifications/
├── train.py                    ← Training script
├── app.py                      ← Flask web application
├── requirements.txt            ← Python dependencies
├── training/
│   ├── 01-minor/
│   ├── 02-moderate/
│   ├── 03-severe/
│   └── 04-whole/
├── validation/
│   ├── 01-minor/
│   ├── 02-moderate/
│   ├── 03-severe/
│   └── 04-whole/
├── model/                      ← Created after training
│   ├── car_damage_model.h5
│   └── class_info.json
├── templates/
│   └── index.html
└── static/
    └── uploads/
```

## 🚀 How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the model
```bash
cd AI-Demaged-Car-Classifications
python train.py
```
This will create `model/car_damage_model.h5`.

### Step 3: Start the web app
```bash
python app.py
```
Then open: **http://localhost:5000**

## 🧠 Model Architecture
- **Base:** MobileNetV2 (pre-trained on ImageNet)
- **Head:** GlobalAveragePooling → BatchNorm → Dense(256) → Dropout → Dense(128) → Dropout → Softmax(4)
- **Training:** 2-phase (feature extraction + fine-tuning)
- **Input size:** 224×224 RGB

## 📈 Dataset
- ~1,613 training images
- ~478 validation images
- 4 balanced classes
