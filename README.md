# 🚗 AI Car Damage Detection System

An end-to-end deep learning system that classifies car damage severity from images using **EfficientNetB3** transfer learning, a **Flask** web application, and deployed live on **AWS Lambda** via Docker container.

🌐 **Live Demo:** https://a3m6jbr82l.execute-api.us-east-1.amazonaws.com/

---

## 📊 Damage Classes

| Class | Label | Description |
|-------|-------|-------------|
| `04-whole` | No Damage | Car is undamaged |
| `01-minor` | Minor Damage | Small scratches, dents, or paint chips |
| `02-moderate` | Moderate Damage | Visible dents, broken parts, panel damage |
| `03-severe` | Severe Damage | Major structural damage |

---

## 🧠 Model Architecture

- **Base model:** EfficientNetB3 (pre-trained on ImageNet)
- **Head:** GlobalAveragePooling → BatchNorm → Dense(256, ReLU) → Dropout(0.4) → Dense(128, ReLU) → Dropout(0.3) → Softmax(4)
- **Training strategy:** 2-phase — feature extraction then fine-tuning
- **Input size:** 300×300 RGB (EfficientNetB3 optimal)
- **Class weights:** Applied to handle class imbalance
- **Accuracy:** ~55–65% validation accuracy

---

## 📁 Project Structure

```
AI-Demaged-Car-Classifications/
├── train.py                  ← Model training script (EfficientNetB3)
├── app.py                    ← Flask web app (local development)
├── lambda_handler.py         ← AWS Lambda handler (production)
├── Dockerfile                ← Docker container for Lambda deployment
├── requirements.txt          ← Production dependencies (CPU-only)
├── requirements-dev.txt      ← Development dependencies (GPU)
├── AWS_DEPLOYMENT.md         ← Full AWS deployment guide
├── .dockerignore             ← Files excluded from Docker build
│
├── model/
│   ├── car_damage_model.h5   ← Trained EfficientNetB3 model (52MB)
│   ├── class_info.json       ← Class labels and indices
│   ├── training_history.png  ← Training/validation accuracy curves
│   ├── confusion_matrix.png  ← Model evaluation matrix
│   └── sample_images.png     ← Sample training images
│
├── training/                 ← Training dataset (~1,613 images)
│   ├── 01-minor/   (452 images)
│   ├── 02-moderate/ (463 images)
│   ├── 03-severe/  (468 images)
│   └── 04-whole/   (230 images)
│
├── validation/               ← Validation dataset (~478 images)
│   ├── 01-minor/   (82 images)
│   ├── 02-moderate/ (75 images)
│   ├── 03-severe/  (91 images)
│   └── 04-whole/   (230 images)
│
├── templates/
│   └── index.html            ← Web UI (dark theme, drag & drop)
└── static/
    └── uploads/              ← Uploaded images (local only)
```

---

## 🚀 Run Locally

### 1. Install dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Train the model (optional — pre-trained model included)
```bash
python train.py
```
Saves model to `model/car_damage_model.h5`.

### 3. Start the web app
```bash
python app.py
```
Open: **http://localhost:5000**

---

## ☁️ AWS Deployment (Lambda + Docker)

The app is deployed as a Docker container on AWS Lambda with API Gateway.

### Architecture
```
User Browser
    │
    ▼
API Gateway (HTTP API)
    │
    ▼
AWS Lambda (car-damage-detection)
    │  - python:3.11-slim container
    │  - EfficientNetB3 model loaded lazily
    │  - 3008 MB memory, 120s timeout
    ▼
Amazon ECR (container registry)
```

### How it works
- Frontend sends image as **base64 JSON** (avoids API Gateway binary encoding issues)
- Lambda loads the TF model on first request (lazy init)
- Subsequent requests reuse the warm container (~2-3s response)
- First cold start takes ~30-45 seconds

### Deploy from scratch

**Prerequisites:** AWS CLI, Docker Desktop

```bash
# 1. Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 876493682275.dkr.ecr.us-east-1.amazonaws.com

# 2. Build and push image
docker buildx build --provenance=false \
  --output "type=image,name=876493682275.dkr.ecr.us-east-1.amazonaws.com/car-damage-detection:latest,push=true" .

# 3. Update Lambda
aws lambda update-function-code \
  --function-name car-damage-detection \
  --image-uri 876493682275.dkr.ecr.us-east-1.amazonaws.com/car-damage-detection:latest \
  --region us-east-1
```

See [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) for the full setup guide.

---

## 🖥️ Web Interface

- Dark theme UI with drag & drop image upload
- Shows damage level label and confidence percentage
- Confidence bar with color coding:
  - 🟢 Green — No Damage
  - 🟡 Yellow — Minor Damage
  - 🟠 Orange — Moderate Damage
  - 🔴 Red — Severe Damage

---

## 📦 Dependencies

**Production (Lambda):**
- `tensorflow-cpu==2.15.0`
- `flask==3.0.3`
- `numpy==1.26.4`
- `pillow==10.3.0`
- `werkzeug==3.0.3`

**Development (local training with GPU):**
- `tensorflow==2.15.0`
- All of the above

---

## 📈 Dataset

| Split | Total | Minor | Moderate | Severe | Whole |
|-------|-------|-------|----------|--------|-------|
| Training | 1,613 | 452 | 463 | 468 | 230 |
| Validation | 478 | 82 | 75 | 91 | 230 |

Images are JPEG format, resized to 300×300 during training and inference.
