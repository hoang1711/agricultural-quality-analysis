# Agricultural Quality Analysis System

AI-powered agricultural quality analysis system for detecting fruit freshness, identifying defects, and generating intelligent Vietnamese recommendations for farmers.

---

## Features

* Fruit quality classification (fresh / rotten)
* Defect detection using object detection
* Vietnamese AI recommendation generation
* Real-time inference pipeline
* REST API support with FastAPI
* Dockerized deployment

---

## Supported Classes

The system currently supports:

* Fresh Apple
* Rotten Apple
* Fresh Banana
* Rotten Banana
* Fresh Orange
* Rotten Orange

---

## Model Performance

| Model             | Metric   | Score      |
| ----------------- | -------- | ---------- |
| EfficientNet-V2-S | Accuracy | **99.41%** |
| YOLOv8-n          | mAP50    | **0.995**  |
| YOLOv8-n          | Recall   | **1.0**    |

---

## Dataset

| Task           | Source                  | Images | Classes |
| -------------- | ----------------------- | ------ | ------- |
| Classification | Kaggle Fruits Dataset   | 13,599 | 6       |
| Detection      | Roboflow Food Detection | ~9,000 | 18      |

---

## System Workflow

```text id="p1m8q4"
Input Image
      ↓
Quality Classification
      ↓
Defect Detection
      ↓
AI Recommendation
      ↓
Result Output
```

---

## Installation

### Clone Repository

```bash id="m7v2q1"
git clone https://github.com/hoang1711/agricultural-quality-analysis.git
cd agricultural-quality-analysis
```

---

### Install Dependencies

```bash id="x4n8m5"
pip install -r requirements.txt
```

---

### Environment Setup

```bash id="w2q9m1"
cp .env.example .env
```

---

## Run Application

```bash id="u5m2x7"
uvicorn api.main:app --reload
```

---

## Tech Stack

* Python
* PyTorch
* EfficientNet-V2-S
* YOLOv8
* FastAPI
* Groq LLM
* Docker
* OpenCV
* Google Colab

---

## Future Improvements

* Real-time webcam support
* Mobile deployment
* Multi-language recommendation system
* Advanced defect segmentation
* Cloud deployment

---

## Author

Nguyen Huy Hoang

GitHub:
https://github.com/hoang1711
