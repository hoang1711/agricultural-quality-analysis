# Agricultural Quality Analysis System

Hệ thống phân tích chất lượng trái cây qua ảnh — tự động nhận diện tươi/hỏng, phát hiện vùng khuyết tật, và tư vấn tiếng Việt cho nông dân.

## 🎯 Chức năng
- **Phân loại chất lượng**: nhận diện 6 loại (táo/chuối/cam tươi hoặc hỏng) với độ chính xác **99.41%**
- **Phát hiện khuyết tật**: khoanh vùng chỗ hỏng trên quả bằng bounding box, mAP50 = **0.995**
- **Tư vấn tự động**: sinh lời khuyên tiếng Việt (bảo quản / xuất bán / loại bỏ) bằng LLM

## 📊 Kết quả
| Model | Metric | Score |
|---|---|---|
| EfficientNet-V2-S | Accuracy | **99.41%** |
| YOLOv8-n | mAP50 | **0.995** |
| YOLOv8-n | Recall | **1.0** |

## 🗂️ Dataset
| | Source | Ảnh | Classes |
|---|---|---|---|
| Classifier | [Kaggle](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) | 13,599 | 6 |
| Detector | [Roboflow](https://universe.roboflow.com/sliit-kuemd/food-detection-3lhd1) | ~9,000 | 18 |

## ⚙️ Cài đặt & Chạy
```bash
git clone https://github.com/hoang1711/agricultural-quality-analysis.git
cd agricultural-quality-analysis
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload
```

## 🛠️ Tech Stack
`PyTorch` · `EfficientNet-V2-S` · `YOLOv8` · `Groq LLM` · `FastAPI` · `Docker` · `Google Colab`

## 👤 Author
**Nguyen Huy Hoang** · [GitHub](https://github.com/hoang1711) · 
