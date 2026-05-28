import torch
from torchvision import transforms
from PIL import Image

from src.models.classifier import FruitQualityClassifier, CLASS_NAMES, QUALITY_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class QualityClassifier:
    def __init__(self, weights_path="weights/efficientnet_best.pth"):
        self.model = FruitQualityClassifier()
        self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
            conf, idx = probs.max(0)
        label = CLASS_NAMES[idx.item()]
        return {
            "label":      label,
            "quality":    QUALITY_MAP[label],
            "confidence": round(conf.item() * 100, 2)
        }