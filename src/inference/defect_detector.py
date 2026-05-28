import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

class DefectDetector:
    def __init__(self, weights_path="weights/yolov8_defect/weights/best.pt"):
        self.model = YOLO(weights_path)

    def detect(self, image: Image.Image, conf_threshold=0.25) -> dict:
        img_np  = np.array(image)
        results = self.model(img_np, conf=conf_threshold, verbose=False)[0]

        detections = [
            {
                "class":      results.names[int(box.cls)],
                "confidence": round(float(box.conf) * 100, 2),
                "bbox":       box.xyxy[0].tolist()
            }
            for box in results.boxes
        ]

        annotated     = results.plot()
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        return {
            "detections":      detections,
            "num_defects":     len(detections),
            "annotated_image": annotated_pil
        }