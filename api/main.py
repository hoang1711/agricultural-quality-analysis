import io, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from src.inference.quality_classifier import QualityClassifier
from src.inference.defect_detector import DefectDetector
from src.inference.llm_advisor import get_advice
from api.schemas import AnalysisResponse

app = FastAPI(title="Agricultural Quality Analysis API", version="1.0.0")

classifier = QualityClassifier()
detector   = DefectDetector()

@app.get("/")
def root():
    return {"message": "Agricultural Quality Analysis API 🌱"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG allowed")

    image      = Image.open(io.BytesIO(await file.read())).convert("RGB")
    clf_result = classifier.predict(image)
    det_result = detector.detect(image)
    advice     = get_advice(clf_result, det_result)

    return AnalysisResponse(
        label       = clf_result["label"],
        quality     = clf_result["quality"],
        confidence  = clf_result["confidence"],
        num_defects = det_result["num_defects"],
        detections  = [
            {"class_name": d["class"], "confidence": d["confidence"], "bbox": d["bbox"]}
            for d in det_result["detections"]
        ],
        advice      = advice
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)