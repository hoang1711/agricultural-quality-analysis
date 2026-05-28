from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox:       List[float]

class AnalysisResponse(BaseModel):
    label:       str
    quality:     str
    confidence:  float
    num_defects: int
    detections:  List[Detection]
    advice:      str