from pydantic import BaseModel

class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    top3: list[dict]