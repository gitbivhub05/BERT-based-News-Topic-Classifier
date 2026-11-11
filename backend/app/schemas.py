from pydantic import BaseModel, Field, field_validator
from typing import List, Dict

class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Article text cannot be blank.")
        return value.strip()

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values):
        cleaned = [v.strip() for v in values]
        if any(not v or len(v) > 20000 for v in cleaned):
            raise ValueError("Each article must contain 1–20,000 characters.")
        return cleaned

class PredictionResult(BaseModel):
    predicted_label: int
    probabilities: Dict[str, float]
    inference_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
