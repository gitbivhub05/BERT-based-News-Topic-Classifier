import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionRequest, BatchPredictionRequest, PredictionResult, BatchPredictionResponse
from .model_service import ModelService

BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / ".env")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/bert-news-classifier"))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = BASE / MODEL_DIR
classifier = ModelService(MODEL_DIR)

app = FastAPI(title="BERT News Topic Classification API", version="1.0.0")
origins = [x.strip() for x in os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=True, allow_methods=["GET", "POST"],
                   allow_headers=["*"])

@app.get("/")
def root():
    return {"message": "BERT News Topic Classification API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier.loaded,
            "device": str(classifier.device)}

@app.get("/model-info")
def model_info():
    return classifier.info()

@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictionRequest):
    try:
        return classifier.predict([request.text])[0]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    try:
        return {"predictions": classifier.predict(request.texts)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Batch prediction failed.") from exc
