"""
Dedicated ABSA inference service.

This service is the long-term serving boundary: application APIs should call it
instead of loading PhoBERT in every web process. The internal predictor can later
be replaced by ONNX Runtime, a distilled student model, or a quantized model
without changing the API contract.
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "app"))

from absa_predictor import ASPECTS, PhoBERTPredictor  # noqa: E402


predictor = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    format: str = "multipolarity"


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    format: str = "multipolarity"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = PhoBERTPredictor()
    predictor.load_model()
    yield
    predictor = None


app = FastAPI(title="ABSA Inference Service", version="1.0.0", lifespan=lifespan)


def _ensure_loaded():
    if not predictor or not predictor.model_loaded:
        raise HTTPException(status_code=503, detail="ABSA model is not loaded")


@app.get("/health")
def health() -> Dict:
    return {
        "status": "ok" if predictor and predictor.model_loaded else "degraded",
        "model_loaded": bool(predictor and predictor.model_loaded),
    }


@app.get("/model-info")
def model_info() -> Dict:
    return {
        "aspects": ASPECTS,
        "sentiment_classes": ["NEG", "POS", "NEU"],
        "multi_polarity": True,
        "device": predictor.device if predictor else None,
        "batch_size": predictor.batch_size if predictor else None,
        "max_length": predictor.max_length if predictor else None,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> Dict:
    _ensure_loaded()
    result = predictor.predict_batch([request.text], format=request.format)[0]
    return {
        "text": request.text,
        "prediction": result,
    }


@app.post("/predict-batch")
def predict_batch(request: PredictBatchRequest) -> Dict:
    _ensure_loaded()
    results = predictor.predict_batch(request.texts, format=request.format)
    return {
        "count": len(results),
        "predictions": results,
    }
