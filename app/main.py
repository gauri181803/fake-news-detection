from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
import asyncio
import uvicorn

from app.predictor import FakeNewsPredictor
from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.classifier import train_classifier, CLASSIFIER_PATH, load_classifier

predictor: FakeNewsPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("⚙️  Loading models...")
    predictor = FakeNewsPredictor()
    predictor.load_all_models()
    print("✅  All models ready.")
    yield
    print("🛑  Shutting down.")

app = FastAPI(
    title="Multilingual Fake News Detection API",
    description="Detects fake news in Hindi, Marathi, Gujarati, and Telugu using XLM-RoBERTa + Heterogeneous R-GCN.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Multilingual Fake News Detection API",
        "docs": "/docs",
        "supported_languages": ["hi", "mr", "gu", "te"],
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    loaded = list(predictor.loaded_models.keys()) if predictor else []
    return HealthResponse(
        status="ok",
        models_loaded=loaded,
        supported_languages=["hi", "mr", "gu", "te"],
    )

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    if not request.text or len(request.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text too short (min 20 chars).")
    try:
        result = await predictor.predict(request.text, language=request.language)
        if result is None:
            raise HTTPException(status_code=500, detail="Predictor returned None.")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
class TrainRequest(BaseModel):
    texts: List[str]
    labels: List[int]
    epochs: int = 10

@app.post("/train", tags=["Training"])
async def train(request: TrainRequest):
    if len(request.texts) != len(request.labels):
        raise HTTPException(status_code=422, detail="texts and labels must be same length")
    if len(request.texts) < 10:
        raise HTTPException(status_code=422, detail="Need at least 10 samples to train")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: train_classifier(
            request.texts, request.labels,
            predictor.tokenizer, predictor.xlm_model,
            epochs=request.epochs
        )
    )
    predictor.classifier = load_classifier()
    return {"status": "trained", "samples": len(request.texts), "path": CLASSIFIER_PATH}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)