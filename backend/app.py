from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import torch
import torch.nn.functional as F
import logging
import random
import numpy as np
from typing import Any, List, Dict
from fastapi.middleware.cors import CORSMiddleware

# Deterministic behavior (CPU)
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Calibration parameters
TEMPERATURE = 3.0  # Increased from 2.0 for softer predictions
CONFIDENCE_THRESHOLD = 0.85  # Threshold for high confidence predictions
NEUTRAL_THRESHOLD = 0.75  # Increased from 0.6 for more neutral classifications

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    score: float

def apply_temperature_scaling(logits: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    return logits / temperature

class CalibratedPipeline:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def __call__(self, text: str) -> Dict[str, Any]:
        # Tokenize and get model outputs
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Apply temperature scaling
        scaled_logits = apply_temperature_scaling(logits, TEMPERATURE)
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Get prediction and confidence (probs shape: [1, num_classes])
        probs = probs.squeeze(0)  # Remove batch dimension
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        
        # Map to labels based on confidence
        if confidence < NEUTRAL_THRESHOLD:
            label = "neutral"
        else:
            label = "negative" if pred_idx == 0 else "positive"
        
        return {"label": label, "score": confidence}

def load_sentiment_pipeline():
    """Load a calibrated sentiment-analysis pipeline."""
    if os.path.isdir(MODEL_DIR) and any(fname for fname in os.listdir(MODEL_DIR) if fname.endswith(".bin")):
        logging.info("Loading model from local path %s", MODEL_DIR)
        model_path = MODEL_DIR
    else:
        logging.info("Local fine-tuned model not found. Falling back to %s", DEFAULT_MODEL_NAME)
        model_path = DEFAULT_MODEL_NAME

    return CalibratedPipeline(model_path)

sentiment_pipeline = load_sentiment_pipeline()

@app.post("/predict", response_model=PredictionOut)
async def predict(payload: TextIn):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Input text is empty")

    # Get prediction from calibrated pipeline
    pred = sentiment_pipeline(payload.text)
    
    return PredictionOut(
        label=pred["label"],
        score=pred["score"]
    )

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API"} 