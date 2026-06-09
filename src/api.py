from __future__ import annotations

import os
from typing import Dict

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "models/domain-classifier")

app = FastAPI(title="Domain Classifier API")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


class ClassifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    domain: str
    confidence: float
    probabilities: Dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    id2label = model.config.id2label
    best_id = int(torch.argmax(probs).item())
    probabilities = {id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

    return {
        "domain": id2label[best_id],
        "confidence": round(float(probs[best_id]), 4),
        "probabilities": probabilities,
    }
