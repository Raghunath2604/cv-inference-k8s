"""
FastAPI service serving the ShapeCNN.

Run locally:
    uvicorn src.api:app --reload --port 8000
"""
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from src.model import CLASSES, IDX2CLASS, ShapeCNN, image_to_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv-inference-api")

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.pt")

_state = {"model": None, "loaded_at": None}

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Time spent running model inference (excludes request parsing)",
    buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0],
)
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made, by predicted class", ["predicted_class"]
)


def load_model() -> ShapeCNN:
    model = ShapeCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model from {MODEL_PATH}...")
    start = time.time()
    _state["model"] = load_model()
    _state["loaded_at"] = time.time()
    logger.info(f"Model loaded in {time.time() - start:.3f}s")
    yield
    _state["model"] = None


app = FastAPI(
    title="Shape Classification API",
    description="Classifies images into circle / square / triangle / star.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    classes: list


@app.middleware("http")
async def track_requests(request, call_next):
    response = await call_next(request)
    REQUEST_COUNT.labels(
        method=request.method, path=request.url.path, status=response.status_code
    ).inc()
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _state["model"] is not None else "loading",
        classes=CLASSES,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)):  # noqa: B008 — this is the documented
    # FastAPI idiom for declaring a required file upload; ruff's B008 rule
    # (no function calls in default args) is a false positive here since
    # FastAPI's dependency-injection system specifically relies on this
    # exact pattern.
    # Deliberately a sync `def`, not `async def`: FastAPI/Starlette runs
    # sync endpoint functions in a worker thread pool automatically,
    # keeping the asyncio event loop free to accept new connections
    # while this CPU-bound torch inference runs. An earlier `async def`
    # version ran inference directly on the event loop, which caused a
    # long tail latency (p99.9 > 2s) under concurrent load in testing —
    # the event loop couldn't accept new connections while blocked on
    # inference. See README's load-testing section for the before/after
    # numbers.
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = file.file.read()
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="File is not a valid image")

    tensor = image_to_tensor(img).unsqueeze(0)  # add batch dim

    start = time.time()
    with torch.no_grad():
        logits = _state["model"](tensor)[0]
        probs = torch.softmax(logits, dim=0)
    latency = time.time() - start
    INFERENCE_LATENCY.observe(latency)

    pred_idx = int(torch.argmax(probs))
    label = IDX2CLASS[pred_idx]
    PREDICTION_COUNT.labels(predicted_class=label).inc()

    scores = {IDX2CLASS[i]: round(float(p), 4) for i, p in enumerate(probs)}
    return PredictResponse(
        label=label,
        confidence=round(float(probs[pred_idx]), 4),
        scores=scores,
        latency_ms=round(latency * 1000, 2),
    )
