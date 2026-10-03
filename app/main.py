"""FastAPI inference service for crop-disease images."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app.model import DiseaseClassifier
from app.schemas import HealthResponse, PredictResponse, ClassScore

MAX_UPLOAD_BYTES = 8 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = DiseaseClassifier()
    yield


app = FastAPI(
    title="Crop Disease Classifier",
    version="0.1.0",
    summary="Classify a leaf photo into a plant-disease or healthy class.",
    description=(
        "CPU inference API. Upload a leaf image to POST /predict. "
        "The checkpoint is a short transfer-learning run (ResNet18 by default) "
        "on a public PlantVillage subset. Predictions reflect that training "
        "distribution: single leaves on a plain background, not a field diagnosis."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    classifier: DiseaseClassifier = app.state.classifier
    return HealthResponse(
        status="ok",
        model_loaded=True,
        arch=classifier.arch,
        num_classes=classifier.num_classes,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Leaf image (JPEG, PNG, or WebP)."),
    topk: int = Query(3, ge=1, le=20, description="How many ranked classes to return."),
) -> PredictResponse:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 8 MB upload limit.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Could not read image.") from exc

    classifier: DiseaseClassifier = app.state.classifier
    ranked = classifier.predict(image, topk=topk)
    top_label, top_confidence = ranked[0]
    return PredictResponse(
        class_name=top_label,
        confidence=top_confidence,
        topk=[ClassScore(class_name=label, confidence=score) for label, score in ranked],
    )
