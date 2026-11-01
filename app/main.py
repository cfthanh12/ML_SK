"""FastAPI service for the student wellbeing demo model."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from app.features import MODEL_VERSION
from app.schemas import PredictionRequest, PredictionResponse
from app.train import DEFAULT_ARTIFACT_PATH


def load_artifact(artifact_path: Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    """Load a locally generated model artifact or report how to create it."""
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. "
            "Run `python -m app.train` before starting the API."
        )
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or "model" not in artifact or "metadata" not in artifact:
        raise ValueError(f"Invalid model artifact: {artifact_path}")
    return artifact


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifact = load_artifact()
    yield


app = FastAPI(
    title="Student Social Media & Wellbeing API",
    description=(
        "Educational demo that estimates a dataset score. "
        "It is not a clinical or diagnostic tool."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    artifact = getattr(request.app.state, "artifact", None)
    return {
        "status": "ok" if artifact is not None else "not_ready",
        "model_loaded": artifact is not None,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    features = pd.DataFrame([payload.to_model_features()])
    score = float(artifact["model"].predict(features)[0])
    return PredictionResponse(
        mental_health_score=round(score, 2),
        model_version=artifact["metadata"].get("model_version", MODEL_VERSION),
    )
