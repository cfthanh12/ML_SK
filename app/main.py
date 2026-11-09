"""FastAPI service for the student wellbeing demo model."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from app.features import FEATURE_COLUMNS, MODEL_VERSION
from app.schemas import (
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.train import DEFAULT_ARTIFACT_PATH

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_handler)
logger.propagate = False

REQUIRED_METADATA_KEYS = (
    "model_version",
    "selected_model",
    "metrics",
    "cv_results",
    "feature_columns",
    "feature_importance",
    "feature_importance_method",
    "training_rows",
    "trained_at_utc",
)


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
    metadata = artifact["metadata"]
    missing_keys = sorted(set(REQUIRED_METADATA_KEYS).difference(metadata))
    if missing_keys:
        raise ValueError(
            "Model artifact metadata is missing keys: "
            f"{', '.join(missing_keys)}. "
            "Run `python -m app.train` to regenerate it."
        )
    if metadata["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            "Model artifact feature columns do not match app/features.py. "
            "Run `python -m app.train` to regenerate it."
        )
    return artifact


def resolve_artifact_path() -> Path:
    """Allow tests and deployments to point at a different artifact."""
    return Path(os.environ.get("MODEL_ARTIFACT_PATH", str(DEFAULT_ARTIFACT_PATH)))


def clamp_score(
    score: float,
    minimum: float = 0.0,
    maximum: float = 10.0,
) -> float:
    """Keep predictions on the dataset's documented scale."""
    return min(maximum, max(minimum, score))


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_path = resolve_artifact_path()
    app.state.artifact = load_artifact(artifact_path)
    logger.info("Loaded model artifact from %s", artifact_path)
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
    version = MODEL_VERSION
    if artifact is not None:
        version = artifact["metadata"].get("model_version", MODEL_VERSION)
    return {
        "status": "ok" if artifact is not None else "not_ready",
        "model_loaded": artifact is not None,
        "model_version": version,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    metadata = artifact["metadata"]
    return ModelInfoResponse(
        model_version=metadata["model_version"],
        selected_model=metadata["selected_model"],
        metrics=metadata["metrics"],
        cv_results=metadata["cv_results"],
        feature_columns=metadata["feature_columns"],
        feature_importance=metadata["feature_importance"],
        feature_importance_method=metadata["feature_importance_method"],
        training_rows=metadata["training_rows"],
        trained_at_utc=metadata["trained_at_utc"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    features = pd.DataFrame([payload.to_model_features()])
    score = round(clamp_score(float(artifact["model"].predict(features)[0])), 2)
    model_version = artifact["metadata"].get("model_version", MODEL_VERSION)
    logger.info(
        "Prediction score=%.2f model_version=%s selected_model=%s",
        score,
        model_version,
        artifact["metadata"].get("selected_model", "unknown"),
    )
    return PredictionResponse(
        mental_health_score=score,
        model_version=model_version,
    )
