"""FastAPI service for the student wellbeing demo model."""

import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from app.features import FEATURE_COLUMNS, MODEL_VERSION, PARAMS, PROJECT_ROOT
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
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

RARE_COUNTRY_THRESHOLD = PARAMS["api"]["rare_country_threshold"]


def add_file_logging(log_path: Path | None = None) -> None:
    """Also write logs to logs/app.log (Logs folder of the course template)."""
    target = log_path or PROJECT_ROOT / PARAMS["api"]["log_file"]
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")
    except OSError:  # read-only filesystem (e.g. some containers): console only
        logger.warning("Cannot write %s; logging to console only.", target)
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)

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
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    return min(maximum, max(minimum, score))


def collect_warnings(features: dict[str, object], profile: dict[str, Any]) -> list[str]:
    """Compare one input row with what the model saw during training."""
    messages: list[str] = []
    for column, (low, high) in profile.get("numeric_ranges", {}).items():
        value = float(features[column])
        if value < low or value > high:
            messages.append(
                f"{column}={value:g} nằm ngoài khoảng dữ liệu huấn luyện "
                f"[{low:g}, {high:g}]; mô hình không ngoại suy nên kết quả không đáng tin."
            )
    counts = profile.get("country_counts", {})
    country = str(features["Country"])
    if counts:
        if country not in counts:
            messages.append(
                f"Country='{country}' chưa có trong dữ liệu huấn luyện; "
                "được xử lý như nhóm chưa biết."
            )
        elif counts[country] < RARE_COUNTRY_THRESHOLD:
            messages.append(
                f"Country='{country}' chỉ có {counts[country]} dòng trong dữ liệu "
                "huấn luyện; độ tin cậy thấp."
            )
    return messages


def _score(raw_score: float) -> float:
    if not math.isfinite(raw_score):
        raise HTTPException(status_code=500, detail="Model returned a non-finite score.")
    return round(clamp_score(raw_score), 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    add_file_logging()
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


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Expose latency per request (monitoring, Chip Huyen ch.8)."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    logger.info("%s %s -> %s in %.1f ms", request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


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
        numeric_ranges=metadata.get("training_profile", {}).get("numeric_ranges", {}),
        categories=metadata.get("training_profile", {}).get("categories", {}),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    row = payload.to_model_features()
    features = pd.DataFrame([row])
    score = _score(float(artifact["model"].predict(features)[0]))
    warnings = collect_warnings(row, artifact["metadata"].get("training_profile", {}))
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
        warnings=warnings,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """Score up to 200 profiles at once (batch prediction, Chip Huyen ch.7)."""
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    rows = [item.to_model_features() for item in payload.items]
    raw_scores = artifact["model"].predict(pd.DataFrame(rows))
    profile = artifact["metadata"].get("training_profile", {})
    version = artifact["metadata"].get("model_version", MODEL_VERSION)
    predictions = [
        PredictionResponse(
            mental_health_score=_score(float(raw)),
            model_version=version,
            warnings=collect_warnings(row, profile),
        )
        for row, raw in zip(rows, raw_scores)
    ]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
