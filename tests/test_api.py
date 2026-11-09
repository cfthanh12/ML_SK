"""Focused tests for API validation and model predictions."""

import logging

import joblib
import pytest
from fastapi.testclient import TestClient

from app.features import MODEL_VERSION
from app.main import app, clamp_score, load_artifact

SAMPLE_INPUT = {
    "age": 21,
    "gender": "Female",
    "country": "Vietnam",
    "academic_level": "Undergraduate",
    "most_used_platform": "Instagram",
    "purpose_of_use": "Education",
    "avg_daily_usage_hours": 4.0,
    "daily_unlocks": 120,
    "study_hours": 4.0,
    "physical_activity_hours": 1.0,
    "sleep_hours_per_night": 7.0,
    "stress_level": "Medium",
}


def test_health_reports_loaded_model(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["status"] == "ok"


def test_predict_returns_score_on_expected_scale(client) -> None:
    response = client.post("/predict", json=SAMPLE_INPUT)

    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["mental_health_score"] <= 10
    assert result["score_scale"] == "0-10"
    assert result["model_version"] == MODEL_VERSION


def test_predict_rejects_age_outside_dataset_range(client) -> None:
    invalid_input = {**SAMPLE_INPUT, "age": 30}
    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_model_info_returns_metadata(client) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == MODEL_VERSION
    assert body["selected_model"] in {
        "baseline_dummy",
        "linear_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert {"mae", "rmse", "r2"} == set(body["metrics"])
    assert len(body["feature_columns"]) == 12
    assert body["cv_results"]
    assert body["feature_importance"]


def test_predict_accepts_unseen_country(client) -> None:
    response = client.post("/predict", json={**SAMPLE_INPUT, "country": "Atlantis"})

    assert response.status_code == 200


def test_clamp_score_bounds() -> None:
    assert clamp_score(12.3) == 10.0
    assert clamp_score(-1.0) == 0.0
    assert clamp_score(6.42) == 6.42


def test_load_artifact_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m app.train"):
        load_artifact(tmp_path / "missing.joblib")


def test_load_artifact_rejects_old_metadata(tmp_path) -> None:
    old_artifact = {
        "model": object(),
        "metadata": {"model_version": "1.0.0", "feature_columns": []},
    }
    path = tmp_path / "old.joblib"
    joblib.dump(old_artifact, path)

    with pytest.raises(ValueError, match="metadata"):
        load_artifact(path)


def test_health_reports_loaded_artifact_version(
    model_artifact, tmp_path, monkeypatch
) -> None:
    artifact = joblib.load(model_artifact)
    artifact["metadata"]["model_version"] = "9.9.9"
    path = tmp_path / "v999.joblib"
    joblib.dump(artifact, path)
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(path))

    with TestClient(app) as local_client:
        response = local_client.get("/health")

    assert response.json()["model_version"] == "9.9.9"


class _FixedModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features):
        return [self.value]


@pytest.mark.parametrize(("raw", "expected"), [(12.3, 10.0), (-1.2, 0.0)])
def test_predict_clamps_out_of_range_prediction(
    client, monkeypatch, raw: float, expected: float
) -> None:
    monkeypatch.setitem(client.app.state.artifact, "model", _FixedModel(raw))

    response = client.post("/predict", json=SAMPLE_INPUT)

    assert response.status_code == 200
    assert response.json()["mental_health_score"] == expected


def test_app_logger_is_configured() -> None:
    logger = logging.getLogger("app")
    assert logger.level == logging.INFO
    assert logger.handlers
