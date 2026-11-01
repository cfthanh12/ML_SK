"""Focused tests for API validation and model predictions."""

from fastapi.testclient import TestClient

from app.main import app


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


def test_health_reports_loaded_model() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["status"] == "ok"


def test_predict_returns_score_on_expected_scale() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE_INPUT)

    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["mental_health_score"] <= 10
    assert result["score_scale"] == "0-10"
    assert result["model_version"] == "1.0.0"


def test_predict_rejects_age_outside_dataset_range() -> None:
    invalid_input = {**SAMPLE_INPUT, "age": 30}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422
