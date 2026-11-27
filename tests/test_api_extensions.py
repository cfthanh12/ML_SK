"""TC-A: tests for the API features added in v1.2.0 (warnings, batch, timing, validation)."""

import statistics
import time

import pytest

from app.features import PARAMS

SAMPLE = {
    "age": 21,
    "gender": "Female",
    "country": "USA",
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


def test_a01_in_range_profile_has_no_warning(client) -> None:
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert response.json()["warnings"] == []


def test_a02_prediction_is_deterministic(client) -> None:
    assert client.post("/predict", json=SAMPLE).json() == client.post("/predict", json=SAMPLE).json()


@pytest.mark.parametrize(
    "change",
    [
        {"age": 17},
        {"gender": "Other"},
        {"stress_level": "Extreme"},
        {"most_used_platform": "Zalo"},
        {"purpose_of_use": "Gaming"},
        {"avg_daily_usage_hours": -1},
        {"sleep_hours_per_night": 25},
        {"daily_unlocks": 5000},
        {"country": "   "},
        {"avg_daily_usage_hours": "nhiều"},
    ],
)
def test_a03_invalid_values_rejected(client, change) -> None:
    assert client.post("/predict", json={**SAMPLE, **change}).status_code == 422


def test_a04_missing_and_unknown_fields_rejected(client) -> None:
    missing = {k: v for k, v in SAMPLE.items() if k != "sleep_hours_per_night"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "income": 1000}).status_code == 422


def test_a05_out_of_training_range_warns(client) -> None:
    body = client.post("/predict", json={**SAMPLE, "avg_daily_usage_hours": 15}).json()
    assert any("Avg_Daily_Usage_Hours" in w and "ngoại suy" in w for w in body["warnings"])


def test_a06_unseen_country_warns(client) -> None:
    body = client.post("/predict", json={**SAMPLE, "country": "Atlantis"}).json()
    assert any("chưa có" in w for w in body["warnings"])


def test_a07_rare_country_warns(client, monkeypatch) -> None:
    profile = dict(client.app.state.artifact["metadata"]["training_profile"])
    profile["country_counts"] = {**profile["country_counts"], "Vietnam": 5}
    monkeypatch.setitem(client.app.state.artifact["metadata"], "training_profile", profile)
    body = client.post("/predict", json={**SAMPLE, "country": "Vietnam"}).json()
    assert any("Vietnam" in w and "5 dòng" in w for w in body["warnings"])


def test_a08_batch_matches_single_predictions(client) -> None:
    items = [SAMPLE, {**SAMPLE, "avg_daily_usage_hours": 7.5}]
    batch = client.post("/predict/batch", json={"items": items}).json()
    singles = [client.post("/predict", json=item).json()["mental_health_score"] for item in items]
    assert batch["count"] == 2
    assert [p["mental_health_score"] for p in batch["predictions"]] == singles


@pytest.mark.parametrize("size", [0, PARAMS["api"]["max_batch_size"] + 1])
def test_a09_batch_size_limits(client, size) -> None:
    assert client.post("/predict/batch", json={"items": [SAMPLE] * size}).status_code == 422


def test_a10_model_info_exposes_training_profile(client) -> None:
    body = client.get("/model-info").json()
    assert set(body["numeric_ranges"]) >= {"Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"}
    assert "Country" in body["categories"]


def test_a11_latency_within_budget(client) -> None:
    client.post("/predict", json=SAMPLE)  # warm-up
    durations = []
    for _ in range(20):
        started = time.perf_counter()
        client.post("/predict", json=SAMPLE)
        durations.append((time.perf_counter() - started) * 1000)
    assert statistics.median(durations) < PARAMS["evaluation"]["max_latency_ms"]


def test_a12_timing_header_present(client) -> None:
    assert "x-process-time-ms" in client.post("/predict", json=SAMPLE).headers
