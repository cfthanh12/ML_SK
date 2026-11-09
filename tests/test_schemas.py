import pytest
from pydantic import ValidationError

from app.schemas import PredictionRequest


def valid_payload() -> dict:
    return {
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


@pytest.mark.parametrize("age", [18, 24])
def test_age_boundaries_accepted(age: int) -> None:
    assert PredictionRequest(**{**valid_payload(), "age": age}).age == age


@pytest.mark.parametrize("age", [17, 25])
def test_age_outside_range_rejected(age: int) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), "age": age})


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), "unexpected": 1})


def test_blank_country_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), "country": "   "})


@pytest.mark.parametrize(
    "field",
    [
        "avg_daily_usage_hours",
        "study_hours",
        "physical_activity_hours",
        "sleep_hours_per_night",
    ],
)
def test_hours_outside_range_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), field: -1.0})
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), field: 25.0})
