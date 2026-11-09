"""Validated JSON schemas for the prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.features import FEATURE_COLUMNS


class PredictionRequest(BaseModel):
    """Student profile accepted by the prediction endpoint."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=24, description="Age in years; supported dataset range is 18-24.")
    gender: Literal["Female", "Male"]
    country: str = Field(min_length=1, max_length=100)
    academic_level: Literal["High School", "Undergraduate", "Graduate"]
    most_used_platform: Literal[
        "Facebook",
        "Instagram",
        "KakaoTalk",
        "LINE",
        "LinkedIn",
        "Snapchat",
        "TikTok",
        "Twitter",
        "VKontakte",
        "WeChat",
        "WhatsApp",
        "YouTube",
    ]
    purpose_of_use: Literal["Education", "Entertainment", "Networking", "News"]
    avg_daily_usage_hours: float = Field(ge=0, le=24)
    daily_unlocks: int = Field(ge=0, le=1000)
    study_hours: float = Field(ge=0, le=24)
    physical_activity_hours: float = Field(ge=0, le=24)
    sleep_hours_per_night: float = Field(ge=0, le=24)
    stress_level: Literal["Low", "Medium", "High", "Very High"]

    @field_validator("country")
    @classmethod
    def country_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("country must not be blank")
        return value

    def to_model_features(self) -> dict[str, object]:
        """Return one feature row using the CSV's column names and order."""
        values = {
            "Age": self.age,
            "Gender": self.gender,
            "Country": self.country,
            "Academic_Level": self.academic_level,
            "Most_Used_Platform": self.most_used_platform,
            "Purpose_Of_Use": self.purpose_of_use,
            "Avg_Daily_Usage_Hours": self.avg_daily_usage_hours,
            "Daily_Unlocks": self.daily_unlocks,
            "Study_Hours": self.study_hours,
            "Physical_Activity_Hours": self.physical_activity_hours,
            "Sleep_Hours_Per_Night": self.sleep_hours_per_night,
            "Stress_Level": self.stress_level,
        }
        return {column: values[column] for column in FEATURE_COLUMNS}


class PredictionResponse(BaseModel):
    """Estimated mental-health score returned by the API."""

    mental_health_score: float = Field(description="Estimated score on the dataset's 0-10 scale.")
    score_scale: str = "0-10"
    model_version: str


class FeatureImportanceItem(BaseModel):
    """One source feature's holdout permutation importance."""

    feature: str
    importance: float


class CVResultItem(BaseModel):
    """Cross-validation summary for one candidate model."""

    model: str
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    r2_mean: float
    r2_std: float


class ModelInfoResponse(BaseModel):
    """Training metadata exposed for the demo UI and reviewers."""

    model_version: str
    selected_model: str
    metrics: dict[str, float]
    cv_results: list[CVResultItem]
    feature_columns: list[str]
    feature_importance: list[FeatureImportanceItem]
    feature_importance_method: str
    training_rows: int
    trained_at_utc: str
