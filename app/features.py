"""Shared feature definitions for model training and API inference."""

TARGET_COLUMN = "Mental_Health_Score"

NUMERIC_FEATURES = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Stress_Level",
]

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
]

# --- Cấu hình dùng chung, đọc từ config/params.yaml ---------------------------
from pathlib import Path as _Path

import yaml as _yaml

PROJECT_ROOT = _Path(__file__).resolve().parents[1]


def load_params(path: _Path | None = None) -> dict:
    """Read config/params.yaml (single source of hyperparameters and thresholds)."""
    config_path = path or PROJECT_ROOT / "config" / "params.yaml"
    with open(config_path, encoding="utf-8") as handle:
        return _yaml.safe_load(handle)


PARAMS = load_params()

MODEL_VERSION = "1.2.0"
RANDOM_STATE = PARAMS["split"]["random_state"]
TEST_SIZE = PARAMS["split"]["test_size"]
DEFAULT_CV_FOLDS = PARAMS["cv_folds"]
