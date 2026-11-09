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

MODEL_VERSION = "1.1.0"
RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5
