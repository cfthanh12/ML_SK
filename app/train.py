"""Train, evaluate, and save the student wellbeing regression model."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "Student Social Media And Mental Health Impact.csv"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "mental_health_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"


def load_dataset(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the provided CSV and fail clearly if its schema is incomplete."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = pd.read_csv(data_path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if dataset.empty:
        raise ValueError("Dataset contains no rows.")
    return dataset


def prepare_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Select model columns and correct negative physical-activity values."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = sorted(set(required_columns).difference(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    prepared = dataset.loc[:, required_columns].copy()
    numeric_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    if prepared[numeric_columns].isna().any().any():
        raise ValueError("Dataset contains missing numeric values.")
    if prepared[CATEGORICAL_FEATURES].isna().any().any():
        raise ValueError("Dataset contains missing categorical values.")

    negative_activity_count = int((prepared["Physical_Activity_Hours"] < 0).sum())
    prepared["Physical_Activity_Hours"] = prepared["Physical_Activity_Hours"].clip(lower=0)
    report = {"negative_physical_activity_values_clipped": negative_activity_count}
    return prepared[FEATURE_COLUMNS], prepared[TARGET_COLUMN], report


def build_pipeline() -> Pipeline:
    """Create the reproducible preprocessing and regression pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(
    dataset: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float], pd.DataFrame, dict[str, int]]:
    """Evaluate on a fixed holdout split, then fit the final model on all rows."""
    features, target, cleaning_report = prepare_dataset(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    evaluation = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": predictions,
        },
        index=y_test.index,
    )
    model.fit(features, target)
    return model, metrics, evaluation, cleaning_report


def save_artifacts(
    model: Pipeline,
    metrics: dict[str, float],
    training_rows: int,
    cleaning_report: dict[str, int],
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Write the trained model and a human-readable training summary."""
    metadata: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "training_rows": training_rows,
        "cleaning_report": cleaning_report,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, artifact_path)
    metrics_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def train_and_save(
    data_path: Path = DEFAULT_DATA_PATH,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Run the full training workflow and return its saved metadata."""
    dataset = load_dataset(data_path)
    model, metrics, _, cleaning_report = train_model(dataset)
    return save_artifacts(
        model=model,
        metrics=metrics,
        training_rows=len(dataset),
        cleaning_report=cleaning_report,
        artifact_path=artifact_path,
        metrics_path=metrics_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    metadata = train_and_save(args.data, args.artifact, args.metrics)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
