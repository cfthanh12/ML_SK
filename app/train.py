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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.features import (
    CATEGORICAL_FEATURES,
    DEFAULT_CV_FOLDS,
    FEATURE_COLUMNS,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    RANDOM_STATE,
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
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Select model columns and correct negative physical-activity values."""
    required_columns = list(feature_columns) + [TARGET_COLUMN]
    missing_columns = sorted(set(required_columns).difference(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    prepared = dataset.loc[:, required_columns].copy()
    numeric_features = [column for column in feature_columns if column in NUMERIC_FEATURES]
    categorical_features = [
        column for column in feature_columns if column in CATEGORICAL_FEATURES
    ]
    numeric_columns = numeric_features + [TARGET_COLUMN]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    if prepared[numeric_columns].isna().any().any():
        raise ValueError("Dataset contains missing numeric values.")
    if prepared[categorical_features].isna().any().any():
        raise ValueError("Dataset contains missing categorical values.")

    negative_activity_count = 0
    if "Physical_Activity_Hours" in numeric_features:
        negative_activity_count = int(
            (prepared["Physical_Activity_Hours"] < 0).sum()
        )
        prepared["Physical_Activity_Hours"] = prepared[
            "Physical_Activity_Hours"
        ].clip(lower=0)
    report = {"negative_physical_activity_values_clipped": negative_activity_count}
    return prepared[feature_columns], prepared[TARGET_COLUMN], report


def build_pipeline(
    regressor: Any | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Create a reproducible preprocessing and regression pipeline."""
    numeric = list(NUMERIC_FEATURES if numeric_features is None else numeric_features)
    categorical = list(
        CATEGORICAL_FEATURES
        if categorical_features is None
        else categorical_features
    )
    if regressor is None:
        regressor = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regressor", regressor),
        ]
    )


def build_candidate_models(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Pipeline]:
    """Return the fixed candidate models compared during model selection."""
    regressors = {
        "baseline_dummy": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: build_pipeline(regressor, numeric_features, categorical_features)
        for name, regressor in regressors.items()
    }


def cross_validate_candidates(
    models: dict[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    cv: int = DEFAULT_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare candidate models with shuffled K-fold cross-validation."""
    splits = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    records: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        scores = cross_validate(
            model,
            features,
            target,
            cv=splits,
            scoring=scoring,
            error_score="raise",
        )
        records[name] = {
            "mae_mean": float(-scores["test_mae"].mean()),
            "mae_std": float(scores["test_mae"].std()),
            "rmse_mean": float(-scores["test_rmse"].mean()),
            "rmse_std": float(scores["test_rmse"].std()),
            "r2_mean": float(scores["test_r2"].mean()),
            "r2_std": float(scores["test_r2"].std()),
        }
    table = pd.DataFrame.from_dict(records, orient="index")
    table.index.name = "model"
    return table.sort_values("rmse_mean")


def select_best_model(cv_table: pd.DataFrame) -> str:
    """Return the candidate with the lowest mean cross-validated RMSE."""
    return str(cv_table["rmse_mean"].idxmin())


def extract_feature_importance(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    """Measure each source column's holdout contribution by permutation."""
    result = permutation_importance(
        model,
        features,
        target,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )
    items = [
        {"feature": str(name), "importance": float(value)}
        for name, value in zip(features.columns, result.importances_mean)
    ]
    items.sort(key=lambda item: item["importance"], reverse=True)
    return items[:top_n]


def train_model(
    dataset: pd.DataFrame,
    cv: int = DEFAULT_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[
    Pipeline,
    dict[str, float],
    pd.DataFrame,
    dict[str, int],
    pd.DataFrame,
    str,
    list[dict[str, Any]],
]:
    """Select a model by CV on train data, then report on a fixed holdout."""
    features, target, cleaning_report = prepare_dataset(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    models = build_candidate_models()
    cv_table = cross_validate_candidates(
        models,
        x_train,
        y_train,
        cv=cv,
        random_state=random_state,
    )
    selected_model = select_best_model(cv_table)
    model = models[selected_model]
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
    feature_importance = extract_feature_importance(model, x_test, y_test)
    model.fit(features, target)
    return (
        model,
        metrics,
        evaluation,
        cleaning_report,
        cv_table,
        selected_model,
        feature_importance,
    )


def run_stress_ablation(
    dataset: pd.DataFrame,
    dropped_feature: str = "Stress_Level",
    cv: int = DEFAULT_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Repeat selection and holdout evaluation without one source feature."""
    feature_columns = [
        column for column in FEATURE_COLUMNS if column != dropped_feature
    ]
    numeric_features = [
        column for column in NUMERIC_FEATURES if column in feature_columns
    ]
    categorical_features = [
        column for column in CATEGORICAL_FEATURES if column in feature_columns
    ]
    features, target, _ = prepare_dataset(dataset, feature_columns=feature_columns)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )
    models = build_candidate_models(numeric_features, categorical_features)
    cv_table = cross_validate_candidates(
        models,
        x_train,
        y_train,
        cv=cv,
        random_state=random_state,
    )
    selected_model = select_best_model(cv_table)
    model = models[selected_model]
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    return {
        "dropped_feature": dropped_feature,
        "selected_model": selected_model,
        "metrics": metrics,
        "cv_results": cv_table.reset_index().to_dict("records"),
    }


def save_artifacts(
    model: Pipeline,
    metrics: dict[str, float],
    training_rows: int,
    cleaning_report: dict[str, int],
    cv_table: pd.DataFrame,
    selected_model: str,
    feature_importance: list[dict[str, Any]],
    stress_ablation: dict[str, Any] | None = None,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Write the trained model and a human-readable training summary."""
    metadata: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "target_column": TARGET_COLUMN,
        "selected_model": selected_model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "cv_results": cv_table.reset_index().to_dict("records"),
        "feature_importance": feature_importance,
        "feature_importance_method": "permutation",
        "stress_ablation": stress_ablation,
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
    cv: int = DEFAULT_CV_FOLDS,
) -> dict[str, Any]:
    """Run the full training workflow and return its saved metadata."""
    dataset = load_dataset(data_path)
    (
        model,
        metrics,
        _,
        cleaning_report,
        cv_table,
        selected_model,
        feature_importance,
    ) = train_model(dataset, cv=cv)
    stress_ablation = run_stress_ablation(dataset, cv=cv)
    return save_artifacts(
        model=model,
        metrics=metrics,
        training_rows=len(dataset),
        cleaning_report=cleaning_report,
        cv_table=cv_table,
        selected_model=selected_model,
        feature_importance=feature_importance,
        stress_ablation=stress_ablation,
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
