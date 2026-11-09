import json

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from app.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from app.train import (
    build_candidate_models,
    build_pipeline,
    cross_validate_candidates,
    extract_feature_importance,
    load_dataset,
    prepare_dataset,
    run_stress_ablation,
    save_artifacts,
    select_best_model,
    train_and_save,
    train_model,
)


def test_prepare_dataset_clips_negative_activity(synthetic_dataset: pd.DataFrame) -> None:
    frame = synthetic_dataset.copy()
    frame["Physical_Activity_Hours"] = frame["Physical_Activity_Hours"].abs()
    frame.loc[0, "Physical_Activity_Hours"] = -1.5
    frame.loc[1, "Physical_Activity_Hours"] = -0.2
    features, target, report = prepare_dataset(frame)
    assert features["Physical_Activity_Hours"].min() >= 0
    assert report["negative_physical_activity_values_clipped"] == 2
    assert len(target) == len(frame)


def test_prepare_dataset_honors_feature_subset(synthetic_dataset: pd.DataFrame) -> None:
    subset = [column for column in FEATURE_COLUMNS if column != "Stress_Level"]
    features, _, _ = prepare_dataset(synthetic_dataset, feature_columns=subset)
    assert list(features.columns) == subset
    assert "Stress_Level" not in features.columns


def test_prepare_dataset_missing_column_raises(
    synthetic_dataset: pd.DataFrame,
) -> None:
    frame = synthetic_dataset.drop(columns=["Stress_Level"])
    with pytest.raises(ValueError, match="Stress_Level"):
        prepare_dataset(frame)


def test_load_dataset_missing_column_raises(
    synthetic_dataset: pd.DataFrame, tmp_path
) -> None:
    path = tmp_path / "bad.csv"
    synthetic_dataset.drop(columns=["Age"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Age"):
        load_dataset(path)


def test_build_candidate_models_returns_four_pipelines() -> None:
    models = build_candidate_models()
    assert set(models) == {
        "baseline_dummy",
        "linear_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert all(
        hasattr(model, "fit") and hasattr(model, "predict")
        for model in models.values()
    )


def test_build_pipeline_with_subset_features(synthetic_dataset: pd.DataFrame) -> None:
    subset = [column for column in FEATURE_COLUMNS if column != "Stress_Level"]
    features, target, _ = prepare_dataset(synthetic_dataset, feature_columns=subset)
    pipeline = build_pipeline(
        regressor=LinearRegression(),
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=[
            c for c in CATEGORICAL_FEATURES if c != "Stress_Level"
        ],
    )
    pipeline.fit(features, target)
    assert pipeline.predict(features).shape == (len(features),)


def test_cross_validate_candidates_returns_row_per_model(
    synthetic_dataset: pd.DataFrame,
) -> None:
    features, target, _ = prepare_dataset(synthetic_dataset)
    table = cross_validate_candidates(build_candidate_models(), features, target, cv=2)
    assert set(table.index) == {
        "baseline_dummy",
        "linear_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert {
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
    } <= set(table.columns)
    assert table["rmse_mean"].is_monotonic_increasing
    assert table["rmse_mean"].min() >= 0
    assert table["mae_mean"].min() >= 0


def test_select_best_model_returns_lowest_rmse() -> None:
    table = pd.DataFrame({"rmse_mean": [1.0, 0.5, 0.8]}, index=["a", "b", "c"])
    table.index.name = "model"
    assert select_best_model(table) == "b"


def test_extract_feature_importance_returns_sorted_top_features(
    synthetic_dataset: pd.DataFrame,
) -> None:
    features, target, _ = prepare_dataset(synthetic_dataset)
    model = build_pipeline(regressor=LinearRegression())
    model.fit(features, target)
    items = extract_feature_importance(model, features, target, top_n=12)
    assert 0 < len(items) <= 12
    assert {item["feature"] for item in items} <= set(FEATURE_COLUMNS)
    importances = [item["importance"] for item in items]
    assert importances == sorted(importances, reverse=True)
    assert all(isinstance(value, float) for value in importances)


def test_train_model_returns_cv_and_selection(
    synthetic_dataset: pd.DataFrame,
) -> None:
    model, metrics, evaluation, report, cv_table, selected, importance = train_model(
        synthetic_dataset, cv=2
    )
    assert {"mae", "rmse", "r2"} == set(metrics)
    assert set(cv_table.index) == {
        "baseline_dummy",
        "linear_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert selected in set(cv_table.index)
    assert len(evaluation) == 40
    assert importance and importance[0]["feature"] in FEATURE_COLUMNS


def test_run_stress_ablation_drops_feature(
    synthetic_dataset: pd.DataFrame,
) -> None:
    result = run_stress_ablation(synthetic_dataset, cv=2)
    assert result["dropped_feature"] == "Stress_Level"
    assert any(row["model"] == result["selected_model"] for row in result["cv_results"])
    assert {"mae", "rmse", "r2"} == set(result["metrics"])


def test_save_artifacts_writes_json_metadata(
    synthetic_dataset: pd.DataFrame, tmp_path
) -> None:
    model, metrics, _, report, cv_table, selected, importance = train_model(
        synthetic_dataset, cv=2
    )
    ablation = run_stress_ablation(synthetic_dataset, cv=2)
    artifact_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    save_artifacts(
        model,
        metrics,
        len(synthetic_dataset),
        report,
        cv_table,
        selected,
        importance,
        stress_ablation=ablation,
        artifact_path=artifact_path,
        metrics_path=metrics_path,
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["model_version"] == "1.1.0"
    assert payload["selected_model"] == selected
    assert payload["feature_importance_method"] == "permutation"
    assert payload["stress_ablation"]["dropped_feature"] == "Stress_Level"
    assert artifact_path.is_file()


def test_train_and_save_end_to_end(
    synthetic_dataset: pd.DataFrame, tmp_path
) -> None:
    data_path = tmp_path / "data.csv"
    synthetic_dataset.to_csv(data_path, index=False)
    metadata = train_and_save(
        data_path=data_path,
        artifact_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        cv=2,
    )
    assert metadata["training_rows"] == len(synthetic_dataset)
