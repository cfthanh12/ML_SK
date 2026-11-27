"""TC-D08..D10 and TC-M01: preprocessing export and metadata.yaml (template structure items)."""

import json

import joblib
import pandas as pd
import pytest
import yaml

from app.features import MODEL_VERSION, TARGET_COLUMN
from app.preprocess import export_processed
from app.train import DEFAULT_ARTIFACT_PATH, DEFAULT_DATA_PATH, DEFAULT_METADATA_YAML_PATH

needs_data = pytest.mark.skipif(not DEFAULT_DATA_PATH.is_file(), reason="dataset not present")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    return out, export_processed(DEFAULT_DATA_PATH, out)


@needs_data
def test_d08_processed_files_written(exported) -> None:
    out, _ = exported
    for name in ["cleaned.csv", "train.csv", "holdout.csv",
                 "encoded_train_sample.csv", "preprocessing_report.json"]:
        assert (out / name).is_file(), name


@needs_data
def test_d09_cleaning_and_split_are_consistent(exported) -> None:
    out, summary = exported
    cleaned = pd.read_csv(out / "cleaned.csv", index_col="row_id")
    train = pd.read_csv(out / "train.csv", index_col="row_id")
    holdout = pd.read_csv(out / "holdout.csv", index_col="row_id")
    assert len(cleaned) == 5000 and (cleaned["Physical_Activity_Hours"] >= 0).all()
    assert (len(train), len(holdout)) == (4000, 1000)
    assert set(train.index).isdisjoint(holdout.index)
    assert set(cleaned.index[cleaned["split"] == "holdout"]) == set(holdout.index)
    assert summary["cleaning"]["negative_physical_activity_values_clipped"] == 10


@needs_data
def test_d10_encoded_sample_matches_report(exported) -> None:
    out, summary = exported
    encoded = pd.read_csv(out / "encoded_train_sample.csv")
    feature_columns = [c for c in encoded.columns if c not in ("row_id", TARGET_COLUMN)]
    assert len(feature_columns) == summary["encoded_columns_total"]
    assert summary["encoded_numeric_columns"] == 6
    one_hot = encoded[feature_columns[6:]]
    # each of the 6 categorical variables contributes exactly one "1" per row
    assert (one_hot.sum(axis=1) == 6).all()
    assert json.loads((out / "preprocessing_report.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(not DEFAULT_ARTIFACT_PATH.is_file() or not DEFAULT_METADATA_YAML_PATH.is_file(),
                    reason="committed artifact/metadata not generated yet")
def test_m01_metadata_yaml_matches_artifact() -> None:
    meta = yaml.safe_load(DEFAULT_METADATA_YAML_PATH.read_text(encoding="utf-8"))
    artifact = joblib.load(DEFAULT_ARTIFACT_PATH)["metadata"]
    assert meta["model_version"] == MODEL_VERSION == artifact["model_version"]
    assert meta["trained_at_utc"] == artifact["trained_at_utc"]
    assert meta["metrics_holdout"]["r2"] == round(artifact["metrics"]["r2"], 4)
