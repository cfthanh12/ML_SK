import joblib
import pytest

from app.features import FEATURE_COLUMNS, MODEL_VERSION
from app.train import DEFAULT_ARTIFACT_PATH, build_candidate_models

pytestmark = pytest.mark.skipif(
    not DEFAULT_ARTIFACT_PATH.is_file(),
    reason="committed artifact not generated yet",
)


def test_committed_artifact_matches_code() -> None:
    artifact = joblib.load(DEFAULT_ARTIFACT_PATH)
    metadata = artifact["metadata"]
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["selected_model"] in build_candidate_models()
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert list(artifact["model"].feature_names_in_) == FEATURE_COLUMNS
