import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.train import train_and_save


def make_synthetic_frame(rows: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stress = rng.choice(["Low", "Medium", "High", "Very High"], rows)
    stress_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
    frame = pd.DataFrame(
        {
            "Age": rng.integers(18, 25, rows),
            "Gender": rng.choice(["Female", "Male"], rows),
            "Country": rng.choice(["Vietnam", "USA", "Other"], rows),
            "Academic_Level": rng.choice(
                ["High School", "Undergraduate", "Graduate"], rows
            ),
            "Most_Used_Platform": rng.choice(
                ["Instagram", "TikTok", "Facebook"], rows
            ),
            "Purpose_Of_Use": rng.choice(
                ["Education", "Entertainment", "Networking", "News"], rows
            ),
            "Avg_Daily_Usage_Hours": rng.uniform(1, 8, rows),
            "Daily_Unlocks": rng.integers(60, 275, rows),
            "Study_Hours": rng.uniform(0.5, 8, rows),
            "Physical_Activity_Hours": rng.uniform(-0.5, 4, rows),
            "Sleep_Hours_Per_Night": rng.uniform(4, 9, rows),
            "Stress_Level": stress,
        }
    )
    stress_score = np.array([stress_map[s] for s in stress])
    target = (
        8.5
        - 0.35 * frame["Avg_Daily_Usage_Hours"]
        + 0.25 * frame["Sleep_Hours_Per_Night"]
        + 0.15 * frame["Study_Hours"]
        + 0.10 * frame["Physical_Activity_Hours"]
        - 0.30 * stress_score
        + rng.normal(0, 0.2, rows)
    )
    frame["Mental_Health_Score"] = target.clip(0, 10)
    return frame


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture(scope="session")
def model_artifact(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("artifact")
    data_path = tmp / "synthetic.csv"
    artifact_path = tmp / "model.joblib"
    metrics_path = tmp / "metrics.json"
    make_synthetic_frame().to_csv(data_path, index=False)
    train_and_save(
        data_path=data_path,
        artifact_path=artifact_path,
        metrics_path=metrics_path,
        cv=2,
    )
    previous = os.environ.get("MODEL_ARTIFACT_PATH")
    os.environ["MODEL_ARTIFACT_PATH"] = str(artifact_path)
    yield artifact_path
    if previous is None:
        os.environ.pop("MODEL_ARTIFACT_PATH", None)
    else:
        os.environ["MODEL_ARTIFACT_PATH"] = previous


@pytest.fixture
def client(model_artifact):
    with TestClient(app) as test_client:
        yield test_client
