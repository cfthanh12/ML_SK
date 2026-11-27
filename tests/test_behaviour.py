"""TC-B: behavioural tests on the real data (Chip Huyen, Designing ML Systems, ch.6).

They check that the model learned the right direction, not only a high R^2.
"""

import numpy as np
import pytest
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from app.features import RANDOM_STATE, TEST_SIZE
from app.train import DEFAULT_DATA_PATH, build_pipeline, load_dataset, prepare_dataset

pytestmark = pytest.mark.skipif(not DEFAULT_DATA_PATH.is_file(), reason="dataset not present")


@pytest.fixture(scope="module")
def fitted():
    features, target, _ = prepare_dataset(load_dataset(DEFAULT_DATA_PATH))
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return build_pipeline().fit(x_train, y_train), x_train, x_test, y_train, y_test


def _with(frame, column, value):
    changed = frame.copy()
    changed[column] = value
    return changed


def test_b01_directional_more_usage_lowers_score(fitted) -> None:
    model, _, x_test, _, _ = fitted
    low = model.predict(_with(x_test, "Avg_Daily_Usage_Hours", 2.0))
    high = model.predict(_with(x_test, "Avg_Daily_Usage_Hours", 7.0))
    assert (high < low).mean() >= 0.95


def test_b02_directional_more_sleep_raises_score(fitted) -> None:
    model, _, x_test, _, _ = fitted
    short = model.predict(_with(x_test, "Sleep_Hours_Per_Night", 5.0))
    long = model.predict(_with(x_test, "Sleep_Hours_Per_Night", 8.0))
    assert (long > short).mean() >= 0.80


def test_b03_invariance_gender_small_effect(fitted) -> None:
    model, _, x_test, _, _ = fitted
    flipped = _with(x_test, "Gender", x_test["Gender"].map({"Male": "Female", "Female": "Male"}))
    change = np.abs(model.predict(flipped) - model.predict(x_test))
    assert change.mean() < 0.15
    assert (change > 0.5).mean() < 0.05


def test_b04_perturbation_small_noise_small_change(fitted) -> None:
    model, _, x_test, _, _ = fitted
    rng = np.random.default_rng(0)
    noisy = x_test.copy()
    for column in ["Avg_Daily_Usage_Hours", "Study_Hours", "Sleep_Hours_Per_Night"]:
        noisy[column] = noisy[column] + rng.normal(0, 0.25, len(noisy))
    assert np.abs(model.predict(noisy) - model.predict(x_test)).mean() < 0.30


def test_b05_holdout_quality_and_overfitting_gap(fitted) -> None:
    model, x_train, x_test, y_train, y_test = fitted
    test_r2 = r2_score(y_test, model.predict(x_test))
    assert test_r2 >= 0.85
    assert r2_score(y_train, model.predict(x_train)) - test_r2 < 0.10


def test_b06_no_extrapolation_beyond_training_range(fitted) -> None:
    """Random Forest cannot extrapolate: usage above the max seen in training (8.8 h) is flat."""
    model, x_train, x_test, _, _ = fitted
    edge = float(x_train["Avg_Daily_Usage_Hours"].max())
    sample = x_test.iloc[:200]
    at_edge = model.predict(_with(sample, "Avg_Daily_Usage_Hours", edge))
    for hours in (12.0, 20.0):
        far = model.predict(_with(sample, "Avg_Daily_Usage_Hours", hours))
        assert np.abs(far - at_edge).max() < 1e-9


def test_b07_stress_level_is_redundant(fitted) -> None:
    """Ablation: dropping Stress_Level barely changes holdout R^2 (info overlaps with usage/sleep)."""
    from app.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    model, x_train, x_test, y_train, y_test = fitted
    full_r2 = r2_score(y_test, model.predict(x_test))
    categorical = [c for c in CATEGORICAL_FEATURES if c != "Stress_Level"]
    reduced = build_pipeline(numeric_features=list(NUMERIC_FEATURES), categorical_features=categorical)
    columns = list(NUMERIC_FEATURES) + categorical
    reduced.fit(x_train[columns], y_train)
    reduced_r2 = r2_score(y_test, reduced.predict(x_test[columns]))
    assert abs(full_r2 - reduced_r2) < 0.01


def test_b08_regression_to_the_mean_at_score_extremes(fitted) -> None:
    """Known limitation, tracked so a future model can show improvement.

    Residual = actual - predicted: negative for low true scores (model too high),
    positive for high true scores (model too low). Each band's bias stays below 0.6.
    """
    model, _, x_test, _, y_test = fitted
    residual = y_test.to_numpy() - model.predict(x_test)
    low = residual[y_test.to_numpy() <= 5]
    high = residual[y_test.to_numpy() > 8]
    assert len(low) >= 30 and len(high) >= 30
    assert low.mean() < 0 < high.mean()
    assert abs(low.mean()) < 0.6 and abs(high.mean()) < 0.6
