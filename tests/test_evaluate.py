"""Unit tests for helpers in app/evaluate.py."""

import pytest

from app.evaluate import adjusted_r2, regression_metrics


def test_adjusted_r2_formula() -> None:
    # 1 - (1 - 0.9) * (100 - 1) / (100 - 10 - 1) = 0.888764...
    assert adjusted_r2(0.9, 100, 10) == pytest.approx(0.888764, abs=1e-6)


def test_adjusted_r2_not_above_r2() -> None:
    assert adjusted_r2(0.8, 1000, 140) <= 0.8


def test_regression_metrics_keys() -> None:
    result = regression_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2], n_predictors=1)
    assert set(result) == {"mae", "rmse", "r2", "adjusted_r2"}
