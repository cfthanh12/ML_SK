import json
from pathlib import Path

import pytest

from scripts.report_figures import build_figures, collect_report_data

ROOT = Path(__file__).resolve().parents[1]


def test_report_data_matches_metrics_json():
    data = collect_report_data()
    metrics = json.loads(
        (ROOT / "reports" / "metrics.json").read_text(encoding="utf-8")
    )
    assert data["model_version"] == metrics["model_version"]
    assert data["selected_model"] == metrics["selected_model"]
    for key in ("mae", "rmse", "r2"):
        assert abs(data["metrics"][key] - metrics["metrics"][key]) < 1e-6
    assert (
        abs(
            data["stress_ablation"]["metrics"]["r2"]
            - metrics["stress_ablation"]["metrics"]["r2"]
        )
        < 1e-6
    )
    assert data["dataset"]["rows"] == 5000
    assert 0.0 <= data["predictions"]["example"]["score"] <= 10.0
    assert len(data["predictions"]["presets"]) == 3
    assert len(data["figures"]) == 12
    assert abs(data["residual_mean"]) < 0.05
    assert data["residual_std"] > 0.3


def test_metric_match_guard_rejects_drift():
    from scripts.report_figures import _assert_metrics_match

    with pytest.raises(RuntimeError):
        _assert_metrics_match(
            {"mae": 0.1, "rmse": 0.2, "r2": 0.9},
            {"mae": 0.1, "rmse": 0.2, "r2": 0.91},
            "ablation",
        )


def test_build_figures_writes_twelve_pngs(tmp_path):
    paths = build_figures(collect_report_data(), tmp_path)
    assert sorted(paths) == sorted(
        [
            "architecture",
            "target_hist",
            "stress_counts",
            "correlation",
            "scatter_numeric",
            "cv_rmse",
            "actual_vs_predicted",
            "residuals",
            "importance",
            "usage_threshold",
            "learning_curve",
            "what_if",
        ]
    )
    for path in paths.values():
        assert path.stat().st_size > 10_000
