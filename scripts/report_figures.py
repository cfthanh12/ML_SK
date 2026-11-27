"""Build the report's data file and figures from the real dataset and artifact."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from app.features import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from app.schemas import PredictionRequest
from app.train import (
    DEFAULT_DATA_PATH,
    DEFAULT_METRICS_PATH,
    load_dataset,
    prepare_dataset,
    run_stress_ablation,
    train_model,
)

FIGURE_FILES = {
    "architecture": "01_architecture.png",
    "target_hist": "02_target_hist.png",
    "stress_counts": "03_stress_counts.png",
    "correlation": "04_correlation.png",
    "scatter_numeric": "05_scatter_numeric.png",
    "cv_rmse": "06_cv_rmse.png",
    "actual_vs_predicted": "07_actual_vs_predicted.png",
    "residuals": "08_residuals.png",
    "importance": "09_importance.png",
    # Ba hình dưới được sinh bởi python -m app.evaluate (reports/figures) và chép sang.
    "usage_threshold": "10_usage_threshold.png",
    "learning_curve": "11_learning_curve.png",
    "what_if": "12_what_if_usage.png",
}
EVALUATE_FIGURES = {
    "usage_threshold": "usage_threshold.png",
    "learning_curve": "learning_curve.png",
    "what_if": "what_if_usage.png",
}
EXPERIMENTS_PATH = ROOT / "reports" / "experiments.json"

SAMPLE_PAYLOAD = {
    "age": 21,
    "gender": "Female",
    "country": "Vietnam",
    "academic_level": "Undergraduate",
    "most_used_platform": "Instagram",
    "purpose_of_use": "Education",
    "avg_daily_usage_hours": 4.0,
    "daily_unlocks": 120,
    "study_hours": 4.0,
    "physical_activity_hours": 1.0,
    "sleep_hours_per_night": 7.0,
    "stress_level": "Medium",
}

_RAW: dict[str, Any] | None = None


def _round_floats(value: Any, digits: int = 6) -> Any:
    if isinstance(value, dict):
        return {key: _round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item, digits) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _load_presets() -> dict[str, dict[str, Any]]:
    """Read PRESETS without importing streamlit_app (which has side effects)."""
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "PRESETS" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("Không tìm thấy PRESETS trong streamlit_app.py")


def _load_experiments() -> dict[str, Any]:
    if not EXPERIMENTS_PATH.is_file():
        raise RuntimeError(
            "Thiếu reports/experiments.json. Chạy python -m app.evaluate trước khi build báo cáo."
        )
    return json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))


def _count_tests() -> int | None:
    """Number of collected pytest cases, shown in the report's testing chapter."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    )
    for line in reversed(result.stdout.splitlines()):
        if "test" in line and line.split()[0].isdigit():
            return int(line.split()[0])
    return None


def _warnings(payload: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    from app.main import collect_warnings

    return collect_warnings(PredictionRequest(**payload).to_model_features(), profile)


def _score(payload: dict[str, Any], model: Any) -> float:
    row = PredictionRequest(**payload).to_model_features()
    frame = pd.DataFrame([row])[FEATURE_COLUMNS]
    raw = float(model.predict(frame)[0])
    return round(min(10.0, max(0.0, raw)), 2)


def _assert_metrics_match(
    reproduced: dict[str, float],
    stored: dict[str, float],
    label: str,
) -> None:
    for key in ("mae", "rmse", "r2"):
        if abs(reproduced[key] - stored[key]) > 1e-6:
            raise RuntimeError(
                f"{label} không tái tạo được: {reproduced} != {stored}. "
                "Chạy lại python -m app.train trước khi build báo cáo."
            )


@lru_cache(maxsize=1)
def _collect_uncached() -> dict[str, Any]:
    global _RAW

    dataset = load_dataset(DEFAULT_DATA_PATH)
    stored = json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
    (
        model,
        holdout_metrics,
        evaluation,
        cleaning_report,
        cv_table,
        selected_model,
        feature_importance,
    ) = train_model(dataset)
    _assert_metrics_match(holdout_metrics, stored["metrics"], "Holdout")
    ablation_recheck = run_stress_ablation(dataset)
    _assert_metrics_match(
        ablation_recheck["metrics"],
        stored["stress_ablation"]["metrics"],
        "Ablation Stress_Level",
    )

    features, target, _ = prepare_dataset(dataset)
    combined = pd.concat([features, target], axis=1)
    correlations = (
        combined[NUMERIC_FEATURES].corrwith(combined[TARGET_COLUMN]).dropna()
    )
    correlations = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )

    describe_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    describe = [
        {
            "column": column,
            "mean": dataset[column].mean(),
            "std": dataset[column].std(),
            "min": dataset[column].min(),
            "p25": dataset[column].quantile(0.25),
            "p50": dataset[column].median(),
            "p75": dataset[column].quantile(0.75),
            "max": dataset[column].max(),
        }
        for column in describe_columns
    ]

    stress_groups = (
        dataset.groupby("Stress_Level")[TARGET_COLUMN]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    platform_groups = (
        dataset.groupby("Most_Used_Platform")[TARGET_COLUMN]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
    )

    residuals = (evaluation["actual"] - evaluation["predicted"]).astype(float)
    artifact = joblib.load(ROOT / "models" / "mental_health_model.joblib")
    api_model = artifact["model"]
    presets = _load_presets()

    data = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_version": stored["model_version"],
        "selected_model": selected_model,
        "trained_at_utc": stored["trained_at_utc"],
        "training_rows": int(stored["training_rows"]),
        "dataset": {
            "rows": int(dataset.shape[0]),
            "columns": int(dataset.shape[1]),
            "feature_count": len(FEATURE_COLUMNS),
            "target": {
                "column": TARGET_COLUMN,
                "min": float(target.min()),
                "max": float(target.max()),
                "mean": float(target.mean()),
                "p25": float(target.quantile(0.25)),
                "p75": float(target.quantile(0.75)),
            },
        },
        "quality": {
            "duplicate_rows": int(dataset.duplicated().sum()),
            "negative_physical_activity": int(
                (dataset["Physical_Activity_Hours"] < 0).sum()
            ),
            "country_count": int(dataset["Country"].nunique()),
            "country_other_rows": int((dataset["Country"] == "Other").sum()),
        },
        "describe": describe,
        "correlations": [
            {"feature": feature, "value": float(value)}
            for feature, value in correlations.items()
        ],
        "stress_groups": [
            {"level": str(level), "mean": float(row["mean"]), "count": int(row["count"])}
            for level, row in stress_groups.iterrows()
        ],
        "platform_groups": [
            {
                "platform": str(platform),
                "mean": float(row["mean"]),
                "count": int(row["count"]),
            }
            for platform, row in platform_groups.iterrows()
        ],
        "metrics": {key: float(value) for key, value in holdout_metrics.items()},
        "cv_results": cv_table.reset_index().to_dict("records"),
        "feature_importance": feature_importance,
        "stress_ablation": stored["stress_ablation"],
        "cleaning_report": cleaning_report,
        "thresholds": {
            "low": float(target.quantile(0.25)),
            "high": float(target.quantile(0.75)),
        },
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "predictions": {
            "example": {
                "payload": SAMPLE_PAYLOAD,
                "score": _score(SAMPLE_PAYLOAD, api_model),
                "warnings": _warnings(
                    SAMPLE_PAYLOAD, artifact["metadata"].get("training_profile", {})
                ),
            },
            "presets": [
                {"name": name, "score": _score(preset, api_model)}
                for name, preset in presets.items()
            ],
        },
        "figures": dict(FIGURE_FILES),
        "extended": _load_experiments(),
        "test_count": _count_tests(),
    }
    _RAW = {
        "dataset": dataset,
        "evaluation": evaluation,
        "cv_table": cv_table,
    }
    return _round_floats(data)


def collect_report_data() -> dict[str, Any]:
    """Return every number the report needs, reproduced from the repo inputs."""
    return _collect_uncached()


def write_report_data(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _architecture(path: Path) -> None:
    figure, axes = plt.subplots(figsize=(11, 4.6))
    axes.set_xlim(0, 11)
    axes.set_ylim(0, 4.6)
    axes.axis("off")

    boxes = [
        ("CSV\n5.000 dòng × 13 cột", 1.35, 3.15),
        ("app/train.py\n5-fold CV + holdout\nchọn mô hình", 3.55, 3.15),
        ("models/*.joblib\n+ reports/", 5.85, 3.15),
        ("FastAPI\n/predict · /predict/batch\n/model-info · cảnh báo", 8.15, 3.15),
        ("Streamlit\nform · what-if", 10.0, 3.15),
        ("notebooks/01–03\n(EDA, mô hình, suy luận)", 3.55, 0.9),
        ("app/evaluate.py\ntinh chỉnh, learning curve,\nkiểm thử hành vi", 5.85, 0.9),
    ]
    width, height = 1.55, 0.95
    for label, x, y in boxes:
        axes.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.1,
                edgecolor="#333333",
                facecolor="#F2F2F2",
            )
        )
        axes.text(x, y, label, ha="center", va="center", fontsize=9)

    arrows = [
        ((2.12, 3.15), (2.78, 3.15)),
        ((4.32, 3.15), (5.08, 3.15)),
        ((6.62, 3.15), (7.38, 3.15)),
        ((8.93, 3.15), (9.22, 3.15)),
        ((3.55, 1.38), (3.55, 2.68)),
        ((5.85, 1.38), (5.85, 2.68)),
    ]
    for start, end in arrows:
        axes.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.1,
                color="#333333",
            )
        )

    axes.text(5.5, 4.28, "Luồng huấn luyện → artifact → phục vụ dự đoán", ha="center", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _target_hist(dataset: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(figsize=(9, 4.2))
    axes.hist(dataset[TARGET_COLUMN], bins=24, color="#4C72B0", edgecolor="white")
    axes.set_xlabel("Mental_Health_Score")
    axes.set_ylabel("Số dòng")
    axes.set_title("Phân bố điểm sức khỏe tinh thần")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _stress_counts(dataset: pd.DataFrame, path: Path) -> None:
    counts = dataset["Stress_Level"].value_counts()
    figure, axes = plt.subplots(figsize=(8, 4.2))
    axes.bar(counts.index, counts.values, color="#55A868")
    axes.set_xlabel("Stress_Level")
    axes.set_ylabel("Số dòng")
    axes.set_title("Số dòng theo mức căng thẳng")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _correlation(dataset: pd.DataFrame, path: Path) -> None:
    columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    matrix = dataset[columns].corr()
    figure, axes = plt.subplots(figsize=(8, 6))
    image = axes.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes.set_xticks(range(len(columns)))
    axes.set_xticklabels(columns, rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(len(columns)))
    axes.set_yticklabels(columns, fontsize=8)
    axes.set_title("Ma trận tương quan các biến số")
    for row in range(len(columns)):
        for col in range(len(columns)):
            axes.text(
                col,
                row,
                f"{matrix.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    figure.colorbar(image, ax=axes, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _scatter_numeric(dataset: pd.DataFrame, path: Path) -> None:
    columns = ["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Study_Hours"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, column in zip(axes, columns):
        axis.scatter(dataset[column], dataset[TARGET_COLUMN], s=8, alpha=0.25)
        axis.set_xlabel(column)
        axis.set_ylabel("Mental_Health_Score")
    figure.suptitle("Quan hệ giữa thói quen và điểm số")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _cv_rmse(data: dict[str, Any], path: Path) -> None:
    rows = sorted(data["cv_results"], key=lambda row: row["rmse_mean"])
    names = [row["model"] for row in rows]
    values = [row["rmse_mean"] for row in rows]
    errors = [row["rmse_std"] for row in rows]
    figure, axes = plt.subplots(figsize=(9, 4.2))
    axes.bar(names, values, yerr=errors, capsize=5, color="#4C72B0")
    axes.set_ylabel("RMSE trung bình (thấp hơn là tốt hơn)")
    axes.set_title("RMSE 5-fold cross-validation")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _actual_vs_predicted(evaluation: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7, 5.5))
    axes.scatter(
        evaluation["actual"], evaluation["predicted"], s=9, alpha=0.35
    )
    low = float(min(evaluation.min()))
    high = float(max(evaluation.max()))
    axes.plot([low, high], [low, high], "--", color="#C44E52", label="Dự đoán hoàn hảo")
    axes.set_xlabel("Điểm thực tế")
    axes.set_ylabel("Điểm dự đoán")
    axes.set_title("Holdout: thực tế và dự đoán")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _residuals(evaluation: pd.DataFrame, path: Path) -> None:
    residuals = evaluation["actual"] - evaluation["predicted"]
    figure, axes = plt.subplots(figsize=(9, 4.2))
    axes.hist(residuals, bins=30, color="#8172B2", edgecolor="white")
    axes.axvline(0, color="#333333", linestyle="--", linewidth=1)
    axes.set_xlabel("Sai số (thực tế − dự đoán)")
    axes.set_ylabel("Số dòng")
    axes.set_title("Phân bố sai số trên holdout")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _importance(data: dict[str, Any], path: Path) -> None:
    rows = sorted(data["feature_importance"], key=lambda row: row["importance"])
    labels = [row["feature"] for row in rows]
    values = [row["importance"] for row in rows]
    figure, axes = plt.subplots(figsize=(9, 5.2))
    axes.barh(labels, values, color="#C44E52")
    axes.set_xlabel("Mức tăng RMSE khi hoán vị biến")
    axes.set_title("Permutation importance trên holdout")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_figures(data: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Write all nine report figures and return their paths."""
    if _RAW is None:
        collect_report_data()
    assert _RAW is not None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = _RAW["dataset"]
    evaluation = _RAW["evaluation"]

    builders = {
        "architecture": lambda path: _architecture(path),
        "target_hist": lambda path: _target_hist(dataset, path),
        "stress_counts": lambda path: _stress_counts(dataset, path),
        "correlation": lambda path: _correlation(dataset, path),
        "scatter_numeric": lambda path: _scatter_numeric(dataset, path),
        "cv_rmse": lambda path: _cv_rmse(data, path),
        "actual_vs_predicted": lambda path: _actual_vs_predicted(evaluation, path),
        "residuals": lambda path: _residuals(evaluation, path),
        "importance": lambda path: _importance(data, path),
    }
    for name, source_name in EVALUATE_FIGURES.items():
        source = ROOT / "reports" / "figures" / source_name
        if not source.is_file():
            raise RuntimeError(f"Thiếu {source}. Chạy python -m app.evaluate trước.")
        builders[name] = lambda path, source=source: shutil.copyfile(source, path)
    paths: dict[str, Path] = {}
    for name, filename in FIGURE_FILES.items():
        figure_path = out_dir / filename
        builders[name](figure_path)
        paths[name] = figure_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir", type=Path, default=ROOT / "docs" / "report" / "figures"
    )
    parser.add_argument(
        "--data-out", type=Path, default=ROOT / "docs" / "report" / "report_data.json"
    )
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    data = collect_report_data()
    write_report_data(data, args.data_out)
    figure_count = 0
    if not args.skip_figures:
        figure_count = len(build_figures(data, args.figures_dir))
    print(
        json.dumps(
            {"data": str(args.data_out), "figures": figure_count},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
