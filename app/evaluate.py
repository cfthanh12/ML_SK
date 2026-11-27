"""Extended evaluation that complements app/train.py.

app/train.py chooses the model (5-fold CV on train) and reports the holdout.
This module adds what the course material asks for beyond one score:
  * holdout comparison incl. the initial vs tuned Random Forest and an ablation
    without Country (Project examples: compare >= 3 methods, several measures);
  * hyperparameter tuning with RandomizedSearchCV on the TRAIN split only (Ch.2);
  * adjusted R^2 (Lab S3_L24: Multiple linear regression and Adjusted R-squared);
  * learning curve for bias/variance (Machine Learning Yearning ch.28-32);
  * slice-based error analysis and error by score band (Chip Huyen ch.6);
  * behavioural tests: directional, invariance, perturbation (Chip Huyen ch.6);
  * EDA facts used in the report (usage threshold, platform vs country).

Run:  python -m app.evaluate            (about 8-10 minutes, mostly tuning)
      python -m app.evaluate --skip-tuning
Writes reports/experiments.json and reports/figures/*.png.
"""

import argparse
import json
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    KFold,
    RandomizedSearchCV,
    learning_curve,
    train_test_split,
)

from app.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PARAMS,
    PROJECT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
)
from app.train import (  # noqa: E402
    DEFAULT_DATA_PATH,
    build_pipeline,
    load_dataset,
    make_random_forest,
    prepare_dataset,
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_PATH = REPORTS_DIR / "experiments.json"
USAGE_BINS = [0, 2, 3, 4, 5, 6, 7, 9]


def regression_metrics(y_true: Any, y_pred: Any, n_predictors: int | None = None) -> dict[str, float]:
    """MAE, RMSE, R^2 and, when the predictor count is given, adjusted R^2."""
    r2 = float(r2_score(y_true, y_pred))
    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
    }
    if n_predictors is not None:
        result["adjusted_r2"] = adjusted_r2(r2, len(y_true), n_predictors)
    return result


def adjusted_r2(r2: float, n_samples: int, n_predictors: int) -> float:
    """Adjusted R^2 = 1 - (1 - R^2)(n - 1)/(n - p - 1)."""
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1)


def candidate_models() -> dict[str, tuple[Any, list[str]]]:
    no_country = [c for c in FEATURE_COLUMNS if c != "Country"]
    cat_no_country = [c for c in CATEGORICAL_FEATURES if c != "Country"]
    return {
        "Baseline (trung bình)": (build_pipeline(DummyRegressor()), FEATURE_COLUMNS),
        "Linear Regression": (build_pipeline(LinearRegression()), FEATURE_COLUMNS),
        "HistGradientBoosting": (
            build_pipeline(HistGradientBoostingRegressor(
                **PARAMS["hist_gradient_boosting"], random_state=RANDOM_STATE)),
            FEATURE_COLUMNS,
        ),
        "Random Forest ban đầu (v1.1.0)": (
            build_pipeline(make_random_forest(PARAMS["random_forest_initial"])),
            FEATURE_COLUMNS,
        ),
        "Random Forest đã tinh chỉnh (v1.2.0)": (build_pipeline(), FEATURE_COLUMNS),
        "Random Forest bỏ Country": (
            build_pipeline(make_random_forest(), NUMERIC_FEATURES, cat_no_country),
            no_country,
        ),
    }


def compare_on_holdout(x_train, x_test, y_train, y_test) -> list[dict[str, Any]]:
    rows = []
    for name, (pipeline, columns) in candidate_models().items():
        started = time.perf_counter()
        pipeline.fit(x_train[columns], y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = pipeline.predict(x_test[columns])
        predict_ms_per_row = (time.perf_counter() - started) * 1000 / len(x_test)
        n_predictors = len(pipeline.named_steps["preprocessing"].get_feature_names_out())
        metrics = regression_metrics(y_test, predictions, n_predictors)
        rows.append({
            "model": name,
            **{f"holdout_{k}": v for k, v in metrics.items()},
            "train_r2": float(r2_score(y_train, pipeline.predict(x_train[columns]))),
            "n_predictors_after_encoding": n_predictors,
            "fit_seconds": fit_seconds,
            "predict_ms_per_row": predict_ms_per_row,
        })
    return rows


def tune_random_forest(x_train, y_train, x_test, y_test, iterations: int) -> dict[str, Any]:
    """Search on the train split only; the holdout is touched once at the end."""
    search = RandomizedSearchCV(
        build_pipeline(make_random_forest(PARAMS["random_forest_initial"])),
        {
            "regressor__n_estimators": [200, 300, 500],
            "regressor__min_samples_leaf": [1, 2, 3, 5, 8],
            "regressor__max_features": [0.3, 0.5, 0.7, 1.0],
            "regressor__max_depth": [None, 12, 20],
        },
        n_iter=iterations, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=1,
    )
    search.fit(x_train, y_train)
    return {
        "search_space": "n_estimators, min_samples_leaf, max_features, max_depth",
        "iterations": iterations,
        "best_params": {k.replace("regressor__", ""): v for k, v in search.best_params_.items()},
        "best_cv_r2_on_train": float(search.best_score_),
        "holdout": regression_metrics(y_test, search.predict(x_test)),
    }


def compute_learning_curve(features, target) -> dict[str, list[float]]:
    sizes, train_scores, valid_scores = learning_curve(
        build_pipeline(), features, target,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=KFold(5, shuffle=True, random_state=0),
        scoring="neg_mean_absolute_error", n_jobs=1,
    )
    return {
        "train_sizes": [int(s) for s in sizes],
        "train_mae": [float(-v) for v in train_scores.mean(axis=1)],
        "valid_mae": [float(-v) for v in valid_scores.mean(axis=1)],
    }


def error_analysis(model, x_test, y_test) -> dict[str, Any]:
    predictions = model.predict(x_test)
    residuals = y_test.to_numpy() - predictions
    frame = x_test.copy()
    frame["abs_error"] = np.abs(residuals)
    frame["residual"] = residuals
    slices = {}
    for column in ["Stress_Level", "Academic_Level", "Gender", "Purpose_Of_Use"]:
        grouped = frame.groupby(column)["abs_error"].agg(["mean", "count"])
        slices[column] = {str(k): {"mae": float(v["mean"]), "n": int(v["count"])}
                          for k, v in grouped.iterrows()}
    bands = pd.cut(y_test, [3, 5, 6, 7, 8, 10])
    by_band = frame.groupby(bands.to_numpy(), observed=True)["residual"].agg(["mean", "count"])
    return {
        "residual_mean": float(residuals.mean()),
        "share_within_0_5": float((np.abs(residuals) <= 0.5).mean()),
        "share_within_1_0": float((np.abs(residuals) <= 1.0).mean()),
        "mae_by_slice": slices,
        "mean_residual_by_actual_band": {
            str(k): {"mean_residual": float(v["mean"]), "n": int(v["count"])}
            for k, v in by_band.iterrows()
        },
    }


def behavioural_tests(model, features: pd.DataFrame) -> dict[str, Any]:
    base = model.predict(features)
    out: dict[str, Any] = {}
    flipped = features.copy()
    flipped["Gender"] = flipped["Gender"].map({"Male": "Female", "Female": "Male"})
    delta = np.abs(model.predict(flipped) - base)
    out["invariance_gender"] = {
        "mean_abs_change": float(delta.mean()),
        "share_change_over_0_5": float((delta > 0.5).mean()),
        "max_abs_change": float(delta.max()),
    }
    low, high = features.copy(), features.copy()
    low["Avg_Daily_Usage_Hours"], high["Avg_Daily_Usage_Hours"] = 2.0, 7.0
    diff = model.predict(high) - model.predict(low)
    out["directional_usage_2h_to_7h"] = {
        "share_score_decreases": float((diff < 0).mean()), "mean_change": float(diff.mean())}
    low, high = features.copy(), features.copy()
    low["Sleep_Hours_Per_Night"], high["Sleep_Hours_Per_Night"] = 5.0, 8.0
    diff = model.predict(high) - model.predict(low)
    out["directional_sleep_5h_to_8h"] = {
        "share_score_increases": float((diff > 0).mean()), "mean_change": float(diff.mean())}
    rng = np.random.default_rng(0)
    noisy = features.copy()
    for column in ["Avg_Daily_Usage_Hours", "Study_Hours", "Sleep_Hours_Per_Night"]:
        noisy[column] = noisy[column] + rng.normal(0, 0.25, len(noisy))
    out["perturbation_noise_0_25h"] = {
        "mean_abs_change": float(np.abs(model.predict(noisy) - base).mean())}
    steps = []
    for index in range(0, min(300, len(features)), 10):
        row = features.iloc[[index]].copy()
        scores = []
        for hours in np.arange(1.0, 9.0, 0.5):
            row["Avg_Daily_Usage_Hours"] = hours
            scores.append(float(model.predict(row)[0]))
        steps.append(max(0.0, float(np.max(np.diff(scores)))))
    out["local_non_monotonicity_usage"] = {
        "max_step_increase": float(np.max(steps)), "mean_max_step_increase": float(np.mean(steps))}
    beyond = features.iloc[:200].copy()
    at_edge, far = beyond.copy(), beyond.copy()
    at_edge["Avg_Daily_Usage_Hours"], far["Avg_Daily_Usage_Hours"] = 8.8, 20.0
    out["no_extrapolation_8_8h_vs_20h"] = {
        "max_abs_difference": float(np.abs(model.predict(far) - model.predict(at_edge)).max())}
    return out


def eda_facts(raw: pd.DataFrame) -> dict[str, Any]:
    usage = raw.groupby(pd.cut(raw["Avg_Daily_Usage_Hours"], USAGE_BINS),
                        observed=True)["Mental_Health_Score"].agg(["mean", "count"])
    platform_country = {}
    for platform in ["LINE", "KakaoTalk", "WeChat", "VKontakte"]:
        share = raw.loc[raw["Most_Used_Platform"] == platform, "Country"].value_counts(normalize=True)
        platform_country[platform] = {"top_country": str(share.index[0]), "share": float(share.iloc[0])}
    return {
        "usage_bin_mean_score": {str(k): {"mean": float(v["mean"]), "n": int(v["count"])}
                                 for k, v in usage.iterrows()},
        "sleep_ge_7_mean": float(raw.loc[raw["Sleep_Hours_Per_Night"] >= 7, "Mental_Health_Score"].mean()),
        "sleep_lt_7_mean": float(raw.loc[raw["Sleep_Hours_Per_Night"] < 7, "Mental_Health_Score"].mean()),
        "stress_mean_score": raw.groupby("Stress_Level")["Mental_Health_Score"].mean().round(3).to_dict(),
        "stress_mean_usage": raw.groupby("Stress_Level")["Avg_Daily_Usage_Hours"].mean().round(3).to_dict(),
        "platform_mean_score": raw.groupby("Most_Used_Platform")["Mental_Health_Score"].mean().round(3).to_dict(),
        "platform_single_country": platform_country,
        "vietnam_rows": int((raw["Country"] == "Vietnam").sum()),
        "n_countries": int(raw["Country"].nunique()),
        "countries_under_30_rows": int((raw["Country"].value_counts() < 30).sum()),
    }


def make_figures(raw, curve, behaviour_model, x_test) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
    written = []

    usage = raw.groupby(pd.cut(raw["Avg_Daily_Usage_Hours"], USAGE_BINS),
                        observed=True)["Mental_Health_Score"].mean()
    labels = ["≤2", "2–3", "3–4", "4–5", "5–6", "6–7", "7–8,8"]
    colors = ["#9e9e9e"] * 5 + ["#1f6fd1"] * 2
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, usage.values, color=colors)
    ax.bar_label(bars, labels=[f"{v:.2f}".replace(".", ",") for v in usage.values], padding=3)
    ax.axhline(raw["Mental_Health_Score"].mean(), ls="--", color="#666", lw=1)
    ax.set(xlabel="Giờ dùng mạng xã hội mỗi ngày", ylabel="Điểm trung bình (0-10)",
           title="Điểm giảm theo giờ dùng rồi đi ngang từ 6 giờ", ylim=(0, 9))
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "usage_threshold.png"); plt.close(fig)
    written.append("usage_threshold.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["train_sizes"], curve["train_mae"], "o-", color="#9e9e9e", label="Huấn luyện")
    ax.plot(curve["train_sizes"], curve["valid_mae"], "o-", color="#1f6fd1", label="Xác thực (5-fold)")
    ax.set(xlabel="Số mẫu huấn luyện", ylabel="MAE (điểm)",
           title="Learning curve: sai số xác thực vẫn giảm khi thêm dữ liệu", ylim=(0, 0.6))
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR / "learning_curve.png"); plt.close(fig)
    written.append("learning_curve.png")

    sample = x_test.iloc[:40]
    grid = np.arange(1.0, 9.01, 0.5)
    curves = []
    for hours in grid:
        changed = sample.copy(); changed["Avg_Daily_Usage_Hours"] = hours
        curves.append(behaviour_model.predict(changed))
    curves = np.array(curves)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, curves, color="#9e9e9e", alpha=0.35, lw=1)
    ax.plot(grid, curves.mean(axis=1), color="#1f6fd1", lw=2.5, label="Trung bình 40 hồ sơ")
    ax.set(xlabel="Giờ dùng mạng xã hội mỗi ngày (giữ nguyên các biến khác)", ylabel="Điểm dự đoán",
           title="What-if: mô hình hạ điểm khi tăng giờ dùng, đi ngang từ khoảng 6 giờ")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR / "what_if_usage.png"); plt.close(fig)
    written.append("what_if_usage.png")
    return written


def run(skip_tuning: bool = False) -> dict[str, Any]:
    raw = load_dataset(DEFAULT_DATA_PATH)
    features, target, cleaning = prepare_dataset(raw)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = build_pipeline().fit(x_train, y_train)
    curve = compute_learning_curve(features, target)
    summary: dict[str, Any] = {
        "n_rows": int(len(raw)),
        "split": {"train": int(len(x_train)), "test": int(len(x_test))},
        "cleaning": cleaning,
        "holdout_comparison": compare_on_holdout(x_train, x_test, y_train, y_test),
        "learning_curve": curve,
        "error_analysis": error_analysis(model, x_test, y_test),
        "behavioural_tests": behavioural_tests(model, x_test),
        "eda": eda_facts(raw),
    }
    if not skip_tuning:
        summary["tuning"] = tune_random_forest(
            x_train, y_train, x_test, y_test, PARAMS["evaluation"]["tuning_iterations"])
    summary["figures"] = make_figures(raw, curve, model, x_test)

    if skip_tuning and EXPERIMENTS_PATH.is_file():  # keep the last tuning result
        previous = json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
        if "tuning" in previous:
            summary["tuning"] = previous["tuning"]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()
    result = run(skip_tuning=args.skip_tuning)
    print(json.dumps({k: result[k] for k in ("holdout_comparison",) + (("tuning",) if "tuning" in result else ())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
