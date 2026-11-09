# Nâng chất lượng đồ án Machine Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nâng đồ án từ "chạy được" thành "bảo vệ được": notebook so sánh mô hình có CV/ablation/diễn giải, API có `/model-info` + clamp + logging, Streamlit hiển thị thông tin mô hình, test độc lập artifact, Docker + README/model card đầy đủ.

**Architecture:** Giữ nguyên 3 phần (notebook → `app/train.py` → FastAPI → Streamlit). Logic huấn luyện/so sánh nằm một chỗ trong `app/train.py`; notebook gọi lại và trực quan hóa. Artifact vẫn là `{model, metadata}` với metadata mở rộng. API chỉ thêm endpoint và clamp, không đổi contract `/predict`.

**Tech Stack:** Python 3.14, scikit-learn 1.9.1, pandas 3.1.0, FastAPI, Uvicorn, Pydantic 2, Streamlit, pytest, Docker.

**Spec:** `docs/superpowers/specs/2026-11-09-ml-project-improvements-design.md`

## Global Constraints

- Windows + PowerShell; dùng `.venv` sẵn có; không thêm dependency nặng.
- Notebook chạy được từ cả repo root và `notebooks/` (giữ bootstrap đường dẫn hiện có).
- Notebook, README, UI, model card giữ tiếng Việt; tên biến/thư viện tiếng Anh.
- `MODEL_VERSION = "1.1.0"`; CV mặc định 5 fold, seed 42; holdout 20% chỉ để báo cáo.
- Ứng viên cố định: `baseline_dummy`, `linear_regression`, `random_forest`, `hist_gradient_boosting`.
- Feature importance = permutation trên holdout, `n_repeats=5`, `random_state=42`, top 12.
- Artifact giữ dạng `{"model": Pipeline, "metadata": {...}}`; `metrics.json` phải đọc được bằng mắt và JSON-serializable.
- Không đổi contract `POST /predict`; chỉ clamp điểm về `[0, 10]`.
- Docker base `python:3.14-slim`; pin đúng phiên bản venv; image copy artifact đã train, không train lúc build.
- Không sửa CSV gốc.
- Không tạo git commit trừ khi người dùng yêu cầu; mỗi task kết thúc bằng test xanh.
- Không chạy `docker build` trong plan này (daemon đang tắt); chỉ kiểm tra tĩnh.

## Review Focus

- Quốc gia/danh mục chưa thấy lúc `/predict` phải trả 200 (OneHot `handle_unknown="ignore"`), không 500. Test: `test_predict_accepts_unseen_country`.
- Model trả điểm ngoài `[0, 10]` phải bị clamp. Test: `test_clamp_score_bounds`.
- Thiếu artifact phải fail nhanh với `FileNotFoundError` có hướng dẫn. Test: `test_load_artifact_missing_raises`.
- Artifact đã commit phải khớp `MODEL_VERSION`/feature list, tránh model cũ lọt vào demo. Test: `test_committed_artifact_matches_code`.
- Metadata phải JSON-serializable (không lọt numpy type) để ghi `metrics.json` thành công. Test: `test_save_artifacts_writes_json_metadata`.

---

### Task 1: Version + tiền xử lý tham số hóa + ứng viên mô hình

**Files:**
- Modify: `app/features.py`
- Modify: `app/train.py`
- Create: `tests/conftest.py`
- Create: `tests/test_training.py`

**Interfaces:**
- Consumes: CSV hiện có, `load_dataset` hiện có.
- Produces:
  - `features.MODEL_VERSION = "1.1.0"`, `features.RANDOM_STATE = 42`, `features.DEFAULT_CV_FOLDS = 5`
  - `prepare_dataset(dataset: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]`
  - `build_pipeline(regressor: Any | None = None, numeric_features: list[str] | None = None, categorical_features: list[str] | None = None) -> Pipeline`
  - `build_candidate_models(numeric_features: list[str] | None = None, categorical_features: list[str] | None = None) -> dict[str, Pipeline]`

- [ ] **Step 1: Tạo fixture dữ liệu tổng hợp trong `tests/conftest.py`**

```python
import numpy as np
import pandas as pd
import pytest


def make_synthetic_frame(rows: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stress = rng.choice(["Low", "Medium", "High", "Very High"], rows)
    stress_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
    frame = pd.DataFrame({
        "Age": rng.integers(18, 25, rows),
        "Gender": rng.choice(["Female", "Male"], rows),
        "Country": rng.choice(["Vietnam", "USA", "Other"], rows),
        "Academic_Level": rng.choice(["High School", "Undergraduate", "Graduate"], rows),
        "Most_Used_Platform": rng.choice(["Instagram", "TikTok", "Facebook"], rows),
        "Purpose_Of_Use": rng.choice(["Education", "Entertainment", "Networking", "News"], rows),
        "Avg_Daily_Usage_Hours": rng.uniform(1, 8, rows),
        "Daily_Unlocks": rng.integers(60, 275, rows),
        "Study_Hours": rng.uniform(0.5, 8, rows),
        "Physical_Activity_Hours": rng.uniform(-0.5, 4, rows),
        "Sleep_Hours_Per_Night": rng.uniform(4, 9, rows),
        "Stress_Level": stress,
    })
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
```

- [ ] **Step 2: Viết test thất bại cho Task 1 trong `tests/test_training.py`**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from app.train import build_candidate_models, build_pipeline, prepare_dataset


def test_prepare_dataset_clips_negative_activity(synthetic_dataset: pd.DataFrame) -> None:
    frame = synthetic_dataset.copy()
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


def test_build_candidate_models_returns_four_pipelines() -> None:
    models = build_candidate_models()
    assert set(models) == {
        "baseline_dummy",
        "linear_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert all(hasattr(model, "fit") and hasattr(model, "predict") for model in models.values())


def test_build_pipeline_with_subset_features(synthetic_dataset: pd.DataFrame) -> None:
    subset = [column for column in FEATURE_COLUMNS if column != "Stress_Level"]
    features, target, _ = prepare_dataset(synthetic_dataset, feature_columns=subset)
    pipeline = build_pipeline(
        regressor=LinearRegression(),
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=[c for c in CATEGORICAL_FEATURES if c != "Stress_Level"],
    )
    pipeline.fit(features, target)
    assert pipeline.predict(features).shape == (len(features),)
```

- [ ] **Step 3: Chạy test để xác nhận fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_training.py -q`
Expected: FAIL — `ImportError`/`TypeError` vì `build_candidate_models` chưa có hoặc `prepare_dataset` chưa nhận `feature_columns`.

- [ ] **Step 4: Cập nhật `app/features.py`**

Thêm `RANDOM_STATE = 42`, `DEFAULT_CV_FOLDS = 5`, đổi `MODEL_VERSION = "1.1.0"`.

- [ ] **Step 5: Sửa `app/train.py`**

- `prepare_dataset` nhận `feature_columns: list[str] = FEATURE_COLUMNS`; validate theo `set(feature_columns) | {TARGET_COLUMN}`; numeric = `[c for c in feature_columns if c in NUMERIC_FEATURES]`, categorical tương tự; clip `Physical_Activity_Hours` nếu có; báo cáo `negative_physical_activity_values_clipped`.
- `build_pipeline(regressor=None, numeric_features=None, categorical_features=None)` dùng mặc định `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES`; regressor mặc định là RandomForest hiện tại; trả `Pipeline([("preprocessing", ColumnTransformer(...)), ("regressor", regressor)])`.
- `build_candidate_models(...)` trả 4 pipeline như bảng dưới:

| key | regressor |
|---|---|
| `baseline_dummy` | `DummyRegressor(strategy="mean")` |
| `linear_regression` | `LinearRegression()` |
| `random_forest` | `RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)` |
| `hist_gradient_boosting` | `HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42)` |

- [ ] **Step 6: Chạy test để xác nhận pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_training.py -q`
Expected: PASS (4 tests).

---

### Task 2: Cross-validation và chọn mô hình

**Files:**
- Modify: `app/train.py`
- Modify: `tests/test_training.py`

**Interfaces:**
- Consumes: `build_candidate_models`, `prepare_dataset`, `DEFAULT_CV_FOLDS`, `RANDOM_STATE`.
- Produces:
  - `cross_validate_candidates(models: dict[str, Pipeline], features: pd.DataFrame, target: pd.Series, cv: int = DEFAULT_CV_FOLDS, random_state: int = RANDOM_STATE) -> pd.DataFrame`
    - index tên `"model"`, cột `mae_mean`, `mae_std`, `rmse_mean`, `rmse_std`, `r2_mean`, `r2_std`; sắp tăng dần theo `rmse_mean`.
  - `select_best_model(cv_table: pd.DataFrame) -> str` trả index của `rmse_mean` nhỏ nhất.

- [ ] **Step 1: Viết test thất bại**

```python
from app.train import build_candidate_models, cross_validate_candidates, prepare_dataset, select_best_model


def test_cross_validate_candidates_returns_row_per_model(synthetic_dataset: pd.DataFrame) -> None:
    features, target, _ = prepare_dataset(synthetic_dataset)
    table = cross_validate_candidates(build_candidate_models(), features, target, cv=2)
    assert set(table.index) == {
        "baseline_dummy", "linear_regression", "random_forest", "hist_gradient_boosting",
    }
    assert {"mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"} <= set(table.columns)
    assert table["rmse_mean"].is_monotonic_increasing
    assert table["rmse_mean"].min() >= 0
    assert table["mae_mean"].min() >= 0


def test_select_best_model_returns_lowest_rmse() -> None:
    table = pd.DataFrame(
        {"rmse_mean": [1.0, 0.5, 0.8]}, index=["a", "b", "c"]
    )
    table.index.name = "model"
    assert select_best_model(table) == "b"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_training.py -q`
Expected: FAIL — `ImportError` cho `cross_validate_candidates`.

- [ ] **Step 3: Triển khai trong `app/train.py`**

Dùng `KFold(n_splits=cv, shuffle=True, random_state=random_state)`, `cross_validate(..., scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}, error_score="raise")`; đảo dấu MAE/RMSE; ép `float`; đặt `table.index.name = "model"`; `select_best_model` = `str(cv_table["rmse_mean"].idxmin())`.

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_training.py -q`
Expected: PASS (6 tests).

---

### Task 3: Permutation feature importance

**Files:**
- Modify: `app/train.py`
- Modify: `tests/test_training.py`

**Interfaces:**
- Consumes: pipeline đã fit, DataFrame feature gốc.
- Produces: `extract_feature_importance(model: Pipeline, features: pd.DataFrame, target: pd.Series, top_n: int = 12) -> list[dict[str, Any]]` — list `{"feature": str, "importance": float}` giảm dần, tối đa `top_n`.

- [ ] **Step 1: Viết test thất bại**

```python
from app.features import FEATURE_COLUMNS
from app.train import build_pipeline, extract_feature_importance
from sklearn.linear_model import LinearRegression


def test_extract_feature_importance_returns_sorted_top_features(synthetic_dataset: pd.DataFrame) -> None:
    features, target, _ = prepare_dataset(synthetic_dataset)
    model = build_pipeline(regressor=LinearRegression())
    model.fit(features, target)
    items = extract_feature_importance(model, features, target, top_n=12)
    assert 0 < len(items) <= 12
    assert set(item["feature"] for item in items) <= set(FEATURE_COLUMNS)
    importances = [item["importance"] for item in items]
    assert importances == sorted(importances, reverse=True)
    assert all(isinstance(value, float) for value in importances)
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_training.py -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Triển khai**

Dùng `sklearn.inspection.permutation_importance(model, features, target, n_repeats=5, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error")`; ghép `features.columns` với `result.importances_mean`; ép `float`; sort giảm dần; cắt `top_n`. Vì permute trên DataFrame gốc 12 cột nên one-hot tự động được gộp về cột nguồn.

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_training.py -q`
Expected: PASS (7 tests).

---

### Task 4: Orchestration, ablation, lưu artifact, CLI

**Files:**
- Modify: `app/train.py`
- Modify: `tests/test_training.py`
- Regenerate: `artifacts/mental_health_model.joblib`, `artifacts/metrics.json`

**Interfaces:**
- Produces:
  - `train_model(dataset: pd.DataFrame, cv: int = DEFAULT_CV_FOLDS, random_state: int = RANDOM_STATE) -> tuple[Pipeline, dict[str, float], pd.DataFrame, dict[str, int], pd.DataFrame, str, list[dict[str, Any]]]`
    (model, holdout metrics, evaluation frame, cleaning report, cv_table, selected name, feature importance)
  - `run_stress_ablation(dataset: pd.DataFrame, dropped_feature: str = "Stress_Level", cv: int = DEFAULT_CV_FOLDS, random_state: int = RANDOM_STATE) -> dict[str, Any]`
    trả `{"dropped_feature", "selected_model", "metrics", "cv_results"}`
  - `save_artifacts(model, metrics, training_rows, cleaning_report, cv_table, selected_model, feature_importance, stress_ablation=None, artifact_path=DEFAULT_ARTIFACT_PATH, metrics_path=DEFAULT_METRICS_PATH) -> dict[str, Any]`
  - `train_and_save(data_path=DEFAULT_DATA_PATH, artifact_path=DEFAULT_ARTIFACT_PATH, metrics_path=DEFAULT_METRICS_PATH, cv=DEFAULT_CV_FOLDS) -> dict[str, Any]`

Metadata keys bắt buộc: `model_version`, `target_column`, `selected_model`, `feature_columns`, `metrics`, `cv_results`, `feature_importance`, `feature_importance_method` (`"permutation"`), `stress_ablation`, `training_rows`, `cleaning_report`, `trained_at_utc`.

- [ ] **Step 1: Viết test thất bại**

```python
import json

from app.features import FEATURE_COLUMNS
from app.train import run_stress_ablation, save_artifacts, train_and_save, train_model


def test_train_model_returns_cv_and_selection(synthetic_dataset: pd.DataFrame) -> None:
    model, metrics, evaluation, report, cv_table, selected, importance = train_model(
        synthetic_dataset, cv=2
    )
    assert {"mae", "rmse", "r2"} == set(metrics)
    assert set(cv_table.index) == {
        "baseline_dummy", "linear_regression", "random_forest", "hist_gradient_boosting",
    }
    assert selected in set(cv_table.index)
    assert len(evaluation) == 40
    assert importance and importance[0]["feature"] in FEATURE_COLUMNS


def test_run_stress_ablation_drops_feature(synthetic_dataset: pd.DataFrame) -> None:
    result = run_stress_ablation(synthetic_dataset, cv=2)
    assert result["dropped_feature"] == "Stress_Level"
    assert any(row["model"] == result["selected_model"] for row in result["cv_results"])
    assert {"mae", "rmse", "r2"} == set(result["metrics"])


def test_save_artifacts_writes_json_metadata(synthetic_dataset: pd.DataFrame, tmp_path) -> None:
    model, metrics, _, report, cv_table, selected, importance = train_model(synthetic_dataset, cv=2)
    ablation = run_stress_ablation(synthetic_dataset, cv=2)
    artifact_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    save_artifacts(
        model, metrics, len(synthetic_dataset), report, cv_table, selected, importance,
        stress_ablation=ablation, artifact_path=artifact_path, metrics_path=metrics_path,
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["model_version"] == "1.1.0"
    assert payload["selected_model"] == selected
    assert payload["feature_importance_method"] == "permutation"
    assert payload["stress_ablation"]["dropped_feature"] == "Stress_Level"
    assert artifact_path.is_file()


def test_train_and_save_end_to_end(synthetic_dataset: pd.DataFrame, tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    synthetic_dataset.to_csv(data_path, index=False)
    metadata = train_and_save(
        data_path=data_path,
        artifact_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        cv=2,
    )
    assert metadata["training_rows"] == len(synthetic_dataset)
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_training.py -q`
Expected: FAIL — chữ ký `train_model`/`save_artifacts` cũ không khớp.

- [ ] **Step 3: Triển khai**

- `train_model`: `prepare_dataset` → `train_test_split(test_size=0.2, random_state=random_state)` → `cross_validate_candidates` → `select_best_model` → fit trên train → metrics holdout → `extract_feature_importance(model, x_test, y_test)` → fit lại toàn bộ → trả 7 giá trị.
- `run_stress_ablation`: subset feature bỏ `Stress_Level`, cùng split, CV, chọn, fit, tính holdout; trả dict; `cv_results = cv_table.reset_index().to_dict("records")`.
- `save_artifacts`: dựng metadata như trên; `joblib.dump({"model": model, "metadata": metadata}, artifact_path)`; ghi `metrics.json` bằng `json.dumps(..., ensure_ascii=False, indent=2)`.
- `train_and_save`: gọi `train_model`, `run_stress_ablation`, rồi `save_artifacts`.
- `main()` giữ `--data/--artifact/--metrics`.
- Xoá `build_pipeline()` cũ nếu bị thay thế hoàn toàn (không để code chết).

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_training.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Sinh lại artifact thật**

Run: `.venv\Scripts\python.exe -m app.train`
Expected: in metadata JSON; `artifacts/metrics.json` có `model_version: "1.1.0"`, `selected_model`, `cv_results`, `feature_importance`, `stress_ablation`.

---

### Task 5: Hạ tầng test độc lập artifact + schema + guard

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/test_schemas.py`
- Create: `tests/test_artifact_guard.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `train_and_save`, `app.main.app`, `MODEL_VERSION`, `build_candidate_models`.
- Produces fixtures: `model_artifact` (session, tạo artifact tạm và set `MODEL_ARTIFACT_PATH`), `client` (TestClient).

- [ ] **Step 1: Mở rộng `tests/conftest.py`**

```python
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.train import train_and_save


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
```

**Ghi chú:** `make_synthetic_frame` đã được tạo ở Task 1 Step 1; fixture `synthetic_dataset` và fixture session dùng chung hàm này. Tên file tạm dùng `pathlib` không cần import vì `tmp_path_factory` trả `Path`.

- [ ] **Step 2: Viết `tests/test_schemas.py`**

```python
import pytest
from pydantic import ValidationError

from app.schemas import PredictionRequest


def valid_payload() -> dict:
    return {
        "age": 21, "gender": "Female", "country": "Vietnam",
        "academic_level": "Undergraduate", "most_used_platform": "Instagram",
        "purpose_of_use": "Education", "avg_daily_usage_hours": 4.0,
        "daily_unlocks": 120, "study_hours": 4.0,
        "physical_activity_hours": 1.0, "sleep_hours_per_night": 7.0,
        "stress_level": "Medium",
    }


@pytest.mark.parametrize("age", [18, 24])
def test_age_boundaries_accepted(age: int) -> None:
    assert PredictionRequest(**{**valid_payload(), "age": age}).age == age


@pytest.mark.parametrize("age", [17, 25])
def test_age_outside_range_rejected(age: int) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), "age": age})


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), "unexpected": 1})


def test_blank_country_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), "country": "   "})


@pytest.mark.parametrize("field", ["avg_daily_usage_hours", "study_hours", "physical_activity_hours", "sleep_hours_per_night"])
def test_hours_outside_range_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), field: -1.0})
    with pytest.raises(ValidationError):
        PredictionRequest(**{**valid_payload(), field: 25.0})
```

- [ ] **Step 3: Viết `tests/test_artifact_guard.py`**

```python
import joblib
import pytest

from app.features import FEATURE_COLUMNS, MODEL_VERSION
from app.train import DEFAULT_ARTIFACT_PATH, build_candidate_models

pytestmark = pytest.mark.skipif(
    not DEFAULT_ARTIFACT_PATH.is_file(), reason="committed artifact not generated yet"
)


def test_committed_artifact_matches_code() -> None:
    metadata = joblib.load(DEFAULT_ARTIFACT_PATH)["metadata"]
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["selected_model"] in build_candidate_models()
    assert metadata["feature_columns"] == FEATURE_COLUMNS
```

- [ ] **Step 4: Viết lại `tests/test_api.py` dùng fixture**

Giữ `SAMPLE_INPUT`; thay `with TestClient(app)` bằng tham số `client`; import `MODEL_VERSION`; assert `result["model_version"] == MODEL_VERSION`. Thêm test unknown country ở Task 6 (chưa thêm bây giờ).

- [ ] **Step 5: Chạy toàn bộ test**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS. Guard test pass vì artifact đã sinh lại ở Task 4.

---

### Task 6: API `/model-info`, clamp, env path, logging

**Files:**
- Modify: `app/main.py`
- Modify: `app/schemas.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Produces in `app/main.py`:
  - `resolve_artifact_path() -> Path` (đọc env `MODEL_ARTIFACT_PATH`)
  - `clamp_score(score: float, minimum: float = 0.0, maximum: float = 10.0) -> float`
  - `GET /model-info` trả `ModelInfoResponse`
- Produces in `app/schemas.py`: `FeatureImportanceItem`, `CVResultItem`, `ModelInfoResponse` với đúng các field trong spec §7.3.

- [ ] **Step 1: Viết test thất bại trong `tests/test_api.py`**

```python
import pytest

from app.features import MODEL_VERSION
from app.main import clamp_score, load_artifact


def test_model_info_returns_metadata(client) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == MODEL_VERSION
    assert body["selected_model"] in {
        "baseline_dummy", "linear_regression", "random_forest", "hist_gradient_boosting",
    }
    assert {"mae", "rmse", "r2"} == set(body["metrics"])
    assert len(body["feature_columns"]) == 12
    assert body["cv_results"]
    assert body["feature_importance"]


def test_predict_accepts_unseen_country(client) -> None:
    response = client.post("/predict", json={**SAMPLE_INPUT, "country": "Atlantis"})
    assert response.status_code == 200


def test_clamp_score_bounds() -> None:
    assert clamp_score(12.3) == 10.0
    assert clamp_score(-1.0) == 0.0
    assert clamp_score(6.42) == 6.42


def test_load_artifact_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m app.train"):
        load_artifact(tmp_path / "missing.joblib")
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api.py -q`
Expected: FAIL — 404 `/model-info`, `ImportError` cho `clamp_score`.

- [ ] **Step 3: Triển khai**

- `app/schemas.py`: thêm 3 model Pydantic; `CVResultItem` khớp cột `cv_results` (`model`, `mae_mean`, `mae_std`, `rmse_mean`, `rmse_std`, `r2_mean`, `r2_std`); `ModelInfoResponse` gồm `model_version`, `selected_model`, `metrics: dict[str, float]`, `cv_results: list[CVResultItem]`, `feature_columns: list[str]`, `feature_importance: list[FeatureImportanceItem]`, `feature_importance_method: str`, `training_rows: int`, `trained_at_utc: str`.
- `app/main.py`: import `os`, `logging`; `logger = logging.getLogger("app")`; lifespan dùng `load_artifact(resolve_artifact_path())`; `predict` dùng `round(clamp_score(float(...)), 2)` và log `score` + version; thêm `@app.get("/model-info", response_model=ModelInfoResponse)` đọc `artifact["metadata"]`, trả 503 nếu chưa load.

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api.py -q`
Expected: PASS (7 tests).

---

### Task 7: Streamlit hiển thị model info + kịch bản mẫu

**Files:**
- Modify: `streamlit_app.py`

**Interfaces:**
- Consumes: `GET /model-info`, `POST /predict`.
- Produces: UI có sidebar thông tin mô hình, preset, kết quả diễn giải.

- [ ] **Step 1: Viết code UI**

- Thêm `import pandas as pd`.
- `API_INFO_TIMEOUT = 10`; hàm `@st.cache_data(ttl=60) def fetch_model_info(api_url: str) -> dict | None` gọi `/model-info`, trả `None` khi lỗi.
- `PRESETS` gồm 3 kịch bản: `"Cân bằng"`, `"Dùng nhiều mạng xã hội"`, `"Học tập cao"`; mỗi dict đủ 12 field với giá trị hợp lệ.
- Selectbox `"Kịch bản mẫu"` ngoài form; mỗi widget trong form dùng `key=f"<field>_{preset_name}"` để đổi preset thì giá trị mặc định đổi theo.
- Sidebar: gọi `fetch_model_info`, hiện version/selected_model/metrics, `st.bar_chart` top feature importance; nếu `None` hiện cảnh báo API.
- Kết quả: `st.metric`, `st.progress(min(max(score / 10, 0), 1))`, nhãn `"thấp"` nếu `< 5.1`, `"cao"` nếu `> 7.1`, còn lại `"trung bình"`; giữ disclaimer hiện có.

- [ ] **Step 2: Kiểm tra cú pháp**

Run: `.venv\Scripts\python.exe -m py_compile streamlit_app.py`
Expected: exit 0.

- [ ] **Step 3 (tuỳ chọn, nếu API đang chạy): chạy thử headless**

Run: `.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.headless true --server.port 8501`
Expected: server khởi động không traceback; dừng process sau khi kiểm tra.

---

### Task 8: Cập nhật và chạy lại notebook

**Files:**
- Modify: `notebooks/model_training.ipynb`
- Regenerate: outputs của notebook, `artifacts/mental_health_model.joblib`, `artifacts/metrics.json`

**Interfaces:**
- Consumes: `load_dataset`, `prepare_dataset`, `train_model`, `run_stress_ablation`, `save_artifacts`, `extract_feature_importance`, `DEFAULT_*`.
- Produces: notebook chạy end-to-end, outputs khớp artifact.

- [ ] **Step 1: Sửa cell import**

Thêm `run_stress_ablation`, `extract_feature_importance` (nếu dùng trực tiếp), giữ `matplotlib.pyplot`, `pandas`.

- [ ] **Step 2: Thêm cell kiểm tra chất lượng dữ liệu**

Sau cell `dataset-overview`, thêm cell in: số dòng trùng, số giá trị `Physical_Activity_Hours < 0`, `Country.nunique()`, số dòng `Country == "Other"`, `Mental_Health_Score` min/max.

- [ ] **Step 3: Thêm cell EDA**

- Heatmap tương quan các biến số (`Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Study_Hours`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night`, `Mental_Health_Score`) bằng `plt.imshow` + `plt.colorbar` (không thêm seaborn).
- Scatter 3 biến (`Avg_Daily_Usage_Hours`, `Sleep_Hours_Per_Night`, `Study_Hours`) vs target.
- Markdown nhận xét: tương quan mạnh usage (−0.82), sleep (0.77), study (0.75), activity (0.52); `Stress_Level` cộng sinh với target và usage/sleep; `"Other"` gộp nhiều nước; `LINE` ít mẫu; dữ liệu khảo sát/tổng hợp.

- [ ] **Step 4: Thay cell `fit-model` bằng orchestration mới**

```python
model, metrics, evaluation, cleaning_report, cv_table, selected_model, feature_importance = train_model(data)
display(cv_table)
print("Selected:", selected_model)
for name, value in metrics.items():
    print(f"{name.upper()}: {value:.4f}")
```

- [ ] **Step 5: Thêm cell so sánh CV**

Vẽ bar `cv_table["rmse_mean"]` với `yerr=cv_table["rmse_std"]`, tiêu đề "5-fold CV RMSE"; in bảng CV.

- [ ] **Step 6: Giữ scatter actual vs predicted; thêm residual**

Thêm histogram `evaluation["actual"] - evaluation["predicted"]` và scatter predicted vs residual.

- [ ] **Step 7: Thêm cell feature importance**

Hiển thị DataFrame từ `feature_importance`, vẽ barh, viết 2–3 nhận xét.

- [ ] **Step 8: Thêm cell ablation**

```python
ablation = run_stress_ablation(data)
comparison = pd.DataFrame([
    {"variant": "with_stress", **metrics},
    {"variant": "without_stress", **ablation["metrics"]},
])
display(comparison.set_index("variant").round(4))
```

- [ ] **Step 9: Cập nhật cell lưu artifact**

```python
metadata = save_artifacts(
    model=model,
    metrics=metrics,
    training_rows=len(data),
    cleaning_report=cleaning_report,
    cv_table=cv_table,
    selected_model=selected_model,
    feature_importance=feature_importance,
    stress_ablation=ablation,
)
metadata
```

- [ ] **Step 10: Thêm cell kết luận & hạn chế**

Markdown: dữ liệu khảo sát/tổng hợp; tương quan ≠ nhân quả; không dùng để chẩn đoán; hỗ trợ yếu với quốc gia ngoài phân phối; `Stress_Level` chi phối mạnh nên kết quả cần đọc thận trọng.

- [ ] **Step 11: Chạy lại notebook**

Run: `.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/model_training.ipynb`
Expected: exit 0; notebook không còn output lỗi; `artifacts/metrics.json` khớp metadata in ra.

- [ ] **Step 12: Xác nhận guard test vẫn pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_artifact_guard.py -q`
Expected: PASS.

---

### Task 9: Pin dependencies + Docker

**Files:**
- Modify: `requirements.txt`
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Consumes: venv hiện tại (Python 3.14.8).
- Produces: `requirements.txt` pin `==`, image chạy được trên máy có Docker.

- [ ] **Step 1: Lấy phiên bản chính xác**

Run: `.venv\Scripts\python.exe -m pip freeze`
Ghi lại phiên bản của: fastapi, uvicorn, pandas, scikit-learn, joblib, numpy, pydantic, streamlit, requests, matplotlib, notebook, pytest, httpx.

- [ ] **Step 2: Viết lại `requirements.txt`**

Mỗi dòng `package==version` (giữ `uvicorn[standard]==...`), giữ nhóm dễ đọc. Bắt buộc khớp: `scikit-learn==1.9.1`, `pandas==3.1.0`, `numpy==2.5.4`, `joblib==1.6.0`.

- [ ] **Step 3: Tạo `Dockerfile`**

```dockerfile
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Tạo `.dockerignore`**

```
.venv
__pycache__
.pytest_cache
.git
tests
notebooks
docs
*.csv
```

- [ ] **Step 5: Kiểm tra resolution**

Run: `.venv\Scripts\python.exe -m pip install -r requirements.txt`
Expected: "Requirement already satisfied" cho các package, exit 0.

- [ ] **Step 6: Ghi chú Docker**

Không chạy `docker build` (daemon tắt). Ghi vào README ở Task 10 lệnh build/run để người dùng tự xác nhận.

---

### Task 10: README + model card

**Files:**
- Modify: `README.md`
- Create: `docs/model_card.md`

**Interfaces:**
- Consumes: `artifacts/metrics.json` (số liệu thật), spec §7.6.
- Produces: tài liệu khớp code và số liệu.

- [ ] **Step 1: Đọc số liệu thật**

Run: `.venv\Scripts\python.exe -c "import json; m=json.load(open('artifacts/metrics.json', encoding='utf-8')); print(json.dumps({'selected': m['selected_model'], 'metrics': m['metrics'], 'ablation': m['stress_ablation']['metrics']}, indent=2))"`

- [ ] **Step 2: Thêm vào README**

- Sơ đồ kiến trúc Mermaid: `CSV → app.train → joblib → FastAPI (main.py) → Streamlit`.
- Bảng "Kết quả mô hình": holdout MAE/RMSE/R² + mô hình được chọn + bảng CV rút gọn.
- Dòng ablation `Stress_Level` (có/không) kèm diễn giải ngắn.
- Bảng ánh xạ: Project 1 → `app/main.py` + `app/schemas.py`; Project 2 → `streamlit_app.py`; Project 3 → `notebooks/model_training.ipynb`.
- Mục Docker: `docker build -t student-wellbeing .` và `docker run -p 8000:8000 student-wellbeing`; ghi rõ chưa được xác nhận trên máy tác giả (daemon tắt).
- Mục `/model-info` + ví dụ JSON.
- Link tới `docs/model_card.md` và notebook.

- [ ] **Step 3: Tạo `docs/model_card.md`**

Gồm: mục đích, dữ liệu (5.000 dòng, 18–24 tuổi), mô hình được chọn + lý do, metrics holdout, CV, ablation `Stress_Level`, hạn chế (khảo sát/tổng hợp, tương quan ≠ nhân quả, `Country="Other"`, ngoài phân phối), cách dùng đúng/sai.

- [ ] **Step 4: Kiểm tra nhất quán**

Run: `.venv\Scripts\python.exe -c "import json; m=json.load(open('artifacts/metrics.json', encoding='utf-8')); print(round(m['metrics']['r2'], 4))"`
Đối chiếu con số R² trong README/model card khớp output.

---

### Task 11: Verification end-to-end

**Files:** không sửa; chỉ chạy kiểm tra.

- [ ] **Step 1: Huấn luyện lại**

Run: `.venv\Scripts\python.exe -m app.train`
Expected: exit 0; metadata JSON in ra có `selected_model`, `cv_results`, `feature_importance`, `stress_ablation`.

- [ ] **Step 2: Toàn bộ test**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: tất cả PASS, không skip ngoài guard (guard phải chạy vì artifact tồn tại).

- [ ] **Step 3: Chạy API và gọi thử**

Start background: `.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000`
Then run:
```powershell
.venv\Scripts\python.exe -c "import requests; print(requests.get('http://127.0.0.1:8000/health').json()); print(requests.get('http://127.0.0.1:8000/model-info').status_code); print(requests.post('http://127.0.0.1:8000/predict', json={'age':21,'gender':'Female','country':'Vietnam','academic_level':'Undergraduate','most_used_platform':'Instagram','purpose_of_use':'Education','avg_daily_usage_hours':4.0,'daily_unlocks':120,'study_hours':4.0,'physical_activity_hours':1.0,'sleep_hours_per_night':7.0,'stress_level':'Medium'}).json())"
```
Expected: `status == "ok"`, `/model-info` 200, `/predict` trả `mental_health_score` trong `[0, 10]`. Dừng process sau kiểm tra.

- [ ] **Step 4: Notebook**

Run: `.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/model_training.ipynb`
Expected: exit 0.

- [ ] **Step 5: Cú pháp Streamlit**

Run: `.venv\Scripts\python.exe -m py_compile streamlit_app.py`
Expected: exit 0.

- [ ] **Step 6: Trạng thái repo**

Run: `git status --short`
Expected: chỉ các file dự kiến thay đổi (app, tests, notebook, streamlit, README, requirements, Dockerfile, .dockerignore, docs, artifacts). Không có file tạm ngoài ý muốn.

---

## Self-Review

- **Spec coverage:** §7.1 → Tasks 1–4; §7.2 → Task 8; §7.3 → Task 6; §7.4 → Task 7; §7.5 → Tasks 5–6; §7.6 → Task 10; §7.3 Docker → Task 9; §8 order → Tasks 1–11.
- **Review Focus:** mỗi dòng có test ở Task 6 (unknown country, clamp, missing artifact), Task 5 (guard), Task 4 (JSON metadata).
- **Type consistency:** `cv_table` index tên `"model"`; metadata `cv_results` list record cùng tên cột; `ModelInfoResponse` khớp; `train_model` trả 7 phần tử được dùng thống nhất ở notebook và `train_and_save`.
- **Không tạo commit** theo quy định hệ thống, dù skill mẫu có bước commit.
