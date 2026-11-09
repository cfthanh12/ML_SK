# Thiết kế: Nâng chất lượng đồ án Machine Learning (3 phần)

Ngày: 2026-11-09
Trạng thái: chờ duyệt

## 1. Mục tiêu và lý do

Đồ án hiện đã đủ cấu trúc theo yêu cầu: FastAPI trả JSON (Project 1), Streamlit
gọi API (Project 2), notebook (Project 3). Vấn đề không phải thiếu phần, mà là
độ sâu học máy và độ hoàn thiện khi trình bày/chấm điểm.

Mục tiêu: nâng đồ án từ "chạy được" thành "bảo vệ được trước hội đồng", theo thứ
tự ưu tiên A rồi B:

- **A. Notebook thực sự là nơi xây dựng và giải thích mô hình** (EDA, chất lượng
  dữ liệu, so sánh mô hình bằng cross-validation, diễn giải, residual).
- **B. Deploy + demo + test + trình bày** đủ chắc: endpoint thông tin mô hình,
  Docker, logging, UI tốt hơn, test độc lập artifact, README/model card.

Người dùng là sinh viên bảo vệ đồ án môn Machine Learning. Thành công là: hội
đồng thấy được quy trình học máy có cơ sở, không chỉ một con số R².

## 2. Goals

- Notebook tái lập được end-to-end: EDA → chất lượng dữ liệu → tiền xử lý →
  so sánh mô hình (5-fold CV trên tập train) → đánh giá holdout → feature
  importance → lưu artifact.
- Bảng so sánh tối thiểu 4 mô hình: Dummy (baseline), Linear Regression,
  Random Forest, HistGradientBoosting.
- Chọn mô hình cuối bằng CV trên tập train; holdout 20% chỉ dùng để báo cáo
  trung thực (không dùng để chọn).
- Notebook có phần diễn giải: vì sao điểm cao, tương quan ≠ nhân quả, cẩn thận
  với `Stress_Level` và `Country="Other"`.
- Có ablation định lượng bỏ `Stress_Level` (cùng giao thức CV/holdout) để trả
  lời trực tiếp câu "vì sao R² cao"; kết quả đưa vào notebook, README và model card.
- API có `/model-info` trả metrics, tên mô hình, danh sách feature, feature
  importance; dự đoán được clamp về [0, 10].
- Streamlit hiển thị thông tin mô hình + metrics + feature importance và cho
  chọn kịch bản mẫu để tương tác nhanh.
- Bộ test không phụ thuộc artifact có sẵn trong repo, phủ preprocessing,
  schema, training và API.
- README có bảng metrics thực tế, sơ đồ kiến trúc, bảng ánh xạ 3 yêu cầu,
  hướng dẫn Docker, và model card.

## 3. Non-goals

- Không đổi bài toán hay target `Mental_Health_Score`.
- Không thêm deep learning, AutoML, hay dịch vụ cloud.
- Không thêm auth, database, hàng đợi, monitoring phức tạp.
- Không sửa CSV gốc.
- Không đổi contract `POST /predict` hiện có (chỉ thêm field vào `/model-info`
  và clamp giá trị trả về).

## 4. Anti-goals (kể cả chạy được vẫn tính là thất bại)

- Dùng holdout để chọn mô hình rồi vẫn báo holdout như "kết quả khách quan".
- Notebook và `app/train.py` cho ra hai mô hình khác nhau (drift).
- Làm số đẹp bằng leakage (fit encoder/preprocessing trên toàn bộ dữ liệu
  trước khi chia).
- Biến repo thành mớ abstraction mới không cần thiết.

## 5. Ràng buộc

- Windows + PowerShell, dùng `.venv` sẵn có; không thêm dependency nặng.
- Notebook chạy được từ `notebooks/` hoặc root (giữ bootstrap đường dẫn hiện có).
- Notebook, README, UI giữ tiếng Việt; tên biến/thư viện tiếng Anh.
- Artifact hiện là `{ "model": Pipeline, "metadata": {...} }`; giữ định dạng
  này, chỉ mở rộng metadata.
- `metrics.json` vẫn là JSON người đọc được.

## 6. Các hướng đã cân nhắc

### 6.1. Nơi đặt logic so sánh mô hình

- **(a) Viết toàn bộ thí nghiệm trong notebook.** Ưu: hội đồng thấy code
  sklearn ngay trong notebook. Nhược: trùng logic với `app/train.py`, dễ lệch;
  sửa một chỗ phải sửa hai nơi.
- **(b) Đưa hàm dùng chung vào `app/train.py`, notebook gọi và trực quan hóa.**
  Ưu: một nguồn sự thật, notebook vẫn kể được câu chuyện bằng bảng/hình. Nhược:
  code định nghĩa mô hình nằm ngoài notebook.
- **(c) Tách module `app/modeling.py`.** Ưu: tách bạch rõ. Nhược: thêm file,
  refactor rộng hơn mức cần cho đồ án.

**Chọn (b).** Lý do: notebook vẫn là nơi trình bày và chạy thí nghiệm, nhưng
không nhân đôi logic huấn luyện; sửa một chỗ là API, CLI và notebook cùng đổi.

### 6.2. Cách chọn mô hình cuối

- **Cố định Random Forest** như hiện tại. Ưu: đơn giản, không đổi gì. Nhược:
  không trả lời được câu "sao không thử mô hình khác?".
- **So sánh bằng CV trên tập train, chọn tốt nhất theo RMSE trung bình, rồi
  đánh giá holdout.** Ưu: có cơ sở học máy, tránh leakage chọn mô hình.
  Nhược: phức tạp hơn một chút, metrics có thể thay đổi.

**Chọn cách thứ hai.** Holdout 20% chỉ để báo cáo sau khi đã chọn.

## 7. Thiết kế chi tiết

### 7.1. `app/train.py` (phần A)

Thêm/xoá các hàm sau, giữ `load_dataset` và `prepare_dataset` (đã đúng, không
fit gì trên toàn bộ dữ liệu trước khi chia):

- `build_candidate_models() -> dict[str, Pipeline]`: 4 pipeline dùng chung một
  `ColumnTransformer` (numeric passthrough + OneHotEncoder `handle_unknown="ignore"`).
  Khoá: `"baseline_dummy"`, `"linear_regression"`, `"random_forest"`,
  `"hist_gradient_boosting"`.
- `cross_validate_candidates(models, X, y, cv=5, seed=42) -> pd.DataFrame`:
  dùng `cross_validate` với scoring MAE, RMSE (neg), R²; trả về mean/std theo
  từng mô hình.
- `select_best_model(cv_table) -> str`: chọn theo RMSE trung bình nhỏ nhất.
- `extract_feature_importance(pipeline, X, y) -> list[dict]`: dùng
  **permutation importance** trên tập holdout (`n_repeats=5`, `random_state=42`),
  không dựa vào `feature_importances_`/`coef_`. Lý do: `HistGradientBoosting`
  không có `feature_importances_`, còn `coef_` của Linear trên numeric chưa
  scale (`Daily_Unlocks` 62–273) không so sánh được giữa các feature.
  Các cột one-hot được **gộp về 12 cột nguồn** (`Country_*` → `Country`) để
  không bị 111 quốc gia chiếm top và không phình metadata; trả
  `{feature, importance}` sắp giảm dần, cắt top 12. Metadata ghi thêm
  `feature_importance_method: "permutation"`.
- `train_model(dataset)`: pipeline mới = chia train/holdout → CV 4 ứng viên trên
  train → chọn tên tốt nhất → fit trên train → tính MAE/RMSE/R² trên holdout →
  fit lại trên toàn bộ dữ liệu → trả `(model, metrics, evaluation, cleaning_report,
  cv_table, selected_model_name, feature_importance)`.

  Để phục vụ ablation, `prepare_dataset` nhận thêm tham số
  `feature_columns=FEATURE_COLUMNS`, và `build_pipeline` nhận
  `numeric_features`/`categorical_features`; mặc định giữ nguyên hành vi.
- `run_stress_ablation(dataset) -> dict`: lặp lại đúng giao thức (cùng
  `test_size=0.2`, `random_state=42`) nhưng bỏ `Stress_Level`; trả
  `{dropped_feature, selected_model, metrics, cv_results}`. Không đổi
  `/predict`: API vẫn nhận và dùng `Stress_Level`.
- `save_artifacts(...)`: metadata thêm `selected_model`, `cv_results` (list of
  dict), `feature_importance`, `feature_importance_method`, `stress_ablation`;
  giữ key `metrics` cho holdout để tương thích. Version
  `MODEL_VERSION = "1.1.0"`.

Lưu ý: `HistGradientBoostingRegressor` có `random_state`. Dùng `max_iter=300`,
`random_state=42`.

### 7.2. Notebook `notebooks/model_training.ipynb` (phần A)

Giữ các cell hiện có, thêm/sửa:

1. Intro: nêu câu hỏi nghiên cứu và cảnh báo dữ liệu khảo sát.
2. Import + bootstrap đường dẫn (giữ nguyên).
3. Tổng quan dữ liệu (giữ) — thêm kiểm tra chất lượng:
   - số dòng trùng (2), giá trị `Physical_Activity_Hours` âm (10),
   - `Country.nunique()` (111) và tỉ lệ `"Other"` (1.880 dòng),
   - khoảng target (3.6–9.4), không phủ hết 0–10.
4. EDA: giữ histogram target + stress; thêm heatmap tương quan biến số và
   scatter `Avg_Daily_Usage_Hours`/`Sleep_Hours_Per_Night`/`Study_Hours` vs target.
5. Markdown "Nhận xét dữ liệu": tương quan mạnh với usage (-0.82), sleep (0.77),
   study (0.75), activity (0.52); `Stress_Level` vừa tương quan với target vừa
   với usage/sleep → cẩn thận diễn giải nhân quả; `"Other"` gộp nhiều nước;
   `LINE` điểm cao nhưng ít mẫu.
6. Tiền xử lý: giải thích vì sao OneHot + clip âm; nhấn mạnh không fit trước khi chia.
7. So sánh mô hình: gọi `build_candidate_models` + `cross_validate_candidates`,
   `display` bảng CV, vẽ bar chart RMSE trung bình ± std.
8. Đánh giá holdout của mô hình được chọn: in metrics, giữ scatter actual vs
   predicted, thêm histogram/ scatter residual.
9. Feature importance: gọi `extract_feature_importance` (permutation trên
   holdout, gộp one-hot về cột nguồn), hiển thị bảng + barh, viết 2–3 nhận xét.
10. Ablation `Stress_Level`: gọi `run_stress_ablation`, hiển thị bảng so sánh
    (có/không `Stress_Level`: MAE, RMSE, R² holdout) và 1 đoạn nhận xét về mức
    đóng góp của biến này.
11. Lưu artifact + in metadata.
12. Kết luận & hạn chế: dữ liệu khảo sát/tổng hợp, tương quan không phải nhân
    quả, model không dùng để chẩn đoán, hỗ trợ yếu với quốc gia ngoài phân phối.

Notebook phải được re-run để outputs khớp artifact mới.

### 7.3. API (phần B)

- `app/main.py`:
  - Đọc artifact path từ env `MODEL_ARTIFACT_PATH` (mặc định giữ nguyên) để test
    độc lập.
  - `POST /predict`: giữ contract; clamp `score = min(10, max(0, score))`.
  - `GET /model-info`: trả `model_version`, `selected_model`, `metrics`
    (holdout), `cv_results`, `feature_columns`, `feature_importance`,
    `training_rows`, `trained_at_utc`.
  - `GET /health`: giữ nguyên.
  - Logging: `logging.getLogger("app")`, log lúc load model và mỗi dự đoán (chỉ
    log version + score, không log toàn bộ payload).
- `Dockerfile` (`python:3.14-slim`, khớp venv): copy `app/`, `artifacts/`,
  `requirements.txt`; `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- `requirements.txt` pin đúng phiên bản đang dùng trong `.venv`
  (scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, joblib 1.6.0, …) để artifact
  unpickle nhất quán giữa máy và image.
- `.dockerignore`: `.venv`, `__pycache__`, `tests`, `notebooks`, `.git`.
- Không thêm CORS (Streamlit gọi server-side).
- Docker daemon đang tắt trên máy này: tôi viết Dockerfile + hướng dẫn và kiểm
  tra tĩnh; việc `docker build`/`run` do người dùng xác nhận.

### 7.4. Streamlit (phần B)

- Sidebar/expander gọi `GET /model-info` (cache `st.cache_data`), hiện:
  version, mô hình được chọn, MAE/RMSE/R² holdout.
- Biểu đồ feature importance (top 10) từ `/model-info` nếu có.
- Thêm selectbox "Kịch bản mẫu" (dùng nhiều MXH / cân bằng / học tập cao) đặt
  giá trị mặc định cho form; vẫn cho chỉnh tay.
- Kết quả: `st.metric` + diễn giải theo ngưỡng tương đối (thấp/trung bình/cao)
  kèm disclaimer rõ không phải chẩn đoán.
- Giữ xử lý API không chạy; hiện trạng thái `/health`.

### 7.5. Test (phần B)

- `tests/conftest.py`: fixture tạo dataset tổng hợp nhỏ (~300 dòng, đủ cột) →
  train model vào `tmp_path/artifact.joblib` → set `MODEL_ARTIFACT_PATH` →
  cung cấp `TestClient`. Nhờ vậy test không cần artifact commit sẵn.
- `test_api.py`: health, predict trong [0,10], age ngoài 18–24 → 422,
  `/model-info` có `selected_model` + metrics.
- `test_training.py`: `prepare_dataset` clip âm và báo số lượng; thiếu cột →
  `ValueError`; `build_candidate_models` đủ 4 khoá; `cross_validate_candidates`
  trả đúng số dòng; `select_best_model` trả tên hợp lệ.
- `test_schemas.py`: age biên 18/24 hợp lệ, 17/25 lỗi; field lạ bị từ chối;
  country rỗng bị từ chối; giới hạn giờ [0,24].
- `test_artifact_guard.py`: nếu artifact trong repo tồn tại thì assert
  `model_version == MODEL_VERSION`, `selected_model` hợp lệ và feature list
  khớp `app/features.py`; bỏ qua (skip) khi test chạy bằng artifact tạm. Mục
  đích: bắt artifact cũ bị commit lệch với code (anti-goal #2).
- `test_api.py` cập nhật assert version từ `"1.0.0"` lên `MODEL_VERSION`.

### 7.6. Trình bày (phần B)

- README thêm:
  - sơ đồ kiến trúc (Mermaid: CSV → train → joblib → FastAPI → Streamlit),
  - bảng metrics holdout + bảng CV rút gọn,
  - dòng ablation `Stress_Level` (có/không) và diễn giải ngắn,
  - bảng ánh xạ Project 1/2/3 → file tương ứng,
  - hướng dẫn Docker,
  - endpoint `/model-info` và ví dụ JSON.
- `docs/model_card.md`: mục đích, dữ liệu, mô hình, metrics, hạn chế, cách dùng
  đúng/sai. README link tới.

## 8. Thứ tự thực hiện

1. Refactor `app/train.py` + cập nhật schema/version.
2. Cập nhật và re-run notebook (gồm ablation `Stress_Level`); kiểm tra metrics
   mới, artifact mới.
3. Làm API (`/model-info`, clamp, logging, env path), Dockerfile.
4. Nâng Streamlit.
5. Viết test mới + chỉnh test cũ; chạy `pytest -q`.
6. Cập nhật README + model card.
7. Chạy lại verification end-to-end: `python -m app.train`, `pytest -q`,
   thử API local, notebook chạy hết cell.

## 9. Quyết định của người viết (có thể bác bỏ)

- Giữ tiếng Việt cho notebook/README/UI.
- Nâng `MODEL_VERSION` lên `1.1.0`.
- Chọn mô hình bằng 5-fold CV trên tập train, seed 42; holdout 20% chỉ báo cáo.
- Ứng viên: Dummy, Linear, Random Forest, HistGradientBoosting.
- Metadata giữ key `metrics` (holdout) và thêm `cv_results`, `selected_model`,
  `feature_importance`.
- `/predict` giữ nguyên contract; chỉ clamp giá trị.
- Test dùng env `MODEL_ARTIFACT_PATH` để không phụ thuộc artifact commit.
- Docker dùng base `python:3.14-slim` khớp venv (Python 3.14.8, scikit-learn
  1.9.1, pandas 3.1.0), `requirements.txt` pin đúng phiên bản; image copy
  artifact đã train, không train trong lúc build. Không tự chạy `docker build`
  vì daemon đang tắt.
- Thêm ablation `Stress_Level` định lượng (cùng giao thức chia dữ liệu); kết quả
  vào metadata, notebook, README, model card.
- Feature importance dùng permutation trên holdout, gộp one-hot về cột nguồn —
  không dùng `feature_importances_`/`coef_`.
- Artifact `.joblib` và `metrics.json` tiếp tục được commit làm model demo;
  thêm guard test để phát hiện artifact lệch version/feature.

## 10. Việc còn để người triển khai quyết

- Số fold CV chính xác (mặc định 5) và tham số cụ thể của các mô hình.
- Cách trình bày ngưỡng "thấp/trung bình/cao" trong Streamlit (dựa trên phân vị
  của target trong dữ liệu train).
- Có thêm ảnh chụp UI vào README hay không (cần chạy Streamlit để chụp).
- Cách đặt tên/định dạng bảng ablation trong model card (một dòng hay một mục riêng).
