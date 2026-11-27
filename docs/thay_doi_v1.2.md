# Thay đổi từ bản nộp (v1.1.0) lên bản hoàn thiện (v1.2.0)

Gốc là bản nộp `ML_SK_NopBai` (trong `3124560085_3124560083.zip`). Mọi phần của bản nộp được giữ,
trừ những chỗ ghi dưới đây. Lý do từng thay đổi xem `docs/yeu_cau_va_doi_chieu.md`.

## Cấu trúc thư mục (theo template môn học)
- CSV: `./` → `data/raw/`. Mô hình: `artifacts/` → `models/`. `metrics.json` → `reports/`.
- Notebook: `model_training.ipynb` → `02_model_training.ipynb`; thêm `01_eda.ipynb`, `03_inference.ipynb`.
- Thêm `config/params.yaml`, `logs/`, `reports/experiments.json`, `reports/figures/`, `.github/workflows/ci.yml`, `.gitignore`.

## Mô hình
- Random Forest dùng siêu tham số tinh chỉnh (200 cây, `max_features=0.5`, `min_samples_leaf=1`) từ `config/params.yaml`.
- Holdout: MAE 0,2854 → 0,2761; RMSE 0,3948 → 0,3762; R² 0,9101 → 0,9184. CV RMSE 0,4166 → 0,3932.
- Metadata có thêm `training_profile` (khoảng giá trị, danh sách nhóm, số dòng mỗi quốc gia) và `duplicate_rows_kept`.
- Lưu nén (`compress=3`), file mô hình khoảng 14 MB.
- `MODEL_VERSION` 1.1.0 → 1.2.0.

## Mã nguồn
- `app/evaluate.py` (mới): so sánh trên holdout, tinh chỉnh, R² hiệu chỉnh, learning curve, lỗi theo nhóm, kiểm thử hành vi, ba hình mới.
- `app/main.py`: thêm `warnings`, `/predict/batch`, header `X-Process-Time-ms`, log ra `logs/app.log`; `/model-info` có `numeric_ranges`, `categories`. Giữ nguyên clamp, `MODEL_ARTIFACT_PATH`, kiểm tra metadata.
- `streamlit_app.py`: giữ sidebar và kịch bản mẫu; thêm ô chọn quốc gia từ `/model-info`, hộp cảnh báo, biểu đồ what-if.
- `Dockerfile`: chép thêm `config/`, tạo `logs/`.

## Kiểm thử: 46 → 86 ca
- Mới: `test_api_extensions.py` (22), `test_behaviour.py` (8), `test_data.py` (7), `test_evaluate.py` (3).
- Sửa: `test_training.py` (phiên bản lấy từ code, kiểm tra `training_profile`), `test_report_figures.py` (12 hình), `test_report_docx.py` (file v2, số liệu đọc từ metrics.json).

## Báo cáo
- `Bao_cao_do_an_ML_v2.docx` (file cũ giữ nguyên ở bản nộp): khoảng 53 trang khi mở bằng LibreOffice, 12 hình, 21 bảng.
- Mục mới: 2.3 đối chiếu tài liệu môn học; 4.7 tinh chỉnh; 4.8 đánh giá mở rộng; 5.3 so sánh holdout và R² hiệu chỉnh; 5.7 kiểm thử hành vi; 5.8 learning curve; 6.5 cấu trúc thư mục; 9.4 kịch bản kiểm thử giao diện. Cập nhật số liệu, hạn chế, kết luận, tài liệu tham khảo, phụ lục.
- Dựng lại: `python -m app.evaluate` → `python scripts/report_figures.py` → `node scripts/build_report.mjs --no-rebuild` (cần `npm install` trong `scripts/`).

## Việc nhóm cần tự làm
- Điền trang bìa trong `scripts/build_report.mjs` (META: trường, giảng viên, họ tên, MSSV, lớp) rồi build lại; hoặc sửa trực tiếp trong Word.
- Mở file Word, bấm Ctrl+A rồi F9 để cập nhật mục lục (hoặc chạy `scripts/finalize_report.ps1`).
- Build thử Docker trên máy có Docker Desktop.
