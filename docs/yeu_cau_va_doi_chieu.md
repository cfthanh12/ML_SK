# Yêu cầu đồ án (đã bổ sung) và đối chiếu với dự án — v1.2.0

Tài liệu này diễn đạt đầy đủ yêu cầu từ đề bài (`yeu_cau_do_an.png`) và toàn bộ tài liệu trong
thư mục `Học Máy/Lý Thuyết`, rồi đối chiếu với bản hoàn thiện (gộp bản nộp ML_SK_NopBai và bản cải tiến).

Ký hiệu: ✅ đã có trong bản nộp · 🆕 bổ sung trong v1.2.0 · ⚠️ giới hạn đã ghi nhận · ➖ không áp dụng

## 0. Tài liệu đã đọc và mức liên quan

| Tài liệu | Liên quan | Dùng cho |
|---|---|---|
| Đề bài `yeu_cau_do_an.png` | Bắt buộc | Ba project; gợi ý FastAPI và cấu trúc dự án |
| Ch1_PhuocNT The ML Landscape | Cao | Thách thức của học máy, kiểm thử và xác thực |
| Ch2_PhuocNT End-to-End ML Project | Cao | Quy trình 8 bước |
| 2024_PHUOCNT ML Project Template Structure + `Machine-Learning-Project-Structure-main.zip` | Cao | Cấu trúc thư mục, công cụ (pytest, GitHub Actions, Docker, config) |
| 2.Project examples | Trung bình | So sánh ≥ 3 phương pháp, nhiều thước đo |
| Building Data Science Applications with FastAPI, 2nd ed. (Voron, 2023) | Cao | Ch.9 kiểm thử, ch.10 triển khai, ch.12 endpoint dự đoán, ch.15 giám sát |
| Designing ML Systems (Chip Huyen, 2022) | Cao | Rò rỉ dữ liệu, đánh giá offline, triển khai, dịch chuyển phân phối, model card |
| Machine Learning Yearning (bản Việt) | Cao | Phép đo, learning curve, độ chệch/phương sai |
| Machine Learning cơ bản (Vũ Hữu Tiệp, 2020) | Trung bình | Xác thực chéo, quá khớp, điều chuẩn |
| EDA with Python Cookbook (Oluleye, 2023) | Trung bình | Phân tích đơn, hai, đa biến; ngoại lai |
| Machine Learning (Srinivasaraghavan & Joseph, Wiley 2019) | Thấp | Kiến thức nền, tiền xử lý |
| Lab `01/02_*.ipynb` (Géron, Hands-On ML) | Trung bình | Bản code của Ch.1, Ch.2 |
| Lab S3_L16, S3_L17, S3_L24 | Trung bình | Hồi quy tuyến tính, **R² hiệu chỉnh** |
| Machine Learning for Imbalanced Data (2023) | ➖ | Dành cho phân loại mất cân bằng; bài toán là hồi quy. Ý liên quan duy nhất: nhóm quốc gia mất cân bằng, đã xử lý bằng cảnh báo |
| Supervised Learning Workshop (code) | Thấp | Ví dụ hồi quy và đánh giá |
| Bao_cao_NCKH_CNTT (Edge-IIoTset) | ➖ | Đề tài khác; chỉ tham khảo cách kiểm tra rò rỉ dữ liệu |
| Seminar Chương 7 (giữa kì) | Trung bình | Batch và online prediction, triển khai |

## A. Đề bài

| # | Yêu cầu | Tình trạng |
|---|---|---|
| A1 | Project 1: REST API Python, không giao diện, input mẫu → JSON | ✅ /predict, /health, /model-info; 🆕 /predict/batch, `warnings`, log, header độ trễ |
| A2 | Project 2: ứng dụng có giao diện gọi API | ✅ Streamlit, sidebar, kịch bản mẫu; 🆕 what-if, cảnh báo, ô chọn quốc gia |
| A3 | Project 3: notebook xây dựng mô hình theo cấu trúc dự án | ✅ notebook mô hình; 🆕 tách 01_eda / 02_model_training / 03_inference |

## B. Quy trình end-to-end (Ch.2)

| # | Bước | Tình trạng |
|---|---|---|
| B1 | Đặt bài toán, chọn RMSE/MAE | ✅ |
| B2 | Tập test cố định | ✅ 80/20 seed 42; 🆕 test kiểm tra không giao nhau |
| B3 | EDA, tương quan (chỉ đo tuyến tính) | ✅; 🆕 ngưỡng 6 giờ, nền tảng trùng quốc gia |
| B4 | Pipeline tiền xử lý | ✅ |
| B5 | Chọn mô hình bằng CV | ✅ 5-fold trên train |
| B6 | Tinh chỉnh, phân tích lỗi, đánh giá test một lần | 🆕 RandomizedSearchCV chỉ trên train; lỗi theo nhóm và mức điểm |
| B7 | Triển khai, giám sát | ✅ API, Docker; 🆕 log file, độ trễ, cảnh báo, CI |

## C. Thách thức của học máy (Ch.1)

| # | Thách thức | Tình trạng |
|---|---|---|
| C1 | Dữ liệu không đại diện | ⚠️ Vietnam 5 dòng, 85/111 quốc gia < 30 dòng; 🆕 cảnh báo trong API |
| C2 | Dữ liệu kém chất lượng | ✅ chặn giá trị âm; 🆕 báo cáo dòng trùng |
| C3 | Đặc trưng không liên quan | ✅ permutation importance, ablation Stress; 🆕 ablation Country |
| C4 | Quá khớp / chưa khớp | 🆕 R² train vs holdout, learning curve |
| C5 | Data mismatch | 🆕 cảnh báo đầu vào ngoài khoảng huấn luyện |

## D. Cấu trúc dự án (template môn học)

| Thành phần | Bản nộp | v1.2.0 |
|---|---|---|
| data/raw | CSV ở thư mục gốc | 🆕 `data/raw/` |
| data/processed | không có | 🆕 `data/processed/` (cleaned, train, holdout, mẫu sau mã hóa, báo cáo) tạo bằng `python -m app.preprocess` |
| data/external | không có | 🆕 `data/external/` (README: hiện không dùng nguồn ngoài) |
| notebooks (exploratory, modeling, inference) | 1 notebook | 🆕 3 notebook |
| models | `artifacts/` | 🆕 `models/` + `metadata.yaml` |
| reports / results | không có | 🆕 `reports/` (metrics, experiments, figures) |
| config | hằng số trong code | 🆕 `config/params.yaml` |
| docs | model_card | ✅ + 🆕 data_dictionary, test_plan, yeu_cau_va_doi_chieu, thay_doi_v1.2, yeu_cau_do_an.png |
| tests | 46 ca | 🆕 90 ca |
| logs | không có | 🆕 `logs/app.log` |
| requirements | ✅ pin phiên bản | ✅ + pyyaml |
| Dockerfile | ✅ | ✅ sửa để chép `config/` |
| GitHub Actions + pytest | không có | 🆕 `.github/workflows/ci.yml` |
| DVC, pre-commit, cookiecutter | không có | ⚠️ chưa làm, không bắt buộc với quy mô đồ án |

## E. So sánh phương pháp (Project examples)

| # | Yêu cầu | Tình trạng |
|---|---|---|
| E1 | ≥ 3 phương pháp | ✅ 4 ứng viên; 🆕 thêm RF ban đầu/tinh chỉnh, RF bỏ Country |
| E2 | Nhiều thước đo | ✅ MAE, RMSE, R², CV; 🆕 R² hiệu chỉnh, thời gian huấn luyện và dự đoán |
| E3 | Bảng, hình, kết luận | ✅; 🆕 báo cáo có 12 hình, 21 bảng |

## F. FastAPI (Voron, 2023)

| # | Nội dung sách | Tình trạng |
|---|---|---|
| F1 | Lưu mô hình bằng joblib, nạp một lần khi khởi động (ch.12) | ✅ lifespan; 🆕 nén `compress=3` |
| F2 | Hàm dự đoán đồng bộ cho tác vụ CPU (ch.12) | ✅ `def predict` |
| F3 | Pydantic cho input/output | ✅ |
| F4 | pytest, fixture, parametrize (ch.9) | ✅; 🆕 thêm parametrize cho giá trị sai và batch |
| F5 | Biến môi trường, Docker (ch.10) | ✅ MODEL_ARTIFACT_PATH, API_URL, Dockerfile |
| F6 | Logging, giám sát (ch.15) | ✅ logger; 🆕 file log, header độ trễ |
| F7 | Cache kết quả bằng joblib Memory (ch.12) | ⚠️ chưa làm: dự đoán đã dưới 100 ms |
| F8 | Gunicorn nhiều worker (ch.10) | ⚠️ chưa làm, không cần ở quy mô demo |

## G. Chip Huyen, Machine Learning Yearning, Lab

| # | Nội dung | Tình trạng |
|---|---|---|
| G1 | Baseline (Huyen ch.6) | ✅ Dummy |
| G2 | Rò rỉ dữ liệu: tiền xử lý trong pipeline (ch.5) | ✅ |
| G3 | Kiểm thử hành vi: directional, invariance, perturbation (ch.6) | 🆕 |
| G4 | Đánh giá theo nhóm (ch.6) | 🆕 |
| G5 | Batch và online prediction (ch.7) | 🆕 /predict/batch |
| G6 | Edge case, dịch chuyển phân phối (ch.8) | 🆕 cảnh báo |
| G7 | Model card (ch.11) | ✅; 🆕 cập nhật số liệu, kiểm thử hành vi |
| G8 | Phép đo đơn trị + phép đo thỏa mãn (ML Yearning ch.9) | 🆕 RMSE + độ trễ < 100 ms |
| G9 | Learning curve (ML Yearning ch.28–32) | 🆕 |
| G10 | R² hiệu chỉnh (Lab S3_L24) | 🆕 |

## H. Việc chưa làm (đề xuất)

1. Build thử Docker trên máy có Docker Desktop.
2. Deploy giao diện lên Hugging Face Spaces hoặc Streamlit Community Cloud.
3. SHAP cho từng dự đoán; khoảng dự đoán (quantile regression).
4. Thu thập dữ liệu sinh viên Việt Nam.
