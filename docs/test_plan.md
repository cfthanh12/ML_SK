# Kế hoạch và kịch bản kiểm thử (v1.2.0)

Chạy toàn bộ: `python -m pytest -q` (90 ca, khoảng 1 phút). GitHub Actions chạy lại mỗi lần push
(`.github/workflows/ci.yml`). Lần chạy gần nhất: **90/90 đạt**.

Bộ test giữ nguyên cách làm của bản nộp: phần API và huấn luyện chạy trên **dữ liệu tổng hợp**
do `tests/conftest.py` tạo và một artifact tạm, nên chạy được trên máy sạch. Các nhóm mới
(dữ liệu, hành vi) chạy trên CSV thật trong `data/raw/` và tự bỏ qua nếu thiếu file.

## Tổng quan theo tệp

| Tệp | Số ca | Nguồn | Phạm vi |
|---|---|---|---|
| tests/test_api.py | 14 | Bản nộp | /health, /predict trong [0,10], tuổi ngoài 18–24 trả 422, /model-info, quốc gia lạ, clamp, NaN trả 500, nạp artifact lỗi, logger |
| tests/test_schemas.py | 10 | Bản nộp | Biên tuổi, trường lạ, country rỗng, giới hạn giờ |
| tests/test_training.py | 14 | Bản nộp (cập nhật) | Chặn giá trị âm, thiếu cột, 4 ứng viên, CV, chọn mô hình, importance, ablation, lưu artifact, `training_profile` |
| tests/test_artifact_guard.py | 1 | Bản nộp | Artifact trong repo khớp phiên bản và danh sách biến |
| tests/test_report_figures.py | 3 | Bản nộp (cập nhật) | Dữ liệu báo cáo khớp metrics.json, đủ 12 hình |
| tests/test_report_docx.py | 4 | Bản nộp (cập nhật) | File Word v2 có 12 hình, 21 bảng, đủ chú thích, chứa chỉ số hiện tại |
| tests/test_api_extensions.py | 22 | Mới | Cảnh báo, batch, 10 ca giá trị sai, thiếu/thừa trường, độ trễ, header thời gian |
| tests/test_behaviour.py | 8 | Mới | Kiểm thử hành vi trên dữ liệu thật |
| tests/test_data.py | 7 | Mới | Kiểm tra dữ liệu thật |
| tests/test_evaluate.py | 3 | Mới | Công thức R² hiệu chỉnh, chỉ số đánh giá |
| tests/test_preprocess.py | 4 | Mới | Xuất dữ liệu từng bước tiền xử lý, metadata.yaml khớp artifact |

## Kịch bản kiểm thử API mới (tests/test_api_extensions.py)

| Mã | Kịch bản | Đầu vào | Kỳ vọng |
|---|---|---|---|
| TC-A01 | Hồ sơ trong vùng dữ liệu | Hồ sơ mẫu (USA) | 200, `warnings` rỗng |
| TC-A02 | Tính tất định | Gọi 2 lần | Kết quả giống hệt |
| TC-A03 | Giá trị sai (10 ca) | tuổi 17, giới tính lạ, stress lạ, nền tảng lạ, mục đích lạ, giờ âm, ngủ 25 giờ, 5.000 lần mở khóa, quốc gia trống, kiểu chữ | 422 |
| TC-A04 | Thiếu hoặc thừa trường | Bỏ `sleep_hours_per_night`; thêm `income` | 422 |
| TC-A05 | Ngoài vùng dữ liệu | Giờ dùng 15 | 200 + cảnh báo "ngoại suy" |
| TC-A06 | Quốc gia chưa gặp | Atlantis | 200 + cảnh báo "chưa có" |
| TC-A07 | Quốc gia hiếm | Vietnam (5 dòng) | 200 + cảnh báo "5 dòng" |
| TC-A08 | Batch khớp đơn lẻ | 2 hồ sơ | Điểm batch = điểm gọi riêng |
| TC-A09 | Giới hạn batch | 0 và 201 hồ sơ | 422 |
| TC-A10 | Thông tin mô hình | GET /model-info | Có `numeric_ranges`, `categories` |
| TC-A11 | Độ trễ | 20 lần sau làm nóng | Trung vị < 100 ms (config/params.yaml) |
| TC-A12 | Header thời gian | POST /predict | Có `X-Process-Time-ms` |

## Kiểm thử hành vi (tests/test_behaviour.py, Chip Huyen ch.6)

| Mã | Loại | Thao tác trên 1.000 mẫu holdout | Ngưỡng | Kết quả đo |
|---|---|---|---|---|
| TC-B01 | Directional | Giờ dùng 2 → 7 giờ | Điểm giảm ở ≥ 95% mẫu | 100%, trung bình −1,51 |
| TC-B02 | Directional | Giờ ngủ 5 → 8 giờ | Điểm tăng ở ≥ 80% mẫu | 96,4%, trung bình +0,83 |
| TC-B03 | Invariance | Đổi giới tính | Thay đổi TB < 0,15; < 5% mẫu lệch > 0,5 | 0,073; 0,1% |
| TC-B04 | Perturbation | Nhiễu ±0,25 giờ vào 3 biến giờ | Thay đổi TB < 0,30 | 0,115 |
| TC-B05 | Chất lượng | R² holdout và khoảng cách train–holdout | R² ≥ 0,85; khoảng cách < 0,10 | 0,918; 0,070 |
| TC-B06 | Không ngoại suy | Giờ dùng 8,8 (lớn nhất trong train) so với 12 và 20 giờ, 200 mẫu | Điểm giống hệt | Chênh 0 |
| TC-B07 | Ablation | Bỏ `Stress_Level`, huấn luyện lại | R² holdout lệch < 0,01 | 0,91837 → 0,91838 |
| TC-B08 | Kéo về trung bình | Phần dư (thật − dự đoán) ở dải điểm ≤ 5 và > 8 | Dải thấp âm, dải cao dương, mỗi dải \|lệch\| < 0,6 | −0,29 (246 mẫu); +0,35 (100 mẫu) |

Ghi nhận (không phải lỗi): ở bước nhỏ 0,5 giờ mô hình không đơn điệu hoàn toàn (bước tăng lớn
nhất 0,23 điểm).

TC-B06 đến TC-B08 khóa lại các phát hiện thú vị của dự án thành test: nếu lần huấn luyện sau
thay đổi tính chất (ví dụ đổi sang mô hình có ngoại suy, hoặc thêm biến làm Stress_Level có ích,
hoặc giảm được hiện tượng kéo về trung bình), test sẽ báo để nhóm cập nhật báo cáo. TC-B08 là
test theo dõi hạn chế đã biết; nếu mô hình mới hết lệch thì sửa ngưỡng chứ không phải lỗi.

## Kiểm tra dữ liệu (tests/test_data.py)

| Mã | Kiểm tra | Kỳ vọng |
|---|---|---|
| TC-D01 | Cột và số dòng | 12 đặc trưng + mục tiêu, 5.000 dòng |
| TC-D02 | Giá trị thiếu | 0 |
| TC-D03 | Thang điểm | 0 ≤ điểm ≤ 10 |
| TC-D04 | Giá trị vận động âm | 10 giá trị được chặn về 0, CSV gốc không đổi |
| TC-D05 | Dòng trùng | Báo cáo đúng 2 dòng |
| TC-D06 | Chia train/holdout | 4.000/1.000, không giao nhau |
| TC-D07 | Biến gây nhiễu nền tảng–quốc gia | LINE (Japan, 50), KakaoTalk (South Korea, 36), WeChat (China, 36), VKontakte (Russia, 40): mỗi nền tảng đúng 1 quốc gia |
| TC-D08 | Xuất data/processed | Đủ 5 file: cleaned, train, holdout, encoded_train_sample, preprocessing_report |
| TC-D09 | Làm sạch và chia tập trong file xuất | 5.000 dòng, không còn giá trị âm; 4.000/1.000 không giao nhau, khớp cột `split` |
| TC-D10 | Dữ liệu sau mã hóa | 134 cột (6 số + 128 one-hot); mỗi dòng có đúng 6 giá trị 1 (một cho mỗi biến phân loại) |
| TC-M01 | models/metadata.yaml | Phiên bản, thời điểm huấn luyện, R² khớp với file mô hình |

## Kịch bản kiểm thử thủ công giao diện

Chuẩn bị: `uvicorn app.main:app` rồi `streamlit run streamlit_app.py`.

| Mã | Bước | Kỳ vọng |
|---|---|---|
| TC-U01 | Mở trang khi API chạy | Sidebar có phiên bản 1.2.0, chỉ số, importance; ô quốc gia có 111 lựa chọn |
| TC-U02 | Kịch bản "Cân bằng" → Ước lượng | Điểm 8,03; cảnh báo Vietnam chỉ có 5 dòng; 2 biểu đồ what-if |
| TC-U03 | Đổi sang India, giờ dùng 12 | Điểm 6,59; cảnh báo giờ dùng ngoài khoảng 1–8,8 |
| TC-U04 | Tab what-if Giờ dùng | Đường đi xuống, đi ngang từ khoảng 6 giờ |
| TC-U05 | Tắt API rồi bấm Ước lượng | Báo lỗi kết nối, trang không sập |
| TC-U06 | Mở /docs | Có /health, /model-info, /predict, /predict/batch |

TC-U01 đến TC-U03 đã chạy tự động bằng `streamlit.testing` với API thật (điểm 8,03 và 6,59 như trên).

## Rủi ro còn lại

| Rủi ro | Biện pháp hiện có | Việc còn lại |
|---|---|---|
| Đầu vào ngoài phân phối | Cảnh báo trong API, log | Tổng hợp tỉ lệ cảnh báo theo ngày |
| Quốc gia hiếm (85/111 dưới 30 dòng) | Cảnh báo khi < 30 dòng | Thu thêm dữ liệu |
| Kéo về trung bình ở hai đầu thang điểm | Ghi trong báo cáo và model card | Thử mô hình hoặc hiệu chỉnh khác |
| Docker chưa build thử | Dockerfile đã thêm `config/` | Build trên máy có Docker Desktop |
