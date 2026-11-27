# Từ điển dữ liệu (Data dictionary)

Nguồn: `data/raw/Student Social Media And Mental Health Impact.csv`, 5.000 dòng, 13 cột
(12 biến đầu vào + 1 biến mục tiêu), không có giá trị thiếu. Số liệu min/max/trung bình
tính trên toàn bộ file gốc.

## Biến số (numeric)

| Cột trong CSV | Trường trong API | Kiểu | Đơn vị | Min | Max | TB | Ý nghĩa | Ghi chú xử lý |
|---|---|---|---|---|---|---|---|---|
| `Age` | `age` | int | năm | 18 | 24 | 20,82 | Tuổi sinh viên | API chỉ nhận 18–24 (ngoài khoảng trả 422) |
| `Avg_Daily_Usage_Hours` | `avg_daily_usage_hours` | float | giờ/ngày | 1,0 | 8,8 | 5,08 | Thời gian dùng mạng xã hội trung bình mỗi ngày | API nhận 0–24, cảnh báo ngoại suy khi ngoài 1,0–8,8 |
| `Daily_Unlocks` | `daily_unlocks` | int | lần/ngày | 62 | 273 | 171,45 | Số lần mở khóa điện thoại mỗi ngày | API nhận 0–1.000 |
| `Study_Hours` | `study_hours` | float | giờ/ngày | 0,3 | 8,3 | 3,01 | Thời gian học mỗi ngày | API nhận 0–24 |
| `Physical_Activity_Hours` | `physical_activity_hours` | float | giờ/ngày | −0,4 | 4,1 | 1,75 | Thời gian vận động mỗi ngày | **10 giá trị âm được chặn về 0** khi huấn luyện; API nhận 0–24 |
| `Sleep_Hours_Per_Night` | `sleep_hours_per_night` | float | giờ/đêm | 3,6 | 9,9 | 6,63 | Thời gian ngủ mỗi đêm | API nhận 0–24 |

## Biến phân loại (categorical)

Mã hóa bằng One-Hot (`handle_unknown="ignore"`), fit trên tập train.

| Cột trong CSV | Trường trong API | Giá trị (số dòng) | Ý nghĩa | Ghi chú |
|---|---|---|---|---|
| `Gender` | `gender` | Male (2.635), Female (2.365) | Giới tính | Kiểm thử invariance TC-B03: đổi giới tính gần như không đổi điểm |
| `Country` | `country` | 111 giá trị; Other (1.880), India (389), USA (355), Canada (230), Australia (198), UK (185), … | Quốc gia | 85/111 quốc gia dưới 30 dòng (Vietnam: 5) → API cảnh báo; quốc gia chưa gặp → cảnh báo, one-hot toàn 0 |
| `Academic_Level` | `academic_level` | Undergraduate (3.632), Graduate (918), High School (450) | Bậc học | |
| `Most_Used_Platform` | `most_used_platform` | Instagram (1.130), TikTok (918), Facebook (719), LinkedIn (523), YouTube (491), Twitter (462), Snapchat (420), WhatsApp (175), LINE (50), VKontakte (40), KakaoTalk (36), WeChat (36) | Nền tảng dùng nhiều nhất | LINE/KakaoTalk/WeChat/VKontakte mỗi nền tảng chỉ ở 1 quốc gia (TC-D07) |
| `Purpose_Of_Use` | `purpose_of_use` | Entertainment (2.552), Education (1.094), Networking (793), News (561) | Mục đích sử dụng chính | |
| `Stress_Level` | `stress_level` | Very High (1.621), High (1.440), Medium (1.295), Low (644) | Mức căng thẳng tự đánh giá | Gần như trùng thông tin với giờ dùng: bỏ biến R² không đổi (TC-B07) |

## Biến mục tiêu

| Cột | Kiểu | Thang | Min | Max | TB | Ý nghĩa |
|---|---|---|---|---|---|---|
| `Mental_Health_Score` | float | 0–10 | 3,6 | 9,4 | 6,23 | Điểm sức khỏe tinh thần tự khảo sát, cao hơn là tốt hơn. **Không phải chẩn đoán y tế.** API kẹp dự đoán về [0, 10] |

## Dữ liệu sau từng bước xử lý

Xem `data/processed/` (tạo bằng `python -m app.preprocess`):

| Bước | File | Nội dung |
|---|---|---|
| 1. Gốc | `data/raw/*.csv` | 5.000 × 13, không sửa |
| 2. Làm sạch | `data/processed/cleaned.csv` | 5.000 dòng; vận động âm → 0; thêm `row_id`, `split` |
| 3. Chia tập | `data/processed/train.csv`, `holdout.csv` | 4.000 / 1.000 dòng (test_size 0,2, random_state 42) |
| 4. Mã hóa | `data/processed/encoded_train_sample.csv` | 20 dòng train sau ColumnTransformer: 6 cột số + 128 cột one-hot = 134 cột |
| Tóm tắt | `data/processed/preprocessing_report.json` | Số dòng, số thay đổi, tên 134 cột sau mã hóa |

Mô hình **không đọc** các file trong `data/processed/`: khi huấn luyện, bước 2–4 chạy lại
trong pipeline và bước mã hóa chỉ fit trên tập train, nên không rò rỉ dữ liệu holdout.
2 dòng trùng hoàn toàn được giữ lại (có thể là hai sinh viên có câu trả lời giống nhau).
