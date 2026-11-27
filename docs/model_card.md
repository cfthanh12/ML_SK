# Model Card — Student Social Media & Wellbeing

## Mục đích

Ước lượng `Mental_Health_Score` (thang 0-10) từ thói quen sử dụng mạng xã hội,
học tập, giấc ngủ, vận động và mức căng thẳng của sinh viên. Đây là bài tập/demo
học máy, không phải công cụ chẩn đoán hay tư vấn sức khỏe tinh thần.

## Dữ liệu

- 5.000 bản ghi, độ tuổi 18-24.
- 12 biến đầu vào: 6 biến số, 6 biến phân loại.
- Target thực tế nằm trong khoảng 3.6-9.4.
- Chất lượng: 2 dòng trùng; 10 giá trị `Physical_Activity_Hours` âm (chặn về 0);
  111 quốc gia nhưng 1.880 dòng là `"Other"`; 85 quốc gia có dưới 30 dòng,
  Vietnam chỉ có 5 dòng.
- LINE, KakaoTalk, WeChat, VKontakte mỗi nền tảng chỉ xuất hiện ở đúng một quốc
  gia, nên không tách được tác động của nền tảng khỏi quốc gia.
- Dữ liệu có dấu hiệu khảo sát/tổng hợp: quan hệ giữa biến và target rất mạnh và
  đều, nên không nên suy rộng ra ngoài phân phối.

## Mô hình và cách chọn

- Ứng viên: Dummy baseline, Linear Regression, Random Forest,
  HistGradientBoosting.
- Mỗi ứng viên dùng chung pipeline: numeric passthrough + One-Hot
  (`handle_unknown="ignore"`, dense) + regressor.
- Chọn bằng 5-fold CV (shuffle, seed 42) trên tập train, tiêu chí RMSE trung bình
  thấp nhất. Holdout 20% chỉ dùng để báo cáo.
- Kết quả: **Random Forest** (`n_estimators=200`, `max_features=0.5`,
  `min_samples_leaf=1`, `max_depth=None`). Siêu tham số chọn bằng
  RandomizedSearchCV (12 tổ hợp, 3-fold CV chỉ trên tập train), lưu trong
  `config/params.yaml`. Bản 1.1.0 dùng 300 cây, `min_samples_leaf=2`.

## Hiệu năng

| Mô hình | CV RMSE (mean ± std) | CV R² |
|---|---|---|
| Random Forest (đã tinh chỉnh) | 0.3932 ± 0.0134 | 0.9036 |
| HistGradientBoosting | 0.4223 ± 0.0121 | 0.8888 |
| Linear Regression | 0.6035 ± 0.0127 | 0.7732 |
| Dummy | 1.2690 ± 0.0248 | -0.0017 |

Holdout của Random Forest: MAE 0.2761, RMSE 0.3762, R² 0.9184,
R² hiệu chỉnh 0.9057 (p = 134 cột sau One-Hot).
Bản 1.1.0 trên cùng holdout: MAE 0.2854, R² 0.9101.

Các chỉ số holdout là của mô hình huấn luyện trên 80% dữ liệu. Sau khi báo cáo,
artifact cuối được fit lại trên toàn bộ 5.000 dòng để phục vụ API; `training_rows`
trong metadata là 5.000, còn metrics là của mô hình 80%.

Trên holdout, 83.1% dự đoán lệch không quá 0,5 điểm. Mô hình kéo dự đoán
về trung bình: sai số trung bình (thực − dự đoán) là -0.29 ở nhóm điểm 3-5 và
+0.35 ở nhóm 8-10.

## Diễn giải

Permutation importance trên holdout (ΔRMSE khi permute một biến):

| Biến | Importance |
|---|---|
| `Avg_Daily_Usage_Hours` | 0.5487 |
| `Sleep_Hours_Per_Night` | 0.3736 |
| `Country` | 0.2093 |
| `Daily_Unlocks` | 0.1598 |
| `Most_Used_Platform` | 0.0541 |

Ablation `Stress_Level`:

| Biến thể | MAE | RMSE | R² |
|---|---|---|---|
| Có `Stress_Level` | 0.2761 | 0.3762 | 0.9184 |
| Bỏ `Stress_Level` | 0.2741 | 0.3762 | 0.9184 |

Bỏ `Country` làm R² holdout giảm từ 0.9184 xuống 0.8845 (R² hiệu chỉnh
0.9057 so với 0.8809), nên biến này được giữ.

Diễn giải: tín hiệu chính đến từ giờ dùng mạng xã hội và giấc ngủ. `Stress_Level`
gần như dư thừa khi đã có hai biến này. `Country` có importance cao nhưng phần
lớn dữ liệu gộp vào `"Other"`, nên cần thận trọng khi diễn giải.

## Kiểm thử hành vi (holdout, 1.000 mẫu)

| Phép thử | Kết quả |
|---|---|
| Giờ dùng 2 → 7 giờ | Điểm giảm ở 100.0% mẫu, trung bình -1.51 |
| Giờ ngủ 5 → 8 giờ | Điểm tăng ở 96.4% mẫu, trung bình +0.83 |
| Đổi giới tính | Thay đổi trung bình 0.073 điểm |
| Nhiễu ±0,25 giờ | Thay đổi trung bình 0.115 điểm |
| 8,8 giờ so với 20 giờ | Chênh lệch 0.0000: mô hình không ngoại suy |

## Hạn chế

- Tương quan không phải nhân quả; không thể kết luận "dùng mạng xã hội nhiều gây
  điểm thấp" từ mô hình này.
- Chỉ huấn luyện cho độ tuổi 18-24; API từ chối tuổi ngoài phạm vi.
- Không ngoại suy: mọi giá trị ngoài khoảng huấn luyện cho cùng kết quả với giá
  trị biên. API vẫn trả điểm nhưng kèm `warnings` khi biến số nằm ngoài khoảng
  huấn luyện hoặc quốc gia chưa gặp/có dưới 30 dòng.
- Kéo về trung bình ở hai đầu thang điểm: không phù hợp để sàng lọc nhóm điểm thấp.
- Dữ liệu có thể là tổng hợp; R² cao không đồng nghĩa mô hình dùng được trong
  thực tế.
- Không dùng cho chẩn đoán, sàng lọc, hay quyết định về một cá nhân.

## Cách dùng đúng và sai

Đúng:

- Minh họa quy trình huấn luyện, chọn mô hình và triển khai API/UI cho đồ án.
- So sánh mô hình, phân tích feature importance, ablation và kiểm thử hành vi trên dữ liệu này.

Sai:

- Kết luận tình trạng sức khỏe tinh thần của một người.
- Đưa ra lời khuyên y tế hoặc can thiệp dựa trên điểm dự đoán.
- Suy rộng cho độ tuổi khác, quốc gia khác, hoặc dữ liệu thu thập theo cách khác.

## Phiên bản

- `model_version`: 1.2.0
- Artifact: `models/mental_health_model.joblib`
- Metadata: `reports/metrics.json`; đánh giá mở rộng: `reports/experiments.json`
