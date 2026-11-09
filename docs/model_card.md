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
  111 quốc gia nhưng 1.880 dòng là `"Other"`.
- Dữ liệu có dấu hiệu khảo sát/tổng hợp: quan hệ giữa biến và target rất mạnh và
  đều, nên không nên suy rộng ra ngoài phân phối.

## Mô hình và cách chọn

- Ứng viên: Dummy baseline, Linear Regression, Random Forest,
  HistGradientBoosting.
- Mỗi ứng viên dùng chung pipeline: numeric passthrough + One-Hot
  (`handle_unknown="ignore"`, dense) + regressor.
- Chọn bằng 5-fold CV (shuffle, seed 42) trên tập train, tiêu chí RMSE trung bình
  thấp nhất. Holdout 20% chỉ dùng để báo cáo.
- Kết quả: **Random Forest** (`n_estimators=300`, `min_samples_leaf=2`).

## Hiệu năng

| Mô hình | CV RMSE (mean ± std) | CV R² |
|---|---|---|
| Random Forest | 0.4166 ± 0.0118 | 0.8918 |
| HistGradientBoosting | 0.4223 ± 0.0121 | 0.8888 |
| Linear Regression | 0.6035 ± 0.0127 | 0.7732 |
| Dummy | 1.2690 ± 0.0248 | -0.0017 |

Holdout của Random Forest: MAE 0.2854, RMSE 0.3948, R² 0.9101.

Các chỉ số holdout là của mô hình huấn luyện trên 80% dữ liệu. Sau khi báo cáo,
artifact cuối được fit lại trên toàn bộ 5.000 dòng để phục vụ API; `training_rows`
trong metadata là 5.000, còn metrics là của mô hình 80%.

## Diễn giải

Permutation importance trên holdout (ΔRMSE khi permute một biến):

| Biến | Importance |
|---|---|
| `Avg_Daily_Usage_Hours` | 0.8765 |
| `Sleep_Hours_Per_Night` | 0.3412 |
| `Country` | 0.2349 |
| `Daily_Unlocks` | 0.0942 |
| `Most_Used_Platform` | 0.0495 |

Ablation `Stress_Level`:

| Biến thể | MAE | RMSE | R² |
|---|---|---|---|
| Có `Stress_Level` | 0.2854 | 0.3948 | 0.9101 |
| Bỏ `Stress_Level` | 0.2855 | 0.3978 | 0.9088 |

Diễn giải: tín hiệu chính đến từ giờ dùng mạng xã hội và giấc ngủ. `Stress_Level`
gần như dư thừa khi đã có hai biến này. `Country` có importance cao nhưng phần
lớn dữ liệu gộp vào `"Other"`, nên đây có thể là mô hình học nhiễu/quốc gia chứ
không phải khác biệt thực sự; cần thận trọng khi diễn giải.

## Hạn chế

- Tương quan không phải nhân quả; không thể kết luận "dùng mạng xã hội nhiều gây
  điểm thấp" từ mô hình này.
- Chỉ huấn luyện cho độ tuổi 18-24; API từ chối tuổi ngoài phạm vi.
- Quốc gia/danh mục chưa thấy được mã hóa thành nhóm chưa biết; chất lượng dự
  đoán ngoài phân phối không được đảm bảo.
- Dữ liệu có thể là tổng hợp; R² cao không đồng nghĩa mô hình dùng được trong
  thực tế.
- Không dùng cho chẩn đoán, sàng lọc, hay quyết định về một cá nhân.

## Cách dùng đúng và sai

Đúng:

- Minh họa quy trình huấn luyện, chọn mô hình và triển khai API/UI cho đồ án.
- So sánh mô hình, phân tích feature importance và ablation trên dữ liệu này.

Sai:

- Kết luận tình trạng sức khỏe tinh thần của một người.
- Đưa ra lời khuyên y tế hoặc can thiệp dựa trên điểm dự đoán.
- Suy rộng cho độ tuổi khác, quốc gia khác, hoặc dữ liệu thu thập theo cách khác.

## Phiên bản

- `model_version`: 1.1.0
- Artifact: `artifacts/mental_health_model.joblib`
- Metadata: `artifacts/metrics.json`
