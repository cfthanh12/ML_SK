# Student Social Media & Wellbeing

Đồ án học máy gồm ba phần theo yêu cầu: notebook xây dựng mô hình, REST API
FastAPI không có giao diện, và ứng dụng Streamlit tương tác với API.

## Bài toán và dữ liệu

- Dữ liệu: `Student Social Media And Mental Health Impact.csv` (5.000 bản ghi).
- Bài toán: hồi quy dự đoán `Mental_Health_Score` trên thang điểm 0-10.
- Đầu vào: thông tin học tập, thói quen sử dụng mạng xã hội, giấc ngủ, vận động
  và mức căng thẳng.
- Tiền xử lý: One-Hot Encoding cho biến phân loại; các giá trị
  `Physical_Activity_Hours` âm được chặn về 0 khi huấn luyện. CSV gốc không bị sửa.
- Chia tập đánh giá cố định 80/20 (random seed 42); báo cáo MAE, RMSE và R².
- Mô hình: Random Forest Regression, huấn luyện lại trên toàn bộ dữ liệu sau khi
  đánh giá. Chỉ số thực tế được ghi trong `artifacts/metrics.json`.

## Cài đặt và chạy trên Windows

Mở PowerShell tại thư mục project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 1. Huấn luyện mô hình

```powershell
python -m app.train
```

Lệnh tạo `artifacts/mental_health_model.joblib` và
`artifacts/metrics.json`. Có thể truyền đường dẫn khác:

```powershell
python -m app.train --data "Student Social Media And Mental Health Impact.csv"
```

### 2. Chạy REST API

Trong một PowerShell:

```powershell
uvicorn app.main:app --reload
```

- Kiểm tra API: <http://127.0.0.1:8000/health>
- Tài liệu Swagger: <http://127.0.0.1:8000/docs>

Ví dụ yêu cầu `POST /predict`:

```json
{
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
  "stress_level": "Medium"
}
```

Ví dụ phản hồi:

```json
{
  "mental_health_score": 6.42,
  "score_scale": "0-10",
  "model_version": "1.0.0"
}
```

Điểm số ở trên chỉ minh họa định dạng; kết quả thực tế phụ thuộc mô hình đã huấn luyện.

### 3. Chạy ứng dụng giao diện

Giữ API đang chạy, mở PowerShell thứ hai tại thư mục project:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Nếu API ở địa chỉ khác, đặt biến môi trường `API_URL` trước khi chạy Streamlit:

```powershell
$env:API_URL = "http://127.0.0.1:8000"
streamlit run streamlit_app.py
```

### Notebook

Mở `notebooks/model_training.ipynb` bằng Jupyter hoặc VS Code. Notebook thực hiện
khảo sát dữ liệu, làm sạch, đánh giá mô hình và lưu cùng artifact mà API sử dụng.

## Kiểm thử

Sau khi đã tạo artifact:

```powershell
python -m pytest -q
```

## Lưu ý sử dụng

Đây là đồ án/demo học máy dựa trên dữ liệu khảo sát, không phải công cụ chẩn đoán,
sàng lọc hay tư vấn sức khỏe tinh thần. Kết quả là ước lượng thống kê, không chứng
minh quan hệ nhân quả và không nên dùng để ra quyết định về cá nhân. Mô hình chỉ
được huấn luyện trên độ tuổi 18-24; API giới hạn tuổi trong phạm vi này. Một số
giá trị phân loại mới (ví dụ quốc gia không xuất hiện trong tập huấn luyện) được
mã hóa an toàn như nhóm chưa thấy, nhưng chất lượng dự đoán ngoài phân phối không
được đảm bảo.
