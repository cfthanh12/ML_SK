"""Interactive Streamlit demo client for the FastAPI prediction service."""

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Student Wellbeing Demo",
    page_icon="📊",
    layout="centered",
)

st.title("Ước lượng điểm sức khỏe tinh thần")
st.caption(
    "Demo học máy dùng dữ liệu thói quen mạng xã hội của sinh viên. "
    "Kết quả không phải chẩn đoán hay tư vấn y tế."
)

api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

PLATFORMS = [
    "Facebook",
    "Instagram",
    "KakaoTalk",
    "LINE",
    "LinkedIn",
    "Snapchat",
    "TikTok",
    "Twitter",
    "VKontakte",
    "WeChat",
    "WhatsApp",
    "YouTube",
]
PURPOSES = ["Education", "Entertainment", "Networking", "News"]
ACADEMIC_LEVELS = ["High School", "Undergraduate", "Graduate"]
STRESS_LEVELS = ["Low", "Medium", "High", "Very High"]

PRESETS = {
    "Cân bằng": {
        "age": 21,
        "gender": "Female",
        "country": "Vietnam",
        "academic_level": "Undergraduate",
        "most_used_platform": "Instagram",
        "purpose_of_use": "Education",
        "avg_daily_usage_hours": 3.0,
        "daily_unlocks": 100,
        "study_hours": 4.0,
        "physical_activity_hours": 1.5,
        "sleep_hours_per_night": 7.5,
        "stress_level": "Medium",
    },
    "Dùng nhiều mạng xã hội": {
        "age": 21,
        "gender": "Female",
        "country": "Vietnam",
        "academic_level": "Undergraduate",
        "most_used_platform": "TikTok",
        "purpose_of_use": "Entertainment",
        "avg_daily_usage_hours": 7.5,
        "daily_unlocks": 240,
        "study_hours": 1.5,
        "physical_activity_hours": 0.5,
        "sleep_hours_per_night": 5.5,
        "stress_level": "Very High",
    },
    "Học tập cao": {
        "age": 21,
        "gender": "Male",
        "country": "Vietnam",
        "academic_level": "Graduate",
        "most_used_platform": "LinkedIn",
        "purpose_of_use": "Education",
        "avg_daily_usage_hours": 2.0,
        "daily_unlocks": 70,
        "study_hours": 7.0,
        "physical_activity_hours": 2.0,
        "sleep_hours_per_night": 8.0,
        "stress_level": "Low",
    },
}

LOW_THRESHOLD = 5.1
HIGH_THRESHOLD = 7.1


@st.cache_data(ttl=60)
def fetch_model_info(url: str) -> dict | None:
    try:
        response = requests.get(f"{url}/model-info", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def score_label(score: float) -> str:
    if score < LOW_THRESHOLD:
        return "thấp"
    if score > HIGH_THRESHOLD:
        return "cao"
    return "trung bình"


model_info = fetch_model_info(api_url)

with st.sidebar:
    st.header("Mô hình")
    if model_info is None:
        st.warning(
            "Không lấy được `/model-info`. Hãy khởi động FastAPI trước khi xem "
            "thông tin mô hình."
        )
    else:
        st.metric("Phiên bản", model_info["model_version"])
        st.caption(f"Mô hình được chọn: `{model_info['selected_model']}`")
        metrics = model_info["metrics"]
        st.write(
            f"Holdout · MAE {metrics['mae']:.3f} · "
            f"RMSE {metrics['rmse']:.3f} · R² {metrics['r2']:.3f}"
        )
        importance = pd.DataFrame(model_info["feature_importance"]).head(10)
        st.caption("Mức ảnh hưởng (permutation importance, holdout)")
        st.bar_chart(importance.set_index("feature"))

    st.divider()
    st.header("Trạng thái API")
    try:
        health = requests.get(f"{api_url}/health", timeout=5).json()
        st.success(f"API: {health.get('status', 'unknown')}")
    except requests.RequestException:
        st.error("API chưa sẵn sàng")

st.subheader("Kịch bản mẫu")
preset_name = st.selectbox(
    "Chọn kịch bản để điền sẵn giá trị",
    list(PRESETS),
    key="preset_name",
)
preset = PRESETS[preset_name]
suffix = list(PRESETS).index(preset_name)

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        age = st.number_input(
            "Tuổi",
            min_value=18,
            max_value=24,
            value=preset["age"],
            step=1,
            key=f"age_{suffix}",
        )
        gender = st.selectbox(
            "Giới tính",
            ["Female", "Male"],
            index=["Female", "Male"].index(preset["gender"]),
            key=f"gender_{suffix}",
        )
        country = st.text_input(
            "Quốc gia", value=preset["country"], key=f"country_{suffix}"
        )
        academic_level = st.selectbox(
            "Bậc học",
            ACADEMIC_LEVELS,
            index=ACADEMIC_LEVELS.index(preset["academic_level"]),
            key=f"academic_{suffix}",
        )
        platform = st.selectbox(
            "Nền tảng dùng nhiều nhất",
            PLATFORMS,
            index=PLATFORMS.index(preset["most_used_platform"]),
            key=f"platform_{suffix}",
        )
        purpose = st.selectbox(
            "Mục đích sử dụng",
            PURPOSES,
            index=PURPOSES.index(preset["purpose_of_use"]),
            key=f"purpose_{suffix}",
        )
    with right:
        usage_hours = st.number_input(
            "Giờ dùng mạng xã hội mỗi ngày",
            min_value=0.0,
            max_value=24.0,
            value=preset["avg_daily_usage_hours"],
            key=f"usage_{suffix}",
        )
        daily_unlocks = st.number_input(
            "Số lần mở khóa mỗi ngày",
            min_value=0,
            max_value=1000,
            value=preset["daily_unlocks"],
            step=1,
            key=f"unlocks_{suffix}",
        )
        study_hours = st.number_input(
            "Giờ học mỗi ngày",
            min_value=0.0,
            max_value=24.0,
            value=preset["study_hours"],
            key=f"study_{suffix}",
        )
        activity_hours = st.number_input(
            "Giờ hoạt động thể chất mỗi ngày",
            min_value=0.0,
            max_value=24.0,
            value=preset["physical_activity_hours"],
            key=f"activity_{suffix}",
        )
        sleep_hours = st.number_input(
            "Giờ ngủ mỗi đêm",
            min_value=0.0,
            max_value=24.0,
            value=preset["sleep_hours_per_night"],
            key=f"sleep_{suffix}",
        )
        stress_level = st.selectbox(
            "Mức độ căng thẳng",
            STRESS_LEVELS,
            index=STRESS_LEVELS.index(preset["stress_level"]),
            key=f"stress_{suffix}",
        )

    submitted = st.form_submit_button("Ước lượng")

if submitted:
    if not country.strip():
        st.error("Vui lòng nhập quốc gia.")
    else:
        payload = {
            "age": age,
            "gender": gender,
            "country": country.strip(),
            "academic_level": academic_level,
            "most_used_platform": platform,
            "purpose_of_use": purpose,
            "avg_daily_usage_hours": usage_hours,
            "daily_unlocks": daily_unlocks,
            "study_hours": study_hours,
            "physical_activity_hours": activity_hours,
            "sleep_hours_per_night": sleep_hours,
            "stress_level": stress_level,
        }
        try:
            response = requests.post(
                f"{api_url}/predict",
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(
                f"Không kết nối được API ({api_url}). "
                f"Hãy khởi động FastAPI trước. Chi tiết: {exc}"
            )
        else:
            score = result["mental_health_score"]
            st.metric(
                "Điểm ước lượng (thang điểm 0-10)",
                f"{score:.2f}",
                help=f"Mức tham khảo: {score_label(score)}",
            )
            st.progress(min(max(score / 10, 0.0), 1.0))
            st.caption(
                "Ngưỡng thấp/trung bình/cao dựa trên phân vị 25/75 của điểm "
                "trong dữ liệu huấn luyện."
            )
            st.info(
                "Đây là dự đoán từ dữ liệu khảo sát và mô hình học máy, "
                "không thể dùng để kết luận tình trạng sức khỏe của một cá nhân."
            )

st.divider()
st.caption(f"API: {api_url} · Dùng `/docs` để thử REST API trực tiếp.")
