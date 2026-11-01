"""Interactive Streamlit demo client for the FastAPI prediction service."""

import os

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

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("Tuổi", min_value=18, max_value=24, value=21, step=1)
        gender = st.selectbox("Giới tính", ["Female", "Male"])
        country = st.text_input("Quốc gia", value="Vietnam")
        academic_level = st.selectbox(
            "Bậc học",
            ["High School", "Undergraduate", "Graduate"],
            index=1,
        )
        platform = st.selectbox(
            "Nền tảng dùng nhiều nhất",
            [
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
            ],
            index=1,
        )
        purpose = st.selectbox(
            "Mục đích sử dụng",
            ["Education", "Entertainment", "Networking", "News"],
            index=1,
        )
    with right:
        usage_hours = st.number_input(
            "Giờ dùng mạng xã hội mỗi ngày", min_value=0.0, max_value=24.0, value=4.0
        )
        daily_unlocks = st.number_input(
            "Số lần mở khóa mỗi ngày", min_value=0, max_value=1000, value=120, step=1
        )
        study_hours = st.number_input(
            "Giờ học mỗi ngày", min_value=0.0, max_value=24.0, value=4.0
        )
        activity_hours = st.number_input(
            "Giờ hoạt động thể chất mỗi ngày",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
        )
        sleep_hours = st.number_input(
            "Giờ ngủ mỗi đêm", min_value=0.0, max_value=24.0, value=7.0
        )
        stress_level = st.selectbox(
            "Mức độ căng thẳng",
            ["Low", "Medium", "High", "Very High"],
            index=1,
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
            st.metric(
                "Điểm ước lượng (thang điểm 0-10)",
                f"{result['mental_health_score']:.2f}",
            )
            st.info(
                "Đây là dự đoán từ dữ liệu khảo sát và mô hình học máy, "
                "không thể dùng để kết luận tình trạng sức khỏe của một cá nhân."
            )

st.divider()
st.caption(f"API: {api_url} · Dùng `/docs` để thử REST API trực tiếp.")
