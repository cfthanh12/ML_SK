import {
  bullets,
  code,
  fmt,
  h1,
  h2,
  pagebreak,
  p,
} from "./helpers.mjs";

export function chaptersTenToThirteen(data) {
  const rf = data.cv_results.find((row) => row.model === "random_forest");
  const linear = data.cv_results.find((row) => row.model === "linear_regression");
  const gap = linear.rmse_mean - rf.rmse_mean;
  const ext = data.extended;
  const eda = ext.eda;
  const bands = Object.values(ext.error_analysis.mean_residual_by_actual_band);
  const tuned = ext.holdout_comparison.find((item) => item.model === "Random Forest đã tinh chỉnh (v1.2.0)");
  const initial = ext.holdout_comparison.find((item) => item.model === "Random Forest ban đầu (v1.1.0)");
  const r2Delta = data.stress_ablation.metrics.r2 - data.metrics.r2;
  const appendix = JSON.stringify(
    {
      model_version: data.model_version,
      selected_model: data.selected_model,
      metrics: data.metrics,
      cv_results: data.cv_results.map((row) => ({
        model: row.model,
        rmse_mean: row.rmse_mean,
        r2_mean: row.r2_mean,
      })),
      stress_ablation: {
        dropped_feature: data.stress_ablation.dropped_feature,
        metrics: data.stress_ablation.metrics,
      },
      training_rows: data.training_rows,
    },
    null,
    2,
  );

  return [
    h1("10. Hạn chế và hướng phát triển"),
    h2("10.1. Hạn chế"),
    bullets([
      "Tương quan không phải nhân quả: mô hình cho biết thói quen nào đi kèm với " +
        "điểm số nào, không cho biết thay đổi thói quen sẽ làm điểm số thay đổi ra sao.",
      "Dữ liệu là khảo sát và có dấu hiệu tổng hợp: quan hệ giữa biến và điểm số " +
        "mạnh và đều bất thường so với dữ liệu hành vi thu thập tự do. Vì vậy chỉ số " +
        "cao trên bộ dữ liệu này không chuyển được sang bối cảnh khác.",
      "Phạm vi tuổi hẹp: toàn bộ mẫu là sinh viên 18–24 tuổi; API chủ động từ chối " +
        "tuổi ngoài khoảng này thay vì đưa ra dự đoán không có cơ sở.",
      "Country bị phân tán: phần lớn dòng nằm trong nhóm Other, nên mức ảnh hưởng " +
        "của biến này khó diễn giải và có thể phản ánh cách gộp dữ liệu. " +
        `${eda.countries_under_30_rows}/${eda.n_countries} quốc gia có dưới 30 dòng, ` +
        `Vietnam chỉ có ${eda.vietnam_rows} dòng, nên mô hình gần như chưa học gì về ` +
        "sinh viên Việt Nam.",
      "Nền tảng và quốc gia trùng nhau: LINE, KakaoTalk, WeChat và VKontakte mỗi nền " +
        "tảng chỉ xuất hiện ở đúng một quốc gia, nên không thể tách tác động của nền " +
        "tảng khỏi tác động của quốc gia.",
      "Kéo về trung bình: mô hình dự đoán cao hơn thực tế khoảng " +
        `${fmt(Math.abs(bands[0].mean_residual), 2)} điểm ở nhóm điểm thấp nhất và thấp ` +
        `hơn khoảng ${fmt(bands[bands.length - 1].mean_residual, 2)} điểm ở nhóm cao nhất. ` +
        "Nếu dùng để sàng lọc, đúng nhóm cần quan tâm lại dễ bị bỏ sót.",
      "Không ngoại suy: Random Forest cho cùng một điểm với mọi giá trị nằm ngoài " +
        "khoảng huấn luyện (ví dụ 8,8 giờ hay 20 giờ dùng mạng xã hội). API cảnh báo " +
        "các trường hợp này nhưng không thể sửa được giới hạn của mô hình.",
      "Điểm mục tiêu là tự đánh giá, không phải thang đo lâm sàng; dữ liệu không " +
        "kèm mô tả phương pháp thu thập hay định nghĩa điểm số.",
      "Chất lượng ngoài phân phối không được đảm bảo, kể cả khi API vẫn trả một " +
        "con số trong [0, 10]; cảnh báo trong phản hồi chỉ giúp người dùng biết điều đó.",
    ]),

    h2("10.2. Hướng phát triển"),
    bullets([
      "Thu thập dữ liệu thật theo bảng hỏi có kiểm định, ghi rõ định nghĩa từng biến " +
        "và quy trình tạo điểm số.",
      "Bổ sung kiểm định thống kê và mô hình nhân quả (ví dụ phân tích hồi quy có " +
        "kiểm soát biến gây nhiễu) thay vì chỉ dự đoán tương quan.",
      "Đánh giá độ tin cậy của dự đoán bằng khoảng tin cậy hoặc calibration, thay vì " +
        "chỉ trả một điểm duy nhất.",
      "Giám sát khi triển khai: API đã ghi log và cảnh báo từng yêu cầu; bước tiếp " +
        "theo là tổng hợp tỉ lệ cảnh báo theo ngày để phát hiện dịch chuyển phân phối, " +
        "hoặc đưa số liệu vào Prometheus và Grafana (Building Data Science Applications " +
        "with FastAPI, chương 15).",
      "Thu thập dữ liệu sinh viên Việt Nam để mô hình dùng được trong bối cảnh trong nước.",
      "Giải thích từng dự đoán riêng lẻ bằng SHAP, và trả về khoảng dự đoán (ví dụ " +
        "quantile regression) thay vì một con số duy nhất.",
      "Deploy giao diện lên Hugging Face Spaces hoặc Streamlit Community Cloud để có " +
        "đường dẫn demo công khai, như đề bài gợi ý.",
      "Xác nhận build Docker trên máy có Docker Desktop, và thêm bước sinh báo cáo " +
        "vào CI.",
    ]),
    pagebreak(),

    h1("11. Kết luận"),
    p(
      "Đồ án đã xây dựng một quy trình học máy hoàn chỉnh cho bài toán dự đoán điểm " +
        "sức khỏe tinh thần từ thói quen sử dụng mạng xã hội của sinh viên, và " +
        "triển khai quy trình đó thành ba sản phẩm theo yêu cầu: notebook xây dựng " +
        "mô hình, REST API trả JSON và ứng dụng Streamlit có giao diện.",
    ),
    p(
      `Về câu hỏi thứ nhất, mô hình được chọn — Random Forest đã tinh chỉnh — đạt RMSE ` +
        `${fmt(rf.rmse_mean, 4)} trong cross-validation và MAE ${fmt(data.metrics.mae, 4)}, ` +
        `RMSE ${fmt(data.metrics.rmse, 4)}, R² ${fmt(data.metrics.r2, 4)} trên holdout. ` +
        `So với baseline dự đoán bằng trung bình (RMSE ${fmt(data.cv_results.find((row) => row.model === "baseline_dummy").rmse_mean, 4)}), ` +
        "mức cải thiện là rất lớn; so với hồi quy tuyến tính, mô hình phi tuyến tốt " +
        `hơn ${fmt(gap, 4)} RMSE, cho thấy dữ liệu có quan hệ phi tuyến và tương tác ` +
        `giữa các biến. Tinh chỉnh siêu tham số nâng R² holdout từ ${fmt(initial.holdout_r2, 4)} ` +
        `lên ${fmt(tuned.holdout_r2, 4)} mà không dùng tập holdout để chọn tham số.`,
    ),
    p(
      "Về câu hỏi thứ hai, hai biến đóng góp nhiều nhất là số giờ dùng mạng xã hội " +
        `và số giờ ngủ; riêng Stress_Level chỉ làm R² thay đổi ${fmt(r2Delta, 5)} khi ` +
        `bị loại bỏ (${fmt(data.metrics.r2, 4)} so với ${fmt(data.stress_ablation.metrics.r2, 4)}), ` +
        "nên gần như dư thừa khi đã có hai biến kia. Về câu hỏi thứ ba, toàn bộ kết " +
        "quả tái lập được bằng một lệnh, có kiểm thử tự động và artifact tự mô tả.",
    ),
    p(
      "Ngoài các chỉ số tổng, kiểm thử hành vi cho thấy mô hình học đúng chiều: tăng " +
        "giờ dùng mạng xã hội làm điểm giảm ở " +
        `${fmt(ext.behavioural_tests.directional_usage_2h_to_7h.share_score_decreases * 100, 1)}% ` +
        "mẫu, và đổi giới tính gần như không làm đổi điểm. Learning curve cho thấy " +
        "phương sai là vấn đề chính, nên thêm dữ liệu là hướng cải thiện hợp lý nhất.",
    ),
    p(
      "Giá trị lớn nhất của đồ án không nằm ở con số R² mà ở cách đi đến con số đó: " +
        "chọn mô hình bằng cross-validation trên tập train, chỉ báo cáo trên holdout, " +
        "đo mức ảnh hưởng bằng permutation importance và kiểm chứng bằng ablation. " +
        "Những bước này giúp phân biệt một mô hình thực sự có cơ sở với một mô hình " +
        "chỉ tình cờ khớp dữ liệu.",
    ),
    pagebreak(),

    h1("12. Tài liệu tham khảo"),
    bullets([
      "Pedregosa, F. và cộng sự (2011). Scikit-learn: Machine Learning in Python. " +
        "Journal of Machine Learning Research, 12, 2825–2830.",
      "Buitinck, L. và cộng sự (2013). API design for machine learning software: " +
        "experiences from the scikit-learn project.",
      "Ramírez, S. (2018). FastAPI Documentation. https://fastapi.tiangolo.com/",
      "Streamlit (2024). Streamlit Documentation. https://docs.streamlit.io/",
      "Pandas Development Team (2024). pandas Documentation. https://pandas.pydata.org/docs/",
      "Pydantic (2024). Pydantic Documentation. https://docs.pydantic.dev/",
      "Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32.",
      "Kaggle (2024). Student Social Media & Mental Health Impact. " +
        "https://www.kaggle.com/datasets/shivasingh4945/student-social-media-and-mental-health-impact",
      "Huyen, C. (2022). Designing Machine Learning Systems: An Iterative Process for " +
        "Production-Ready Applications. O'Reilly Media.",
      "Voron, F. (2023). Building Data Science Applications with FastAPI, 2nd ed. " +
        "Packt Publishing.",
      "Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and " +
        "TensorFlow, 3rd ed. O'Reilly Media.",
      "Ng, A. Machine Learning Yearning (bản dịch tiếng Việt Khát khao học máy, nhóm " +
        "dịch Machine Learning Cơ Bản).",
      "Vũ Hữu Tiệp (2020). Machine Learning cơ bản (bản cập nhật 20/01/2020).",
      "Oluleye, A. (2023). Exploratory Data Analysis with Python Cookbook. Packt Publishing.",
      "Srinivasaraghavan, A. và Joseph, V. (2019). Machine Learning. Wiley India.",
      "Nguyễn Thanh Phước. The Machine Learning Landscape; End-to-End Machine " +
        "Learning Project (bài giảng chương 1 và 2, v1.0); ML Project Template " +
        "Structure (2024).",
      "Mitchell, M. và cộng sự (2019). Model Cards for Model Reporting. Proceedings " +
        "of the Conference on Fairness, Accountability, and Transparency, 220–229.",
    ]),
    pagebreak(),

    h1("Phụ lục A. Cấu trúc thư mục"),
    code(
      "app/\n" +
        "  features.py      # tên cột, phiên bản, đọc config/params.yaml\n" +
        "  preprocess.py    # xuất dữ liệu từng bước tiền xử lý ra data/processed\n" +
        "  schemas.py       # Pydantic request/response\n" +
        "  train.py         # pipeline, cross-validation, chọn mô hình, ablation, lưu artifact\n" +
        "  evaluate.py      # đánh giá mở rộng, tinh chỉnh, learning curve, kiểm thử hành vi\n" +
        "  main.py          # FastAPI: /health, /model-info, /predict, /predict/batch\n" +
        "config/params.yaml # siêu tham số, ngưỡng cảnh báo, ngưỡng độ trễ\n" +
        "data/\n" +
        "  raw/             # CSV gốc, không chỉnh sửa\n" +
        "  processed/       # cleaned, train, holdout, encoded_train_sample, preprocessing_report\n" +
        "  external/        # dành cho dữ liệu ngoài (hiện không dùng)\n" +
        "models/\n" +
        "  mental_health_model.joblib, metadata.yaml\n" +
        "reports/\n" +
        "  metrics.json, experiments.json, figures/\n" +
        "notebooks/\n" +
        "  01_eda.ipynb, 02_model_training.ipynb, 03_inference.ipynb\n" +
        "docs/\n" +
        "  data_dictionary.md, model_card.md, test_plan.md, yeu_cau_va_doi_chieu.md, report/\n" +
        "logs/              # app.log của API\n" +
        "scripts/\n" +
        "  report_figures.py    # sinh dữ liệu và hình cho báo cáo\n" +
        "  build_report.mjs     # dựng file .docx\n" +
        "  lib/render.mjs       # style, mục lục, bảng, hình\n" +
        "  content/             # nội dung từng chương\n" +
        "tests/\n" +
        "  conftest.py, test_api.py, test_api_extensions.py, test_schemas.py,\n" +
        "  test_training.py, test_evaluate.py, test_data.py, test_behaviour.py, test_preprocess.py,\n" +
        "  test_artifact_guard.py, test_report_figures.py, test_report_docx.py\n" +
        ".github/workflows/ci.yml\n" +
        "streamlit_app.py\n" +
        "Dockerfile, requirements.txt, requirements-api.txt, pytest.ini",
    ),

    h2("Phụ lục B. Trích metadata mô hình"),
    code(appendix),
  ];
}
