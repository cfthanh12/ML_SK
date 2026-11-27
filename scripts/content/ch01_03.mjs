import {
  bullets,
  describeRow,
  dot,
  figure,
  fmt,
  h1,
  h2,
  pagebreak,
  p,
  table,
} from "./helpers.mjs";

const corr = (data, feature) =>
  fmt(data.correlations.find((row) => row.feature === feature).value, 4);

const describeValues = (data, column) => {
  const row = describeRow(data, column);
  return `${fmt(row.min, 1)}–${fmt(row.max, 1)}`;
};

export function chaptersOneToThree(data) {
  const eda = data.extended.eda;
  const usageBins = Object.entries(eda.usage_bin_mean_score);
  const firstBin = usageBins[0][1];
  const lastBin = usageBins[usageBins.length - 1][1];
  const stressUsage = eda.stress_mean_usage;
  return [
    h1("1. Giới thiệu"),
    p(
      "Mạng xã hội đã trở thành một phần thường ngày trong đời sống sinh viên: " +
        "học tập, giải trí, liên lạc và cả những khoảng thời gian rảnh đều đi qua " +
        "màn hình điện thoại. Câu hỏi đặt ra không phải là mạng xã hội tốt hay xấu, " +
        "mà là những thói quen số cụ thể — dùng bao nhiêu giờ mỗi ngày, mở khóa bao " +
        "nhiêu lần, ngủ bao nhiêu, học bao nhiêu, vận động bao nhiêu — đi kèm với " +
        "mức điểm sức khỏe tinh thần như thế nào.",
    ),

    h2("1.1. Bối cảnh và động lực"),
    p(
      "Bộ dữ liệu được sử dụng trong đồ án là một cuộc khảo sát " +
        `${dot(data.dataset.rows)} sinh viên, ghi lại đồng thời thói quen sử dụng ` +
        "mạng xã hội, thói quen học tập, giấc ngủ, vận động và một điểm tự đánh giá " +
        `về sức khỏe tinh thần trên thang 0–10. Điểm này nằm trong khoảng ` +
        `${fmt(data.dataset.target.min, 1)} đến ${fmt(data.dataset.target.max, 1)}, ` +
        `trung bình ${fmt(data.dataset.target.mean, 4)}.`,
    ),
    p(
      "Vì dữ liệu đã có sẵn nhãn, bài toán phù hợp để giải bằng học có giám sát. " +
        "Đồ án chọn hướng hồi quy thay vì phân loại vì nhãn là một điểm liên tục: " +
        "dự đoán một con số giữ lại được nhiều thông tin hơn là gán sinh viên vào " +
        "ba bốn nhóm rời rạc, và cho phép đánh giá bằng các chỉ số sai số có đơn vị " +
        "rõ ràng trên thang điểm gốc.",
    ),

    h2("1.2. Bài toán"),
    p(
      "Cho hồ sơ của một sinh viên gồm 12 biến đầu vào (6 biến số và 6 biến phân " +
        "loại), dự đoán điểm " +
        "Mental_Health_Score trên thang 0–10. Đầu vào mô tả độ tuổi, giới tính, " +
        "quốc gia, bậc học, nền tảng được dùng nhiều nhất, mục đích sử dụng, số giờ " +
        "dùng mạng xã hội mỗi ngày, số lần mở khóa điện thoại, số giờ học, số giờ " +
        "vận động, số giờ ngủ mỗi đêm và mức căng thẳng tự khai.",
    ),
    p(
      "Đầu ra là một số thực trong [0, 10]. Đây là bài toán hồi quy nhiều biến, có " +
        "cả biến số và biến phân loại, và có yêu cầu triển khai thực tế: mô hình " +
        "phải chạy được sau một REST API và được một ứng dụng giao diện gọi tới.",
    ),

    h2("1.3. Câu hỏi nghiên cứu"),
    bullets([
      "Mô hình dự đoán điểm sức khỏe tinh thần tốt đến mức nào trên dữ liệu này, " +
        "so với một baseline dự đoán bằng trung bình?",
      "Biến nào đóng góp nhiều nhất vào khả năng dự đoán, và biến Stress_Level có " +
        "thực sự cần thiết khi đã biết số giờ dùng mạng xã hội và số giờ ngủ?",
      "Có thể đóng gói toàn bộ quy trình thành một hệ thống gồm notebook, REST API " +
        "và giao diện demo, với kết quả tái lập được không?",
    ]),

    h2("1.4. Phạm vi và giới hạn"),
    p(
      "Dữ liệu chỉ gồm sinh viên 18–24 tuổi, là dữ liệu khảo sát và có dấu hiệu " +
        "được tổng hợp sẵn: quan hệ giữa các biến và điểm số rất mạnh và khá đều. " +
        "Vì vậy kết quả của đồ án là một bài tập học máy có kiểm chứng, không phải " +
        "công cụ chẩn đoán, sàng lọc hay tư vấn sức khỏe tinh thần. Mô hình chỉ mô " +
        "tả tương quan trong dữ liệu, không chứng minh quan hệ nhân quả, và không " +
        "được dùng để đưa ra quyết định về một cá nhân cụ thể.",
    ),
    bullets([
      "Dự đoán ngoài phân phối — quốc gia chưa từng xuất hiện, độ tuổi ngoài 18–24 — " +
        "không được đảm bảo.",
      "R² cao trên dữ liệu khảo sát không đồng nghĩa mô hình dùng được trong thực tế.",
      "Điểm số là tự đánh giá của người trả lời, không phải thang đo lâm sàng.",
    ]),

    h2("1.5. Ba hạng mục của đồ án"),
    p(
      "Đồ án được tổ chức thành ba hạng mục theo yêu cầu của môn học, và cả ba dùng " +
        "chung một nguồn sự thật về mô hình:",
    ),
    bullets([
      "Một REST API không có giao diện, trả JSON, xây bằng FastAPI.",
      "Một ứng dụng demo có giao diện, gọi API ở trên, xây bằng Streamlit.",
      "Một Jupyter Notebook trình bày toàn bộ quá trình xây dựng mô hình.",
    ]),
    figure(
      data.figures.architecture,
      "Kiến trúc tổng thể của hệ thống",
    ),
    p(
      "Hình 1 cho thấy ba sản phẩm không đứng riêng lẻ: notebook và API cùng dùng " +
        "một pipeline huấn luyện, API chỉ nạp artifact do pipeline đó sinh ra, còn " +
        "giao diện chỉ gọi API. Kiến trúc này được mô tả chi tiết ở chương 6.",
    ),
    p(
      "Điểm nhấn kỹ thuật của đồ án nằm ở cách chọn mô hình và cách diễn giải: so " +
        "sánh bốn ứng viên bằng 5-fold cross-validation trên tập train, chỉ dùng " +
        "holdout 20% để báo cáo, đo mức ảnh hưởng của từng biến bằng permutation " +
        "importance, và chạy thêm một ablation bỏ Stress_Level để trả lời trực tiếp " +
        "câu hỏi biến căng thẳng có cần thiết hay không.",
    ),

    h2("1.6. Cấu trúc báo cáo"),
    p(
      "Chương 2 đối chiếu ba yêu cầu của đồ án và các tài liệu của môn học với các " +
        "thành phần đã xây. Chương 3 khảo sát dữ liệu và chất lượng dữ liệu. Chương 4 " +
        "trình bày phương pháp: giao thức chia dữ liệu, tiền xử lý, bốn ứng viên mô " +
        "hình, chỉ số đánh giá (gồm R² hiệu chỉnh), cách đo mức ảnh hưởng, ablation, " +
        "tinh chỉnh siêu tham số và các phép đánh giá mở rộng. Chương 5 báo cáo kết " +
        "quả thực nghiệm, kiểm thử hành vi và learning curve. " +
        "Chương 6 mô tả kiến trúc hệ thống; chương 7 và 8 lần lượt mô tả REST API " +
        "và ứng dụng Streamlit. Chương 9 trình bày kiểm thử và triển khai. Chương " +
        "10 nêu hạn chế và hướng phát triển, chương 11 kết luận. Phụ lục gồm cấu " +
        "trúc thư mục và trích metadata của mô hình.",
    ),
    pagebreak(),

    h1("2. Yêu cầu đồ án và cách đáp ứng"),
    h2("2.1. Ba yêu cầu"),
    p(
      "Đề bài yêu cầu ba sản phẩm độc lập nhưng liên quan chặt chẽ. Thứ nhất là " +
        "triển khai mô hình để sử dụng dưới dạng backend không có giao diện, nhận " +
        "dữ liệu đầu vào mẫu và trả về JSON. Thứ hai là một tình huống sử dụng: " +
        "ứng dụng demo có giao diện, gọi API đã xây, có hỗ trợ tương tác người " +
        "dùng. Thứ ba là notebook xây dựng mô hình, trình bày quá trình từ dữ liệu " +
        "đến mô hình cuối.",
    ),
    p(
      "Cả ba sản phẩm dùng chung một pipeline huấn luyện trong app/train.py. " +
        "Notebook gọi lại đúng các hàm đó để trình bày, API nạp artifact do chính " +
        "pipeline sinh ra, và giao diện chỉ gọi API. Cách tổ chức này tránh một lỗi " +
        "thường gặp của đồ án: notebook và API cho ra hai mô hình khác nhau.",
    ),

    h2("2.2. Ánh xạ yêu cầu sang thành phần"),
    table(
      "Ánh xạ ba yêu cầu của đồ án sang thành phần và tệp",
      ["Hạng mục", "Sản phẩm", "Tệp chính"],
      [
        [
          "Project 1",
          "REST API không giao diện: /predict và /predict/batch trả JSON kèm cảnh báo, /model-info trả metadata, /health kiểm tra trạng thái",
          "app/main.py, app/schemas.py, app/train.py",
        ],
        [
          "Project 2",
          "Ứng dụng demo có giao diện, gọi API, form 12 trường, ba kịch bản mẫu, hộp cảnh báo và biểu đồ what-if",
          "streamlit_app.py",
        ],
        [
          "Project 3",
          "Ba notebook theo template: khảo sát dữ liệu; so sánh mô hình bằng cross-validation, đánh giá và ablation; suy luận và gọi API",
          "notebooks/01_eda.ipynb, 02_model_training.ipynb, 03_inference.ipynb",
        ],
      ],
      ["left", "left", "left"],
    ),
    p(
      "Ngoài ba hạng mục bắt buộc, đồ án có thêm bộ kiểm thử tự động, Dockerfile " +
        "để đóng gói API, và tài liệu model card mô tả mục đích, hạn chế và cách " +
        "dùng đúng của mô hình.",
    ),

    h2("2.3. Đối chiếu với tài liệu môn học"),
    p(
      "Đề bài chỉ nêu ba sản phẩm, nhưng tài liệu lý thuyết của môn đặt thêm nhiều " +
        "yêu cầu ngầm về quy trình, cấu trúc và cách đánh giá. Bảng dưới tóm tắt yêu " +
        "cầu rút ra từ từng tài liệu và nơi đồ án đáp ứng; bảng chi tiết nằm trong " +
        "docs/yeu_cau_va_doi_chieu.md.",
    ),
    table(
      "Yêu cầu rút ra từ tài liệu môn học và nơi đáp ứng",
      ["Tài liệu", "Yêu cầu rút ra", "Nơi đáp ứng"],
      [
        ["Đề bài (yeu_cau_do_an.png)", "Ba sản phẩm: API, giao diện, notebook; tham khảo Building Data Science Applications with FastAPI và cấu trúc một dự án học máy", "Chương 6–8"],
        ["Ch.2 End-to-End ML Project", "Tám bước từ đặt bài toán đến giám sát; tập test cố định; cross-validation; tinh chỉnh; phân tích lỗi", "Chương 3–5"],
        ["Ch.1 The ML Landscape", "Dữ liệu không đại diện, dữ liệu kém chất lượng, quá khớp và chưa khớp, sai số tổng quát hóa", "Mục 3.3, 5.3, 5.8, 10.1"],
        ["ML Project Template Structure", "Thư mục data/raw, notebooks, models, reports, config, docs, tests, logs; requirements; Dockerfile; GitHub Actions và pytest", "Mục 6.5, Phụ lục A"],
        ["Project examples (slide 5)", "So sánh ít nhất 3 phương pháp bằng nhiều thước đo, có bảng và hình", "Mục 5.1, 5.3"],
        ["Building Data Science Applications with FastAPI", "Lưu mô hình bằng joblib, nạp một lần khi khởi động, endpoint dự đoán dạng hàm đồng bộ, kiểm thử bằng pytest, cấu hình qua biến môi trường, Docker", "Chương 7, mục 9.1–9.2"],
        ["Designing ML Systems (Chip Huyen)", "Baseline, rò rỉ dữ liệu, kiểm thử hành vi, đánh giá theo nhóm, batch và online prediction, dịch chuyển phân phối, model card", "Mục 4.1, 5.4, 5.7, 7.1, 7.4"],
        ["Machine Learning Yearning", "Phép đo đơn trị và phép đo thỏa mãn (độ trễ), learning curve, phân tích lỗi", "Mục 5.8, 9.1"],
        ["Lab S3_L24 và Machine Learning cơ bản", "R² hiệu chỉnh khi có nhiều biến; xác thực chéo; điều chuẩn", "Mục 4.4, 5.3"],
      ],
      ["left", "left", "left"],
    ),
    pagebreak(),

    h1("3. Dữ liệu"),
    h2("3.1. Nguồn và quy mô"),
    p(
      "Dữ liệu là bộ Student Social Media & Mental Health Impact công bố trên " +
        "Kaggle (tác giả shivasingh4945, đường dẫn " +
        "kaggle.com/datasets/shivasingh4945/student-social-media-and-mental-health-impact). " +
        "Tệp CSV gồm " +
        `${dot(data.dataset.rows)} dòng và ${data.dataset.columns} cột: ` +
        `${data.dataset.feature_count} biến đầu vào và một biến mục tiêu. ` +
        "Trang dataset mô tả đây là dữ liệu hành vi của sinh viên nhằm khảo sát " +
        "quan hệ giữa thói quen mạng xã hội và sức khỏe tinh thần, nhưng không kèm " +
        "mô tả chi tiết về phương pháp thu thập hay cách tạo điểm số. Điều này được " +
        "ghi nhận như một hạn chế của đồ án và ảnh hưởng trực tiếp đến cách diễn " +
        "giải kết quả.",
    ),

    h2("3.2. Mô tả biến"),
    table(
      "Mô tả 12 biến đầu vào và biến mục tiêu",
      ["Biến", "Kiểu", "Miền giá trị", "Ý nghĩa"],
      [
        ["Age", "số nguyên", "18–24", "Tuổi của sinh viên"],
        ["Gender", "phân loại", "Female, Male", "Giới tính tự khai"],
        [
          "Country",
          "phân loại",
          `${data.quality.country_count} giá trị`,
          "Quốc gia; phần lớn gộp vào nhóm Other",
        ],
        [
          "Academic_Level",
          "phân loại",
          "High School, Undergraduate, Graduate",
          "Bậc học hiện tại",
        ],
        [
          "Most_Used_Platform",
          "phân loại",
          "12 nền tảng",
          "Nền tảng mạng xã hội dùng nhiều nhất",
        ],
        [
          "Purpose_Of_Use",
          "phân loại",
          "Education, Entertainment, Networking, News",
          "Mục đích sử dụng chính",
        ],
        [
          "Avg_Daily_Usage_Hours",
          "số thực",
          `${describeValues(data, "Avg_Daily_Usage_Hours")} giờ`,
          "Số giờ dùng mạng xã hội mỗi ngày",
        ],
        [
          "Daily_Unlocks",
          "số nguyên",
          `${describeValues(data, "Daily_Unlocks")} lần`,
          "Số lần mở khóa điện thoại mỗi ngày",
        ],
        [
          "Study_Hours",
          "số thực",
          `${describeValues(data, "Study_Hours")} giờ`,
          "Số giờ học mỗi ngày",
        ],
        [
          "Physical_Activity_Hours",
          "số thực",
          `${describeValues(data, "Physical_Activity_Hours")} giờ`,
          "Số giờ vận động thể chất mỗi ngày",
        ],
        [
          "Sleep_Hours_Per_Night",
          "số thực",
          `${describeValues(data, "Sleep_Hours_Per_Night")} giờ`,
          "Số giờ ngủ mỗi đêm",
        ],
        [
          "Stress_Level",
          "phân loại",
          "Low, Medium, High, Very High",
          "Mức độ căng thẳng tự khai",
        ],
        [
          "Mental_Health_Score",
          "số thực (mục tiêu)",
          `${describeValues(data, "Mental_Health_Score")}`,
          "Điểm sức khỏe tinh thần tự đánh giá",
        ],
      ],
      ["left", "left", "left", "left"],
    ),
    table(
      "Thống kê mô tả các biến số",
      ["Biến", "Trung bình", "Độ lệch chuẩn", "Nhỏ nhất", "Q1", "Trung vị", "Q3", "Lớn nhất"],
      data.describe.map((row) => [
        row.column,
        fmt(row.mean, 2),
        fmt(row.std, 2),
        fmt(row.min, 2),
        fmt(row.p25, 2),
        fmt(row.p50, 2),
        fmt(row.p75, 2),
        fmt(row.max, 2),
      ]),
      ["left", "center", "center", "center", "center", "center", "center", "center"],
    ),
    p(
      `Sáu biến số và sáu biến phân loại tạo thành ${data.dataset.feature_count} ` +
        "chiều đầu vào. Các biến số có đơn vị khác nhau — giờ, số lần, tuổi — nên " +
        "mô hình dạng cây không cần chuẩn hóa, còn hồi quy tuyến tính sẽ chịu ảnh " +
        "hưởng của khác biệt thang đo. Điều này được tính đến khi đọc kết quả so " +
        "sánh mô hình ở chương 5.",
    ),

    h2("3.3. Chất lượng dữ liệu"),
    p(
      "Kiểm tra sơ bộ phát hiện bốn vấn đề. Thứ nhất, có " +
        `${data.quality.duplicate_rows} dòng trùng hoàn toàn. Thứ hai, có ` +
        `${data.quality.negative_physical_activity} giá trị ` +
        "Physical_Activity_Hours âm, là giá trị không hợp lệ về mặt vật lý. Thứ ba, " +
        `cột Country có ${data.quality.country_count} giá trị nhưng ` +
        `${dot(data.quality.country_other_rows)} dòng (khoảng ` +
        `${fmt((data.quality.country_other_rows / data.dataset.rows) * 100, 1)}%) ` +
        "được gộp vào nhóm Other, nên thông tin quốc gia bị phân tán mạnh. Thứ tư, " +
        `điểm mục tiêu chỉ nằm trong ${fmt(data.dataset.target.min, 1)}–` +
        `${fmt(data.dataset.target.max, 1)} chứ không phủ hết 0–10.`,
    ),
    table(
      "Vấn đề chất lượng dữ liệu và cách xử lý",
      ["Vấn đề", "Số lượng", "Cách xử lý"],
      [
        [
          "Dòng trùng hoàn toàn",
          `${data.quality.duplicate_rows} dòng`,
          "Giữ nguyên trong báo cáo; không sửa CSV gốc",
        ],
        [
          "Physical_Activity_Hours âm",
          `${data.quality.negative_physical_activity} giá trị`,
          "Chặn về 0 ngay trong bước chuẩn bị dữ liệu huấn luyện; CSV gốc không đổi",
        ],
        [
          "Country phân tán",
          `${data.quality.country_count} giá trị, ${dot(data.quality.country_other_rows)} dòng Other`,
          "Giữ nguyên; One-Hot với handle_unknown=\"ignore\" để quốc gia mới được mã hóa an toàn",
        ],
        [
          "Target không phủ 0–10",
          `${fmt(data.dataset.target.min, 1)}–${fmt(data.dataset.target.max, 1)}`,
          "Dự đoán được clamp về [0, 10] ở API; không suy rộng ngoài miền dữ liệu",
        ],
      ],
      ["left", "left", "left"],
    ),
    p(
      "Nguyên tắc chung là không sửa dữ liệu gốc: mọi hiệu chỉnh diễn ra trong " +
        "pipeline huấn luyện, để có thể chạy lại từ CSV và kiểm chứng từng bước. " +
        "Việc chặn giá trị âm được thực hiện trước khi chia tập, nhưng chỉ là phép " +
        "biến đổi tất định trên từng dòng nên không gây rò rỉ thông tin giữa tập " +
        "train và tập đánh giá.",
    ),

    h2("3.4. Phân tích khám phá"),
    p(
      "Phân bố điểm mục tiêu khá đối xứng quanh trung bình " +
        `${fmt(data.dataset.target.mean, 2)}, tập trung chủ yếu trong khoảng ` +
        `${fmt(data.dataset.target.p25, 1)}–${fmt(data.dataset.target.p75, 1)}. ` +
        "Hai đầu thang điểm ít mẫu hơn, nên sai số ở các vùng cực có thể lớn hơn.",
    ),
    figure(
      data.figures.target_hist,
      "Phân bố điểm Mental_Health_Score",
    ),
    p(
      "Mức căng thẳng phân bố lệch về phía cao: nhóm Very High có " +
        `${dot(data.stress_groups.find((row) => row.level === "Very High").count)} ` +
        "dòng, nhiều hơn nhóm Low. Đây là biến tự khai và có quan hệ chặt với điểm " +
        "số, nên cần thận trọng khi diễn giải.",
    ),
    figure(
      data.figures.stress_counts,
      "Số dòng theo mức Stress_Level",
    ),
    table(
      "Hệ số tương quan với Mental_Health_Score",
      ["Biến", "Tương quan Pearson"],
      data.correlations.map((row) => [row.feature, fmt(row.value, 4)]),
      ["left", "center"],
    ),
    p(
      `Số giờ dùng mạng xã hội tương quan âm mạnh nhất (${corr(data, "Avg_Daily_Usage_Hours")}), ` +
        `tiếp theo là số lần mở khóa (${corr(data, "Daily_Unlocks")}). ` +
        `Giờ ngủ (${corr(data, "Sleep_Hours_Per_Night")}) và giờ học ` +
        `(${corr(data, "Study_Hours")}) tương quan dương, vận động ở mức trung bình ` +
        `(${corr(data, "Physical_Activity_Hours")}), còn tuổi gần như không liên quan ` +
        `(${corr(data, "Age")}). Các biến số cũng tương quan mạnh với nhau — ví dụ ` +
        "số giờ dùng mạng xã hội và số lần mở khóa gần như đo cùng một thứ — nên " +
        "không thể đọc hệ số của từng biến như hiệu ứng riêng rẽ.",
    ),
    figure(
      data.figures.correlation,
      "Ma trận tương quan giữa các biến số và điểm số",
    ),
    p(
      "Ba biểu đồ phân tán dưới đây cho thấy quan hệ gần như tuyến tính giữa các " +
        "thói quen chính và điểm số, kèm dao động ngẫu nhiên đáng kể. Chính phần " +
        "dao động này là giới hạn của mọi mô hình: dù có biết chính xác thói quen, " +
        "điểm số vẫn không được xác định hoàn toàn.",
    ),
    figure(
      data.figures.scatter_numeric,
      "Quan hệ giữa giờ dùng mạng xã hội, giờ ngủ, giờ học và điểm số",
    ),
    p(
      "Khi gom theo nhóm giờ dùng, quan hệ không còn tuyến tính: điểm trung bình " +
        `khoảng ${fmt(firstBin.mean, 2)} ở nhóm dùng tối đa 2 giờ, giảm dần, rồi đi ` +
        `ngang ở khoảng ${fmt(lastBin.mean, 2)} từ 6 giờ trở lên. Hệ số tương quan ` +
        "Pearson không thể hiện được đoạn đi ngang này vì nó chỉ đo quan hệ tuyến " +
        "tính (Ch.2, slide 23). Đây cũng là lý do hồi quy tuyến tính kém hơn hẳn các " +
        "mô hình cây ở chương 5.",
    ),
    figure(
      data.figures.usage_threshold,
      "Điểm trung bình theo nhóm giờ dùng mạng xã hội",
    ),
    table(
      "Điểm trung bình theo Stress_Level",
      ["Stress_Level", "Điểm trung bình", "Số dòng"],
      data.stress_groups.map((row) => [
        row.level,
        fmt(row.mean, 4),
        dot(row.count),
      ]),
      ["left", "center", "center"],
    ),
    table(
      "Thống kê theo nền tảng được dùng nhiều nhất",
      ["Nền tảng", "Số dòng", "Điểm trung bình"],
      data.platform_groups.map((row) => [
        row.platform,
        dot(row.count),
        fmt(row.mean, 4),
      ]),
      ["left", "center", "center"],
    ),

    h2("3.5. Nhận xét từ dữ liệu"),
    bullets([
      "Hai biến mạnh nhất là số giờ dùng mạng xã hội và số giờ ngủ; số lần mở khóa " +
        "gần như là một cách đo khác của cường độ sử dụng.",
      "Stress_Level vừa liên quan đến điểm số vừa liên quan đến thói quen, nên " +
        "tương quan không cho biết biến nào gây ra biến nào.",
      `LINE có điểm trung bình cao nhất nhưng chỉ ${dot(data.platform_groups.find((row) => row.platform === "LINE").count)} dòng, ` +
        `và ${fmt(eda.platform_single_country.LINE.share * 100, 0)}% người dùng LINE đến từ ` +
        `${eda.platform_single_country.LINE.top_country}. Tương tự, KakaoTalk chỉ có ở ` +
        `${eda.platform_single_country.KakaoTalk.top_country}, WeChat chỉ ở ` +
        `${eda.platform_single_country.WeChat.top_country} và VKontakte chỉ ở ` +
        `${eda.platform_single_country.VKontakte.top_country}. Vì vậy dữ liệu không cho ` +
        "phép tách tác động của nền tảng khỏi tác động của quốc gia.",
      `Giờ dùng mạng xã hội trung bình tăng theo mức stress, từ ${fmt(stressUsage.Low, 2)} ` +
        `giờ (Low) lên ${fmt(stressUsage["Very High"], 2)} giờ (Very High), nên Stress_Level ` +
        "mang phần lớn thông tin đã có trong giờ dùng.",
      `Dữ liệu có ${eda.n_countries} quốc gia nhưng ${eda.countries_under_30_rows} quốc ` +
        `gia có dưới 30 dòng; Vietnam chỉ có ${eda.vietnam_rows} dòng. Đây là ví dụ của ` +
        "dữ liệu không đại diện (Ch.1): kết quả cho sinh viên Việt Nam có độ tin cậy thấp.",
      "Quan hệ mạnh và đều như vậy là dấu hiệu dữ liệu được tổng hợp hoặc mô phỏng " +
        "theo một công thức; vì thế R² cao ở chương 5 là kết quả trên dữ liệu này, " +
        "không phải bằng chứng cho quan hệ nhân quả ngoài đời thực.",
    ]),
    pagebreak(),
  ];
}
