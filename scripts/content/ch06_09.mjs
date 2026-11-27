import {
  bullets,
  code,
  fmt,
  h1,
  h2,
  numbered,
  pagebreak,
  p,
  table,
} from "./helpers.mjs";

export function chaptersSixToNine(data) {
  const testCount = data.test_count;
  const example = data.predictions.example;
  const requestJson = JSON.stringify(example.payload, null, 2);
  const responseJson = JSON.stringify(
    {
      mental_health_score: example.score,
      score_scale: "0-10",
      model_version: data.model_version,
      warnings: example.warnings ?? [],
    },
    null,
    2,
  );

  return [
    h1("6. Kiến trúc hệ thống"),
    h2("6.1. Tổng quan"),
    p(
      "Hệ thống gồm bốn khối nối tiếp: dữ liệu CSV, bước huấn luyện trong " +
        "app/train.py, artifact mô hình kèm metadata, và hai lớp phục vụ là FastAPI " +
        "và Streamlit. Notebook không phải một khối riêng trong luồng chạy thật: nó " +
        "gọi lại đúng các hàm của app/train.py để trình bày quá trình, nhờ vậy kết " +
        "quả trong notebook và mô hình phục vụ API luôn là một.",
    ),
    p(
      "Sơ đồ ở Hình 1 (chương 1) mô tả kiến trúc tổng thể. Mũi tên liền là luồng dữ " +
        "liệu chính khi huấn luyện và phục vụ; nhánh từ notebook đi vào bước huấn " +
        "luyện thể hiện việc notebook dùng chung pipeline chứ không tự định nghĩa " +
        "mô hình.",
    ),
    p(
      "Kiến trúc này giữ ba nguyên tắc. Một là một nguồn sự thật cho mô hình: chỉ " +
        "app/train.py định nghĩa cách huấn luyện. Hai là artifact tự mô tả: tệp " +
        ".joblib chứa cả pipeline đã fit lẫn metadata về phiên bản, danh sách biến " +
        "và chỉ số đánh giá. Ba là lớp phục vụ không huấn luyện: API chỉ nạp artifact " +
        "và dự đoán, nên thời gian khởi động ngắn và hành vi ổn định.",
    ),

    h2("6.2. Thành phần và trách nhiệm"),
    bullets([
      "app/features.py: tên cột, danh sách biến số và biến phân loại, phiên bản mô " +
        "hình và random seed; là nơi duy nhất định nghĩa thứ tự 12 biến.",
      "app/schemas.py: schema Pydantic cho request và response, kèm ràng buộc miền " +
        "giá trị và hàm chuyển payload sang đúng tên cột của CSV.",
      "app/train.py: đọc dữ liệu, chuẩn bị dữ liệu, xây pipeline, so sánh bốn ứng " +
        "viên bằng cross-validation, đánh giá holdout, đo permutation importance, " +
        "chạy ablation và lưu artifact.",
      "app/evaluate.py: đánh giá mở rộng (so sánh trên holdout, tinh chỉnh, R² hiệu " +
        "chỉnh, learning curve, phân tích lỗi theo nhóm, kiểm thử hành vi) và sinh hình " +
        "cho báo cáo.",
      "config/params.yaml: siêu tham số, tỉ lệ chia dữ liệu, ngưỡng cảnh báo và ngưỡng " +
        "độ trễ; mọi thay đổi cấu hình nằm ở một chỗ.",
      "app/main.py: ứng dụng FastAPI, nạp artifact khi khởi động, cung cấp bốn " +
        "endpoint, so đầu vào với vùng dữ liệu huấn luyện để cảnh báo, và ghi log.",
      "streamlit_app.py: ứng dụng demo, dựng form nhập liệu, gọi API và hiển thị " +
        "kết quả kèm thông tin mô hình.",
      "tests/: bộ kiểm thử tự động cho schema, huấn luyện, API và tính nhất quán " +
        "của artifact.",
    ]),

    h2("6.3. Định dạng artifact"),
    p(
      "Artifact là một từ điển gồm hai khóa: model là pipeline scikit-learn đã fit " +
        "(gồm bước tiền xử lý và bộ hồi quy), metadata là thông tin mô tả. Lưu cả " +
        "pipeline thay vì chỉ lưu bộ hồi quy giúp bảo đảm lúc dự đoán dùng đúng " +
        "phép biến đổi đã dùng lúc huấn luyện — đây là lỗi phổ biến khi triển khai " +
        "mô hình có One-Hot Encoding.",
    ),
    bullets([
      "model_version: phiên bản mô hình, hiện là " + data.model_version + ".",
      "selected_model và cv_results: tên mô hình được chọn và bảng so sánh 5-fold CV.",
      "metrics: MAE, RMSE, R² trên holdout của mô hình 80%.",
      "feature_columns: danh sách 12 biến theo đúng thứ tự pipeline chờ đợi.",
      "feature_importance và feature_importance_method: mức ảnh hưởng theo permutation.",
      "stress_ablation: kết quả chạy lại giao thức khi bỏ Stress_Level.",
      "training_profile: khoảng giá trị của từng biến số, danh sách giá trị của từng " +
        "biến phân loại và số dòng của mỗi quốc gia trong dữ liệu huấn luyện; API dùng " +
        "phần này để cảnh báo đầu vào nằm ngoài vùng mô hình đã thấy.",
      "training_rows và trained_at_utc: số dòng dùng huấn luyện và thời điểm huấn luyện.",
    ]),

    h2("6.4. Hai luồng chạy"),
    numbered([
      "Huấn luyện: python -m app.train đọc CSV, chia train/holdout, chạy CV bốn ứng " +
        "viên, chọn mô hình tốt nhất, đo importance, chạy ablation, rồi ghi " +
        "models/mental_health_model.joblib và reports/metrics.json.",
      "Suy luận: uvicorn nạp artifact một lần khi khởi động; mỗi yêu cầu POST " +
        "/predict được chuyển thành một dòng dữ liệu đúng thứ tự cột, đưa qua " +
        "pipeline, clamp về [0, 10], so với training_profile để sinh cảnh báo và trả JSON.",
    ]),

    h2("6.5. Cấu trúc thư mục theo template của môn"),
    p(
      "Từ phiên bản 1.2.0, thư mục dự án được sắp xếp theo tài liệu ML Project " +
        "Template Structure của môn: dữ liệu gốc nằm trong data/raw và không bị sửa; " +
        "mô hình đã huấn luyện nằm trong models; chỉ số, kết quả thực nghiệm và hình " +
        "nằm trong reports; cấu hình nằm trong config; log của API nằm trong logs; " +
        "notebook chia theo ba vai trò khám phá, xây dựng mô hình và suy luận. Tệp " +
        ".github/workflows/ci.yml chạy toàn bộ kiểm thử trên GitHub Actions mỗi lần " +
        "push, đúng với bảng công cụ (GitHub Actions, Pytest) trong tài liệu template. " +
        "Cây thư mục đầy đủ ở Phụ lục A.",
    ),
    pagebreak(),

    h1("7. REST API"),
    h2("7.1. Danh sách endpoint"),
    p(
      "API được xây bằng FastAPI, chạy bằng uvicorn. Bốn endpoint phục vụ các mục " +
        "đích khác nhau: kiểm tra trạng thái, xem thông tin mô hình, dự đoán một hồ " +
        "sơ (online prediction) và dự đoán nhiều hồ sơ một lượt (batch prediction, " +
        "Chip Huyen chương 7).",
    ),
    table(
      "Danh sách endpoint của REST API",
      ["Endpoint", "Phương thức", "Chức năng", "Mã lỗi chính"],
      [
        [
          "/health",
          "GET",
          "Báo trạng thái dịch vụ và việc artifact đã nạp hay chưa",
          "Không có",
        ],
        [
          "/model-info",
          "GET",
          "Trả phiên bản, mô hình được chọn, metrics, CV, importance, danh sách biến",
          "503 nếu artifact chưa nạp",
        ],
        [
          "/predict",
          "POST",
          "Nhận hồ sơ 12 trường, trả điểm dự đoán đã làm tròn 2 chữ số",
          "422 khi dữ liệu sai; 500 nếu điểm không hữu hạn",
        ],
        [
          "/predict/batch",
          "POST",
          "Nhận 1–200 hồ sơ, trả danh sách điểm kèm cảnh báo; giao diện dùng cho biểu đồ what-if",
          "422 khi danh sách rỗng, quá 200 hồ sơ hoặc có hồ sơ sai",
        ],
        [
          "/docs",
          "GET",
          "Giao diện Swagger để thử API trực tiếp",
          "Không có",
        ],
      ],
      ["left", "center", "left", "left"],
    ),

    h2("7.2. Schema đầu vào và đầu ra"),
    p(
      "Request là một object JSON gồm đúng 12 trường, không chấp nhận trường lạ. " +
        "Các trường phân loại bị giới hạn bằng Literal nên giá trị ngoài danh sách " +
        "bị từ chối với mã 422; các trường số có ràng buộc ge/le (tuổi 18–24, giờ " +
        "0–24, số lần mở khóa 0–1000). Riêng country là chuỗi tự do 1–100 ký tự, " +
        "được strip và không được để trống — nhờ vậy quốc gia chưa có trong dữ liệu " +
        "vẫn dự đoán được nhờ handle_unknown=\"ignore\" của bộ mã hóa.",
    ),
    code(requestJson),
    p(
      "Response gồm bốn trường: điểm dự đoán, thang điểm, phiên bản mô hình và danh " +
        "sách cảnh báo. Điểm được làm tròn hai chữ số thập phân và luôn nằm trong " +
        "[0, 10]. Danh sách cảnh báo rỗng khi hồ sơ nằm trong vùng dữ liệu huấn luyện.",
    ),
    code(responseJson),
    p(
      "Ví dụ trên là kết quả thật khi gọi artifact hiện có với payload mẫu, không " +
        "phải số minh họa. Giá trị cụ thể phụ thuộc phiên bản mô hình đang nạp, nên " +
        "response luôn kèm model_version để phía gọi đối chiếu.",
    ),

    h2("7.3. Endpoint thông tin mô hình"),
    p(
      "/model-info trả về metadata huấn luyện: phiên bản, tên mô hình được chọn, " +
        "metrics holdout, bảng CV của bốn ứng viên, danh sách 12 biến, permutation " +
        "importance, số dòng huấn luyện và thời điểm huấn luyện. Endpoint này tồn " +
        "tại vì giao diện cần hiển thị thông tin mô hình mà không phải nhúng cứng " +
        "số liệu; khi artifact được huấn luyện lại, giao diện tự cập nhật theo.",
    ),

    h2("7.4. Xử lý lỗi và vận hành"),
    bullets([
      "Dữ liệu sai kiểu hoặc ngoài miền giá trị trả 422 kèm mô tả lỗi của Pydantic.",
      "Artifact thiếu hoặc hỏng làm ứng dụng không khởi động được, vì việc nạp " +
        "artifact diễn ra trong lifespan; đây là lựa chọn có chủ ý để lỗi cấu hình " +
        "lộ ra ngay thay vì trả kết quả sai. Các nhánh 503 trong code chỉ là phòng vệ.",
      "Mọi dự đoán được clamp về [0, 10] để không bao giờ trả điểm ngoài thang đo " +
        "của dữ liệu.",
      "Logging ghi phiên bản, tên mô hình và điểm dự đoán, không ghi toàn bộ payload, " +
        "để tránh lưu thông tin cá nhân không cần thiết.",
      "Đường dẫn artifact có thể đổi bằng biến môi trường MODEL_ARTIFACT_PATH, phục " +
        "vụ kiểm thử và triển khai nhiều phiên bản.",
      "Cảnh báo ngoài phân phối: nếu một biến số nằm ngoài khoảng đã thấy khi huấn " +
        "luyện (ví dụ giờ dùng lớn hơn 8,8), hoặc quốc gia chưa có trong dữ liệu hay " +
        "có dưới 30 dòng (ví dụ Vietnam), API vẫn trả điểm nhưng kèm cảnh báo. Mục 5.7 " +
        "cho thấy Random Forest không ngoại suy, nên đây là cách xử lý edge case và " +
        "dịch chuyển phân phối đầu vào (covariate shift) theo Chip Huyen chương 8.",
      "Mỗi phản hồi có header X-Process-Time-ms, và mỗi yêu cầu được ghi vào " +
        "logs/app.log cùng mã trạng thái và độ trễ, phục vụ giám sát.",
      "Endpoint dự đoán được viết dạng hàm đồng bộ vì scikit-learn tính toán bằng " +
        "CPU; FastAPI chạy hàm đồng bộ trong thread pool nên không chặn vòng lặp sự " +
        "kiện (Building Data Science Applications with FastAPI, chương 12). Mô hình " +
        "được nạp đúng một lần trong lifespan thay vì ở mỗi yêu cầu.",
      "Phép đo thỏa mãn về hiệu năng: trung vị độ trễ của /predict phải dưới 100 ms " +
        "(Machine Learning Yearning, mục 9); ngưỡng này nằm trong config/params.yaml " +
        "và được kiểm tra tự động.",
    ]),
    pagebreak(),

    h1("8. Ứng dụng Streamlit"),
    h2("8.1. Mục tiêu"),
    p(
      "Ứng dụng demo minh họa cách một người dùng cuối tương tác với mô hình: nhập " +
        "hồ sơ, nhận điểm ước lượng và thấy được bối cảnh của con số đó. Tiêu đề " +
        "giao diện là Ước lượng điểm sức khỏe tinh thần, kèm dòng ghi chú rằng kết " +
        "quả không phải chẩn đoán hay tư vấn y tế. Giao diện không tự tính toán gì " +
        "về mô hình — mọi dự đoán đều đi qua REST API, đúng với kiến trúc ở chương 6.",
    ),

    h2("8.2. Thành phần giao diện"),
    bullets([
      "Sidebar hiển thị phiên bản mô hình, tên mô hình được chọn, ba chỉ số holdout " +
        "và biểu đồ 10 biến ảnh hưởng nhất, lấy từ /model-info.",
      "Sidebar cũng hiển thị trạng thái /health để người dùng biết API đã sẵn sàng " +
        "hay chưa.",
      "Form nhập liệu chia hai cột: thông tin nhân khẩu và học tập bên trái, thói " +
        "quen số và giấc ngủ bên phải, tổng cộng 12 trường.",
      "Ô chọn kịch bản mẫu điền sẵn một bộ giá trị để thử nhanh.",
      "Ô quốc gia là danh sách chọn gồm 111 quốc gia lấy từ /model-info, nên người " +
        "dùng không nhập được quốc gia mà mô hình chưa từng thấy.",
      "Khu vực kết quả hiển thị điểm, thanh tiến trình theo thang 0–10, các cảnh báo " +
        "màu vàng do API trả về và một ghi chú rằng đây không phải chẩn đoán y tế.",
      "Hai biểu đồ what-if: giữ nguyên hồ sơ, chỉ thay đổi giờ dùng mạng xã hội " +
        "(1–8,5 giờ) hoặc giờ ngủ (4–9,5 giờ), gửi một yêu cầu /predict/batch và vẽ " +
        "đường điểm dự đoán. Giao diện ghi rõ đây là phản ứng của mô hình, không phải " +
        "quan hệ nhân quả.",
    ]),

    h2("8.3. Kịch bản mẫu"),
    p(
      "Ba kịch bản được thiết kế để cho kết quả khác nhau rõ rệt, giúp người xem " +
        "thấy mô hình phản ứng với thói quen như thế nào. Điểm dưới đây là kết quả " +
        "thật khi gọi artifact hiện tại.",
    ),
    table(
      "Ba kịch bản mẫu và điểm dự đoán tương ứng",
      ["Kịch bản", "Đặc điểm chính", "Điểm dự đoán"],
      [
        [
          "Cân bằng",
          "Dùng mạng xã hội 3 giờ/ngày, ngủ 7.5 giờ, vận động 1.5 giờ, căng thẳng trung bình",
          fmt(data.predictions.presets[0].score, 2),
        ],
        [
          "Dùng nhiều mạng xã hội",
          "Dùng 7.5 giờ/ngày, mở khóa 240 lần, ngủ 5.5 giờ, căng thẳng rất cao",
          fmt(data.predictions.presets[1].score, 2),
        ],
        [
          "Học tập cao",
          "Dùng 2 giờ/ngày, học 7 giờ, ngủ 8 giờ, căng thẳng thấp",
          fmt(data.predictions.presets[2].score, 2),
        ],
      ],
      ["left", "left", "center"],
    ),

    h2("8.4. Diễn giải kết quả"),
    p(
      `Điểm được gắn nhãn tham khảo theo phân vị 25 và 75 của điểm trong dữ liệu ` +
        `huấn luyện: dưới ${fmt(data.thresholds.low, 1)} là thấp, trên ` +
        `${fmt(data.thresholds.high, 1)} là cao, ở giữa là trung bình. Cách gắn nhãn ` +
        "này chỉ giúp người xem định vị con số so với phân bố khảo sát, không phải " +
        "ngưỡng lâm sàng.",
    ),
    p(
      "Giao diện cũng nêu rõ giới hạn: kết quả là ước lượng thống kê từ dữ liệu " +
        "khảo sát và mô hình học máy, không thể dùng để kết luận tình trạng sức " +
        "khỏe của một cá nhân.",
    ),

    h2("8.5. Cấu hình và xử lý lỗi"),
    p(
      "Địa chỉ API lấy từ biến môi trường API_URL, mặc định http://127.0.0.1:8000. " +
        "Nếu không lấy được /model-info, sidebar hiện cảnh báo thay vì làm hỏng " +
        "trang. Nếu người dùng bấm dự đoán khi API chưa chạy, giao diện bắt lỗi kết " +
        "nối và hướng dẫn khởi động FastAPI; nếu API trả 422, thông điệp lỗi của " +
        "Pydantic được hiển thị để người dùng sửa dữ liệu.",
    ),
    pagebreak(),

    h1("9. Kiểm thử và triển khai"),
    h2("9.1. Bộ kiểm thử"),
    p(
      "Bộ test chạy bằng pytest và không phụ thuộc artifact commit sẵn: một fixture " +
        "tạo dataset tổng hợp nhỏ, huấn luyện một mô hình tạm rồi trỏ " +
        "MODEL_ARTIFACT_PATH vào đó. Nhờ vậy test chạy được trên máy sạch và không " +
        "che giấu lỗi do artifact cũ.",
    ),
    table(
      "Nhóm kiểm thử và phạm vi",
      ["Tệp test", "Phạm vi"],
      [
        [
          "tests/test_api.py",
          "Trạng thái /health, dự đoán nằm trong [0, 10], tuổi ngoài 18–24 trả 422, /model-info có mô hình và metrics",
        ],
        [
          "tests/test_schemas.py",
          "Biên tuổi 18/24 hợp lệ và 17/25 lỗi, trường lạ bị từ chối, country rỗng bị từ chối, giới hạn giờ",
        ],
        [
          "tests/test_training.py",
          "Chặn giá trị vận động âm, thiếu cột báo lỗi, đủ bốn ứng viên, bảng CV đúng số dòng, chọn mô hình hợp lệ",
        ],
        [
          "tests/test_artifact_guard.py",
          "Artifact trong repo phải khớp phiên bản và danh sách biến với app/features.py",
        ],
        [
          "tests/test_report_figures.py",
          "Dữ liệu báo cáo khớp metrics.json, holdout tái tạo được, đủ hình cho báo cáo",
        ],
        [
          "tests/test_api_extensions.py",
          "Cảnh báo ngoài khoảng, quốc gia hiếm và chưa gặp; batch khớp gọi đơn lẻ; giới hạn batch; 10 ca giá trị sai trả 422; độ trễ trung vị dưới 100 ms; header thời gian",
        ],
        [
          "tests/test_behaviour.py",
          "Trên dữ liệu thật: directional (giờ dùng, giờ ngủ), invariance (giới tính), perturbation, R² holdout ≥ 0,85 và khoảng cách train–holdout dưới 0,10; không ngoại suy (giờ dùng 8,8 / 12 / 20 giờ cho cùng điểm); ablation Stress_Level (R² lệch dưới 0,01); kéo về trung bình ở hai đầu thang điểm (độ lệch mỗi dải dưới 0,6)",
        ],
        [
          "tests/test_data.py",
          "Trên dữ liệu thật: đủ cột và 5.000 dòng, không thiếu giá trị, điểm trong 0–10, chặn 10 giá trị âm mà không sửa CSV, báo 2 dòng trùng, train/holdout không giao nhau; LINE, KakaoTalk, WeChat, VKontakte mỗi nền tảng chỉ gắn với 1 quốc gia (kiểm tra biến gây nhiễu)",
        ],
        [
          "tests/test_evaluate.py",
          "Công thức R² hiệu chỉnh và các chỉ số của module đánh giá",
        ],
      ],
      ["left", "left"],
    ),
    p(
      `Toàn bộ ${testCount} test đạt ở lần chạy cuối. Bộ test được chạy tự động bằng ` +
        "GitHub Actions mỗi lần push (.github/workflows/ci.yml). Kế hoạch kiểm thử chi " +
        "tiết với mã từng ca nằm trong docs/test_plan.md.",
    ),

    h2("9.2. Đóng gói bằng Docker"),
    p(
      "Dockerfile dùng base python:3.14-slim, chỉ cài dependency runtime từ " +
        "requirements-api.txt, chạy bằng user không phải root và có HEALTHCHECK gọi " +
        "/health. Image copy artifact đã huấn luyện thay vì huấn luyện lúc build, " +
        "nên build nhanh và kết quả mô hình không phụ thuộc môi trường build. Lệnh " +
        "build chưa được xác nhận trên máy tác giả vì Docker daemon đang tắt; đây " +
        "là phần cần chạy lại ở máy có Docker Desktop.",
    ),

    h2("9.3. Hướng dẫn chạy từ đầu"),
    p(
      "Phần này hướng dẫn chạy lại toàn bộ đồ án trên Windows từ một thư mục mới " +
        "giải nén. Các lệnh dùng PowerShell, chạy tại thư mục gốc của project. Nếu " +
        "gói nộp kèm artifact mental_health_model.joblib thì có thể bỏ qua bước 2.",
    ),
    p("Bước 1 — Tạo môi trường và cài thư viện:"),
    code(
      "python -m venv .venv\n" +
        ".\\.venv\\Scripts\\Activate.ps1\n" +
        "python -m pip install -r requirements.txt",
    ),
    p(
      "Kết quả mong đợi: pip cài xong không lỗi. requirements.txt đã pin đúng phiên " +
        "bản dùng để tạo artifact (Python 3.14.8, scikit-learn 1.9.1, pandas 3.1.0); " +
        "nếu máy dùng phiên bản Python khác, có thể cần nới pin để pip tìm được " +
        "wheel tương thích.",
    ),
    p("Bước 2 — Huấn luyện mô hình và ghi artifact:"),
    code("python -m app.train"),
    p(
      "Kết quả mong đợi: lệnh chạy khoảng 40 giây, in ra một object JSON chứa " +
        "model_version, selected_model, metrics, cv_results, feature_importance và " +
        "stress_ablation, đồng thời tạo models/mental_health_model.joblib và " +
        "reports/metrics.json. Nếu artifact đã có trong gói, bước này không bắt buộc.",
    ),
    p("Bước 2b (tùy chọn) — Chạy đánh giá mở rộng và sinh hình:"),
    code("python -m app.evaluate"),
    p(
      "Kết quả mong đợi: lệnh chạy khoảng 8–10 phút (phần lớn là tinh chỉnh) và ghi " +
        "reports/experiments.json cùng ba hình trong reports/figures. Thêm " +
        "--skip-tuning để bỏ bước tinh chỉnh khi chỉ cần cập nhật hình.",
    ),
    p("Bước 3 — Chạy REST API:"),
    code("uvicorn app.main:app --reload"),
    p(
      "Kết quả mong đợi: mở http://127.0.0.1:8000/health thấy " +
        "{\"status\":\"ok\",\"model_loaded\":true,...}; /model-info trả metadata mô " +
        "hình; /docs mở giao diện Swagger để thử POST /predict. Nếu muốn dùng " +
        "artifact ở đường dẫn khác, đặt biến môi trường MODEL_ARTIFACT_PATH trước " +
        "khi khởi động.",
    ),
    p("Bước 4 — Chạy giao diện demo (giữ API đang chạy, mở PowerShell thứ hai):"),
    code(
      ".\\.venv\\Scripts\\Activate.ps1\n" +
        "streamlit run streamlit_app.py",
    ),
    p(
      "Kết quả mong đợi: Streamlit mở http://localhost:8501, sidebar hiển thị phiên " +
        "bản, mô hình được chọn, MAE/RMSE/R² và trạng thái API. Chọn một kịch bản " +
        "mẫu rồi bấm Ước lượng để nhận điểm dự đoán. Nếu API chạy ở địa chỉ khác, " +
        "đặt biến môi trường API_URL trước khi chạy Streamlit.",
    ),
    p("Bước 5 — Chạy toàn bộ kiểm thử:"),
    code("python -m pytest -q"),
    p(
      `Kết quả mong đợi: toàn bộ test pass (${testCount} passed tại thời điểm đóng gói). Bộ ` +
        "test tự tạo một dataset tổng hợp nhỏ và một artifact tạm, nên chạy được trên " +
        "máy sạch mà không cần artifact trong repo.",
    ),
    p("Bước 6 (tùy chọn) — Đóng gói và chạy bằng Docker:"),
    code(
      "docker build -t student-wellbeing .\n" +
        "docker run -p 8000:8000 student-wellbeing",
    ),
    p(
      "Image dùng python:3.14-slim, chạy bằng user không phải root và có " +
        "HEALTHCHECK gọi /health. Lệnh build chưa được xác nhận trên máy tác giả vì " +
        "Docker daemon đang tắt, nên cần chạy lại ở máy có Docker Desktop.",
    ),

    h2("9.4. Kịch bản kiểm thử thủ công giao diện"),
    p(
      "Các kịch bản dưới đây dùng khi demo hoặc kiểm tra lại giao diện bằng tay, với " +
        "API và Streamlit cùng đang chạy.",
    ),
    table(
      "Kịch bản kiểm thử thủ công giao diện",
      ["Mã", "Bước thực hiện", "Kết quả mong đợi"],
      [
        ["TC-U01", "Mở trang khi API đang chạy", "Sidebar hiện phiên bản, mô hình và chỉ số; ô quốc gia có 111 lựa chọn"],
        ["TC-U02", "Chọn kịch bản Cân bằng, bấm Ước lượng", "Hiện điểm, thanh tiến trình và hai biểu đồ what-if; có cảnh báo vì Vietnam chỉ có 5 dòng"],
        ["TC-U03", "Đổi quốc gia sang India, giờ dùng 12", "Hiện cảnh báo giờ dùng nằm ngoài khoảng 1–8,8 giờ"],
        ["TC-U04", "Mở tab what-if Giờ dùng mạng xã hội", "Đường điểm đi xuống khi giờ dùng tăng và đi ngang từ khoảng 6 giờ"],
        ["TC-U05", "Tắt API rồi bấm Ước lượng", "Thông báo lỗi kết nối, trang không bị sập"],
        ["TC-U06", "Mở http://127.0.0.1:8000/docs", "Swagger liệt kê /health, /model-info, /predict, /predict/batch"],
      ],
      ["left", "left", "left"],
    ),

    h2("9.5. Lỗi thường gặp khi chạy"),
    bullets([
      "API báo không tìm thấy artifact: chạy python -m app.train trước, hoặc trỏ " +
        "MODEL_ARTIFACT_PATH tới file .joblib có sẵn.",
      "Cổng 8000 đã bị chiếm: chạy uvicorn với --port 8001 và đặt API_URL tương ứng " +
        "cho giao diện.",
      "Giao diện báo không kết nối được API: kiểm tra API còn chạy không và API_URL " +
        "có đúng địa chỉ không; giao diện không tự dự đoán khi thiếu API.",
      "API trả 422: dữ liệu ngoài miền cho phép (tuổi 18–24, giờ 0–24, số lần mở " +
        "khóa 0–1000) hoặc sai giá trị phân loại; thông điệp lỗi chỉ rõ trường sai.",
      "Cài đặt lỗi trên Python khác 3.14: nới các dòng pin trong requirements.txt " +
        "hoặc dùng đúng phiên bản Python ghi trong tệp.",
    ]),
    p(
      `Kết quả tham chiếu để đối chiếu: mô hình ${data.selected_model} đạt MAE ` +
        `${fmt(data.metrics.mae, 4)}, RMSE ${fmt(data.metrics.rmse, 4)}, R² ` +
        `${fmt(data.metrics.r2, 4)} trên holdout; ba kịch bản mẫu cho điểm ` +
        `${fmt(data.predictions.presets[0].score, 2)}, ` +
        `${fmt(data.predictions.presets[1].score, 2)} và ` +
        `${fmt(data.predictions.presets[2].score, 2)}.`,
    ),
    pagebreak(),
  ];
}
