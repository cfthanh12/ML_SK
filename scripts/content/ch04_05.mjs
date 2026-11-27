import {
  bullets,
  cvRow,
  figure,
  fmt,
  h1,
  h2,
  numbered,
  pagebreak,
  p,
  table,
} from "./helpers.mjs";

const pm = (mean, std) => `${fmt(mean, 4)} ± ${fmt(std, 4)}`;
const pct = (value, digits = 1) => `${fmt(value * 100, digits)}%`;
const change = (before, after, digits = 4) => {
  const delta = after - before;
  if (Math.abs(delta) < 5e-4) {
    return `gần như không đổi (${fmt(before, digits)} → ${fmt(after, digits)})`;
  }
  return `${delta > 0 ? "tăng" : "giảm"} từ ${fmt(before, digits)} ${delta > 0 ? "lên" : "xuống"} ${fmt(after, digits)}`;
};

export function chaptersFourToFive(data) {
  const rf = cvRow(data, "random_forest");
  const hgb = cvRow(data, "hist_gradient_boosting");
  const linear = cvRow(data, "linear_regression");
  const dummy = cvRow(data, "baseline_dummy");
  const ablation = data.stress_ablation;
  const gap = linear.rmse_mean - rf.rmse_mean;
  const ext = data.extended;
  const row = (name) => ext.holdout_comparison.find((item) => item.model === name);
  const tuned = row("Random Forest đã tinh chỉnh (v1.2.0)");
  const initial = row("Random Forest ban đầu (v1.1.0)");
  const noCountry = row("Random Forest bỏ Country");
  const curve = ext.learning_curve;
  const beh = ext.behavioural_tests;
  const bands = ext.error_analysis.mean_residual_by_actual_band;
  const lastValid = curve.valid_mae[curve.valid_mae.length - 1];
  const lastTrain = curve.train_mae[curve.train_mae.length - 1];

  return [
    h1("4. Phương pháp"),
    h2("4.1. Giao thức đánh giá"),
    p(
      "Toàn bộ dữ liệu được chia ngẫu nhiên thành hai phần: 80% dùng để huấn luyện " +
        "và 20% dùng làm holdout, với random seed 42 để lần chạy nào cũng cho cùng " +
        "một cách chia. Holdout được tách ra trước và chỉ mở đúng một lần để báo " +
        "cáo, không được dùng để chọn mô hình.",
    ),
    p(
      "Việc chọn mô hình diễn ra bên trong tập train bằng 5-fold cross-validation " +
        "có xáo trộn, cũng với seed 42. Mỗi fold dùng 4/5 tập train để fit và 1/5 " +
        "để tính điểm; kết quả cuối là trung bình và độ lệch chuẩn của 5 lần. Cách " +
        "làm này tránh hai lỗi thường gặp: chọn mô hình trên chính tập dùng để báo " +
        "cáo khiến kết quả báo cáo bị lạc quan, và chọn mô hình dựa trên một lần " +
        "chia duy nhất khiến quyết định phụ thuộc may mắn.",
    ),
    p(
      "Một nguồn rò rỉ khác là tiền xử lý. Bộ mã hóa One-Hot và mọi phép biến đổi " +
        "được đặt bên trong pipeline scikit-learn, nên chúng chỉ được fit trên phần " +
        "train của từng fold. Nhờ vậy tập đánh giá luôn sạch, kể cả khi có quốc gia " +
        "chỉ xuất hiện trong một fold.",
    ),

    h2("4.2. Tiền xử lý"),
    numbered([
      "Sáu biến số được giữ nguyên giá trị (numeric passthrough) vì mô hình dạng cây " +
        "không nhạy với thang đo, còn hồi quy tuyến tính được đánh giá công bằng " +
        "nhờ cùng một pipeline.",
      "Sáu biến phân loại được mã hóa One-Hot với handle_unknown=\"ignore\" và đầu " +
        "ra dày (sparse_output=False). Nhờ handle_unknown, một quốc gia chưa từng " +
        "xuất hiện khi huấn luyện vẫn được mã hóa thành vector toàn 0 thay vì gây lỗi.",
      "Giá trị Physical_Activity_Hours âm được chặn về 0 ngay trong bước chuẩn bị " +
        "dữ liệu; CSV gốc không bị sửa.",
      "Dòng trùng được giữ nguyên vì không ảnh hưởng đến tính đúng đắn của phép " +
        "đánh giá, và việc xóa chúng sẽ làm thay đổi phân phối so với dữ liệu thật.",
    ]),

    h2("4.3. Bốn ứng viên mô hình"),
    p(
      "Bốn ứng viên được chọn để bao phủ các mức độ linh hoạt khác nhau, từ baseline " +
        "không học gì đến mô hình phi tuyến mạnh:",
    ),
    table(
      "Cấu hình bốn ứng viên mô hình",
      ["Mô hình", "Cấu hình", "Vai trò"],
      [
        [
          "Dummy (baseline)",
          "DummyRegressor(strategy=\"mean\")",
          "Dự đoán bằng trung bình tập train; mốc so sánh thấp nhất",
        ],
        [
          "Linear Regression",
          "Hồi quy tuyến tính trên toàn bộ đặc trưng sau One-Hot",
          "Kiểm tra liệu quan hệ có gần tuyến tính hay không",
        ],
        [
          "Random Forest",
          "200 cây, max_features=0.5, min_samples_leaf=1, random_state=42 (tinh chỉnh ở mục 4.7)",
          "Mô hình phi tuyến mạnh, ổn định, không cần chuẩn hóa",
        ],
        [
          "HistGradientBoosting",
          "max_iter=300, learning_rate=0.1, random_state=42",
          "Boosting trên dữ liệu dạng bảng, huấn luyện nhanh",
        ],
      ],
      ["left", "left", "left"],
    ),
    p(
      "Phiên bản 1.1.0 dùng Random Forest 300 cây với min_samples_leaf=2. Từ phiên " +
        "bản 1.2.0, siêu tham số Random Forest được lấy từ bước tinh chỉnh ở mục 4.7 " +
        "và ghi trong config/params.yaml, nên mọi thay đổi đều nằm ở một chỗ.",
    ),

    h2("4.4. Chỉ số đánh giá"),
    p(
      "Ba chỉ số được dùng, tất cả tính trên thang điểm gốc 0–10:",
    ),
    bullets([
      `MAE (mean absolute error): sai số tuyệt đối trung bình, dễ đọc nhất — ví dụ ` +
        `MAE ${fmt(data.metrics.mae, 4)} nghĩa là mỗi dự đoán lệch khoảng ` +
        `${fmt(data.metrics.mae, 4)} điểm.`,
      "RMSE (root mean squared error): phạt nặng các sai số lớn, dùng làm tiêu chí " +
        "chọn mô hình vì nhạy với dự đoán lệch xa.",
      "R²: tỉ lệ phương sai của điểm số được mô hình giải thích; 0 là ngang baseline " +
        "trung bình, 1 là hoàn hảo, âm là tệ hơn baseline.",
      "R² hiệu chỉnh (adjusted R²) = 1 − (1 − R²)(n − 1)/(n − p − 1), với n là số mẫu " +
        "đánh giá và p là số cột đầu vào sau One-Hot. Chỉ số này phạt mô hình dùng " +
        "nhiều biến, theo bài thực hành S3_L24 (Multiple linear regression and Adjusted " +
        `R-squared). Với ${tuned.n_predictors_after_encoding} cột sau mã hóa, phần phạt ` +
        "này đáng kể và giúp đánh giá xem biến Country có thật sự đáng giữ hay không.",
    ]),

    h2("4.5. Đo mức ảnh hưởng của từng biến"),
    p(
      "Đồ án dùng permutation importance thay vì feature_importances_ của Random " +
        "Forest hay hệ số của hồi quy tuyến tính. Lý do: HistGradientBoosting không " +
        "có thuộc tính feature_importances_, còn hệ số hồi quy tuyến tính trên các " +
        "biến chưa chuẩn hóa (Daily_Unlocks có thang 62–273) không so sánh được với " +
        "nhau. Permutation importance đặt mọi biến về cùng một đơn vị: mức tăng RMSE " +
        "khi xáo trộn ngẫu nhiên giá trị của biến đó trên tập holdout.",
    ),
    p(
      "Cách đo: với mỗi biến, xáo trộn giá trị của cột đó, dự đoán lại toàn bộ " +
        "holdout, ghi nhận mức tăng RMSE, lặp 5 lần rồi lấy trung bình " +
        "(n_repeats=5, seed 42). Việc hoán vị thực hiện trên 12 cột nguồn của " +
        "pipeline, nên mọi cột One-Hot của cùng một biến bị xáo cùng lúc — kết quả " +
        "đọc được theo từng biến gốc chứ không bị phân tán thành hàng trăm cột.",
    ),

    h2("4.6. Ablation Stress_Level"),
    p(
      "Để trả lời câu hỏi biến căng thẳng có cần thiết hay không, đồ án chạy lại " +
        "toàn bộ giao thức — chia 80/20 seed 42, 5-fold CV chọn mô hình, đánh giá " +
        "holdout — nhưng bỏ hẳn Stress_Level khỏi danh sách đầu vào. Nếu chất lượng " +
        "giảm không đáng kể, biến này gần như dư thừa khi đã có số giờ dùng mạng xã " +
        "hội và giờ ngủ; nếu giảm mạnh, nó mang thông tin riêng.",
    ),

    h2("4.7. Tinh chỉnh siêu tham số"),
    p(
      "Sau khi CV cho thấy Random Forest là họ mô hình tốt nhất, nhóm tinh chỉnh " +
        `siêu tham số bằng RandomizedSearchCV với ${ext.tuning.iterations} tổ hợp ngẫu ` +
        `nhiên trên bốn tham số (${ext.tuning.search_space}), mỗi tổ hợp được chấm bằng ` +
        "3-fold CV chỉ trên tập train. Tập holdout không tham gia vào việc chọn " +
        "tham số; nó chỉ được dùng một lần sau đó để xác nhận (Machine Learning " +
        "Yearning, mục 11: không dùng tập kiểm tra để ra quyết định).",
    ),
    p(
      `Tổ hợp tốt nhất là n_estimators=${ext.tuning.best_params.n_estimators}, ` +
        `max_features=${ext.tuning.best_params.max_features}, ` +
        `min_samples_leaf=${ext.tuning.best_params.min_samples_leaf}, ` +
        `max_depth=${ext.tuning.best_params.max_depth ?? "None"}. Giới hạn max_features ` +
        "khiến mỗi cây chỉ xét một nửa số cột ở mỗi lần tách, nên các cây ít giống " +
        "nhau hơn và trung bình của chúng tổng quát tốt hơn.",
    ),

    h2("4.8. Đánh giá mở rộng"),
    p(
      "Ngoài các chỉ số tổng, module app/evaluate.py thực hiện bốn phép đánh giá " +
        "mà tài liệu môn học khuyến nghị:",
    ),
    bullets([
      "Learning curve: huấn luyện trên 10% đến 100% tập dữ liệu (5-fold) để xem sai " +
        "số đến từ độ chệch hay phương sai (Machine Learning Yearning, mục 28–32).",
      "Phân tích lỗi theo nhóm (slice-based evaluation) và theo mức điểm thực tế " +
        "(Chip Huyen, chương 6).",
      "Kiểm thử hành vi: directional (đổi một biến theo chiều đã biết), invariance " +
        "(đổi giới tính, kỳ vọng điểm gần như không đổi) và perturbation (thêm nhiễu " +
        "nhỏ) (Chip Huyen, chương 6).",
      "So sánh trên holdout có thêm thời gian huấn luyện, thời gian dự đoán và " +
        "R² hiệu chỉnh, theo gợi ý so sánh nhiều phương pháp bằng nhiều thước đo " +
        "trong tài liệu Project examples.",
    ]),

    h2("4.9. Tái lập"),
    p(
      `Mô hình cuối có phiên bản ${data.model_version}, được huấn luyện với random ` +
        "seed 42, lưu thành một artifact gồm pipeline đã fit và metadata " +
        `(thời điểm huấn luyện ${data.trained_at_utc}). Toàn bộ kết quả trong báo ` +
        "cáo được sinh lại từ CSV và artifact hiện có bằng một lệnh, và bước sinh " +
        "dữ liệu sẽ dừng nếu các chỉ số holdout tính lại không khớp metadata.",
    ),
    pagebreak(),

    h1("5. Kết quả thực nghiệm"),
    h2("5.1. So sánh mô hình bằng cross-validation"),
    p(
      `Kết quả 5-fold CV trên tập train cho thấy Random Forest đứng đầu với RMSE ` +
        `${fmt(rf.rmse_mean, 4)} ± ${fmt(rf.rmse_std, 4)} và R² ${fmt(rf.r2_mean, 4)}. ` +
        `HistGradientBoosting đứng thứ hai (RMSE ${fmt(hgb.rmse_mean, 4)}, kém ` +
        `${fmt(hgb.rmse_mean - rf.rmse_mean, 4)}). ` +
        `Hồi quy tuyến tính kém hơn rõ rệt (RMSE ${fmt(linear.rmse_mean, 4)}), còn ` +
        `baseline dự đoán bằng trung bình có RMSE ${fmt(dummy.rmse_mean, 4)} và R² ` +
        `xấp xỉ 0, đúng như kỳ vọng.`,
    ),
    table(
      "Kết quả 5-fold cross-validation",
      ["Mô hình", "CV MAE", "CV RMSE", "CV R²"],
      [
        ["Random Forest", pm(rf.mae_mean, rf.mae_std), pm(rf.rmse_mean, rf.rmse_std), pm(rf.r2_mean, rf.r2_std)],
        ["HistGradientBoosting", pm(hgb.mae_mean, hgb.mae_std), pm(hgb.rmse_mean, hgb.rmse_std), pm(hgb.r2_mean, hgb.r2_std)],
        ["Linear Regression", pm(linear.mae_mean, linear.mae_std), pm(linear.rmse_mean, linear.rmse_std), pm(linear.r2_mean, linear.r2_std)],
        ["Dummy (baseline)", pm(dummy.mae_mean, dummy.mae_std), pm(dummy.rmse_mean, dummy.rmse_std), pm(dummy.r2_mean, dummy.r2_std)],
      ],
      ["left", "center", "center", "center"],
    ),
    figure(
      data.figures.cv_rmse,
      "RMSE 5-fold cross-validation của bốn ứng viên",
    ),
    p(
      `Mô hình được chọn theo tiêu chí RMSE trung bình thấp nhất là ` +
        `${data.selected_model}. Độ lệch chuẩn giữa các fold của Random Forest chỉ ` +
        `${fmt(rf.rmse_std, 4)}, tức khoảng ${fmt((rf.rmse_std / rf.rmse_mean) * 100, 1)}% ` +
        "giá trị trung bình, cho thấy kết quả ổn định và không phụ thuộc vào một " +
        "cách chia may mắn.",
    ),

    h2("5.2. Hiệu năng trên tập holdout"),
    p(
      "Trên 20% dữ liệu chưa từng dùng để chọn mô hình, Random Forest đạt MAE " +
        `${fmt(data.metrics.mae, 4)}, RMSE ${fmt(data.metrics.rmse, 4)} và R² ` +
        `${fmt(data.metrics.r2, 4)}. So với CV, kết quả holdout tốt hơn một chút; ` +
        "chênh lệch này nằm trong dao động thông thường giữa các lần chia dữ liệu " +
        "và không phải dấu hiệu của rò rỉ.",
    ),
    table(
      "Chỉ số holdout của mô hình được chọn",
      ["Chỉ số", "Giá trị", "Ý nghĩa"],
      [
        ["MAE", fmt(data.metrics.mae, 4), "Sai số tuyệt đối trung bình trên thang 0–10"],
        ["RMSE", fmt(data.metrics.rmse, 4), "Sai số bình phương trung bình, phạt nặng sai số lớn"],
        ["R²", fmt(data.metrics.r2, 4), "Tỉ lệ phương sai điểm số được giải thích"],
      ],
      ["left", "center", "left"],
    ),

    h2("5.3. So sánh trên holdout, tinh chỉnh và R² hiệu chỉnh"),
    p(
      "Bảng dưới huấn luyện từng cấu hình trên cùng 80% train và chấm trên cùng 20% " +
        "holdout, kèm R² hiệu chỉnh và thời gian huấn luyện (cột Giây). Thời gian đo " +
        "trên máy chạy thực nghiệm nên chỉ dùng để so sánh tương đối.",
    ),
    table(
      "So sánh các cấu hình trên tập holdout",
      ["Cấu hình", "MAE", "RMSE", "R²", "R² hiệu chỉnh", "Giây"],
      ext.holdout_comparison.map((item) => [
        item.model,
        fmt(item.holdout_mae, 3),
        fmt(item.holdout_rmse, 3),
        fmt(item.holdout_r2, 3),
        fmt(item.holdout_adjusted_r2, 3),
        fmt(item.fit_seconds, 1),
      ]),
      ["left", "center", "center", "center", "center", "center"],
    ),
    bullets([
      `Tinh chỉnh làm MAE ${change(initial.holdout_mae, tuned.holdout_mae)} và R² ` +
        `${change(initial.holdout_r2, tuned.holdout_r2)}, đồng thời thời gian huấn ` +
        `luyện giảm từ ${fmt(initial.fit_seconds, 2)} xuống ${fmt(tuned.fit_seconds, 2)} giây.`,
      `Bỏ Country làm R² ${change(tuned.holdout_r2, noCountry.holdout_r2)}. Ngay cả ` +
        `khi xét R² hiệu chỉnh (mô hình bỏ Country chỉ còn ${noCountry.n_predictors_after_encoding} ` +
        `cột thay vì ${tuned.n_predictors_after_encoding}), mô hình có Country vẫn hơn ` +
        `(${fmt(tuned.holdout_adjusted_r2, 4)} so với ${fmt(noCountry.holdout_adjusted_r2, 4)}), ` +
        "nên biến này được giữ lại.",
      `R² trên tập train của Random Forest là ${fmt(tuned.train_r2, 3)}, cao hơn R² holdout ` +
        `khoảng ${fmt(tuned.train_r2 - tuned.holdout_r2, 3)}, lớn hơn nhiều so với Linear Regression; đây là ` +
        "dấu hiệu phương sai, được kiểm tra thêm bằng learning curve ở mục 5.8.",
    ]),

    h2("5.4. Phân tích sai số"),
    p(
      "Biểu đồ thực tế và dự đoán cho thấy các điểm bám sát đường chéo lý tưởng " +
        "trên toàn bộ miền điểm, với độ tản rộng hơn ở vùng điểm thấp và vùng điểm " +
        "cao — nơi dữ liệu ít mẫu hơn. Không có dấu hiệu mô hình chỉ đoán đúng một " +
        "khoảng hẹp rồi bỏ qua hai đầu.",
    ),
    figure(
      data.figures.actual_vs_predicted,
      "Điểm thực tế và điểm dự đoán trên tập holdout",
    ),
    p(
      `Sai số trung bình trên holdout là ${fmt(data.residual_mean, 4)}, gần như bằng 0, ` +
        `với độ lệch chuẩn ${fmt(data.residual_std, 4)}. Phân bố sai số đối xứng quanh 0 ` +
        "và không có đuôi dài bất thường, nghĩa là mô hình không thiên lệch hệ thống " +
        "theo hướng dự đoán cao hay thấp.",
    ),
    figure(
      data.figures.residuals,
      "Phân bố sai số (residual) trên tập holdout",
    ),
    p(
      `Trên holdout, ${pct(ext.error_analysis.share_within_0_5)} dự đoán lệch không quá ` +
        `0,5 điểm và ${pct(ext.error_analysis.share_within_1_0)} lệch không quá 1 điểm. ` +
        "Tuy sai số trung bình toàn bộ gần 0, khi chia theo mức điểm thực tế thì mô " +
        "hình kéo dự đoán về phía trung bình: dự đoán cao hơn thực tế ở nhóm điểm thấp " +
        "và thấp hơn thực tế ở nhóm điểm cao. Đây là đặc tính của Random Forest, vốn " +
        "lấy trung bình giá trị trong các lá.",
    ),
    table(
      "Sai số trung bình theo mức điểm thực tế (thực tế − dự đoán)",
      ["Mức điểm thực tế", "Số mẫu", "Sai số trung bình"],
      Object.entries(bands).map(([band, value]) => [band, String(value.n), fmt(value.mean_residual, 3)]),
      ["left", "center", "center"],
    ),
    table(
      "MAE theo nhóm (slice-based evaluation)",
      ["Nhóm", "Giá trị", "Số mẫu", "MAE"],
      ["Stress_Level", "Academic_Level", "Gender"].flatMap((column) =>
        Object.entries(ext.error_analysis.mae_by_slice[column]).map(([level, value]) => [
          column,
          level,
          String(value.n),
          fmt(value.mae, 3),
        ]),
      ),
      ["left", "left", "center", "center"],
    ),

    h2("5.5. Mức ảnh hưởng của từng biến"),
    table(
      "Permutation importance trên 12 biến",
      ["Biến", "Mức tăng RMSE khi hoán vị"],
      data.feature_importance.map((row) => [row.feature, fmt(row.importance, 4)]),
      ["left", "center"],
    ),
    figure(
      data.figures.importance,
      "Permutation importance trên tập holdout",
    ),
    p(
      `Hai biến đóng góp lớn nhất là số giờ dùng mạng xã hội ` +
        `(${fmt(data.feature_importance[0].importance, 4)}) và số giờ ngủ ` +
        `(${fmt(data.feature_importance[1].importance, 4)}). Đứng thứ ba là Country ` +
        `(${fmt(data.feature_importance[2].importance, 4)}) — cần thận trọng, vì ` +
        `${fmt((data.quality.country_other_rows / data.dataset.rows) * 100, 1)}% số dòng ` +
        "nằm trong nhóm Other, nên mức ảnh hưởng này có thể phản ánh cách dữ liệu " +
        "được gộp chứ không phải khác biệt thực sự giữa các quốc gia.",
    ),
    p(
      `Stress_Level đứng cuối bảng với ${fmt(data.feature_importance[data.feature_importance.length - 1].importance, 4)}. ` +
        "Kết quả này không có nghĩa biến vô nghĩa về mặt nội dung, mà là nó gần như " +
        "không thêm thông tin khi mô hình đã biết số giờ dùng mạng xã hội và giờ ngủ " +
        "— hai biến có tương quan mạnh với nó.",
    ),

    h2("5.6. Ablation Stress_Level"),
    p(
      "Chạy lại toàn bộ giao thức khi bỏ Stress_Level cho kết quả gần như không " +
        "đổi:",
    ),
    table(
      "So sánh có và không có Stress_Level",
      ["Biến thể", "MAE", "RMSE", "R²"],
      [
        [
          "Có Stress_Level",
          fmt(data.metrics.mae, 4),
          fmt(data.metrics.rmse, 4),
          fmt(data.metrics.r2, 4),
        ],
        [
          "Bỏ Stress_Level",
          fmt(ablation.metrics.mae, 4),
          fmt(ablation.metrics.rmse, 4),
          fmt(ablation.metrics.r2, 4),
        ],
      ],
      ["left", "center", "center", "center"],
    ),
    p(
      `Khi bỏ Stress_Level, R² ${change(data.metrics.r2, ablation.metrics.r2)} và RMSE ` +
        `${change(data.metrics.rmse, ablation.metrics.rmse)}. ` +
        "Mức chênh lệch này nhỏ hơn nhiều so với độ lệch chuẩn giữa các fold CV, nên " +
        "có thể kết luận Stress_Level gần như dư thừa trên dữ liệu này. Lý do: giờ " +
        "dùng mạng xã hội trung bình tăng đều theo mức stress (mục 3.5), nên thông tin " +
        "của stress đã nằm sẵn trong giờ dùng.",
    ),

    h2("5.7. Kiểm thử hành vi"),
    p(
      "Các kiểm thử hành vi trả lời câu hỏi mà R² không trả lời được: mô hình có học " +
        "đúng chiều không. Mỗi phép thử thay đổi một biến trên toàn bộ 1.000 mẫu " +
        "holdout và giữ nguyên các biến còn lại.",
    ),
    table(
      "Kết quả kiểm thử hành vi trên tập holdout",
      ["Phép thử", "Thao tác", "Kết quả"],
      [
        [
          "Directional",
          "Giờ dùng mạng xã hội 2 → 7 giờ",
          `Điểm giảm ở ${pct(beh.directional_usage_2h_to_7h.share_score_decreases)} mẫu, ` +
            `trung bình ${fmt(beh.directional_usage_2h_to_7h.mean_change, 2)} điểm`,
        ],
        [
          "Directional",
          "Giờ ngủ 5 → 8 giờ",
          `Điểm tăng ở ${pct(beh.directional_sleep_5h_to_8h.share_score_increases)} mẫu, ` +
            `trung bình +${fmt(beh.directional_sleep_5h_to_8h.mean_change, 2)} điểm`,
        ],
        [
          "Invariance",
          "Đổi giới tính",
          `Thay đổi trung bình ${fmt(beh.invariance_gender.mean_abs_change, 3)} điểm; ` +
            `${pct(beh.invariance_gender.share_change_over_0_5)} mẫu lệch quá 0,5`,
        ],
        [
          "Perturbation",
          "Nhiễu ±0,25 giờ vào 3 biến giờ",
          `Thay đổi trung bình ${fmt(beh.perturbation_noise_0_25h.mean_abs_change, 3)} điểm`,
        ],
        [
          "Ngoại suy",
          "Giờ dùng 8,8 giờ so với 20 giờ",
          `Chênh lệch lớn nhất ${fmt(beh.no_extrapolation_8_8h_vs_20h.max_abs_difference, 4)} ` +
            "điểm: mô hình cho cùng một kết quả",
        ],
      ],
      ["left", "left", "left"],
    ),
    figure(
      data.figures.what_if,
      "What-if: điểm dự đoán khi thay đổi giờ dùng mạng xã hội (40 hồ sơ holdout)",
    ),
    p(
      "Mô hình học đúng chiều và gần như không phụ thuộc giới tính. Hai giới hạn được " +
        "ghi nhận. Thứ nhất, ở bước nhỏ 0,5 giờ mô hình không đơn điệu hoàn toàn (bước " +
        `tăng lớn nhất ${fmt(beh.local_non_monotonicity_usage.max_step_increase, 2)} điểm) ` +
        "vì Random Forest tạo hàm bậc thang. Thứ hai, mô hình không ngoại suy: mọi giá " +
        "trị giờ dùng lớn hơn 8,8 giờ (giá trị lớn nhất trong dữ liệu) đều cho cùng " +
        "một điểm. Đây là lý do API trả cảnh báo khi đầu vào nằm ngoài khoảng huấn " +
        "luyện (mục 7.4).",
    ),
    p(
      "Các phát hiện trên không chỉ được ghi trong báo cáo mà còn được khóa thành " +
        "test tự động (mục 9.1, docs/test_plan.md): TC-B06 kiểm tra mô hình không ngoại " +
        "suy (8,8, 12 và 20 giờ cho cùng điểm), TC-B07 kiểm tra Stress_Level dư thừa " +
        "(R² lệch dưới 0,01 khi bỏ biến), TC-B08 theo dõi hiện tượng kéo về trung bình " +
        "ở hai đầu thang điểm, và TC-D07 khẳng định LINE, KakaoTalk, WeChat, VKontakte " +
        "mỗi nền tảng chỉ xuất hiện ở một quốc gia. Nếu lần huấn luyện sau làm thay đổi " +
        "các tính chất này, bộ test sẽ báo ngay.",
    ),

    h2("5.8. Learning curve"),
    figure(
      data.figures.learning_curve,
      "Learning curve: MAE huấn luyện và MAE xác thực theo số mẫu",
    ),
    p(
      `Ở 4.000 mẫu, MAE xác thực là ${fmt(lastValid, 3)} trong khi MAE huấn luyện chỉ ` +
        `${fmt(lastTrain, 3)}, và đường xác thực vẫn còn đi xuống. Theo Machine ` +
        "Learning Yearning (mục 30–31), đây là dấu hiệu phương sai là vấn đề chính, " +
        "không phải độ chệch: thu thập thêm dữ liệu nhiều khả năng còn cải thiện được " +
        "mô hình, còn tăng độ phức tạp thì ít tác dụng.",
    ),

    h2("5.9. Thảo luận"),
    bullets([
      `Mô hình phi tuyến tốt hơn tuyến tính rõ rệt: chênh RMSE CV trung bình là ` +
        `${fmt(gap, 4)} (${fmt(linear.rmse_mean, 4)} so với ${fmt(rf.rmse_mean, 4)}). ` +
        "Điều này phù hợp với việc các biến tương tác với nhau — ví dụ tác động của " +
        "giờ dùng mạng xã hội khác nhau tùy mức giờ ngủ.",
      `R² ${fmt(data.metrics.r2, 4)} là cao, nhưng phải đọc trong bối cảnh dữ liệu ` +
        "khảo sát có quan hệ rất mạnh và khá đều. Trên dữ liệu thực tế thu thập tự " +
        "do, con số này khó giữ được.",
      "Hai biến quyết định là cường độ sử dụng mạng xã hội và giấc ngủ; các biến " +
        "còn lại đóng góp ít, và Stress_Level gần như dư thừa.",
      "Country có mức ảnh hưởng đáng kể nhưng phần lớn dữ liệu bị gộp vào Other, " +
        "nên đây là điểm cần kiểm tra lại nếu có dữ liệu chi tiết hơn.",
      "Mô hình kéo dự đoán về trung bình ở hai đầu thang điểm. Nếu dùng để sàng lọc, " +
        "đây là lỗi đáng lo nhất vì đúng nhóm điểm thấp lại được ước lượng cao hơn thực tế.",
    ]),
    pagebreak(),
  ];
}
