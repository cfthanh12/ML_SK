import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { writeReport } from "./lib/render.mjs";
import { chaptersOneToThree } from "./content/ch01_03.mjs";
import { chaptersFourToFive } from "./content/ch04_05.mjs";
import { chaptersSixToNine } from "./content/ch06_09.mjs";
import { chaptersTenToThirteen } from "./content/ch10_13.mjs";

const ROOT = fileURLToPath(new URL("..", import.meta.url));

const META = {
  title: "BÁO CÁO ĐỒ ÁN CUỐI KÌ",
  subject: "[Tên môn học]",
  subtitle:
    "Student Social Media & Wellbeing: dự đoán điểm sức khỏe tinh thần của sinh viên từ thói quen sử dụng mạng xã hội",
  fields: [
    ["Trường/Khoa", "[Tên trường/khoa]"],
    ["Giảng viên hướng dẫn", "[Họ tên giảng viên]"],
    ["Sinh viên thực hiện", "[Họ và tên sinh viên]"],
    ["Mã số sinh viên", "[MSSV]"],
    ["Lớp", "[Lớp]"],
  ],
  place: "[Thành phố], tháng 11 năm 2026",
};

function parseArgs(argv) {
  const options = {
    out: path.join(ROOT, "Bao_cao_do_an_ML_v2.docx"),
    data: path.join(ROOT, "docs", "report", "report_data.json"),
    figures: path.join(ROOT, "docs", "report", "figures"),
    rebuild: true,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === "--out") options.out = path.resolve(argv[(index += 1)]);
    else if (flag === "--data") options.data = path.resolve(argv[(index += 1)]);
    else if (flag === "--figures") options.figures = path.resolve(argv[(index += 1)]);
    else if (flag === "--no-rebuild") options.rebuild = false;
    else throw new Error(`Tham số không hợp lệ: ${flag}`);
  }
  return options;
}

function pythonExecutable() {
  const venvPython = path.join(ROOT, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(venvPython)) return venvPython;
  return process.env.PYTHON || "python";
}

function rebuildData() {
  const result = spawnSync(
    pythonExecutable(),
    [path.join(ROOT, "scripts", "report_figures.py")],
    { cwd: ROOT, stdio: "inherit" },
  );
  if (result.status !== 0) {
    throw new Error("Sinh dữ liệu/hình thất bại. Chạy lại python scripts/report_figures.py để xem lỗi.");
  }
}

function warnIfStale(options) {
  const metricsPath = path.join(ROOT, "reports", "metrics.json");
  if (!fs.existsSync(options.data) || !fs.existsSync(metricsPath)) return;
  if (fs.statSync(metricsPath).mtimeMs > fs.statSync(options.data).mtimeMs) {
    console.warn(
      "STALE report_data.json — chạy lại python scripts/report_figures.py",
    );
  }
}

const options = parseArgs(process.argv.slice(2));
if (!fs.existsSync(options.data)) {
  if (!options.rebuild) {
    throw new Error(`Không thấy ${options.data} và đã tắt --rebuild.`);
  }
  rebuildData();
}
warnIfStale(options);

const data = JSON.parse(fs.readFileSync(options.data, "utf8"));

const REQUIRED_KEYS = [
  "metrics",
  "cv_results",
  "feature_importance",
  "stress_ablation",
  "selected_model",
  "training_rows",
  "model_version",
  "cleaning_report",
  "trained_at_utc",
];
const missingKeys = REQUIRED_KEYS.filter((key) => !(key in data));
if (missingKeys.length > 0) {
  throw new Error(
    `Thiếu key trong ${options.data}: ${missingKeys.join(", ")}. ` +
      "Chạy lại: python scripts/report_figures.py",
  );
}

const blocks = [
  ...chaptersOneToThree(data),
  ...chaptersFourToFive(data),
  ...chaptersSixToNine(data),
  ...chaptersTenToThirteen(data),
];

await writeReport({
  blocks,
  figuresDir: options.figures,
  outPath: options.out,
  meta: META,
});
console.log(`Wrote ${options.out}`);
