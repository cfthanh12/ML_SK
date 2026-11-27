export const fmt = (value, digits = 4) => Number(value).toFixed(digits);

export const int = (value) =>
  new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 0 }).format(Number(value));

export const dot = (value) =>
  new Intl.NumberFormat("de-DE", { maximumFractionDigits: 0 }).format(Number(value));

export const h1 = (text) => ({ type: "h1", text });
export const h2 = (text) => ({ type: "h2", text });
export const h3 = (text) => ({ type: "h3", text });
export const p = (text) => ({ type: "p", text });
export const bullets = (items) => ({ type: "bullets", items });
export const numbered = (items) => ({ type: "numbered", items });
export const table = (caption, headers, rows, align) => ({
  type: "table",
  caption,
  headers,
  rows,
  align,
});
export const figure = (file, caption, widthCm) => ({
  type: "figure",
  file,
  caption,
  widthCm,
});
export const code = (text) => ({ type: "code", text });
export const pagebreak = () => ({ type: "pagebreak" });

export function describeRow(data, column) {
  return data.describe.find((row) => row.column === column);
}

export function cvRow(data, model) {
  return data.cv_results.find((row) => row.model === model);
}

export function importanceRow(data, feature) {
  return data.feature_importance.find((row) => row.feature === feature);
}
