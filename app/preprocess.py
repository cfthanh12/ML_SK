"""Export the data at each preprocessing stage to data/processed/ so it can be inspected.

The model does NOT read these files: training always starts from data/raw/ and the
preprocessing runs inside the scikit-learn Pipeline, fitted on the training split only
(no data leakage). This script reproduces exactly the same steps and saves the result
of each one:

    data/raw/*.csv                       1. raw data (never modified)
    data/processed/cleaned.csv           2. after cleaning (prepare_dataset) + split column
    data/processed/train.csv             3a. training split (80%)
    data/processed/holdout.csv           3b. holdout split (20%), only used for reporting
    data/processed/encoded_train_sample.csv
                                         4. what the model actually receives: numeric
                                            columns + one-hot columns (first 20 train rows)
    data/processed/preprocessing_report.json
                                         summary: rows, changes, encoded column names

Usage: python -m app.preprocess
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from app.features import FEATURE_COLUMNS, PROJECT_ROOT, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from app.train import DEFAULT_DATA_PATH, build_pipeline, load_dataset, prepare_dataset

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ENCODED_SAMPLE_ROWS = 20


def export_processed(
    data_path: Path = DEFAULT_DATA_PATH,
    out_dir: Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, Any]:
    """Write every preprocessing stage to out_dir and return a summary."""
    raw = load_dataset(data_path)
    features, target, cleaning_report = prepare_dataset(raw)
    cleaned = features.copy()
    cleaned[TARGET_COLUMN] = target

    # Same split as app/train.py (same test_size and random_state from config/params.yaml).
    train_index, holdout_index = train_test_split(
        cleaned.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cleaned["split"] = "train"
    cleaned.loc[holdout_index, "split"] = "holdout"

    # Step 4: fit the ColumnTransformer on the training split only, as in training.
    preprocessing = build_pipeline().named_steps["preprocessing"]
    preprocessing.fit(features.loc[train_index])
    encoded_columns = [
        name.split("__", 1)[1] for name in preprocessing.get_feature_names_out()
    ]
    sample_index = train_index[:ENCODED_SAMPLE_ROWS]
    encoded = pd.DataFrame(
        preprocessing.transform(features.loc[sample_index]),
        columns=encoded_columns,
        index=sample_index,
    )
    encoded.insert(0, "row_id", sample_index)
    encoded[TARGET_COLUMN] = target.loc[sample_index].to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_dir / "cleaned.csv", index_label="row_id")
    cleaned.loc[train_index].drop(columns="split").to_csv(
        out_dir / "train.csv", index_label="row_id")
    cleaned.loc[holdout_index].drop(columns="split").to_csv(
        out_dir / "holdout.csv", index_label="row_id")
    encoded.to_csv(out_dir / "encoded_train_sample.csv", index=False)

    numeric_count = sum(1 for name in preprocessing.get_feature_names_out()
                        if name.startswith("numeric__"))
    summary = {
        "source": str(data_path.relative_to(PROJECT_ROOT)) if data_path.is_relative_to(PROJECT_ROOT) else str(data_path),
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "cleaning": cleaning_report,
        "missing_values": int(raw[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum()),
        "train_rows": int(len(train_index)),
        "holdout_rows": int(len(holdout_index)),
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "encoded_columns_total": len(encoded_columns),
        "encoded_numeric_columns": numeric_count,
        "encoded_one_hot_columns": len(encoded_columns) - numeric_count,
        "encoded_column_names": encoded_columns,
        "note": "Mô hình không đọc các file này; chúng chỉ để xem kết quả từng bước tiền xử lý.",
    }
    (out_dir / "preprocessing_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args()
    summary = export_processed(args.data, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "encoded_column_names"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
