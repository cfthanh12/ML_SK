import json
from pathlib import Path

import pytest

from scripts.verify_report import inspect

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Bao_cao_do_an_ML_v2.docx"
METRICS = ROOT / "reports" / "metrics.json"
EXPECTED_FIGURES = 12
EXPECTED_TABLES = 21

pytestmark = pytest.mark.skipif(
    not DOCX.is_file(), reason="Chưa build Bao_cao_do_an_ML_v2.docx"
)


@pytest.fixture(scope="module")
def report():
    return inspect(DOCX)


def test_report_has_expected_figures_and_tables(report):
    assert report["images"] == EXPECTED_FIGURES
    assert report["tables"] == EXPECTED_TABLES


def test_every_figure_and_table_is_captioned(report):
    text = report["text"]
    for number in range(1, EXPECTED_FIGURES + 1):
        assert f"Hình {number}." in text
    for number in range(1, EXPECTED_TABLES + 1):
        assert f"Bảng {number}." in text


def test_report_contains_current_metrics(report):
    stored = json.loads(METRICS.read_text(encoding="utf-8"))
    for key in ("mae", "rmse", "r2"):
        assert f"{stored['metrics'][key]:.4f}" in report["text"]
    assert stored["model_version"] in report["text"]


def test_report_is_vietnamese_with_east_asian_font(report):
    assert "Ước lượng" in report["text"]
    assert report["has_east_asia"]
