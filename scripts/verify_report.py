"""Check a generated report .docx for required text, images, tables, and fonts."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import unescape

TAG_PATTERN = re.compile(r"<[^>]+>")
BLOCK_BREAK = re.compile(r"</w:p>|</w:tc>|</w:tr>")
TABLE_PATTERN = re.compile(r"<w:tbl[ >]")


def extract_text(xml: str) -> str:
    separated = BLOCK_BREAK.sub("\n", xml)
    return unescape(TAG_PATTERN.sub("", separated))


def inspect(docx_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(docx_path) as archive:
        names = archive.namelist()
        document_xml = archive.read("word/document.xml").decode("utf-8")
        try:
            styles_xml = archive.read("word/styles.xml").decode("utf-8")
        except KeyError:
            styles_xml = ""
    return {
        "docx": str(docx_path),
        "chars": len(extract_text(document_xml)),
        "images": sum(1 for name in names if name.startswith("word/media/")),
        "tables": len(TABLE_PATTERN.findall(document_xml)),
        "has_east_asia": "w:eastAsia" in document_xml or "w:eastAsia" in styles_xml,
        "text": extract_text(document_xml),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--must-contain", nargs="*", default=[])
    parser.add_argument("--min-images", type=int, default=0)
    parser.add_argument("--min-tables", type=int, default=0)
    parser.add_argument("--expect-eastasia", action="store_true")
    args = parser.parse_args()

    if not args.docx.is_file():
        print(f"Lỗi: không tìm thấy {args.docx}", file=sys.stderr)
        return 1

    report = inspect(args.docx)
    text = str(report["text"])
    missing = [needle for needle in args.must_contain if needle not in text]
    failures: list[str] = []
    if missing:
        failures.append("MISSING: " + " | ".join(missing))
    if int(report["images"]) < args.min_images:
        failures.append(f"images {report['images']} < min {args.min_images}")
    if int(report["tables"]) < args.min_tables:
        failures.append(f"tables {report['tables']} < min {args.min_tables}")
    if args.expect_eastasia and not report["has_east_asia"]:
        failures.append("Không tìm thấy khai báo font eastAsia (tiếng Việt có thể bị lỗi dấu)")

    summary = {
        "docx": report["docx"],
        "chars": report["chars"],
        "images": report["images"],
        "tables": report["tables"],
        "has_east_asia": report["has_east_asia"],
        "missing": missing,
    }
    print(json.dumps(summary, ensure_ascii=False))
    for failure in failures:
        print(failure, file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
