"""FAQ import and export files (docs/faq.md, Import format): CSV (UTF-8, BOM accepted) and
XLSX (openpyxl, from the `parsers` extra). Multiple values in one cell are joined with `##`."""

from __future__ import annotations

import csv
import io
import re

from agenticwf.kb.runtime.errors import Invalid

HEADERS = {
    "standard_question": ("问题", "standard_question", "question"),
    "similar_questions": ("相似问题", "similar_questions"),
    "negative_questions": ("反例问题", "negative_questions"),
    "answers": ("机器人回答", "answers", "answer"),
    "tag_name": ("标签", "分类", "tag_name", "tag"),
    "answer_all": ("是否全部回复", "answer_all"),
    "disabled": ("是否停用", "disabled", "is_disabled"),
    "not_recommended": ("是否禁止被推荐", "not_recommended"),
}
EXPORT_HEADERS = ("standard_question", "similar_questions", "negative_questions", "answers", "tag_name",
                  "answer_all", "disabled", "not_recommended")
_HINT = re.compile(r"[(（].*?[)）]")
_TRUE = ("true", "1", "yes", "y", "是")


def _field(header: str) -> str | None:
    name = _HINT.sub("", str(header or "")).strip().lower()
    for field, names in HEADERS.items():
        if name in names:
            return field
    return None


def _rows(name: str, data: bytes) -> list[list[str]]:
    lower = name.lower()
    if lower.endswith(".csv"):
        text = data.decode("utf-8-sig", errors="replace")
        return [row for row in csv.reader(io.StringIO(text))]
    if lower.endswith(".xlsx"):
        try:
            import openpyxl
        except ImportError:
            raise Invalid("XLSX import needs openpyxl (the parsers extra)") from None
        book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sheet = book.worksheets[0]
        return [["" if v is None else str(v) for v in row] for row in sheet.iter_rows(values_only=True)]
    raise Invalid("import a .csv or .xlsx file")


def parse(name: str, data: bytes) -> list[dict]:
    """Rows → raw entries (the shape of a batch upsert)."""
    rows = [r for r in _rows(name, data) if any(str(c).strip() for c in r)]
    if not rows:
        raise Invalid("the file has no rows")
    fields = [_field(h) for h in rows[0]]
    if "standard_question" not in fields or "answers" not in fields:
        raise Invalid("the file needs a question column and an answer column")
    entries = []
    for row in rows[1:]:
        values = {f: (row[i] if i < len(row) else "").strip() for i, f in enumerate(fields) if f}
        entry = {"standard_question": values.get("standard_question", ""),
                 "similar_questions": values.get("similar_questions", ""),
                 "negative_questions": values.get("negative_questions", ""),
                 "answers": values.get("answers", "")}
        if values.get("tag_name"):
            entry["tag_name"] = values["tag_name"]
        if values.get("answer_all"):
            entry["answer_strategy"] = "all" if values["answer_all"].lower() in _TRUE else "random"
        if "disabled" in values:
            entry["is_enabled"] = values["disabled"].lower() not in _TRUE
        if "not_recommended" in values:
            entry["is_recommended"] = values["not_recommended"].lower() not in _TRUE
        entries.append(entry)
    return entries


def export(entries: list[dict], tag_names: dict[str, str]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(EXPORT_HEADERS)
    for e in entries:
        writer.writerow([e["standard_question"], "##".join(e["similar_questions"]), "##".join(e["negative_questions"]),
                         "##".join(e["answers"]), tag_names.get(e.get("tag_id") or "", ""),
                         "TRUE" if e["answer_strategy"] == "all" else "FALSE",
                         "FALSE" if e["is_enabled"] else "TRUE", "FALSE" if e["is_recommended"] else "TRUE"])
    return ("﻿" + out.getvalue()).encode("utf-8")
