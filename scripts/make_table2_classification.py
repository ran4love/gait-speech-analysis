#!/usr/bin/env python3
"""Create a manuscript-ready Table 2 from classification performance output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


METRICS = [
    ("auroc", "AUROC"),
    ("acc", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1 score"),
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def format_metric(row: pd.Series, metric: str, digits: int) -> str:
    mean = row[f"mean_{metric}"]
    std = row[f"std_{metric}"]
    low = row[f"ci95_low_{metric}"]
    high = row[f"ci95_high_{metric}"]
    return f"{mean:.{digits}f} +/- {std:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def build_table2(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    required = {"model"}
    for metric, _ in METRICS:
        required.update(
            {
                f"mean_{metric}",
                f"std_{metric}",
                f"ci95_low_{metric}",
                f"ci95_high_{metric}",
            }
        )
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    rows = []
    include_feature_set = "feature_set" in df.columns and df["feature_set"].nunique(dropna=False) > 1
    for _, row in df.iterrows():
        out = {}
        if include_feature_set:
            out["Feature set"] = row["feature_set"]
        out["Model"] = row["model"]
        for metric, label in METRICS:
            out[f"{label}, mean +/- SD (95% CI)"] = format_metric(row, metric, digits)
        rows.append(out)
    return pd.DataFrame(rows)


def write_markdown(table: pd.DataFrame, out_path: Path, caption: str, source_note: str) -> None:
    lines = []
    if caption:
        lines.extend([caption, ""])
    headers = [str(col) for col in table.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table.iterrows():
        values = [str(row[col]) for col in table.columns]
        lines.append("| " + " | ".join(values) + " |")
    if source_note:
        lines.extend(["", f"Note: {source_note}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def style_workbook(path: Path, table_rows: int, table_cols: int, source_note: str) -> None:
    wb = load_workbook(path)
    ws = wb["Table 2"]
    raw_ws = wb["Source metrics"]
    notes_ws = wb["Notes"]

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    title_fill = PatternFill("solid", fgColor="1F4E79")
    title_font = Font(color="FFFFFF", bold=True, size=12)
    header_font = Font(bold=True)
    thin = Side(style="thin", color="B7B7B7")
    border = Border(top=thin, bottom=thin, left=thin, right=thin)

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=table_cols)
    ws["A1"] = "Table 2. Classification performance of publication models"
    ws["A1"].font = title_font
    ws["A1"].fill = title_fill
    ws["A1"].alignment = Alignment(horizontal="center")
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=table_cols)
    ws["A2"] = source_note
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")

    for cell in ws[4]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border

    for row in ws.iter_rows(min_row=5, max_row=4 + table_rows, max_col=table_cols):
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(vertical="center", wrap_text=True)

    widths = {
        1: 18,
        2: 31,
        3: 31,
        4: 31,
        5: 31,
        6: 31,
    }
    for idx in range(1, table_cols + 1):
        ws.column_dimensions[get_column_letter(idx)].width = widths.get(idx, 24)
    ws.row_dimensions[2].height = 42
    ws.freeze_panes = "A5"

    for worksheet in (raw_ws, notes_ws):
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for idx in range(1, worksheet.max_column + 1):
            worksheet.column_dimensions[get_column_letter(idx)].width = min(42, max(12, idx * 2 + 10))

    wb.save(path)


def write_outputs(table2: pd.DataFrame, raw: pd.DataFrame, out_prefix: Path, source_note: str) -> Iterable[Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_prefix.with_suffix(".xlsx")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    table2.to_csv(csv_path, index=False)
    write_markdown(
        table2,
        md_path,
        "Table 2. Classification performance of publication models.",
        source_note,
    )
    notes = pd.DataFrame(
        {
            "item": ["Source", "Format", "Threshold", "Feature selection"],
            "description": [
                source_note,
                "Values are mean +/- SD with bootstrap 95% confidence intervals in parentheses.",
                "Accuracy, precision, recall and F1 score use a probability threshold of 0.5.",
                "When nested outputs are used, feature selection is performed inside each outer training fold only.",
            ],
        }
    )
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table2.to_excel(writer, sheet_name="Table 2", startrow=3, index=False)
        raw.to_excel(writer, sheet_name="Source metrics", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    style_workbook(xlsx_path, len(table2), len(table2.columns), source_note)
    return xlsx_path, csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manuscript Table 2 from model_performance_table output.")
    parser.add_argument("--input", required=True, help="Path to model_performance_table.csv/xlsx.")
    parser.add_argument("--out-prefix", required=True, help="Output path without extension.")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument(
        "--source-note",
        default=(
            "Repeated stratified five-fold cross-validation with ten repeats; "
            "bootstrap 95% confidence intervals from held-out predictions."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_prefix = Path(args.out_prefix).expanduser().resolve()
    raw = read_table(input_path)
    table2 = build_table2(raw, args.digits)
    outputs = write_outputs(table2, raw, out_prefix, args.source_note)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
