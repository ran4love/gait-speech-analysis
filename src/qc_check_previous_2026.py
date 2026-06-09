#!/usr/bin/env python3
"""Quality check for gait metrics xlsx using physics-informed rules."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import zipfile
import xml.etree.ElementTree as ET

from compare_gait_metrics_to_xlsx import write_xlsx


NUMERIC_COLS = [
    "average_step_length",
    "average_stride_length",
    "step_length_variability",
    "stride_length_variability",
    "average_step_time",
    "average_stride_time",
    "step_time_variability",
    "stride_time_variability",
    "average_step_assymetry",
    "average_stride_assymetry",
    "walking_speed",
    "cadence",
    "stance_times",
    "swing_times",
    "double_support_times",
]


XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def parse_first_sheet(path: Path) -> Tuple[List[str], List[Dict[str, object]]]:
    ns = {"a": XLSX_NS, "r": XLSX_REL_NS}
    with zipfile.ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst_root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared_strings.append("".join(texts))

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = wb.find("a:sheets/a:sheet", ns)
        if first_sheet is None:
            raise ValueError("No sheet in workbook.")
        rel_id = first_sheet.attrib.get(f"{{{XLSX_REL_NS}}}id")
        if not rel_id:
            raise ValueError("Cannot resolve first sheet relation.")

        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall(f"{{{PKG_REL_NS}}}Relationship"):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            raise ValueError("Cannot find first sheet target.")

        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        elif target.startswith("xl/"):
            sheet_path = target
        else:
            sheet_path = "xl/" + target
        sheet_root = ET.fromstring(zf.read(sheet_path))

        rows_raw: List[Dict[int, object]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", ns):
            row_map: Dict[int, object] = {}
            for c in row.findall("a:c", ns):
                ref = c.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                col_idx = 0
                for ch in col:
                    col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)
                ctype = c.attrib.get("t")
                value: object = ""
                v = c.find("a:v", ns)
                if ctype == "s" and v is not None and v.text is not None:
                    idx = int(v.text)
                    value = shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
                elif ctype == "inlineStr":
                    is_tag = c.find("a:is", ns)
                    if is_tag is not None:
                        value = "".join((t.text or "") for t in is_tag.findall(".//a:t", ns))
                elif v is not None and v.text is not None:
                    value = v.text
                row_map[col_idx] = value
            rows_raw.append(row_map)

    if not rows_raw:
        return [], []
    max_col = max((max(r.keys()) if r else 0) for r in rows_raw)
    header_row = rows_raw[0]
    headers = [str(header_row.get(i, "")).strip() for i in range(1, max_col + 1)]
    records: List[Dict[str, object]] = []
    for rr in rows_raw[1:]:
        rec = {}
        for i, h in enumerate(headers, start=1):
            if h:
                rec[h] = rr.get(i, "")
        records.append(rec)
    return headers, records


def to_float(v: object) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        x = float(s)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return a / b


def label_from_score(score: float) -> str:
    if score >= 80.0:
        return "GREEN"
    if score >= 60.0:
        return "YELLOW"
    return "RED"


def rule_three_zone(
    value: Optional[float],
    good_low: float,
    good_high: float,
    warn_low: float,
    warn_high: float,
) -> Tuple[float, str]:
    if value is None:
        return 0.0, "missing"
    if good_low <= value <= good_high:
        return 1.0, "good"
    if warn_low <= value <= warn_high:
        return 0.5, "warn"
    return 0.0, "fail"


def rule_upper_better(value: Optional[float], good_max: float, warn_max: float) -> Tuple[float, str]:
    if value is None:
        return 0.0, "missing"
    if value <= good_max:
        return 1.0, "good"
    if value <= warn_max:
        return 0.5, "warn"
    return 0.0, "fail"


def evaluate_case(rec: Dict[str, Optional[float]]) -> Dict[str, object]:
    step_len = rec.get("average_step_length")
    stride_len = rec.get("average_stride_length")
    step_t = rec.get("average_step_time")
    stride_t = rec.get("average_stride_time")
    cadence = rec.get("cadence")
    stance = rec.get("stance_times")
    swing = rec.get("swing_times")
    ds = rec.get("double_support_times")
    step_asym = rec.get("average_step_assymetry")
    stride_asym = rec.get("average_stride_assymetry")
    speed = rec.get("walking_speed")

    length_ratio = ratio(stride_len, step_len)
    time_ratio = ratio(stride_t, step_t)
    cadence_expected = None if stride_t is None or stride_t <= 0 else 120.0 / stride_t
    cadence_abs_err = None
    if cadence is not None and cadence_expected is not None:
        cadence_abs_err = abs(cadence - cadence_expected)

    phase_total = None
    if stance is not None and swing is not None:
        phase_total = stance + swing
    stance_ratio = ratio(stance, phase_total)
    ds_ratio_total = ratio(ds, phase_total)
    ds_le_stance_ok = None
    if ds is not None and stance is not None:
        ds_le_stance_ok = ds <= stance + 1e-9

    # Each rule returns 1.0 good / 0.5 warn / 0 fail.
    rule_scores: Dict[str, Tuple[float, str]] = {
        "length_ratio": rule_three_zone(length_ratio, 1.6, 2.4, 1.4, 2.8),
        "time_ratio": rule_three_zone(time_ratio, 1.8, 2.2, 1.6, 2.4),
        "cadence_formula": rule_upper_better(cadence_abs_err, 3.0, 8.0),
        "stance_ratio": rule_three_zone(stance_ratio, 0.45, 0.75, 0.35, 0.85),
        "double_support_ratio": rule_three_zone(ds_ratio_total, 0.05, 0.40, 0.01, 0.50),
        "step_asymmetry": rule_upper_better(step_asym, 40.0, 60.0),
        "stride_asymmetry": rule_upper_better(stride_asym, 60.0, 90.0),
        "walking_speed": rule_three_zone(speed, 0.40, 2.00, 0.20, 2.50),
    }
    if ds_le_stance_ok is None:
        rule_scores["ds_le_stance"] = (0.0, "missing")
    elif ds_le_stance_ok:
        rule_scores["ds_le_stance"] = (1.0, "good")
    else:
        rule_scores["ds_le_stance"] = (0.0, "fail")

    weights = {
        "length_ratio": 15.0,
        "time_ratio": 10.0,
        "cadence_formula": 10.0,
        "stance_ratio": 20.0,
        "double_support_ratio": 10.0,
        "ds_le_stance": 10.0,
        "step_asymmetry": 10.0,
        "stride_asymmetry": 10.0,
        "walking_speed": 5.0,
    }
    total_w = sum(weights.values())
    weighted = sum(rule_scores[k][0] * weights[k] for k in weights)
    score = 100.0 * clamp01(weighted / total_w)

    failed = [k for k, (_, s) in rule_scores.items() if s in {"fail", "missing"}]
    warned = [k for k, (_, s) in rule_scores.items() if s == "warn"]

    out: Dict[str, object] = {
        "length_ratio_stride_over_step": length_ratio,
        "time_ratio_stride_over_step": time_ratio,
        "cadence_expected_120_over_stride_t": cadence_expected,
        "cadence_abs_error": cadence_abs_err,
        "stance_ratio": stance_ratio,
        "double_support_ratio_total": ds_ratio_total,
        "double_support_le_stance": ds_le_stance_ok,
        "quality_score": round(score, 4),
        "quality_level": label_from_score(score),
        "failed_rules": ",".join(failed),
        "warn_rules": ",".join(warned),
    }
    for k in sorted(rule_scores):
        out[f"rule_{k}_status"] = rule_scores[k][1]
    return out


def rows_from_dicts(dict_rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> List[List[object]]:
    out: List[List[object]] = [list(columns)]
    for d in dict_rows:
        out.append([d.get(c) for c in columns])
    return out


def build_qc_report(input_xlsx: Path, output_xlsx: Path) -> None:
    headers, rows = parse_first_sheet(input_xlsx)
    if "ID" not in headers:
        raise ValueError("Input sheet missing ID column.")

    records: List[Dict[str, object]] = []
    for row in rows:
        rec: Dict[str, object] = {"ID": str(row.get("ID", "")).strip()}
        for c in NUMERIC_COLS:
            rec[c] = to_float(row.get(c))
        records.append(rec)

    qc_rows: List[Dict[str, object]] = []
    for rec in records:
        qc = evaluate_case(rec)
        merged = {**rec, **qc}
        qc_rows.append(merged)

    qc_rows.sort(key=lambda d: (str(d["quality_level"]), float(d["quality_score"])))

    level_counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    for r in qc_rows:
        level_counts[str(r["quality_level"])] += 1

    n = len(qc_rows) if qc_rows else 1
    avg_score = sum(float(r["quality_score"]) for r in qc_rows) / n
    rule_cols = [k for k in qc_rows[0].keys() if k.startswith("rule_") and k.endswith("_status")] if qc_rows else []
    viol_counts: Dict[str, int] = {}
    for rc in rule_cols:
        viol_counts[rc] = sum(1 for r in qc_rows if r.get(rc) in {"fail", "missing"})

    summary = [
        ["item", "value"],
        ["input_xlsx", str(input_xlsx)],
        ["total_cases", len(qc_rows)],
        ["quality_score_mean", round(avg_score, 4)],
        ["green_count", level_counts["GREEN"]],
        ["yellow_count", level_counts["YELLOW"]],
        ["red_count", level_counts["RED"]],
        ["green_ratio_pct", round(100.0 * level_counts["GREEN"] / n, 4)],
        ["yellow_ratio_pct", round(100.0 * level_counts["YELLOW"] / n, 4)],
        ["red_ratio_pct", round(100.0 * level_counts["RED"] / n, 4)],
    ]
    for rc in sorted(viol_counts):
        summary.append([f"violations_{rc}", viol_counts[rc]])

    rules_sheet = [
        ["rule_name", "good", "warn", "fail_or_missing"],
        ["length_ratio (stride/step)", "1.6-2.4", "1.4-2.8", "outside warn"],
        ["time_ratio (stride/step)", "1.8-2.2", "1.6-2.4", "outside warn"],
        ["cadence formula abs err", "<=3", "<=8", ">8 or missing"],
        ["stance_ratio stance/(stance+swing)", "0.45-0.75", "0.35-0.85", "outside warn"],
        ["double_support_ratio_total ds/(stance+swing)", "0.05-0.40", "0.01-0.50", "outside warn"],
        ["ds <= stance", "true", "-", "false or missing"],
        ["step asymmetry (%)", "<=40", "<=60", ">60 or missing"],
        ["stride asymmetry (%)", "<=60", "<=90", ">90 or missing"],
        ["walking speed (m/s)", "0.40-2.00", "0.20-2.50", "outside warn"],
    ]

    case_cols = [
        "ID",
        "quality_score",
        "quality_level",
        "failed_rules",
        "warn_rules",
        "length_ratio_stride_over_step",
        "time_ratio_stride_over_step",
        "cadence_abs_error",
        "stance_ratio",
        "double_support_ratio_total",
        "double_support_le_stance",
        "average_step_assymetry",
        "average_stride_assymetry",
        "walking_speed",
        "rule_length_ratio_status",
        "rule_time_ratio_status",
        "rule_cadence_formula_status",
        "rule_stance_ratio_status",
        "rule_double_support_ratio_status",
        "rule_ds_le_stance_status",
        "rule_step_asymmetry_status",
        "rule_stride_asymmetry_status",
        "rule_walking_speed_status",
    ]
    # Fix names to exactly match generated keys.
    case_cols = [c.replace("rule_double_support_ratio_status", "rule_double_support_ratio_status") for c in case_cols]

    # Generated keys include rule_double_support_ratio and rule_walking_speed.
    # Keep explicit columns and fallback with dict.get(None).
    top_bad = sorted(qc_rows, key=lambda d: float(d["quality_score"]))[:100]
    violations = [
        {k: r.get(k) for k in case_cols}
        for r in top_bad
    ]

    write_xlsx(
        output_xlsx,
        [
            ("Summary", summary),
            ("Rules", rules_sheet),
            ("CaseQC", rows_from_dicts(qc_rows, case_cols)),
            ("Top100Worst", rows_from_dicts(violations, case_cols)),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QC report xlsx for gait metrics.")
    parser.add_argument(
        "--input-xlsx",
        default="check_previous_format_new_2026.xlsx",
        help="Input gait metrics xlsx.",
    )
    parser.add_argument(
        "--output-xlsx",
        default="check_previous_format_new_2026_qc.xlsx",
        help="Output QC xlsx path.",
    )
    args = parser.parse_args()

    build_qc_report(Path(args.input_xlsx), Path(args.output_xlsx))
    print(f"written: {Path(args.output_xlsx).resolve()}")


if __name__ == "__main__":
    main()
