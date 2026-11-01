#!/usr/bin/env python3
"""Read-only ADualNet2606 provenance check. Never fits or scores a model."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def is_close(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8)
    except (TypeError, ValueError):
        return False


def inspect_rows(rows, contract):
    if not rows:
        return {"rows": 0, "publication_ready": False, "blocking_items": ["Empty feature table"]}
    missing = sorted(set(contract["columns_in_original_order"]) - set(rows[0]))
    report = {"rows": len(rows), "missing_columns": missing, "publication_ready": False,
              "blocking_items": list(contract["blocking_items"]),
              "author_confirmation": contract["author_confirmation"]}
    if missing:
        report["blocking_items"].insert(0, "Schema mismatch; no feature reinterpretation attempted")
        return report
    identities = {
        "step_ratio": ("average_step_time", "average_step_length", "walking_speed", 1),
        "stride_ratio": ("average_stride_time", "average_stride_length", "walking_speed", 1),
        "step_variability_ratio": ("step_time_variability", "step_length_variability", "walking_speed", 1),
        "stride_variability_ratio": ("stride_time_variability", "stride_length_variability", "walking_speed", 1),
        "legacy_cadence": ("cadence", None, "average_stride_time", 60),
    }
    counts = {}
    for name, (out, numerator, denominator, scale) in identities.items():
        matches = 0
        for r in rows:
            try:
                expected = (float(r[numerator]) if numerator else scale) / float(r[denominator])
                matches += is_close(r[out], expected)
            except (ZeroDivisionError, ValueError, TypeError):
                pass
        counts[name] = matches
    report["legacy_formula_match_counts"] = counts
    report["duplicate_identifier_count"] = len(rows) - len({r["ID"] for r in rows})
    report["speech_flags_without_identifiers"] = []
    for row_number, r in enumerate(rows, 2):
        flags = []
        try:
            if float(r["Correct_Num_DT"]) > float(r["Num_DT"]):
                flags.append("correct_exceeds_total")
            if float(r["Correct_Num_20s_DT"]) > float(r["Num_20s_DT"]):
                flags.append("correct20_exceeds_total20")
            if float(r["Timecount_DT"]) <= 20 and (r["Num_DT"] != r["Num_20s_DT"] or r["Correct_Num_DT"] != r["Correct_Num_20s_DT"]):
                flags.append("within20s_window_disagrees_with_full_task")
            if float(r["Timecount_DT_1stNum"]) > float(r["Timecount_DT"]):
                flags.append("first_response_outside_task")
        except (ValueError, TypeError):
            flags.append("speech_value_missing_or_nonnumeric")
        if flags:
            report["speech_flags_without_identifiers"].append({"excel_row": row_number, "flags": flags})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, required=True)
    parser.add_argument("--sheet", default="info")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--require-publication-ready", action="store_true")
    args = parser.parse_args()
    from openpyxl import load_workbook
    contract = json.loads((Path(__file__).resolve().parents[1] / "docs/adualnet_2606_feature_contract.json").read_text())
    before = hashlib.sha256(args.input_table.read_bytes()).hexdigest()
    wb = load_workbook(args.input_table, read_only=True, data_only=True)
    values = list(wb[args.sheet].values)
    wb.close()
    rows = [dict(zip(values[0], r)) for r in values[1:] if r[0] is not None]
    report = inspect_rows(rows, contract)
    after = hashlib.sha256(args.input_table.read_bytes()).hexdigest()
    assert before == after, "Read-only source hash changed"
    report["source_sha256"] = before
    report["recognized_recorded_version"] = before in contract["source_sha256"].values()
    report["source_unchanged"] = True
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n")
    print(text)
    if args.require_publication_ready and not report["publication_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
