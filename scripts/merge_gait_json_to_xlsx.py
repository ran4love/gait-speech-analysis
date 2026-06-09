#!/usr/bin/env python3
"""Merge gait feature JSON files into check_previous-format xlsx."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from common import SRC_DIR

sys.path.insert(0, str(SRC_DIR))
from compare_gait_metrics_to_xlsx import write_xlsx  # noqa: E402


WITHOUT_DS_METRICS = [
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
]

DS_METRICS = [
    "stance_times",
    "swing_times",
    "double_support_times",
]


def load_feature_map(path: Path, allow_missing: bool = False) -> Dict[str, Dict[str, object]]:
    if allow_missing and not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return {str(k): v for k, v in obj.items() if isinstance(v, dict)}


def clean_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and (value != value):
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert gait JSON features to check_previous-format xlsx.")
    parser.add_argument("--without-ds-json", required=True)
    parser.add_argument("--ds-json", required=True)
    parser.add_argument("--output-xlsx", required=True)
    parser.add_argument("--allow-missing-ds", action="store_true")
    args = parser.parse_args()

    without_ds_path = Path(args.without_ds_json).expanduser().resolve()
    ds_path = Path(args.ds_json).expanduser().resolve()
    output_xlsx = Path(args.output_xlsx).expanduser().resolve()

    without_ds = load_feature_map(without_ds_path)
    ds = load_feature_map(ds_path, allow_missing=args.allow_missing_ds)
    header = ["ID", *WITHOUT_DS_METRICS, *DS_METRICS, "label"]
    rows = [header]
    for case_id in sorted(set(without_ds) | set(ds)):
        wo = without_ds.get(case_id, {})
        dd = ds.get(case_id, {})
        row = [case_id]
        row.extend(clean_value(wo.get(metric, "")) for metric in WITHOUT_DS_METRICS)
        row.extend(clean_value(dd.get(metric, "")) for metric in DS_METRICS)
        row.append("")
        rows.append(row)

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(output_xlsx, [("gait_features", rows)])
    print(f"written: {output_xlsx} rows={len(rows) - 1}")


if __name__ == "__main__":
    main()
