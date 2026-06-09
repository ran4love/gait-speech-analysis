#!/usr/bin/env python3
"""Generate corrected xlsx by aligning previous metrics with corrected rules output."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from compare_gait_metrics_to_xlsx import parse_xlsx_first_sheet, write_xlsx


METRIC_COLS = [
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


def to_float(v) -> Optional[float]:
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.mean(values)


def median_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.median(values)


def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return a / b


def cadence_from_stride(stride_time: Optional[float]) -> Optional[float]:
    if stride_time is None or stride_time <= 0:
        return None
    return 120.0 / stride_time


def build_corrected_xlsx(previous_xlsx: Path, corrected_json: Path, output_xlsx: Path) -> None:
    _, prev_rows = parse_xlsx_first_sheet(previous_xlsx)
    corrected = json.loads(corrected_json.read_text())

    prev_map: Dict[str, Dict[str, Optional[float]]] = {}
    for row in prev_rows:
        case_id = str(row.get("ID", "")).strip()
        if not case_id:
            continue
        prev_map[case_id] = {k: to_float(row.get(k)) for k in METRIC_COLS}

    common_ids = sorted(set(prev_map.keys()) & set(corrected.keys()))

    details_header = [
        "ID",
        "prev_average_step_length",
        "corr_average_step_length",
        "prev_average_stride_length",
        "corr_average_stride_length",
        "prev_stride_step_ratio",
        "corr_stride_step_ratio",
        "prev_average_step_time",
        "corr_average_step_time",
        "prev_average_stride_time",
        "corr_average_stride_time",
        "prev_time_ratio_stride_over_step",
        "corr_time_ratio_stride_over_step",
        "prev_cadence",
        "corr_cadence",
        "prev_cadence_physical_120_over_stride",
        "corr_cadence_physical_120_over_stride",
        "prev_cadence_abs_error_physical",
        "corr_cadence_abs_error_physical",
        "prev_walking_speed",
        "corr_walking_speed",
        "corr_step_count_estimate",
        "corr_stride_count_estimate",
    ]

    detail_rows = [details_header]

    prev_ratio_err = []
    corr_ratio_err = []
    prev_cad_err = []
    corr_cad_err = []
    prev_ratio_bad = 0
    corr_ratio_bad = 0

    for cid in common_ids:
        p = prev_map[cid]
        c = corrected[cid]

        p_step = to_float(p.get("average_step_length"))
        p_stride = to_float(p.get("average_stride_length"))
        c_step = to_float(c.get("average_step_length"))
        c_stride = to_float(c.get("average_stride_length"))

        p_step_t = to_float(p.get("average_step_time"))
        p_stride_t = to_float(p.get("average_stride_time"))
        c_step_t = to_float(c.get("average_step_time"))
        c_stride_t = to_float(c.get("average_stride_time"))

        p_ratio = ratio(p_stride, p_step)
        c_ratio = ratio(c_stride, c_step)
        p_time_ratio = ratio(p_stride_t, p_step_t)
        c_time_ratio = ratio(c_stride_t, c_step_t)

        p_cad = to_float(p.get("cadence"))
        c_cad = to_float(c.get("cadence"))
        p_cad_phy = cadence_from_stride(p_stride_t)
        c_cad_phy = cadence_from_stride(c_stride_t)

        p_cad_err = None if p_cad is None or p_cad_phy is None else abs(p_cad - p_cad_phy)
        c_cad_err = None if c_cad is None or c_cad_phy is None else abs(c_cad - c_cad_phy)

        if p_ratio is not None:
            prev_ratio_err.append(abs(p_ratio - 2.0))
            if p_ratio < 1.6 or p_ratio > 2.4:
                prev_ratio_bad += 1
        if c_ratio is not None:
            corr_ratio_err.append(abs(c_ratio - 2.0))
            if c_ratio < 1.6 or c_ratio > 2.4:
                corr_ratio_bad += 1
        if p_cad_err is not None:
            prev_cad_err.append(p_cad_err)
        if c_cad_err is not None:
            corr_cad_err.append(c_cad_err)

        detail_rows.append(
            [
                cid,
                p_step,
                c_step,
                p_stride,
                c_stride,
                p_ratio,
                c_ratio,
                p_step_t,
                c_step_t,
                p_stride_t,
                c_stride_t,
                p_time_ratio,
                c_time_ratio,
                p_cad,
                c_cad,
                p_cad_phy,
                c_cad_phy,
                p_cad_err,
                c_cad_err,
                to_float(p.get("walking_speed")),
                to_float(c.get("walking_speed")),
                c.get("step_count_estimate"),
                c.get("stride_count_estimate"),
            ]
        )

    summary_rows = [
        ["item", "value"],
        ["previous_cases", len(prev_map)],
        ["corrected_cases", len(corrected)],
        ["common_cases", len(common_ids)],
        ["prev_mean_abs(stride/step-2)", mean_or_none(prev_ratio_err)],
        ["corr_mean_abs(stride/step-2)", mean_or_none(corr_ratio_err)],
        ["prev_median_abs(stride/step-2)", median_or_none(prev_ratio_err)],
        ["corr_median_abs(stride/step-2)", median_or_none(corr_ratio_err)],
        ["prev_ratio_outside_[1.6,2.4]", prev_ratio_bad],
        ["corr_ratio_outside_[1.6,2.4]", corr_ratio_bad],
        ["prev_mean_abs(cadence-120/stride_time)", mean_or_none(prev_cad_err)],
        ["corr_mean_abs(cadence-120/stride_time)", mean_or_none(corr_cad_err)],
        ["prev_median_abs(cadence-120/stride_time)", median_or_none(prev_cad_err)],
        ["corr_median_abs(cadence-120/stride_time)", median_or_none(corr_cad_err)],
    ]

    rules_rows = [
        ["rule", "description"],
        ["R1", "Step/stride time are estimated from frame intervals (event frame delta / FPS)."],
        ["R2", "Cadence (steps/min) = 120 / average_stride_time."],
        ["R3", "Walking speed = distance_meters / (frame_count / FPS)."],
        ["R4", "Discard stride events closer than 0.8 seconds to reduce false triggers."],
    ]

    write_xlsx(
        output_xlsx,
        [
            ("Rules", rules_rows),
            ("Summary", summary_rows),
            ("Details", detail_rows),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create corrected previous-vs-recalculated xlsx.")
    parser.add_argument(
        "--previous-xlsx",
        default="/Users/Reese/Documents/my codex project/gait/check_previous.xlsx",
        help="Path to previous xlsx.",
    )
    parser.add_argument(
        "--corrected-json",
        default="/Users/Reese/Documents/my codex project/gait/gait_features_pred_2025_without_ds_corrected.json",
        help="Path to corrected json produced by gaitfeature.py without_ds_json.",
    )
    parser.add_argument(
        "--output-xlsx",
        default="/Users/Reese/Documents/my codex project/gait/check_previous_corrected_recalculated.xlsx",
        help="Output corrected xlsx path.",
    )
    args = parser.parse_args()

    build_corrected_xlsx(
        previous_xlsx=Path(args.previous_xlsx),
        corrected_json=Path(args.corrected_json),
        output_xlsx=Path(args.output_xlsx),
    )
    print(f"written: {Path(args.output_xlsx).resolve()}")


if __name__ == "__main__":
    main()
