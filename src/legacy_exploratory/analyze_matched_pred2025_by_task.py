#!/usr/bin/env python3
"""Analyze matched pred_2025 gait features split by single/dual task."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PARENT_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC_DIR))

from analyze_matched_pred2025_gait_vs_demo import (
    GAIT_FEATURES,
    gait_group_analysis,
    matrix_from_rows,
    normalize_id,
    regression_cv_subset,
    top_classification_candidates,
    top_regression_candidates,
    classification_cv_subset,
    forward_select_classification,
    forward_select_regression,
    write_csv,
)
from analyze_new_data_check import (
    LinearRegressionClosedForm,
    LinearSVM,
    LogisticRegressionGD,
    RandomForest,
    to_float,
)
from qc_check_previous_2026 import parse_first_sheet


def task_of(raw_id: object) -> str:
    s = normalize_id(raw_id)
    return "single" if ("_as" in s or "_bs" in s or s.endswith("s")) else "dual"


def gait_base_id(raw_id: object) -> str:
    s = normalize_id(raw_id)
    s = s.replace("_as", "_a").replace("_bs", "_b")
    if s.endswith("s"):
        s = s[:-1]
    return s


def dedupe_demo_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    demo_map: Dict[str, Dict[str, object]] = {}
    for row in rows:
        nid = normalize_id(row.get("ID", ""))
        if nid and nid not in demo_map:
            demo_map[nid] = row
    return demo_map


def build_task_rows(
    demo_rows: Sequence[Dict[str, object]],
    gait_rows: Sequence[Dict[str, object]],
    task: str,
) -> List[Dict[str, object]]:
    demo_map = dedupe_demo_rows(demo_rows)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in gait_rows:
        if task_of(row.get("ID", "")) != task:
            continue
        bid = gait_base_id(row.get("ID", ""))
        if bid in demo_map:
            grouped.setdefault(bid, []).append(row)

    aggregated: List[Dict[str, object]] = []
    for bid, items in sorted(grouped.items()):
        demo = demo_map[bid]
        row: Dict[str, object] = {
            "ID": bid,
            "task": task,
            "source_gait_ids": ";".join(str(x.get("ID", "")).strip() for x in items),
            "source_gait_count": len(items),
            "label": to_float(demo.get("label")),
            "MoCA": to_float(demo.get("MoCA")),
            "MMSE": to_float(demo.get("MMSE")),
            "age": to_float(demo.get("age")),
            "edu": to_float(demo.get("edu")),
            "gender": str(demo.get("gender", "")).strip(),
            "学历": str(demo.get("学历", "")).strip(),
        }
        for feat in GAIT_FEATURES:
            vals = [to_float(x.get(feat)) for x in items]
            vals = [v for v in vals if v is not None]
            row[feat] = float(sum(vals) / len(vals)) if vals else None
        aggregated.append(row)
    return aggregated


def save_task_dataset(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    data: List[List[object]] = []
    for row in rows:
        data.append(
            [
                row.get("ID"),
                row.get("task"),
                row.get("source_gait_ids"),
                row.get("source_gait_count"),
                row.get("label"),
                row.get("MoCA"),
                row.get("MMSE"),
                row.get("age"),
                row.get("edu"),
                row.get("gender"),
                row.get("学历"),
            ]
            + [row.get(f) for f in GAIT_FEATURES]
        )
    write_csv(
        path,
        ["ID", "task", "source_gait_ids", "source_gait_count", "label", "MoCA", "MMSE", "age", "edu", "gender", "学历"] + GAIT_FEATURES,
        data,
    )


def analyze_task(rows: List[Dict[str, object]], out_dir: Path, regression_target: str, n_perm: int, max_selected: int, top_candidates: int) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_task_dataset(out_dir / "matched_dataset.csv", rows)

    labels = sorted({int(r["label"]) for r in rows if r.get("label") in {0.0, 1.0}})
    summary: Dict[str, object] = {
        "n_rows": len(rows),
        "label_counts": {str(lbl): sum(1 for r in rows if r.get("label") == float(lbl)) for lbl in labels},
        "regression_target": regression_target,
    }

    if len(labels) >= 2:
        group_rows = gait_group_analysis(rows, n_perm=n_perm)
        write_csv(
            out_dir / "gait_group_analysis.csv",
            [
                "feature",
                "n_label0",
                "mean_label0",
                "std_label0",
                "median_label0",
                "n_label1",
                "mean_label1",
                "std_label1",
                "median_label1",
                "mean_diff_label1_minus_label0",
                "permutation_pvalue",
            ],
            group_rows,
        )
        X_cls = matrix_from_rows(rows, GAIT_FEATURES)
        y_cls = np.array([int(r["label"]) for r in rows], dtype=int)
        cls_candidate_names = top_classification_candidates(group_rows, top_n=top_candidates)
        cls_candidate_idx = [GAIT_FEATURES.index(x) for x in cls_candidate_names]
        cls_models = {
            "logistic_regression": lambda: LogisticRegressionGD(lr=0.05, epochs=2500, l2=1e-3),
            "linear_svm": lambda: LinearSVM(lr=0.02, epochs=2500, c=1.0),
            "random_forest": lambda: RandomForest(task="classification", n_estimators=24, max_depth=5, min_samples_split=6, max_features=3, random_state=42),
        }
        cls_all_rows: List[List[object]] = []
        cls_best_rows: List[List[object]] = []
        for model_name, builder in cls_models.items():
            all_metrics = classification_cv_subset(X_cls, y_cls, list(range(len(GAIT_FEATURES))), builder)
            best_idx, _ = forward_select_classification(
                X_cls, y_cls, GAIT_FEATURES, builder, max_features=max_selected, candidate_indices=cls_candidate_idx
            )
            best_metrics = classification_cv_subset(X_cls, y_cls, best_idx, builder)
            cls_all_rows.append(
                [model_name, "all_features", ";".join(GAIT_FEATURES)]
                + [all_metrics.get(k) for k in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"]]
            )
            cls_best_rows.append(
                [model_name, len(best_idx), ";".join(GAIT_FEATURES[i] for i in best_idx)]
                + [best_metrics.get(k) for k in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"]]
            )
        write_csv(
            out_dir / "classification_all_features_cv.csv",
            ["model", "feature_mode", "features", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"],
            cls_all_rows,
        )
        write_csv(
            out_dir / "classification_best_feature_sets.csv",
            ["model", "n_features", "features", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"],
            cls_best_rows,
        )
        summary["classification_candidate_features"] = cls_candidate_names
        summary["classification_all_features"] = cls_all_rows
        summary["classification_best_feature_sets"] = cls_best_rows
    else:
        summary["classification_skipped_reason"] = "Only one label present in this task subset."
        summary["group_analysis_skipped_reason"] = "Only one label present in this task subset."

    regression_rows = [r for r in rows if r.get(regression_target) is not None]
    if len(regression_rows) >= 8:
        X_reg = matrix_from_rows(regression_rows, GAIT_FEATURES)
        y_reg = np.array([float(r[regression_target]) for r in regression_rows], dtype=float)
        reg_candidate_names = top_regression_candidates(regression_rows, top_n=top_candidates)
        reg_candidate_idx = [GAIT_FEATURES.index(x) for x in reg_candidate_names]
        reg_models = {
            "linear_regression": lambda: LinearRegressionClosedForm(l2=1e-6),
            "random_forest_regressor": lambda: RandomForest(task="regression", n_estimators=24, max_depth=5, min_samples_split=6, max_features=3, random_state=42),
        }
        reg_all_rows: List[List[object]] = []
        reg_best_rows: List[List[object]] = []
        for model_name, builder in reg_models.items():
            all_metrics = regression_cv_subset(X_reg, y_reg, list(range(len(GAIT_FEATURES))), builder)
            best_idx, _ = forward_select_regression(
                X_reg, y_reg, GAIT_FEATURES, builder, max_features=max_selected, candidate_indices=reg_candidate_idx
            )
            best_metrics = regression_cv_subset(X_reg, y_reg, best_idx, builder)
            reg_all_rows.append([model_name, "all_features", ";".join(GAIT_FEATURES)] + [all_metrics.get(k) for k in ["rmse", "mae", "r2"]])
            reg_best_rows.append([model_name, len(best_idx), ";".join(GAIT_FEATURES[i] for i in best_idx)] + [best_metrics.get(k) for k in ["rmse", "mae", "r2"]])
        write_csv(
            out_dir / "regression_all_features_cv.csv",
            ["model", "feature_mode", "features", "rmse", "mae", "r2"],
            reg_all_rows,
        )
        write_csv(
            out_dir / "regression_best_feature_sets.csv",
            ["model", "n_features", "features", "rmse", "mae", "r2"],
            reg_best_rows,
        )
        summary["regression_rows"] = len(regression_rows)
        summary["regression_candidate_features"] = reg_candidate_names
        summary["regression_all_features"] = reg_all_rows
        summary["regression_best_feature_sets"] = reg_best_rows
    else:
        summary["regression_skipped_reason"] = "Too few rows with regression target."

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched pred_2025 gait features split by task")
    parser.add_argument("--demo-xlsx", default="../new data_check.xlsx")
    parser.add_argument("--gait-xlsx", default="check_previous_format_pred_2025_adaptive.xlsx")
    parser.add_argument("--out-dir", default="matched_pred2025_by_task_analysis")
    parser.add_argument("--regression-target", default="MoCA", choices=["MMSE", "MoCA"])
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--max-selected-features", type=int, default=4)
    parser.add_argument("--top-candidates", type=int, default=6)
    args = parser.parse_args()

    _, demo_rows = parse_first_sheet(Path(args.demo_xlsx))
    _, gait_rows = parse_first_sheet(Path(args.gait_xlsx))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_summary: Dict[str, object] = {
        "demo_xlsx": str(Path(args.demo_xlsx).resolve()),
        "gait_xlsx": str(Path(args.gait_xlsx).resolve()),
        "regression_target": args.regression_target,
        "tasks": {},
    }
    for task in ["dual", "single"]:
        task_rows = build_task_rows(demo_rows, gait_rows, task)
        summary = analyze_task(task_rows, out_dir / task, args.regression_target, args.n_perm, args.max_selected_features, args.top_candidates)
        full_summary["tasks"][task] = summary

    (out_dir / "summary.json").write_text(json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output dir: {out_dir.resolve()}")
    for task in ["dual", "single"]:
        task_summary = full_summary["tasks"][task]
        print(f"{task}: n_rows={task_summary['n_rows']} label_counts={task_summary['label_counts']}")


if __name__ == "__main__":
    main()
