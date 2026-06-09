#!/usr/bin/env python3
"""Match demographic info to pred_2025 gait features and run analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PARENT_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC_DIR))

from analyze_new_data_check import (
    LinearRegressionClosedForm,
    LinearSVM,
    LogisticRegressionGD,
    RandomForest,
    aggregate_metric_dicts,
    classification_metrics,
    make_kfold,
    make_stratified_folds,
    permutation_pvalue_mean_diff,
    regression_metrics,
    safe_mean,
    safe_median,
    safe_std,
    standardize_train_test,
    to_float,
)
from qc_check_previous_2026 import parse_first_sheet


GAIT_FEATURES = [
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


def normalize_id(raw: object) -> str:
    s = str(raw or "").strip()
    s = re.sub(r"-1(?:_1920p_30fps)?$", "", s)
    s = re.sub(r"[.#](?:1|2)$", "", s)
    return s


def dedupe_demo_rows(rows: Sequence[Dict[str, object]]) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    demo_map: Dict[str, Dict[str, object]] = {}
    conflicts: List[str] = []
    for row in rows:
        nid = normalize_id(row.get("ID", ""))
        if not nid:
            continue
        if nid not in demo_map:
            demo_map[nid] = row
            continue
        prev = demo_map[nid]
        keys = ["label", "edu", "age", "MMSE", "MoCA", "gender", "学历"]
        if any(str(prev.get(k, "")).strip() != str(row.get(k, "")).strip() for k in keys):
            conflicts.append(nid)
    return demo_map, conflicts


def aggregate_gait_rows(rows: Sequence[Dict[str, object]], valid_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
    valid = set(valid_ids)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        nid = normalize_id(row.get("ID", ""))
        if nid and nid in valid:
            grouped.setdefault(nid, []).append(row)

    aggregated: Dict[str, Dict[str, object]] = {}
    for nid, items in grouped.items():
        out: Dict[str, object] = {
            "ID": nid,
            "source_ids": ";".join(str(x.get("ID", "")).strip() for x in items),
            "source_count": len(items),
        }
        for feat in GAIT_FEATURES:
            vals = [to_float(x.get(feat)) for x in items]
            vals = [v for v in vals if v is not None]
            out[feat] = safe_mean(vals)
        aggregated[nid] = out
    return aggregated


def build_merged_rows(
    demo_rows: Sequence[Dict[str, object]],
    gait_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    demo_map, conflicts = dedupe_demo_rows(demo_rows)
    gait_map = aggregate_gait_rows(gait_rows, demo_map.keys())
    merged: List[Dict[str, object]] = []
    recovered_by_normalization = 0

    exact_demo_ids = {str(r.get("ID", "")).strip() for r in demo_rows}
    exact_gait_ids = {str(r.get("ID", "")).strip() for r in gait_rows}

    for nid, demo in demo_map.items():
        gait = gait_map.get(nid)
        if gait is None:
            continue
        row: Dict[str, object] = {
            "ID": nid,
            "source_gait_ids": gait["source_ids"],
            "source_gait_count": gait["source_count"],
            "label": to_float(demo.get("label")),
            "edu": to_float(demo.get("edu")),
            "age": to_float(demo.get("age")),
            "MMSE": to_float(demo.get("MMSE")),
            "MoCA": to_float(demo.get("MoCA")),
            "gender": str(demo.get("gender", "")).strip(),
            "学历": str(demo.get("学历", "")).strip(),
        }
        for feat in GAIT_FEATURES:
            row[feat] = gait.get(feat)
        if nid not in exact_demo_ids or nid not in exact_gait_ids:
            recovered_by_normalization += 1
        merged.append(row)

    merged.sort(key=lambda r: str(r["ID"]))
    qc = {
        "demo_rows": len(demo_rows),
        "gait_rows": len(gait_rows),
        "demo_unique_normalized_ids": len(demo_map),
        "gait_matched_normalized_ids": len(gait_map),
        "merged_rows": len(merged),
        "conflicting_demo_duplicates": sorted(set(conflicts)),
        "recovered_by_normalization_count": recovered_by_normalization,
    }
    return merged, qc


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def gait_group_analysis(rows: Sequence[Dict[str, object]], n_perm: int) -> List[List[object]]:
    label0 = [r for r in rows if r.get("label") == 0.0]
    label1 = [r for r in rows if r.get("label") == 1.0]
    out: List[List[object]] = []
    for feat in GAIT_FEATURES:
        x0 = [r.get(feat) for r in label0 if isinstance(r.get(feat), (int, float))]
        x1 = [r.get(feat) for r in label1 if isinstance(r.get(feat), (int, float))]
        p = permutation_pvalue_mean_diff(x0, x1, n_perm=n_perm, seed=42)
        mean0 = safe_mean(x0)
        mean1 = safe_mean(x1)
        out.append(
            [
                feat,
                len(x0),
                mean0,
                safe_std(x0),
                safe_median(x0),
                len(x1),
                mean1,
                safe_std(x1),
                safe_median(x1),
                None if mean0 is None or mean1 is None else mean1 - mean0,
                p,
            ]
        )
    out.sort(key=lambda r: (1e9 if r[-1] is None else r[-1], r[0]))
    return out


def impute_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_train_imp = np.where(np.isnan(X_train), med, X_train)
    X_test_imp = np.where(np.isnan(X_test), med, X_test)
    return X_train_imp, X_test_imp


def matrix_from_rows(rows: Sequence[Dict[str, object]], features: Sequence[str]) -> np.ndarray:
    data: List[List[float]] = []
    for row in rows:
        vals = []
        for feat in features:
            x = row.get(feat)
            vals.append(float(x) if isinstance(x, (int, float)) else float("nan"))
        data.append(vals)
    return np.array(data, dtype=float)


def classification_cv_subset(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: Sequence[int],
    model_builder: Callable[[], object],
) -> Dict[str, Optional[float]]:
    X_sub = X[:, feature_idx]
    folds = make_stratified_folds(y.tolist(), n_splits=5, seed=42)
    fold_metrics: List[Dict[str, Optional[float]]] = []
    for test_idx in folds:
        train_idx = [i for i in range(len(X_sub)) if i not in test_idx]
        X_train = X_sub[train_idx]
        X_test = X_sub[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        X_train, X_test = impute_train_test(X_train, X_test)
        X_train, X_test, _, _ = standardize_train_test(X_train, X_test)
        model = model_builder()
        model.fit(X_train, y_train)
        if hasattr(model, "predict_score"):
            scores = model.predict_score(X_test)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)  # type: ignore[attr-defined]
        else:
            scores = model.predict(X_test)
        y_pred = model.predict(X_test)
        fold_metrics.append(classification_metrics(y_test.tolist(), y_pred.tolist(), np.asarray(scores).tolist()))
    return aggregate_metric_dicts(fold_metrics)


def regression_cv_subset(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: Sequence[int],
    model_builder: Callable[[], object],
) -> Dict[str, Optional[float]]:
    X_sub = X[:, feature_idx]
    folds = make_kfold(len(X_sub), n_splits=5, seed=42)
    fold_metrics: List[Dict[str, Optional[float]]] = []
    for test_idx in folds:
        train_idx = [i for i in range(len(X_sub)) if i not in test_idx]
        X_train = X_sub[train_idx]
        X_test = X_sub[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        X_train, X_test = impute_train_test(X_train, X_test)
        X_train, X_test, _, _ = standardize_train_test(X_train, X_test)
        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=float)
        fold_metrics.append(regression_metrics(y_test.tolist(), y_pred.tolist()))
    return aggregate_metric_dicts(fold_metrics)


def forward_select_classification(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    model_builder: Callable[[], object],
    max_features: int,
    candidate_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[Dict[str, object]]]:
    selected: List[int] = []
    remaining = list(candidate_indices) if candidate_indices is not None else list(range(len(feature_names)))
    history: List[Dict[str, object]] = []
    best_score = -1.0
    for step in range(1, max_features + 1):
        best_candidate = None
        best_metrics = None
        best_candidate_score = -1.0
        for idx in remaining:
            feat_idx = selected + [idx]
            metrics = classification_cv_subset(X, y, feat_idx, model_builder)
            score = float(metrics.get("balanced_accuracy") or -1.0)
            auc = float(metrics.get("auc") or -1.0)
            if (
                score > best_candidate_score + 1e-12
                or (abs(score - best_candidate_score) <= 1e-12 and auc > float((best_metrics or {}).get("auc") or -1.0))
            ):
                best_candidate = idx
                best_candidate_score = score
                best_metrics = metrics
        if best_candidate is None or best_metrics is None:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        history.append(
            {
                "step": step,
                "selected_features": [feature_names[i] for i in selected],
                **best_metrics,
            }
        )
        if best_candidate_score < best_score - 1e-12:
            break
        best_score = best_candidate_score
    best_entry = max(history, key=lambda x: (float(x.get("balanced_accuracy") or -1.0), float(x.get("auc") or -1.0)))
    best_selected = best_entry["selected_features"]
    return [feature_names.index(name) for name in best_selected], history


def forward_select_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    model_builder: Callable[[], object],
    max_features: int,
    candidate_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[Dict[str, object]]]:
    selected: List[int] = []
    remaining = list(candidate_indices) if candidate_indices is not None else list(range(len(feature_names)))
    history: List[Dict[str, object]] = []
    best_score = -1e18
    for step in range(1, max_features + 1):
        best_candidate = None
        best_metrics = None
        best_candidate_score = -1e18
        for idx in remaining:
            feat_idx = selected + [idx]
            metrics = regression_cv_subset(X, y, feat_idx, model_builder)
            score = float(metrics.get("r2") or -1e18)
            rmse = -float(metrics.get("rmse") or 1e18)
            if (
                score > best_candidate_score + 1e-12
                or (abs(score - best_candidate_score) <= 1e-12 and rmse > -float((best_metrics or {}).get("rmse") or 1e18))
            ):
                best_candidate = idx
                best_candidate_score = score
                best_metrics = metrics
        if best_candidate is None or best_metrics is None:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        history.append(
            {
                "step": step,
                "selected_features": [feature_names[i] for i in selected],
                **best_metrics,
            }
        )
        if best_candidate_score < best_score - 1e-12:
            break
        best_score = best_candidate_score
    best_entry = max(history, key=lambda x: (float(x.get("r2") or -1e18), -float(x.get("rmse") or 1e18)))
    best_selected = best_entry["selected_features"]
    return [feature_names.index(name) for name in best_selected], history


def safe_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(ys) < 3 or len(xs) != len(ys):
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return num / (den_x * den_y)


def top_classification_candidates(group_rows: Sequence[Sequence[object]], top_n: int) -> List[str]:
    ranked = sorted(group_rows, key=lambda r: (1e9 if r[-1] is None else float(r[-1]), r[0]))
    return [str(r[0]) for r in ranked[:top_n]]


def top_regression_candidates(rows: Sequence[Dict[str, object]], top_n: int) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for feat in GAIT_FEATURES:
        xs: List[float] = []
        ys: List[float] = []
        for row in rows:
            x = row.get(feat)
            y = row.get("MMSE")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                xs.append(float(x))
                ys.append(float(y))
        corr = safe_corr(xs, ys)
        scored.append((abs(corr) if corr is not None else -1.0, feat))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [feat for _, feat in scored[:top_n]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched pred_2025 gait features against demographic labels")
    parser.add_argument("--demo-xlsx", default="../new data_check.xlsx")
    parser.add_argument("--gait-xlsx", default="check_previous_format_pred_2025_adaptive.xlsx")
    parser.add_argument("--out-dir", default="matched_pred2025_gait_analysis")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--max-selected-features", type=int, default=5)
    parser.add_argument("--top-candidates", type=int, default=6)
    parser.add_argument("--regression-target", default="MMSE", choices=["MMSE", "MoCA"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, demo_rows = parse_first_sheet(Path(args.demo_xlsx))
    _, gait_rows = parse_first_sheet(Path(args.gait_xlsx))
    merged_rows, qc = build_merged_rows(demo_rows, gait_rows)

    merged_csv_rows: List[List[object]] = []
    for row in merged_rows:
        merged_csv_rows.append(
            [
                row.get("ID"),
                row.get("source_gait_ids"),
                row.get("source_gait_count"),
                row.get("label"),
                row.get("MMSE"),
                row.get("MoCA"),
                row.get("edu"),
                row.get("age"),
                row.get("gender"),
                row.get("学历"),
            ]
            + [row.get(f) for f in GAIT_FEATURES]
        )
    write_csv(
        out_dir / "matched_dataset.csv",
        ["ID", "source_gait_ids", "source_gait_count", "label", "MMSE", "MoCA", "edu", "age", "gender", "学历"] + GAIT_FEATURES,
        merged_csv_rows,
    )

    group_rows = gait_group_analysis(merged_rows, n_perm=args.n_perm)
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

    cls_candidate_names = top_classification_candidates(group_rows, top_n=args.top_candidates)
    reg_candidate_names = top_regression_candidates(merged_rows, top_n=args.top_candidates)
    cls_candidate_idx = [GAIT_FEATURES.index(x) for x in cls_candidate_names]
    reg_candidate_idx = [GAIT_FEATURES.index(x) for x in reg_candidate_names]

    classification_rows = [r for r in merged_rows if r.get("label") in {0.0, 1.0}]
    regression_target = args.regression_target
    regression_rows = [r for r in merged_rows if r.get(regression_target) is not None]
    X_cls = matrix_from_rows(classification_rows, GAIT_FEATURES)
    y_cls = np.array([int(r["label"]) for r in classification_rows], dtype=int)
    X_reg = matrix_from_rows(regression_rows, GAIT_FEATURES)
    y_reg = np.array([float(r[regression_target]) for r in regression_rows], dtype=float)

    cls_models: Dict[str, Callable[[], object]] = {
        "logistic_regression": lambda: LogisticRegressionGD(lr=0.05, epochs=2500, l2=1e-3),
        "linear_svm": lambda: LinearSVM(lr=0.02, epochs=2500, c=1.0),
        "random_forest": lambda: RandomForest(
            task="classification",
            n_estimators=40,
            max_depth=5,
            min_samples_split=6,
            max_features=3,
            random_state=42,
        ),
    }
    reg_models: Dict[str, Callable[[], object]] = {
        "linear_regression": lambda: LinearRegressionClosedForm(l2=1e-6),
        "random_forest_regressor": lambda: RandomForest(
            task="regression",
            n_estimators=40,
            max_depth=5,
            min_samples_split=6,
            max_features=3,
            random_state=42,
        ),
    }

    cls_all_rows: List[List[object]] = []
    cls_best_rows: List[List[object]] = []
    cls_progress_rows: List[List[object]] = []
    for model_name, builder in cls_models.items():
        all_metrics = classification_cv_subset(X_cls, y_cls, list(range(len(GAIT_FEATURES))), builder)
        best_idx, history = forward_select_classification(
            X_cls,
            y_cls,
            GAIT_FEATURES,
            builder,
            max_features=args.max_selected_features,
            candidate_indices=cls_candidate_idx,
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
        for entry in history:
            cls_progress_rows.append(
                [model_name, entry["step"], len(entry["selected_features"]), ";".join(entry["selected_features"])]
                + [entry.get(k) for k in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"]]
            )

    reg_all_rows: List[List[object]] = []
    reg_best_rows: List[List[object]] = []
    reg_progress_rows: List[List[object]] = []
    for model_name, builder in reg_models.items():
        all_metrics = regression_cv_subset(X_reg, y_reg, list(range(len(GAIT_FEATURES))), builder)
        best_idx, history = forward_select_regression(
            X_reg,
            y_reg,
            GAIT_FEATURES,
            builder,
            max_features=args.max_selected_features,
            candidate_indices=reg_candidate_idx,
        )
        best_metrics = regression_cv_subset(X_reg, y_reg, best_idx, builder)
        reg_all_rows.append(
            [model_name, "all_features", ";".join(GAIT_FEATURES)]
            + [all_metrics.get(k) for k in ["rmse", "mae", "r2"]]
        )
        reg_best_rows.append(
            [model_name, len(best_idx), ";".join(GAIT_FEATURES[i] for i in best_idx)]
            + [best_metrics.get(k) for k in ["rmse", "mae", "r2"]]
        )
        for entry in history:
            reg_progress_rows.append(
                [model_name, entry["step"], len(entry["selected_features"]), ";".join(entry["selected_features"])]
                + [entry.get(k) for k in ["rmse", "mae", "r2"]]
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
    write_csv(
        out_dir / "classification_feature_selection_progress.csv",
        ["model", "step", "n_features", "features", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"],
        cls_progress_rows,
    )

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
    write_csv(
        out_dir / "regression_feature_selection_progress.csv",
        ["model", "step", "n_features", "features", "rmse", "mae", "r2"],
        reg_progress_rows,
    )

    summary = {
        "demo_xlsx": str(Path(args.demo_xlsx).resolve()),
        "gait_xlsx": str(Path(args.gait_xlsx).resolve()),
        "gait_features": GAIT_FEATURES,
        "regression_target": regression_target,
        "qc": qc,
        "classification_candidate_features": cls_candidate_names,
        "regression_candidate_features": reg_candidate_names,
        "classification_rows": len(classification_rows),
        "regression_rows": len(regression_rows),
        "classification_all_features": cls_all_rows,
        "classification_best_feature_sets": cls_best_rows,
        "regression_all_features": reg_all_rows,
        "regression_best_feature_sets": reg_best_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Output dir: {out_dir.resolve()}")
    print(f"merged_rows={len(merged_rows)} classification_rows={len(classification_rows)} regression_rows={len(regression_rows)}")


if __name__ == "__main__":
    main()
