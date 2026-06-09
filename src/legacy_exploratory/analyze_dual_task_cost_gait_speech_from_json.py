#!/usr/bin/env python3
"""Joint analysis using server-rerun task8/task9 JSON features and gait paired cost."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PARENT_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC_DIR))

from analyze_matched_pred2025_by_task import gait_base_id, task_of
from analyze_matched_pred2025_gait_vs_demo import (
    GAIT_FEATURES,
    gait_group_analysis,
    matrix_from_rows,
    normalize_id,
)
from analyze_new_data_check import (
    LinearRegressionClosedForm,
    LinearSVM,
    LogisticRegressionGD,
    RandomForest,
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


SPEECH_BASE_FEATURES = [
    "ACC",
    "Correct_persec",
    "Correct_Num",
    "Num",
    "Timecount",
    "Timecount_1stNum",
    "Num_20s",
    "CorrectNum_20s",
]


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def speech_key_to_id_and_task(key: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"([A-Z]{4}_\d{3}(?:_[ab])?)_+(8|9)(?!\d)", key)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"([A-Z]{4}_\d{3}(?:_[ab])?)_(8|9)(?!\d)", key)
    if m:
        return m.group(1), m.group(2)
    return None, None


def aggregate_speech_json(path: Path, expected_task: str) -> Dict[str, Dict[str, Optional[float]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for key, feat in obj.get("features", {}).items():
        if not isinstance(feat, dict) or "error" in feat:
            continue
        bid, task = speech_key_to_id_and_task(str(key))
        if not bid or task != expected_task:
            continue
        grouped[bid].append(feat)

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for bid, items in grouped.items():
        agg: Dict[str, Optional[float]] = {}
        for feat_name in SPEECH_BASE_FEATURES:
            vals = [to_float(x.get(feat_name)) for x in items]
            vals = [v for v in vals if v is not None]
            agg[feat_name] = float(sum(vals) / len(vals)) if vals else None
        out[bid] = agg
    return out


def aggregate_gait_rows(rows: Sequence[Dict[str, object]]) -> Dict[Tuple[str, str], Dict[str, Optional[float]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(gait_base_id(row.get("ID", "")), task_of(row.get("ID", "")))].append(row)
    out: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    for key, items in grouped.items():
        agg: Dict[str, Optional[float]] = {}
        for feat in GAIT_FEATURES:
            vals = [to_float(x.get(feat)) for x in items]
            vals = [v for v in vals if v is not None]
            agg[feat] = float(sum(vals) / len(vals)) if vals else None
        out[key] = agg
    return out


def dual_task_cost(dual: Optional[float], single: Optional[float]) -> Optional[float]:
    if dual is None or single is None:
        return None
    if abs(single) < 1e-12:
        return 0.0 if abs(dual) < 1e-12 else None
    return (dual - single) / single


def build_joint_rows(
    demo_rows: Sequence[Dict[str, object]],
    gait_rows: Sequence[Dict[str, object]],
    speech8: Dict[str, Dict[str, Optional[float]]],
    speech9: Dict[str, Dict[str, Optional[float]]],
) -> List[Dict[str, object]]:
    demo_map = {normalize_id(r.get("ID", "")): r for r in demo_rows if normalize_id(r.get("ID", ""))}
    gait_map = aggregate_gait_rows(gait_rows)
    speech_paired_ids = set(speech8) & set(speech9)

    paired_ids = sorted(
        bid
        for bid in demo_map
        if (bid, "single") in gait_map and (bid, "dual") in gait_map and bid in speech_paired_ids
    )
    rows: List[Dict[str, object]] = []
    for bid in paired_ids:
        demo = demo_map[bid]
        row: Dict[str, object] = {
            "ID": bid,
            "label": to_float(demo.get("label")),
            "MoCA": to_float(demo.get("MoCA")),
            "MMSE": to_float(demo.get("MMSE")),
        }
        for feat in GAIT_FEATURES:
            row[f"gait_cost::{feat}"] = dual_task_cost(gait_map[(bid, "dual")].get(feat), gait_map[(bid, "single")].get(feat))
        for feat in SPEECH_BASE_FEATURES:
            row[f"speech_cost::{feat}"] = dual_task_cost(speech9[bid].get(feat), speech8[bid].get(feat))
        rows.append(row)
    return rows


def matrix_from_feature_list(rows: Sequence[Dict[str, object]], features: Sequence[str]) -> np.ndarray:
    return np.array(
        [
            [float(row.get(f)) if isinstance(row.get(f), (int, float)) else float("nan") for f in features]
            for row in rows
        ],
        dtype=float,
    )


def impute_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(X_train), med, X_train), np.where(np.isnan(X_test), med, X_test)


def safe_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= 1e-12 or deny <= 1e-12:
        return None
    return num / (denx * deny)


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
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
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
        preds = model.predict(X_test)
        fold_metrics.append(classification_metrics(y_test.tolist(), preds.tolist(), np.asarray(scores).tolist()))
    keys = fold_metrics[0].keys()
    return {k: statistics.mean([m[k] for m in fold_metrics if m.get(k) is not None]) if any(m.get(k) is not None for m in fold_metrics) else None for k in keys}


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
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = impute_train_test(X_train, X_test)
        X_train, X_test, _, _ = standardize_train_test(X_train, X_test)
        model = model_builder()
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test), dtype=float)
        fold_metrics.append(regression_metrics(y_test.tolist(), pred.tolist()))
    keys = fold_metrics[0].keys()
    return {k: statistics.mean([m[k] for m in fold_metrics if m.get(k) is not None]) if any(m.get(k) is not None for m in fold_metrics) else None for k in keys}


def rank_by_group_pvalue(rows: Sequence[Dict[str, object]], features: Sequence[str]) -> List[str]:
    label0 = [r for r in rows if r.get("label") == 0.0]
    label1 = [r for r in rows if r.get("label") == 1.0]
    ranked: List[Tuple[float, str]] = []
    for feat in features:
        x0 = [float(r[feat]) for r in label0 if isinstance(r.get(feat), (int, float))]
        x1 = [float(r[feat]) for r in label1 if isinstance(r.get(feat), (int, float))]
        p = permutation_pvalue_mean_diff(x0, x1, n_perm=2000, seed=42)
        ranked.append((1e9 if p is None else float(p), feat))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [feat for _, feat in ranked]


def rank_by_corr(rows: Sequence[Dict[str, object]], features: Sequence[str], target: str) -> List[str]:
    ranked: List[Tuple[float, str]] = []
    for feat in features:
        xs = [float(r[feat]) for r in rows if isinstance(r.get(feat), (int, float)) and isinstance(r.get(target), (int, float))]
        ys = [float(r[target]) for r in rows if isinstance(r.get(feat), (int, float)) and isinstance(r.get(target), (int, float))]
        corr = safe_corr(xs, ys)
        ranked.append((-(abs(corr) if corr is not None else -1.0), feat))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [feat for _, feat in ranked]


def forward_select_classification(
    rows: Sequence[Dict[str, object]],
    features: Sequence[str],
    model_builder: Callable[[], object],
    max_features: int,
    top_n: int,
) -> Tuple[List[str], Dict[str, Optional[float]]]:
    ranked = rank_by_group_pvalue(rows, features)[:top_n]
    X = matrix_from_feature_list(rows, features)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    selected: List[str] = []
    best_feats: List[str] = []
    best_metrics: Dict[str, Optional[float]] = {}
    best_score = -1e18
    remaining = ranked[:]
    for _ in range(min(max_features, len(ranked))):
        step_best_feat = None
        step_best_metrics = None
        step_best_score = -1e18
        step_best_auc = -1e18
        for feat in remaining:
            idx = [features.index(x) for x in selected + [feat]]
            metrics = classification_cv_subset(X, y, idx, model_builder)
            score = float(metrics.get("balanced_accuracy") or -1e18)
            auc = float(metrics.get("auc") or -1e18)
            if score > step_best_score + 1e-12 or (abs(score - step_best_score) <= 1e-12 and auc > step_best_auc):
                step_best_feat = feat
                step_best_metrics = metrics
                step_best_score = score
                step_best_auc = auc
        if step_best_feat is None or step_best_metrics is None:
            break
        selected.append(step_best_feat)
        remaining.remove(step_best_feat)
        if step_best_score > best_score + 1e-12 or (abs(step_best_score - best_score) <= 1e-12 and float(step_best_metrics.get("auc") or -1e18) > float(best_metrics.get("auc") or -1e18)):
            best_score = step_best_score
            best_feats = selected[:]
            best_metrics = step_best_metrics
    return best_feats, best_metrics


def forward_select_regression(
    rows: Sequence[Dict[str, object]],
    features: Sequence[str],
    target: str,
    model_builder: Callable[[], object],
    max_features: int,
    top_n: int,
) -> Tuple[List[str], Dict[str, Optional[float]]]:
    ranked = rank_by_corr(rows, features, target)[:top_n]
    X = matrix_from_feature_list(rows, features)
    y = np.array([float(r[target]) for r in rows], dtype=float)
    selected: List[str] = []
    best_feats: List[str] = []
    best_metrics: Dict[str, Optional[float]] = {}
    best_score = -1e18
    remaining = ranked[:]
    for _ in range(min(max_features, len(ranked))):
        step_best_feat = None
        step_best_metrics = None
        step_best_score = -1e18
        step_best_rmse = 1e18
        for feat in remaining:
            idx = [features.index(x) for x in selected + [feat]]
            metrics = regression_cv_subset(X, y, idx, model_builder)
            score = float(metrics.get("r2") or -1e18)
            rmse = float(metrics.get("rmse") or 1e18)
            if score > step_best_score + 1e-12 or (abs(score - step_best_score) <= 1e-12 and rmse < step_best_rmse):
                step_best_feat = feat
                step_best_metrics = metrics
                step_best_score = score
                step_best_rmse = rmse
        if step_best_feat is None or step_best_metrics is None:
            break
        selected.append(step_best_feat)
        remaining.remove(step_best_feat)
        if step_best_score > best_score + 1e-12 or (abs(step_best_score - best_score) <= 1e-12 and float(step_best_metrics.get("rmse") or 1e18) < float(best_metrics.get("rmse") or 1e18)):
            best_score = step_best_score
            best_feats = selected[:]
            best_metrics = step_best_metrics
    return best_feats, best_metrics


def group_analysis(rows: Sequence[Dict[str, object]], features: Sequence[str]) -> List[List[object]]:
    label0 = [r for r in rows if r.get("label") == 0.0]
    label1 = [r for r in rows if r.get("label") == 1.0]
    out: List[List[object]] = []
    for feat in features:
        x0 = [float(r[feat]) for r in label0 if isinstance(r.get(feat), (int, float))]
        x1 = [float(r[feat]) for r in label1 if isinstance(r.get(feat), (int, float))]
        p = permutation_pvalue_mean_diff(x0, x1, n_perm=2000, seed=42)
        mean0 = safe_mean(x0)
        mean1 = safe_mean(x1)
        out.append([feat, len(x0), mean0, safe_std(x0), safe_median(x0), len(x1), mean1, safe_std(x1), safe_median(x1), None if mean0 is None or mean1 is None else mean1 - mean0, p])
    out.sort(key=lambda r: (1e9 if r[-1] is None else r[-1], r[0]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint analysis from rerun task8/task9 JSON and gait cost")
    parser.add_argument("--demo-xlsx", default="../new data_check.xlsx")
    parser.add_argument("--gait-xlsx", nargs="+", default=["check_previous_format_pred_2025_adaptive.xlsx", "check_previous_format_new_2026_adaptive.xlsx"])
    parser.add_argument("--task8-json", default="task8_features_rerun_20260311.json")
    parser.add_argument("--task9-json", default="task9_features_rerun_20260311.json")
    parser.add_argument("--out-dir", default="dual_task_cost_gait_speech_json_analysis")
    parser.add_argument("--max-selected-features", type=int, default=3)
    parser.add_argument("--top-candidates", type=int, default=6)
    parser.add_argument("--target", default="MoCA", choices=["MoCA", "MMSE"])
    args = parser.parse_args()

    _, demo_rows = parse_first_sheet(Path(args.demo_xlsx))
    gait_rows: List[Dict[str, object]] = []
    for xlsx in args.gait_xlsx:
        _, rows = parse_first_sheet(Path(xlsx))
        gait_rows.extend(rows)

    speech8 = aggregate_speech_json(Path(args.task8_json), "8")
    speech9 = aggregate_speech_json(Path(args.task9_json), "9")
    rows = build_joint_rows(demo_rows, gait_rows, speech8, speech9)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gait_cost_features = [f"gait_cost::{x}" for x in GAIT_FEATURES]
    speech_cost_features = [f"speech_cost::{x}" for x in SPEECH_BASE_FEATURES]
    all_features = gait_cost_features + speech_cost_features

    dataset_rows = [[r.get("ID"), r.get("label"), r.get("MoCA"), r.get("MMSE")] + [r.get(f) for f in all_features] for r in rows]
    write_csv(out_dir / "joint_dataset.csv", ["ID", "label", "MoCA", "MMSE"] + all_features, dataset_rows)

    group_rows = group_analysis(rows, all_features)
    write_csv(out_dir / "group_analysis.csv", ["feature", "n_label0", "mean_label0", "std_label0", "median_label0", "n_label1", "mean_label1", "std_label1", "median_label1", "mean_diff_label1_minus_label0", "permutation_pvalue"], group_rows)

    cls_models = {
        "logistic_regression": lambda: LogisticRegressionGD(lr=0.05, epochs=2500, l2=1e-3),
        "linear_svm": lambda: LinearSVM(lr=0.02, epochs=2500, c=1.0),
        "random_forest": lambda: RandomForest(task="classification", n_estimators=24, max_depth=5, min_samples_split=6, max_features=3, random_state=42),
    }
    reg_models = {
        "linear_regression": lambda: LinearRegressionClosedForm(l2=1e-6),
        "random_forest_regressor": lambda: RandomForest(task="regression", n_estimators=24, max_depth=5, min_samples_split=6, max_features=3, random_state=42),
    }
    feature_sets = {
        "gait_cost_only": gait_cost_features,
        "speech_cost_only": speech_cost_features,
        "joint": all_features,
    }

    cls_rows: List[List[object]] = []
    reg_rows: List[List[object]] = []
    for set_name, feature_list in feature_sets.items():
        X_cls = matrix_from_feature_list(rows, feature_list)
        y_cls = np.array([int(r["label"]) for r in rows], dtype=int)
        X_reg = matrix_from_feature_list(rows, feature_list)
        y_reg = np.array([float(r[args.target]) for r in rows if isinstance(r.get(args.target), (int, float))], dtype=float)
        reg_rows_data = [r for r in rows if isinstance(r.get(args.target), (int, float))]

        for model_name, builder in cls_models.items():
            best_feats, best_metrics = forward_select_classification(rows, feature_list, builder, args.max_selected_features, args.top_candidates)
            cls_rows.append([set_name, model_name, len(best_feats), ";".join(best_feats)] + [best_metrics.get(k) for k in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"]])

        for model_name, builder in reg_models.items():
            best_feats, best_metrics = forward_select_regression(reg_rows_data, feature_list, args.target, builder, args.max_selected_features, args.top_candidates)
            reg_rows.append([set_name, model_name, len(best_feats), ";".join(best_feats)] + [best_metrics.get(k) for k in ["rmse", "mae", "r2"]])

    write_csv(out_dir / "classification_results.csv", ["feature_set", "model", "n_features", "features", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"], cls_rows)
    write_csv(out_dir / "regression_results.csv", ["feature_set", "model", "n_features", "features", "rmse", "mae", "r2"], reg_rows)

    summary = {
        "demo_xlsx": str(Path(args.demo_xlsx).resolve()),
        "gait_xlsx": [str(Path(x).resolve()) for x in args.gait_xlsx],
        "task8_json": str(Path(args.task8_json).resolve()),
        "task9_json": str(Path(args.task9_json).resolve()),
        "n_joint_rows": len(rows),
        "label_counts": {
            "0": sum(1 for r in rows if r.get("label") == 0.0),
            "1": sum(1 for r in rows if r.get("label") == 1.0),
        },
        "target": args.target,
        "classification_results": cls_rows,
        "regression_results": reg_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"n_joint_rows={len(rows)} label0={summary['label_counts']['0']} label1={summary['label_counts']['1']}")


if __name__ == "__main__":
    main()
