#!/usr/bin/env python3
"""Group analysis and simple ML baselines for new data_check.xlsx.

No third-party ML dependencies are required beyond numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from qc_check_previous_2026 import parse_first_sheet


CONTINUOUS_COLS = ["edu", "age", "MMSE", "MoCA"]
CATEGORICAL_COLS = ["gender", "学历"]


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


def safe_mean(xs: Sequence[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def safe_median(xs: Sequence[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def safe_std(xs: Sequence[float]) -> Optional[float]:
    return statistics.stdev(xs) if len(xs) >= 2 else None


def permutation_pvalue_mean_diff(
    x0: Sequence[float],
    x1: Sequence[float],
    n_perm: int = 10000,
    seed: int = 42,
) -> Optional[float]:
    if len(x0) < 2 or len(x1) < 2:
        return None
    rng = random.Random(seed)
    pooled = list(x0) + list(x1)
    n0 = len(x0)
    obs = abs(statistics.mean(x1) - statistics.mean(x0))
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        a = pooled[:n0]
        b = pooled[n0:]
        diff = abs(statistics.mean(b) - statistics.mean(a))
        if diff >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def permutation_pvalue_chi_square(
    values: Sequence[str],
    labels: Sequence[int],
    n_perm: int = 10000,
    seed: int = 42,
) -> Optional[float]:
    if len(values) != len(labels) or len(values) < 4:
        return None
    categories = sorted(set(values))
    if len(categories) < 2 or len(set(labels)) < 2:
        return None

    def chisq(vals: Sequence[str], labs: Sequence[int]) -> float:
        cats = categories
        lab_levels = [0, 1]
        table = {
            (c, l): 0 for c in cats for l in lab_levels
        }
        for v, y in zip(vals, labs):
            table[(v, y)] += 1
        row_tot = {c: sum(table[(c, l)] for l in lab_levels) for c in cats}
        col_tot = {l: sum(table[(c, l)] for c in cats) for l in lab_levels}
        n = len(vals)
        score = 0.0
        for c in cats:
            for l in lab_levels:
                exp = row_tot[c] * col_tot[l] / n if n else 0.0
                if exp > 1e-12:
                    score += (table[(c, l)] - exp) ** 2 / exp
        return score

    obs = chisq(values, labels)
    rng = random.Random(seed)
    lab_copy = list(labels)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(lab_copy)
        score = chisq(values, lab_copy)
        if score >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def auc_score(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pairs = sorted(zip(scores, y_true), key=lambda t: t[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        pos_in_tie = sum(y for _, y in pairs[i:j])
        rank_sum += avg_rank * pos_in_tie
        rank += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], scores: Sequence[float]) -> Dict[str, Optional[float]]:
    n = len(y_true)
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / n if n else None
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else None
    bal_acc = None
    if sensitivity is not None and specificity is not None:
        bal_acc = (sensitivity + specificity) / 2.0
    auc = auc_score(y_true, scores)
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Optional[float]]:
    if not y_true:
        return {"rmse": None, "mae": None, "r2": None}
    errors = [a - b for a, b in zip(y_true, y_pred)]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mae = sum(abs(e) for e in errors) / len(errors)
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    return {"rmse": rmse, "mae": mae, "r2": r2}


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


class LogisticRegressionGD:
    def __init__(self, lr: float = 0.05, epochs: int = 2000, l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: Optional[np.ndarray] = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p = X.shape
        self.w = np.zeros(p, dtype=float)
        self.b = 0.0
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p_hat = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            grad_w = (X.T @ (p_hat - y)) / n + self.l2 * self.w
            grad_b = float(np.mean(p_hat - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_score(X) >= 0.5).astype(int)


class LinearSVM:
    def __init__(self, lr: float = 0.02, epochs: int = 2000, c: float = 1.0):
        self.lr = lr
        self.epochs = epochs
        self.c = c
        self.w: Optional[np.ndarray] = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p = X.shape
        y2 = np.where(y > 0, 1.0, -1.0)
        self.w = np.zeros(p, dtype=float)
        self.b = 0.0
        for _ in range(self.epochs):
            margins = y2 * (X @ self.w + self.b)
            mask = margins < 1.0
            grad_w = self.w - self.c * (X[mask].T @ y2[mask]) / n
            grad_b = -self.c * float(np.sum(y2[mask])) / n
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(int)


class LinearRegressionClosedForm:
    def __init__(self, l2: float = 1e-6):
        self.l2 = l2
        self.beta: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X1 = np.column_stack([np.ones(len(X)), X])
        reg = self.l2 * np.eye(X1.shape[1])
        reg[0, 0] = 0.0
        self.beta = np.linalg.pinv(X1.T @ X1 + reg) @ X1.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.beta is not None
        X1 = np.column_stack([np.ones(len(X)), X])
        return X1 @ self.beta


@dataclass
class TreeNode:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[float] = None


class DecisionTree:
    def __init__(
        self,
        task: str,
        max_depth: int = 5,
        min_samples_split: int = 6,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root: Optional[TreeNode] = None
        self.rng = random.Random(random_state)

    def _leaf_value(self, y: np.ndarray) -> float:
        if self.task == "classification":
            return float(np.mean(y))
        return float(np.mean(y))

    def _score(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            p = float(np.mean(y))
            return 1.0 - p * p - (1.0 - p) * (1.0 - p)
        mean_y = float(np.mean(y))
        return float(np.mean((y - mean_y) ** 2))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        n, p = X.shape
        feature_ids = list(range(p))
        if self.max_features is not None and self.max_features < p:
            feature_ids = self.rng.sample(feature_ids, self.max_features)
        parent_score = self._score(y)
        best_gain = 0.0
        best_feature = None
        best_thresh = None
        for j in feature_ids:
            col = X[:, j]
            uniq = np.unique(col)
            if len(uniq) <= 1:
                continue
            if len(uniq) > 20:
                quantiles = np.quantile(uniq, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                thresholds = np.unique(quantiles)
            else:
                thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for t in thresholds:
                left = col <= t
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue
                score = (left.sum() / n) * self._score(y[left]) + (right.sum() / n) * self._score(y[right])
                gain = parent_score - score
                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_thresh = float(t)
        return best_feature, best_thresh, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode(value=self._leaf_value(y))
        if depth >= self.max_depth or len(y) < self.min_samples_split or self._score(y) < 1e-12:
            return node
        feat, thresh, gain = self._best_split(X, y)
        if feat is None or thresh is None or gain <= 1e-12:
            return node
        mask = X[:, feat] <= thresh
        node.feature = feat
        node.threshold = thresh
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root = self._build(X, y, 0)

    def _predict_one(self, node: TreeNode, x: np.ndarray) -> float:
        while node.feature is not None and node.left is not None and node.right is not None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        assert node.value is not None
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        return np.array([self._predict_one(self.root, x) for x in X], dtype=float)


class RandomForest:
    def __init__(
        self,
        task: str,
        n_estimators: int = 200,
        max_depth: int = 5,
        min_samples_split: int = 6,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees: List[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.trees = []
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = DecisionTree(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + i,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([t.predict(X) for t in self.trees])
        return preds.mean(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_score(X)
        if self.task == "classification":
            return (scores >= 0.5).astype(int)
        return scores


def make_stratified_folds(y: Sequence[int], n_splits: int = 5, seed: int = 42) -> List[List[int]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(y):
        by_class[int(label)].append(i)
    folds = [[] for _ in range(n_splits)]
    for idxs in by_class.values():
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            folds[i % n_splits].append(idx)
    for fold in folds:
        fold.sort()
    return folds


def make_kfold(n: int, n_splits: int = 5, seed: int = 42) -> List[List[int]]:
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    folds = [[] for _ in range(n_splits)]
    for i, idx in enumerate(idxs):
        folds[i % n_splits].append(idx)
    for fold in folds:
        fold.sort()
    return folds


def aggregate_metric_dicts(rows: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    keys = rows[0].keys() if rows else []
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = statistics.mean(vals) if vals else None
    return out


def make_category_levels(rows: Sequence[Dict[str, object]], col: str) -> List[str]:
    levels = sorted({str(r.get(col, "")).strip() for r in rows if str(r.get(col, "")).strip()})
    return levels


def encode_rows(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    gender_levels = make_category_levels(rows, "gender")
    edu_levels = make_category_levels(rows, "学历")
    feature_names = (
        ["edu", "age", "MoCA"]
        + [f"gender::{level}" for level in gender_levels]
        + [f"学历::{level}" for level in edu_levels]
    )
    X = []
    y_cls = []
    y_reg = []
    for row in rows:
        label = to_float(row.get("label"))
        mmse = to_float(row.get("MMSE"))
        edu = to_float(row.get("edu"))
        age = to_float(row.get("age"))
        moca = to_float(row.get("MoCA"))
        gender_raw = str(row.get("gender", "")).strip()
        edu_raw = str(row.get("学历", "")).strip()
        if None in (label, mmse, edu, age, moca):
            continue
        feats = [float(edu), float(age), float(moca)]
        feats.extend(1.0 if gender_raw == level else 0.0 for level in gender_levels)
        feats.extend(1.0 if edu_raw == level else 0.0 for level in edu_levels)
        X.append(feats)
        y_cls.append(int(label))
        y_reg.append(float(mmse))
    return np.array(X, dtype=float), np.array(y_cls, dtype=int), np.array(y_reg, dtype=float), feature_names


def classification_cv(X: np.ndarray, y: np.ndarray) -> List[Dict[str, object]]:
    folds = make_stratified_folds(y.tolist(), n_splits=5, seed=42)
    results = []
    models = [
        ("logistic_regression", LogisticRegressionGD(lr=0.05, epochs=2500, l2=1e-3)),
        ("linear_svm", LinearSVM(lr=0.02, epochs=2500, c=1.0)),
        ("random_forest", RandomForest(task="classification", n_estimators=200, max_depth=5, min_samples_split=6, max_features=2, random_state=42)),
    ]
    for name, model in models:
        fold_metrics: List[Dict[str, Optional[float]]] = []
        for test_idx in folds:
            train_idx = [i for i in range(len(X)) if i not in test_idx]
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            X_train_z, X_test_z, _, _ = standardize_train_test(X_train, X_test)
            model.fit(X_train_z, y_train)
            if hasattr(model, "predict_score"):
                scores = model.predict_score(X_test_z)
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test_z)  # type: ignore[attr-defined]
            else:
                scores = model.predict(X_test_z)
            y_pred = model.predict(X_test_z)
            fold_metrics.append(classification_metrics(y_test.tolist(), y_pred.tolist(), np.asarray(scores).tolist()))
        results.append({"model": name, **aggregate_metric_dicts(fold_metrics)})
    return results


def regression_cv(X: np.ndarray, y: np.ndarray) -> List[Dict[str, object]]:
    folds = make_kfold(len(X), n_splits=5, seed=42)
    results = []
    models = [
        ("linear_regression", LinearRegressionClosedForm(l2=1e-6)),
        ("random_forest_regressor", RandomForest(task="regression", n_estimators=200, max_depth=5, min_samples_split=6, max_features=2, random_state=42)),
    ]
    for name, model in models:
        fold_metrics: List[Dict[str, Optional[float]]] = []
        for test_idx in folds:
            train_idx = [i for i in range(len(X)) if i not in test_idx]
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            X_train_z, X_test_z, _, _ = standardize_train_test(X_train, X_test)
            model.fit(X_train_z, y_train)
            y_pred = np.asarray(model.predict(X_test_z), dtype=float)
            fold_metrics.append(regression_metrics(y_test.tolist(), y_pred.tolist()))
        results.append({"model": name, **aggregate_metric_dicts(fold_metrics)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze new data_check.xlsx")
    parser.add_argument("--xlsx", default="../new data_check.xlsx")
    parser.add_argument("--out-dir", default="new_data_check_analysis")
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers, rows = parse_first_sheet(Path(args.xlsx))
    usable_rows = [r for r in rows if str(r.get("label", "")).strip() in {"0", "1"}]

    label0 = [r for r in usable_rows if str(r.get("label", "")).strip() == "0"]
    label1 = [r for r in usable_rows if str(r.get("label", "")).strip() == "1"]

    group_summary_rows: List[List[object]] = []
    for col in CONTINUOUS_COLS:
        x0 = [to_float(r.get(col)) for r in label0]
        x1 = [to_float(r.get(col)) for r in label1]
        x0 = [x for x in x0 if x is not None]
        x1 = [x for x in x1 if x is not None]
        p = permutation_pvalue_mean_diff(x0, x1, n_perm=args.n_perm, seed=42)
        group_summary_rows.append(
            [
                col,
                len(x0),
                safe_mean(x0),
                safe_std(x0),
                safe_median(x0),
                len(x1),
                safe_mean(x1),
                safe_std(x1),
                safe_median(x1),
                None if safe_mean(x0) is None or safe_mean(x1) is None else safe_mean(x1) - safe_mean(x0),
                p,
            ]
        )

    categorical_rows: List[List[object]] = []
    for col in CATEGORICAL_COLS:
        vals = [str(r.get(col, "")).strip() for r in usable_rows]
        labs = [int(str(r.get("label", "")).strip()) for r in usable_rows]
        p = permutation_pvalue_chi_square(vals, labs, n_perm=args.n_perm, seed=42)
        c0 = Counter(str(r.get(col, "")).strip() for r in label0)
        c1 = Counter(str(r.get(col, "")).strip() for r in label1)
        cats = sorted(set(c0) | set(c1))
        for cat in cats:
            categorical_rows.append([col, cat, c0.get(cat, 0), c1.get(cat, 0), p])

    X, y_cls, y_reg, feature_names = encode_rows(usable_rows)
    classification_results = classification_cv(X, y_cls)
    regression_results = regression_cv(X, y_reg)

    with (out_dir / "continuous_group_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
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
            ]
        )
        w.writerows(group_summary_rows)

    with (out_dir / "categorical_group_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "category", "count_label0", "count_label1", "permutation_pvalue"])
        w.writerows(categorical_rows)

    with (out_dir / "classification_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"])
        for row in classification_results:
            w.writerow([row.get(k) for k in ["model", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"]])

    with (out_dir / "regression_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "rmse", "mae", "r2"])
        for row in regression_results:
            w.writerow([row.get(k) for k in ["model", "rmse", "mae", "r2"]])

    summary = {
        "xlsx": str(Path(args.xlsx).resolve()),
        "row_count_total": len(rows),
        "row_count_used": len(usable_rows),
        "label_counts": {"0": len(label0), "1": len(label1)},
        "features_for_modeling": feature_names,
        "classification_results": classification_results,
        "regression_results": regression_results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Output dir: {out_dir.resolve()}")
    print(f"used_rows={len(usable_rows)} label0={len(label0)} label1={len(label1)}")


if __name__ == "__main__":
    main()
