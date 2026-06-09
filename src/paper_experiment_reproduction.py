#!/usr/bin/env python3
"""Paper-level experiment reproduction with repeated CV, OOF ROC, and SHAP.

This script is adapted from the manuscript experiment script
`xp_hand_vi_au_250819.py`, but removes hard-coded local paths and writes all
tables/figures into a user-provided output directory.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    r2_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42

SCRIPT_ALL_FEATURES = [
    "age",
    "edu",
    "ACC_DT",
    "Correct_persec_DT",
    "Correct_Num_DT",
    "Num_DT",
    "Timecount_DT",
    "Timecount_DT_1stNum",
    "Num_20s_DT",
    "Correct_Num_20s_DT",
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

# Original manuscript ADualselected definition:
# {v1, v10, v12, v13, v16, a1-a7, education, age}.
# Kept for backward-compatible re-analysis; it is no longer the default
# manuscript panel because it was not selected inside the nested CV protocol.
ADUALSELECTED_FEATURES = [
    "Timecount_DT",  # v1: dual-task total time
    "average_step_assymetry",  # v10
    "stance_times",  # v12
    "swing_times",  # v13
    "walking_speed",  # v16
    "Num_DT",  # a1
    "Correct_Num_DT",  # a2
    "Num_20s_DT",  # a3
    "Correct_Num_20s_DT",  # a4
    "Correct_persec_DT",  # a5
    "ACC_DT",  # a6
    "Timecount_DT_1stNum",  # a7
    "edu",
    "age",
]

# Fixed panel derived from the nested feature-selection frequency analysis
# on dataset_2606: top 14 features across 5-fold x 10 repeated training-fold
# rankings. For unbiased model performance, still report the nested CV output.
NESTED_SELECTED_FEATURES = [
    "ACC_DT",
    "Correct_Num_20s_DT",
    "Correct_Num_DT",
    "Correct_persec_DT",
    "Num_20s_DT",
    "Num_DT",
    "Timecount_DT_1stNum",
    "age",
    "edu",
    "swing_times",
    "average_step_assymetry",
    "average_stride_assymetry",
    "stance_times",
    "average_stride_length",
]


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,;\n]", value) if x.strip()]


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def read_table(path: Path, sheet_name: str) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    return pd.read_excel(path, sheet_name=sheet_name)


def select_features(df: pd.DataFrame, feature_set: str, custom_features: str) -> List[str]:
    if feature_set == "script_all":
        features = SCRIPT_ALL_FEATURES
    elif feature_set == "nestedselected":
        features = NESTED_SELECTED_FEATURES
    elif feature_set == "adualselected":
        features = ADUALSELECTED_FEATURES
    elif feature_set == "custom":
        features = parse_csv_list(custom_features)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    missing = [x for x in features if x not in df.columns]
    if missing:
        raise ValueError(
            "Missing feature columns required by "
            f"{feature_set}: {missing}\nAvailable columns: {list(df.columns)}"
        )
    return features


def select_candidate_features(df: pd.DataFrame, candidate_set: str, custom_features: str) -> List[str]:
    return select_features(df, candidate_set, custom_features)


def prepare_xy(
    df: pd.DataFrame,
    features: Sequence[str],
    label_col: str,
    id_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    data = df.copy()
    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[label_col] = pd.to_numeric(data[label_col], errors="coerce")

    keep = data[label_col].notna()
    keep &= data[list(features)].notna().any(axis=1)
    data = data.loc[keep].copy()
    if data.empty:
        raise ValueError("No usable rows after numeric conversion and missing-value filtering.")

    y = data[label_col].astype(int).to_numpy()
    unique = sorted(set(y.tolist()))
    if unique != [0, 1]:
        raise ValueError(f"Classification label must contain binary 0/1 values; got {unique}")

    X = data[list(features)].to_numpy(dtype=float)
    ids = data[id_col].astype(str).tolist() if id_col in data.columns else [str(i) for i in data.index]
    return data, X, y, ids


def prepare_regression_xy(
    df: pd.DataFrame,
    features: Sequence[str],
    target_col: str,
    id_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Regression target column not found: {target_col}")

    data = df.copy()
    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")

    keep = data[target_col].notna()
    keep &= data[list(features)].notna().any(axis=1)
    data = data.loc[keep].copy()
    if data.empty:
        raise ValueError("No usable rows for regression after numeric conversion and missing-value filtering.")

    X = data[list(features)].to_numpy(dtype=float)
    y = data[target_col].to_numpy(dtype=float)
    ids = data[id_col].astype(str).tolist() if id_col in data.columns else [str(i) for i in data.index]
    return data, X, y, ids


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_pred.tolist())) > 1 else 0.0,
        "thr": float(thr),
    }


def youden_j_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    order = np.argsort(-j)
    for idx in order:
        thr = float(thresholds[idx])
        if math.isfinite(thr):
            return thr
    return 0.5


def make_adaboost(stump: DecisionTreeClassifier) -> AdaBoostClassifier:
    kwargs = {"n_estimators": 200, "learning_rate": 0.1, "random_state": RANDOM_STATE}
    if "estimator" in inspect.signature(AdaBoostClassifier).parameters:
        kwargs["estimator"] = stump
    else:
        kwargs["base_estimator"] = stump
    return AdaBoostClassifier(**kwargs)


def make_models(use_robust: bool = False) -> Dict[str, object]:
    scaler = RobustScaler() if use_robust else StandardScaler()

    def scaled_pipe(clf: object) -> Pipeline:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", scaler), ("clf", clf)])

    def tree_pipe(clf: object) -> Pipeline:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", clf)])

    m_lr = scaled_pipe(
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
    )
    m_svm = scaled_pipe(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))
    m_knn = scaled_pipe(KNeighborsClassifier())
    m_lda = scaled_pipe(LinearDiscriminantAnalysis())
    m_mlp = scaled_pipe(
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=1e-3,
            early_stopping=True,
            max_iter=500,
            random_state=RANDOM_STATE,
        )
    )
    m_dt = tree_pipe(DecisionTreeClassifier(min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE))
    m_rf = tree_pipe(
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    )
    m_gbdt = tree_pipe(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE))
    m_ada = tree_pipe(make_adaboost(DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)))
    m_hgb = tree_pipe(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=RANDOM_STATE))
    m_nb = tree_pipe(GaussianNB())

    m_vote = VotingClassifier(
        estimators=[("lr", clone(m_lr)), ("svm", clone(m_svm)), ("rf", clone(m_rf)), ("gbdt", clone(m_gbdt))],
        voting="soft",
        n_jobs=1,
    )
    m_stack = StackingClassifier(
        estimators=[("lr", clone(m_lr)), ("svm", clone(m_svm)), ("rf", clone(m_rf))],
        final_estimator=LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
        ),
        passthrough=False,
        cv=5,
        n_jobs=1,
    )

    return {
        "Logistic Regression": m_lr,
        "SVM (RBF)": m_svm,
        "KNN": m_knn,
        "Decision Tree": m_dt,
        "Random Forest": m_rf,
        "Naive Bayes": m_nb,
        "LDA": m_lda,
        "GBDT": m_gbdt,
        "AdaBoost": m_ada,
        "HistGB": m_hgb,
        "MLP (shallow)": m_mlp,
        "Soft Voting": m_vote,
        "Stacking": m_stack,
    }


def make_regression_models(use_robust: bool = False) -> Dict[str, object]:
    scaler = RobustScaler() if use_robust else StandardScaler()

    def scaled_pipe(model: object) -> Pipeline:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", scaler), ("model", model)])

    def tree_pipe(model: object) -> Pipeline:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("model", model)])

    return {
        "Linear Regression": scaled_pipe(LinearRegression()),
        "SVR (RBF)": scaled_pipe(SVR(kernel="rbf")),
        "Random Forest Regressor": tree_pipe(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "GBDT Regressor": tree_pipe(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE,
            )
        ),
    }


def make_publication_classification_models(use_robust: bool = False) -> Dict[str, object]:
    scaler = RobustScaler() if use_robust else StandardScaler()

    def scaled_pipe(clf: object) -> Pipeline:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", scaler), ("clf", clf)])

    def tree_pipe(clf: object) -> Pipeline:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", clf)])

    return {
        "AdaBoost": tree_pipe(make_adaboost(DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE))),
        "SVM (RBF)": scaled_pipe(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        "RF": tree_pipe(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "DT": tree_pipe(DecisionTreeClassifier(min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE)),
    }


def has_balanced_class_weight(model: object) -> bool:
    params = model.get_params() if hasattr(model, "get_params") else {}
    return any(k.endswith("class_weight") and v == "balanced" for k, v in params.items())


def safe_fit(model: object, X_train: np.ndarray, y_train: np.ndarray) -> object:
    if not has_balanced_class_weight(model):
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        for key in ("clf__sample_weight", "sample_weight"):
            try:
                model.fit(X_train, y_train, **{key: weights})
                return model
            except TypeError:
                continue
            except Exception:
                break
    model.fit(X_train, y_train)
    return model


def predict_probability(estimator: object, X_test: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        s_min, s_max = float(np.min(scores)), float(np.max(scores))
        return (scores - s_min) / (s_max - s_min + 1e-12)
    return estimator.predict(X_test).astype(float)


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def summarize_repeated_metrics(
    metric_df: pd.DataFrame,
    metrics: Sequence[str],
    sort_cols: Sequence[str],
    ascending: Sequence[bool],
    bootstrap_repeats: int,
    bootstrap_ci: float,
    group_cols: Sequence[str] = ("model",),
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group_idx, (keys, group) in enumerate(metric_df.groupby(list(group_cols), dropna=False)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = {col: key for col, key in zip(group_cols, keys)}
        for metric_idx, metric in enumerate(metrics):
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"std_{metric}"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            lo, hi = bootstrap_mean_ci(
                vals,
                n_bootstrap=bootstrap_repeats,
                ci=bootstrap_ci,
                seed=RANDOM_STATE + group_idx * 1000 + metric_idx,
            )
            pct = int(round(bootstrap_ci * 100))
            row[f"ci{pct}_low_{metric}"] = lo
            row[f"ci{pct}_high_{metric}"] = hi
        rows.append(row)

    summary = pd.DataFrame(rows)
    available_sort_cols = [c for c in sort_cols if c in summary.columns]
    available_ascending = [ascending[i] for i, c in enumerate(sort_cols) if c in summary.columns]
    if available_sort_cols:
        summary = summary.sort_values(available_sort_cols, ascending=available_ascending)
    return summary.reset_index(drop=True)


def evaluate_all_models(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    n_splits: int,
    n_repeats: int,
    use_robust: bool,
    bootstrap_repeats: int,
    bootstrap_ci: float,
) -> pd.DataFrame:
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    for model_name, proto in make_models(use_robust=use_robust).items():
        for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            try:
                est = safe_fit(clone(proto), X_tr, y_tr)
                y_prob = predict_probability(est, X_te)
                thr_j = youden_j_threshold(y_te, y_prob)
                default = metrics_at_threshold(y_te, y_prob, 0.5)
                youden = metrics_at_threshold(y_te, y_prob, thr_j)
                rows.append(
                    {
                        "model": model_name,
                        "fold": fold_id,
                        "roc_auc": roc_auc_score(y_te, y_prob),
                        "pr_auc": average_precision_score(y_te, y_prob),
                        "brier": brier_score_loss(y_te, y_prob),
                        "f1": default["f1"],
                        "recall": default["recall"],
                        "specificity": default["specificity"],
                        "bal_acc": default["bal_acc"],
                        "mcc": default["mcc"],
                        "f1@J": youden["f1"],
                        "recall@J": youden["recall"],
                        "specificity@J": youden["specificity"],
                        "bal_acc@J": youden["bal_acc"],
                        "mcc@J": youden["mcc"],
                        "thr@J": youden["thr"],
                    }
                )
            except Exception as exc:
                errors.append({"model": model_name, "fold": fold_id, "error": str(exc)})

    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        raise RuntimeError(f"All model folds failed. First errors: {errors[:5]}")

    metrics = [c for c in fold_df.columns if c not in {"model", "fold"}]
    summary = summarize_repeated_metrics(
        fold_df,
        metrics,
        sort_cols=["mean_roc_auc", "mean_pr_auc"],
        ascending=[False, False],
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_ci=bootstrap_ci,
    )

    fold_df.to_csv(out_dir / "classification_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "classification_repeated_cv.csv", index=False)
    summary.to_excel(out_dir / "classification_repeated_cv.xlsx", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "classification_fold_errors.csv", index=False)
    return summary


def evaluate_stratified_train_test_models(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    n_repeats: int,
    test_size: float,
    use_robust: bool,
    bootstrap_repeats: int,
    bootstrap_ci: float,
) -> pd.DataFrame:
    splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=RANDOM_STATE)
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    for model_name, proto in make_models(use_robust=use_robust).items():
        for split_id, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            try:
                est = safe_fit(clone(proto), X_tr, y_tr)
                y_prob = predict_probability(est, X_te)
                thr_j = youden_j_threshold(y_te, y_prob)
                default = metrics_at_threshold(y_te, y_prob, 0.5)
                youden = metrics_at_threshold(y_te, y_prob, thr_j)
                rows.append(
                    {
                        "model": model_name,
                        "split": split_id,
                        "roc_auc": roc_auc_score(y_te, y_prob),
                        "pr_auc": average_precision_score(y_te, y_prob),
                        "brier": brier_score_loss(y_te, y_prob),
                        "f1": default["f1"],
                        "recall": default["recall"],
                        "specificity": default["specificity"],
                        "bal_acc": default["bal_acc"],
                        "mcc": default["mcc"],
                        "f1@J": youden["f1"],
                        "recall@J": youden["recall"],
                        "specificity@J": youden["specificity"],
                        "bal_acc@J": youden["bal_acc"],
                        "mcc@J": youden["mcc"],
                        "thr@J": youden["thr"],
                    }
                )
            except Exception as exc:
                errors.append({"model": model_name, "split": split_id, "error": str(exc)})

    split_df = pd.DataFrame(rows)
    if split_df.empty:
        raise RuntimeError(f"All train-test split runs failed. First errors: {errors[:5]}")

    metrics = [c for c in split_df.columns if c not in {"model", "split"}]
    summary = summarize_repeated_metrics(
        split_df,
        metrics,
        sort_cols=["mean_roc_auc", "mean_pr_auc"],
        ascending=[False, False],
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_ci=bootstrap_ci,
    )

    split_df.to_csv(out_dir / "classification_train_test_split_metrics.csv", index=False)
    summary.to_csv(out_dir / "classification_train_test_split_summary.csv", index=False)
    summary.to_excel(out_dir / "classification_train_test_split_summary.xlsx", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "classification_train_test_split_errors.csv", index=False)
    return summary


def publication_feature_sets() -> Dict[str, List[str]]:
    return {
        "all_features": SCRIPT_ALL_FEATURES,
        "Nested-selected": NESTED_SELECTED_FEATURES,
    }


def infer_column(df: pd.DataFrame, requested: str, candidates: Sequence[str]) -> Optional[str]:
    if requested and requested in df.columns:
        return requested
    lower_map = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = lower_map.get(candidate.lower())
        if found:
            return found
    return None


def add_numeric_covariate(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    column: Optional[str],
    out_name: str,
) -> List[str]:
    if not column or column not in source.columns:
        warnings.warn(f"Confounder column not found and skipped: {out_name}")
        return []
    frame[out_name] = pd.to_numeric(source[column], errors="coerce")
    return [out_name]


def add_sex_covariates(frame: pd.DataFrame, source: pd.DataFrame, sex_col: Optional[str]) -> List[str]:
    if not sex_col or sex_col not in source.columns:
        warnings.warn("Sex/gender column not found and skipped.")
        return []
    raw = source[sex_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    non_missing = raw.notna() & (raw.astype(str).str.strip() != "")
    if numeric[non_missing].notna().all():
        frame["sex"] = numeric
        return ["sex"]

    cleaned = raw.fillna("missing").astype(str).str.strip().replace("", "missing")
    dummies = pd.get_dummies(cleaned, prefix="sex", drop_first=True, dtype=float)
    if dummies.empty:
        return []
    for col in dummies.columns:
        frame[col] = dummies[col].astype(float)
    return dummies.columns.tolist()


def make_confounder_adjusted_feature_sets(
    df: pd.DataFrame,
    selected_features: Sequence[str],
    age_col: str,
    education_col: str,
    sex_col: str,
) -> Tuple[Dict[str, Tuple[pd.DataFrame, List[str]]], List[Dict[str, object]]]:
    age_source = infer_column(df, age_col, ["age", "Age", "AGE"])
    edu_source = infer_column(df, education_col, ["edu", "education", "Education", "years_of_education", "学历年限"])
    sex_source = infer_column(df, sex_col, ["sex", "Sex", "gender", "Gender", "性别"])
    confounder_source_cols = {x for x in [age_source, edu_source, sex_source] if x}
    confounder_names = {"age", "edu", "education", "sex", "gender", "Gender", "Sex", "性别"}
    base_features = [
        feature
        for feature in selected_features
        if feature not in confounder_source_cols and feature.lower() not in {x.lower() for x in confounder_names}
    ]
    if not base_features:
        raise ValueError("No biomarker features remain after removing age/education/sex confounders.")

    out: Dict[str, Tuple[pd.DataFrame, List[str]]] = {}
    manifest: List[Dict[str, object]] = []

    def build(name: str, include_age: bool, include_edu: bool, include_sex: bool) -> None:
        frame = df.copy()
        feature_cols = list(base_features)
        covariate_cols: List[str] = []
        if include_age:
            covariate_cols.extend(add_numeric_covariate(frame, df, age_source, "age"))
        if include_edu:
            covariate_cols.extend(add_numeric_covariate(frame, df, edu_source, "edu"))
        if include_sex:
            covariate_cols.extend(add_sex_covariates(frame, df, sex_source))
        feature_cols.extend(covariate_cols)
        out[name] = (frame, feature_cols)
        manifest.append(
            {
                "adjustment_set": name,
                "n_biomarker_features": len(base_features),
                "covariates_requested": ",".join(
                    [x for x, flag in [("age", include_age), ("education", include_edu), ("sex", include_sex)] if flag]
                ),
                "covariate_columns_included": ",".join(covariate_cols),
                "source_age_col": age_source or "",
                "source_education_col": edu_source or "",
                "source_sex_col": sex_source or "",
                "feature_columns": ",".join(feature_cols),
            }
        )

    build("unadjusted_biomarkers", include_age=False, include_edu=False, include_sex=False)
    build("age_adjusted", include_age=True, include_edu=False, include_sex=False)
    build("education_adjusted", include_age=False, include_edu=True, include_sex=False)
    build("age_sex_education_adjusted", include_age=True, include_edu=True, include_sex=True)
    return out, manifest


def evaluate_publication_classification(
    df: pd.DataFrame,
    label_col: str,
    id_col: str,
    out_dir: Path,
    n_repeats: int,
    test_size: float,
    use_robust: bool,
    bootstrap_repeats: int,
    bootstrap_ci: float,
    requested_models: Sequence[str],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_models = make_publication_classification_models(use_robust=use_robust)
    model_names = [name for name in requested_models if name in all_models]
    missing_models = [name for name in requested_models if name not in all_models]
    if missing_models:
        warnings.warn(f"Publication models not found and skipped: {missing_models}")
    if not model_names:
        raise ValueError("No valid publication classification models selected.")

    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for feature_set_name, feature_list in publication_feature_sets().items():
        missing_features = [col for col in feature_list if col not in df.columns]
        if missing_features:
            raise ValueError(
                f"Missing columns for publication feature set {feature_set_name}: {missing_features}"
            )
        dataset, X, y, ids = prepare_xy(df, feature_list, label_col, id_col)
        dataset_cols = ([id_col] if id_col in dataset.columns else []) + [label_col] + feature_list
        dataset[dataset_cols].to_csv(out_dir / f"dataset_{safe_name(feature_set_name)}.csv", index=False)

        splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=RANDOM_STATE)
        splits = list(splitter.split(X, y))
        for model_name in model_names:
            proto = all_models[model_name]
            for split_id, (tr_idx, te_idx) in enumerate(splits, start=1):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                try:
                    est = safe_fit(clone(proto), X_tr, y_tr)
                    y_prob = predict_probability(est, X_te)
                    y_pred = (y_prob >= 0.5).astype(int)
                    metric_rows.append(
                        {
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "split": split_id,
                            "auroc": roc_auc_score(y_te, y_prob),
                            "acc": accuracy_score(y_te, y_pred),
                            "precision": precision_score(y_te, y_pred, zero_division=0),
                            "recall": recall_score(y_te, y_pred, zero_division=0),
                            "f1": f1_score(y_te, y_pred, zero_division=0),
                        }
                    )
                    for local_idx, case_idx in enumerate(te_idx):
                        prediction_rows.append(
                            {
                                "feature_set": feature_set_name,
                                "model": model_name,
                                "split": split_id,
                                "ID": ids[case_idx],
                                "y_true": int(y_te[local_idx]),
                                "y_prob": float(y_prob[local_idx]),
                                "y_pred": int(y_pred[local_idx]),
                            }
                        )
                except Exception as exc:
                    error_rows.append(
                        {
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "split": split_id,
                            "error": str(exc),
                        }
                    )

    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        raise RuntimeError(f"All publication classification runs failed. First errors: {error_rows[:5]}")

    summary = summarize_repeated_metrics(
        metric_df,
        metrics=["auroc", "acc", "precision", "recall", "f1"],
        sort_cols=["feature_set", "mean_auroc"],
        ascending=[True, False],
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_ci=bootstrap_ci,
        group_cols=("feature_set", "model"),
    )
    metric_df.to_csv(out_dir / "model_performance_split_metrics.csv", index=False)
    metric_df.to_csv(out_dir / "fixed_panel_model_performance_split_metrics.csv", index=False)
    summary.to_csv(out_dir / "model_performance_table.csv", index=False)
    summary.to_excel(out_dir / "model_performance_table.xlsx", index=False)
    summary.to_csv(out_dir / "fixed_panel_model_performance_table.csv", index=False)
    summary.to_excel(out_dir / "fixed_panel_model_performance_table.xlsx", index=False)
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(out_dir / "publication_predictions.csv", index=False)
    pred_df.to_csv(out_dir / "fixed_panel_publication_predictions.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(out_dir / "publication_classification_errors.csv", index=False)

    write_publication_confusion_matrices(pred_df, out_dir)
    write_publication_roc_curves(pred_df, out_dir)
    write_publication_calibration_curves(pred_df, out_dir)
    return summary


def evaluate_confounder_adjustment(
    df: pd.DataFrame,
    selected_features: Sequence[str],
    label_col: str,
    id_col: str,
    age_col: str,
    education_col: str,
    sex_col: str,
    out_dir: Path,
    n_repeats: int,
    test_size: float,
    use_robust: bool,
    bootstrap_repeats: int,
    bootstrap_ci: float,
    requested_models: Sequence[str],
    stability_margin: float,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    adjusted_sets, manifest = make_confounder_adjusted_feature_sets(
        df, selected_features, age_col, education_col, sex_col
    )
    pd.DataFrame(manifest).to_csv(out_dir / "confounder_feature_manifest.csv", index=False)

    all_models = make_publication_classification_models(use_robust=use_robust)
    model_names = [name for name in requested_models if name in all_models]
    missing_models = [name for name in requested_models if name not in all_models]
    if missing_models:
        warnings.warn(f"Confounder adjustment models not found and skipped: {missing_models}")
    if not model_names:
        raise ValueError("No valid confounder adjustment models selected.")

    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for adjustment_set, (feature_df, feature_cols) in adjusted_sets.items():
        dataset, X, y, ids = prepare_xy(feature_df, feature_cols, label_col, id_col)
        dataset_cols = ([id_col] if id_col in dataset.columns else []) + [label_col] + feature_cols
        dataset[dataset_cols].to_csv(out_dir / f"dataset_{safe_name(adjustment_set)}.csv", index=False)
        splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=RANDOM_STATE)
        splits = list(splitter.split(X, y))

        for model_name in model_names:
            proto = all_models[model_name]
            for split_id, (tr_idx, te_idx) in enumerate(splits, start=1):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                try:
                    est = safe_fit(clone(proto), X_tr, y_tr)
                    y_prob = predict_probability(est, X_te)
                    y_pred = (y_prob >= 0.5).astype(int)
                    metric_rows.append(
                        {
                            "adjustment_set": adjustment_set,
                            "model": model_name,
                            "split": split_id,
                            "auroc": roc_auc_score(y_te, y_prob),
                            "acc": accuracy_score(y_te, y_pred),
                            "precision": precision_score(y_te, y_pred, zero_division=0),
                            "recall": recall_score(y_te, y_pred, zero_division=0),
                            "f1": f1_score(y_te, y_pred, zero_division=0),
                        }
                    )
                    for local_idx, case_idx in enumerate(te_idx):
                        prediction_rows.append(
                            {
                                "adjustment_set": adjustment_set,
                                "model": model_name,
                                "split": split_id,
                                "ID": ids[case_idx],
                                "y_true": int(y_te[local_idx]),
                                "y_prob": float(y_prob[local_idx]),
                                "y_pred": int(y_pred[local_idx]),
                            }
                        )
                except Exception as exc:
                    error_rows.append(
                        {
                            "adjustment_set": adjustment_set,
                            "model": model_name,
                            "split": split_id,
                            "error": str(exc),
                        }
                    )

    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        raise RuntimeError(f"All confounder-adjusted runs failed. First errors: {error_rows[:5]}")

    summary = summarize_repeated_metrics(
        metric_df,
        metrics=["auroc", "acc", "precision", "recall", "f1"],
        sort_cols=["model", "adjustment_set"],
        ascending=[True, True],
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_ci=bootstrap_ci,
        group_cols=("adjustment_set", "model"),
    )
    metric_df.to_csv(out_dir / "confounder_adjusted_split_metrics.csv", index=False)
    summary.to_csv(out_dir / "confounder_adjusted_performance_table.csv", index=False)
    summary.to_excel(out_dir / "confounder_adjusted_performance_table.xlsx", index=False)
    pd.DataFrame(prediction_rows).to_csv(out_dir / "confounder_adjusted_predictions.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(out_dir / "confounder_adjustment_errors.csv", index=False)

    stability = build_auroc_stability_table(summary, stability_margin)
    stability.to_csv(out_dir / "auroc_stability_after_adjustment.csv", index=False)
    stability.to_excel(out_dir / "auroc_stability_after_adjustment.xlsx", index=False)
    plot_auroc_stability(summary, out_dir)
    return summary


def impute_for_univariate_ranking(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(X_train), med, X_train), med


def select_top_k_features_train_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    k: int,
) -> Tuple[List[int], pd.DataFrame]:
    X_imp, _ = impute_for_univariate_ranking(X_train)
    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(feature_names):
        values = X_imp[:, idx]
        if len(np.unique(values)) <= 1:
            auc = 0.5
        else:
            try:
                auc = float(roc_auc_score(y_train, values))
            except Exception:
                auc = 0.5
        score = abs(auc - 0.5)
        direction = 1 if auc >= 0.5 else -1
        rows.append(
            {
                "feature": name,
                "train_auc": auc,
                "selection_score_abs_auc_minus_0.5": score,
                "direction": direction,
            }
        )
    rank_df = pd.DataFrame(rows).sort_values(
        ["selection_score_abs_auc_minus_0.5", "feature"], ascending=[False, True]
    )
    selected_names = rank_df.head(max(1, min(k, len(rank_df))))["feature"].tolist()
    selected_idx = [feature_names.index(name) for name in selected_names]
    return selected_idx, rank_df.reset_index(drop=True)


def evaluate_nested_feature_selection_cv(
    df: pd.DataFrame,
    label_col: str,
    id_col: str,
    candidate_set: str,
    custom_features: str,
    select_k: int,
    out_dir: Path,
    n_splits: int,
    n_repeats: int,
    use_robust: bool,
    bootstrap_repeats: int,
    bootstrap_ci: float,
    requested_models: Sequence[str],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_features = select_candidate_features(df, candidate_set, custom_features)
    dataset, X, y, ids = prepare_xy(df, candidate_features, label_col, id_col)
    dataset_cols = ([id_col] if id_col in dataset.columns else []) + [label_col] + candidate_features
    dataset[dataset_cols].to_csv(out_dir / "nested_feature_selection_dataset.csv", index=False)

    all_models = make_publication_classification_models(use_robust=use_robust)
    model_names = [name for name in requested_models if name in all_models]
    if not model_names:
        raise ValueError("No valid models selected for nested feature selection.")

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
        selected_idx, rank_df = select_top_k_features_train_only(
            X[tr_idx],
            y[tr_idx],
            candidate_features,
            select_k,
        )
        selected_features = [candidate_features[i] for i in selected_idx]
        selection_rows.append(
            {
                "fold": fold_id,
                "selected_features": ",".join(selected_features),
                "n_selected": len(selected_features),
            }
        )
        rank_df.insert(0, "fold", fold_id)
        rank_df.insert(1, "selected", rank_df["feature"].isin(selected_features))
        ranking_rows.extend(rank_df.to_dict("records"))

        X_tr = X[tr_idx][:, selected_idx]
        X_te = X[te_idx][:, selected_idx]
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        test_ids = [ids[i] for i in te_idx]
        for model_name in model_names:
            proto = all_models[model_name]
            try:
                est = safe_fit(clone(proto), X_tr, y_tr)
                y_prob = predict_probability(est, X_te)
                y_pred = (y_prob >= 0.5).astype(int)
                metric_rows.append(
                    {
                        "feature_set": "Nested-selection CV",
                        "model": model_name,
                        "fold": fold_id,
                        "auroc": roc_auc_score(y_te, y_prob),
                        "acc": accuracy_score(y_te, y_pred),
                        "precision": precision_score(y_te, y_pred, zero_division=0),
                        "recall": recall_score(y_te, y_pred, zero_division=0),
                        "f1": f1_score(y_te, y_pred, zero_division=0),
                        "selected_features": ",".join(selected_features),
                    }
                )
                for local_idx, case_id in enumerate(test_ids):
                    prediction_rows.append(
                        {
                            "feature_set": "Nested-selection CV",
                            "model": model_name,
                            "fold": fold_id,
                            "ID": case_id,
                            "y_true": int(y_te[local_idx]),
                            "y_prob": float(y_prob[local_idx]),
                            "y_pred": int(y_pred[local_idx]),
                            "selected_features": ",".join(selected_features),
                        }
                    )
            except Exception as exc:
                error_rows.append({"model": model_name, "fold": fold_id, "error": str(exc)})

    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        raise RuntimeError(f"All nested feature-selection folds failed. First errors: {error_rows[:5]}")
    summary = summarize_repeated_metrics(
        metric_df,
        metrics=["auroc", "acc", "precision", "recall", "f1"],
        sort_cols=["mean_auroc"],
        ascending=[False],
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_ci=bootstrap_ci,
        group_cols=("model",),
    )
    strict_summary = summary.copy()
    strict_summary.insert(0, "feature_set", "Nested-selection CV")
    metric_df.to_csv(out_dir / "nested_feature_selection_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "nested_feature_selection_performance.csv", index=False)
    summary.to_excel(out_dir / "nested_feature_selection_performance.xlsx", index=False)
    strict_summary.to_csv(out_dir / "strict_nested_model_performance_table.csv", index=False)
    strict_summary.to_excel(out_dir / "strict_nested_model_performance_table.xlsx", index=False)
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(out_dir / "nested_feature_selection_predictions.csv", index=False)
    pred_df.to_csv(out_dir / "strict_nested_predictions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out_dir / "nested_selected_features_by_fold.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(out_dir / "nested_feature_rankings_by_fold.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(out_dir / "nested_feature_selection_errors.csv", index=False)
    write_publication_confusion_matrices(pred_df, out_dir)
    write_publication_roc_curves(pred_df, out_dir)
    write_publication_calibration_curves(pred_df, out_dir)

    freq_rows: List[Dict[str, object]] = []
    for feature in candidate_features:
        count = sum(feature in row["selected_features"].split(",") for row in selection_rows)
        freq_rows.append(
            {
                "feature": feature,
                "selected_fold_count": count,
                "selected_fold_rate": count / max(1, len(selection_rows)),
            }
        )
    pd.DataFrame(freq_rows).sort_values(
        ["selected_fold_count", "feature"], ascending=[False, True]
    ).to_csv(out_dir / "nested_selected_feature_frequency.csv", index=False)

    protocol = {
        "leakage_control": "Feature ranking and top-k selection are run separately inside each outer training fold only.",
        "test_set_role": "The held-out fold is used only once for final evaluation after features have been selected from the training fold.",
        "candidate_set": candidate_set,
        "candidate_features": candidate_features,
        "selection_method": "univariate_train_auc_abs_distance_from_0.5",
        "select_k": select_k,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "models": model_names,
    }
    (out_dir / "nested_feature_selection_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def promote_nested_selection_outputs(nested_dir: Path, classification_dir: Path) -> None:
    """Make strict nested held-out outputs the main manuscript classification outputs."""
    mapping = {
        "strict_nested_model_performance_table.csv": "model_performance_table.csv",
        "strict_nested_model_performance_table.xlsx": "model_performance_table.xlsx",
        "nested_feature_selection_fold_metrics.csv": "model_performance_split_metrics.csv",
        "strict_nested_predictions.csv": "publication_predictions.csv",
        "confusion_matrix_summary.csv": "confusion_matrix_summary.csv",
        "roc_curve_points.csv": "roc_curve_points.csv",
        "calibration_curve_points.csv": "calibration_curve_points.csv",
        "confusion_matrices.png": "confusion_matrices.png",
        "confusion_matrices.svg": "confusion_matrices.svg",
        "roc_curves.png": "roc_curves.png",
        "roc_curves.svg": "roc_curves.svg",
        "calibration_curves.png": "calibration_curves.png",
        "calibration_curves.svg": "calibration_curves.svg",
    }
    missing = []
    for src_name, dst_name in mapping.items():
        src = nested_dir / src_name
        if not src.exists():
            missing.append(src_name)
            continue
        shutil.copy2(src, classification_dir / dst_name)
    manifest = {
        "main_classification_outputs": "strict_nested_feature_selection_cv",
        "source_dir": str(nested_dir),
        "copied_files": mapping,
        "missing_files": missing,
        "note": (
            "The main classification table, ROC, calibration and confusion matrix are copied from "
            "nested feature-selection held-out predictions. Fixed-panel publication outputs are "
            "retained as fixed_panel_* files."
        ),
    }
    (classification_dir / "main_output_source.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_auroc_stability_table(summary: pd.DataFrame, stability_margin: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base = summary[summary["adjustment_set"] == "unadjusted_biomarkers"].set_index("model")
    for _, row in summary.iterrows():
        model = row["model"]
        if model not in base.index:
            continue
        base_row = base.loc[model]
        delta = float(row["mean_auroc"] - base_row["mean_auroc"])
        rows.append(
            {
                "model": model,
                "adjustment_set": row["adjustment_set"],
                "unadjusted_mean_auroc": base_row["mean_auroc"],
                "adjusted_mean_auroc": row["mean_auroc"],
                "delta_auroc": delta,
                "abs_delta_auroc": abs(delta),
                "stability_margin": stability_margin,
                "stable_within_margin": abs(delta) <= stability_margin,
                "adjusted_ci95_low_auroc": row.get("ci95_low_auroc"),
                "adjusted_ci95_high_auroc": row.get("ci95_high_auroc"),
            }
        )
    return pd.DataFrame(rows)


def plot_auroc_stability(summary: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "unadjusted_biomarkers",
        "age_adjusted",
        "education_adjusted",
        "age_sex_education_adjusted",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    x = np.arange(len(order))
    for model, group in summary.groupby("model", sort=True):
        group = group.set_index("adjustment_set").reindex(order)
        means = group["mean_auroc"].to_numpy(dtype=float)
        low = group["ci95_low_auroc"].to_numpy(dtype=float)
        high = group["ci95_high_auroc"].to_numpy(dtype=float)
        yerr = np.vstack([means - low, high - means])
        ax.errorbar(x, means, yerr=yerr, marker="o", capsize=3, linewidth=2, label=model)
    ax.set_xticks(x, ["Unadjusted", "Age", "Education", "Age+sex+education"], rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC Stability After Confounder Adjustment")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "auroc_stability_after_adjustment.png", bbox_inches="tight")
    fig.savefig(out_dir / "auroc_stability_after_adjustment.svg", bbox_inches="tight")
    plt.close(fig)


def write_publication_confusion_matrices(pred_df: pd.DataFrame, out_dir: Path) -> None:
    rows: List[Dict[str, object]] = []
    groups = list(pred_df.groupby(["feature_set", "model"], sort=True))
    n = len(groups)
    ncols = 3
    nrows = int(math.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), dpi=160)
    axes_arr = np.asarray(axes).reshape(-1)

    for ax, ((feature_set, model), group) in zip(axes_arr, groups):
        cm = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{model} | {feature_set}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    for ax in axes_arr[len(groups) :]:
        ax.axis("off")

    pd.DataFrame(rows).to_csv(out_dir / "confusion_matrix_summary.csv", index=False)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", bbox_inches="tight")
    fig.savefig(out_dir / "confusion_matrices.svg", bbox_inches="tight")
    plt.close(fig)


def write_publication_roc_curves(pred_df: pd.DataFrame, out_dir: Path) -> None:
    curve_rows: List[Dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for (feature_set, model), group in pred_df.groupby(["feature_set", "model"], sort=True):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_prob = group["y_prob"].to_numpy(dtype=float)
        if len(set(y_true.tolist())) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        label = f"{model} | {feature_set} (AUROC={auc:.3f})"
        ax.plot(fpr, tpr, lw=2, label=label)
        for point, (f, t, thr) in enumerate(zip(fpr, tpr, thresholds)):
            curve_rows.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "point": point,
                    "fpr": float(f),
                    "tpr": float(t),
                    "threshold": float(thr),
                    "auroc": float(auc),
                }
            )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="0.5")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Publication ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    pd.DataFrame(curve_rows).to_csv(out_dir / "roc_curve_points.csv", index=False)
    plt.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", bbox_inches="tight")
    fig.savefig(out_dir / "roc_curves.svg", bbox_inches="tight")
    plt.close(fig)


def write_publication_calibration_curves(pred_df: pd.DataFrame, out_dir: Path) -> None:
    curve_rows: List[Dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for (feature_set, model), group in pred_df.groupby(["feature_set", "model"], sort=True):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_prob = group["y_prob"].to_numpy(dtype=float)
        if len(set(y_true.tolist())) < 2:
            continue
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", lw=2, label=f"{model} | {feature_set}")
        for point, (pp, pt) in enumerate(zip(prob_pred, prob_true)):
            curve_rows.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "point": point,
                    "mean_predicted_probability": float(pp),
                    "fraction_of_positives": float(pt),
                }
            )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="0.5")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Publication Calibration Curves")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    pd.DataFrame(curve_rows).to_csv(out_dir / "calibration_curve_points.csv", index=False)
    plt.tight_layout()
    fig.savefig(out_dir / "calibration_curves.png", bbox_inches="tight")
    fig.savefig(out_dir / "calibration_curves.svg", bbox_inches="tight")
    plt.close(fig)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def evaluate_regression_models(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    n_splits: int,
    n_repeats: int,
    use_robust: bool,
    bootstrap_repeats: int,
    bootstrap_ci: float,
) -> pd.DataFrame:
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    for model_name, proto in make_regression_models(use_robust=use_robust).items():
        for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(X), start=1):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            try:
                est = clone(proto)
                est.fit(X_tr, y_tr)
                pred = est.predict(X_te)
                rows.append({"model": model_name, "fold": fold_id, **regression_metrics(y_te, pred)})
            except Exception as exc:
                errors.append({"model": model_name, "fold": fold_id, "error": str(exc)})

    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        raise RuntimeError(f"All regression folds failed. First errors: {errors[:5]}")

    metrics = ["r2", "rmse", "mae"]
    summary = summarize_repeated_metrics(
        fold_df,
        metrics,
        sort_cols=["mean_r2", "mean_rmse"],
        ascending=[False, True],
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_ci=bootstrap_ci,
    )

    fold_df.to_csv(out_dir / "regression_fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "regression_repeated_cv.csv", index=False)
    summary.to_excel(out_dir / "regression_repeated_cv.xlsx", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "regression_fold_errors.csv", index=False)
    return summary


def plot_regression_metric_bars(results: pd.DataFrame, out_dir: Path) -> None:
    plot_specs = [
        ("mean_r2", "std_r2", "R2 (mean +/- SD)"),
        ("mean_rmse", "std_rmse", "RMSE (mean +/- SD)"),
        ("mean_mae", "std_mae", "MAE (mean +/- SD)"),
    ]
    res_plot = results.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(res_plot))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=160)
    for ax, (mean_col, std_col, title) in zip(axes, plot_specs):
        means = res_plot[mean_col].to_numpy(dtype=float)
        stds = res_plot[std_col].to_numpy(dtype=float) if std_col in res_plot else np.zeros_like(means)
        ax.barh(y_pos, means, xerr=stds, alpha=0.9)
        ax.invert_yaxis()
        ax.set_yticks(y_pos, res_plot["model"])
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_dir / "table2_regression_metric_bars.png", bbox_inches="tight")
    fig.savefig(out_dir / "table2_regression_metric_bars.svg", bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(results: pd.DataFrame, out_dir: Path) -> None:
    plot_specs = [
        ("mean_roc_auc", "std_roc_auc", "AUROC (mean +/- SD)"),
        ("mean_pr_auc", "std_pr_auc", "AUPRC (mean +/- SD)"),
        ("mean_bal_acc", "std_bal_acc", "Balanced Accuracy (mean +/- SD)"),
        ("mean_mcc", "std_mcc", "MCC (mean +/- SD)"),
    ]
    res_plot = results.sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(res_plot))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.ravel()
    for ax, (mean_col, std_col, title) in zip(axes, plot_specs):
        means = res_plot[mean_col].to_numpy(dtype=float)
        stds = res_plot[std_col].to_numpy(dtype=float) if std_col in res_plot else np.zeros_like(means)
        ax.barh(y_pos, means, xerr=stds, alpha=0.9)
        ax.invert_yaxis()
        ax.set_yticks(y_pos, res_plot["model"])
        ax.set_title(title)
        ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_metric_bars_4panels.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_metric_bars_4panels.svg", bbox_inches="tight")
    plt.close(fig)


def get_oof_probabilities(
    models: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
) -> Dict[str, np.ndarray]:
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    oof_sum = {name: np.zeros(len(y), dtype=float) for name in models}
    oof_count = {name: np.zeros(len(y), dtype=float) for name in models}

    for tr_idx, te_idx in splitter.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]
        for name, proto in models.items():
            try:
                est = safe_fit(clone(proto), X_tr, y_tr)
                prob = predict_probability(est, X_te)
            except Exception as exc:
                warnings.warn(f"OOF ROC skipped {name} fold: {exc}")
                continue
            oof_sum[name][te_idx] += prob
            oof_count[name][te_idx] += 1.0
    return {name: oof_sum[name] / np.maximum(oof_count[name], 1.0) for name in models}


def plot_oof_roc(
    models: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    ids: Sequence[str],
    out_dir: Path,
    n_splits: int,
    n_repeats: int,
) -> None:
    oof = get_oof_probabilities(models, X, y, n_splits, n_repeats)
    prob_df = pd.DataFrame({"ID": ids, "label": y})
    curve_rows: List[Dict[str, object]] = []

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for name, prob in oof.items():
        prob_df[name] = prob
        fpr, tpr, thresholds = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={auc:.3f})")
        for i, (f, t, thr) in enumerate(zip(fpr, tpr, thresholds)):
            curve_rows.append({"model": name, "point": i, "fpr": f, "tpr": t, "threshold": thr, "auc": auc})

    prob_df.to_csv(out_dir / "oof_probabilities.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out_dir / "roc_curves.csv", index=False)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="0.5")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("OOF ROC Curves (Repeated Stratified CV)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_oof_roc_curves.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_oof_roc_curves.svg", bbox_inches="tight")
    plt.close(fig)


def pipeline_prefix_transform(estimator: object, X: np.ndarray) -> np.ndarray:
    if isinstance(estimator, Pipeline):
        return estimator[:-1].transform(X) if len(estimator.steps) > 1 else X
    return X


def extract_final_estimator(estimator: object) -> object:
    return estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator


def is_tree_model(model: object) -> bool:
    return isinstance(
        model,
        (
            RandomForestClassifier,
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
            AdaBoostClassifier,
            DecisionTreeClassifier,
        ),
    )


def normalize_shap_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def oof_shap_for_model(
    model_name: str,
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    out_dir: Path,
    n_splits: int,
    max_display: int,
) -> Optional[pd.DataFrame]:
    try:
        import shap  # type: ignore
    except ImportError:
        raise RuntimeError("shap is not installed. Install requirements-paper.txt or set run_shap = no.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    n, p = X.shape
    oof_shap = np.full((n, p), np.nan, dtype=float)
    oof_features = np.full((n, p), np.nan, dtype=float)

    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]
        try:
            est = safe_fit(clone(model), X_tr, y_tr)
            base = extract_final_estimator(est)
            X_tr_t = pipeline_prefix_transform(est, X_tr)
            X_te_t = pipeline_prefix_transform(est, X_te)

            if is_tree_model(base):
                explainer = shap.TreeExplainer(base)
                shap_te = normalize_shap_values(explainer.shap_values(X_te_t))
            elif isinstance(base, LogisticRegression):
                explainer = shap.LinearExplainer(base, X_tr_t)
                shap_te = normalize_shap_values(explainer.shap_values(X_te_t))
            else:
                warnings.warn(f"SHAP skipped for unsupported model type: {model_name}")
                return None

            if shap_te.shape != X_te_t.shape:
                raise ValueError(f"SHAP shape {shap_te.shape} does not match transformed X {X_te_t.shape}")
            oof_shap[te_idx, :] = shap_te
            oof_features[te_idx, :] = X_te_t
        except Exception as exc:
            warnings.warn(f"SHAP fold failed for {model_name} fold {fold_id}: {exc}")

    if np.isnan(oof_shap).all():
        warnings.warn(f"SHAP all-NaN for {model_name}")
        return None

    mean_abs = np.nanmean(np.abs(oof_shap), axis=0)
    norm = mean_abs / (np.nansum(mean_abs) + 1e-12)
    order = np.argsort(-mean_abs)
    top_idx = order[:max_display]
    top = pd.DataFrame(
        {
            "Feature": np.asarray(feature_names)[order],
            "mean_abs_SHAP": mean_abs[order],
            "avg_norm_abs_SHAP": norm[order],
            "avg_norm_abs_SHAP_pct": norm[order] * 100.0,
        }
    )

    safe = safe_name(model_name)
    pd.DataFrame(oof_shap, columns=feature_names).to_csv(out_dir / f"shap_values_{safe}.csv", index=False)
    top.to_csv(out_dir / f"shap_importance_{safe}.csv", index=False)

    name_to_pct = dict(zip(top["Feature"], top["avg_norm_abs_SHAP_pct"]))
    plt.figure(figsize=(8, 6), dpi=160)
    shap.summary_plot(
        oof_shap[:, top_idx],
        features=oof_features[:, top_idx],
        feature_names=np.asarray(feature_names)[top_idx],
        show=False,
        plot_type="dot",
    )
    ax = plt.gca()
    ax.set_yticklabels(
        [
            f"{label.get_text()} ({name_to_pct.get(label.get_text(), float('nan')):.1f}%)"
            if math.isfinite(name_to_pct.get(label.get_text(), float("nan")))
            else label.get_text()
            for label in ax.get_yticklabels()
        ]
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"fig4_shap_beeswarm_{safe}.png", bbox_inches="tight")
    plt.savefig(out_dir / f"fig4_shap_beeswarm_{safe}.svg", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 6), dpi=160)
    shap.summary_plot(
        oof_shap[:, top_idx],
        feature_names=np.asarray(feature_names)[top_idx],
        show=False,
        plot_type="bar",
    )
    ax = plt.gca()
    ax.set_yticklabels(
        [
            f"{label.get_text()} ({name_to_pct.get(label.get_text(), float('nan')):.1f}%)"
            if math.isfinite(name_to_pct.get(label.get_text(), float("nan")))
            else label.get_text()
            for label in ax.get_yticklabels()
        ]
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_{safe}.png", bbox_inches="tight")
    plt.savefig(out_dir / f"shap_bar_{safe}.svg", bbox_inches="tight")
    plt.close()
    return top


def run_shap_tables(
    model_names: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    out_dir: Path,
    n_splits: int,
    max_display: int,
    use_robust: bool,
) -> None:
    models = make_models(use_robust=use_robust)
    tables: Dict[str, pd.DataFrame] = {}
    for name in model_names:
        if name not in models:
            warnings.warn(f"SHAP model not found: {name}")
            continue
        table = oof_shap_for_model(name, models[name], X, y, feature_names, out_dir, n_splits, max_display)
        if table is not None:
            tables[name] = table
    if tables:
        with pd.ExcelWriter(out_dir / "shap_top_features.xlsx") as writer:
            for name, table in tables.items():
                table.to_excel(writer, sheet_name=safe_name(name)[:31], index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce paper-level classification, OOF ROC, and SHAP outputs.")
    parser.add_argument("--input", required=True, help="Input xlsx/csv containing age, edu, gait, speech, and label columns.")
    parser.add_argument("--sheet-name", default="age-edu")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument(
        "--feature-set",
        choices=["nestedselected", "adualselected", "script_all", "custom"],
        default="nestedselected",
    )
    parser.add_argument("--custom-features", default="")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--run-train-test-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-publication-outputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-confounder-adjustment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-nested-feature-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--promote-nested-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--publication-models", default="AdaBoost,RF,SVM (RBF),DT")
    parser.add_argument(
        "--nested-candidate-set",
        choices=["nestedselected", "adualselected", "script_all", "custom"],
        default="script_all",
    )
    parser.add_argument("--nested-custom-features", default="")
    parser.add_argument("--nested-select-k", type=int, default=14)
    parser.add_argument("--train-test-repeats", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-ci", type=float, default=0.95)
    parser.add_argument("--age-col", default="age")
    parser.add_argument("--education-col", default="edu")
    parser.add_argument("--sex-col", default="")
    parser.add_argument("--auroc-stability-margin", type=float, default=0.02)
    parser.add_argument("--use-robust", action="store_true")
    parser.add_argument("--run-shap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regression-target", default="MoCA")
    parser.add_argument("--shap-models", default="GBDT,Random Forest,Logistic Regression")
    parser.add_argument("--roc-models", default="GBDT,Random Forest,Logistic Regression,SVM (RBF)")
    parser.add_argument("--max-shap-display", type=int, default=20)
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    classification_dir = out_dir / "classification"
    regression_dir = out_dir / "regression"
    confounder_dir = classification_dir / "confounder_adjustment"
    nested_fs_dir = classification_dir / "nested_feature_selection"
    classification_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input).expanduser().resolve()
    df = read_table(input_path, args.sheet_name)
    features = select_features(df, args.feature_set, args.custom_features)
    dataset, X, y, ids = prepare_xy(df, features, args.label_col, args.id_col)
    dataset_cols = ([args.id_col] if args.id_col in dataset.columns else []) + [args.label_col] + features
    dataset[dataset_cols].to_csv(out_dir / f"{safe_name(args.feature_set)}_dataset.csv", index=False)

    results = evaluate_all_models(
        X,
        y,
        out_dir,
        args.n_splits,
        args.n_repeats,
        args.use_robust,
        args.bootstrap_repeats,
        args.bootstrap_ci,
    )
    plot_metric_bars(results, out_dir)

    train_test_summary = None
    if args.run_train_test_split:
        train_test_summary = evaluate_stratified_train_test_models(
            X,
            y,
            out_dir,
            args.train_test_repeats,
            args.test_size,
            args.use_robust,
            args.bootstrap_repeats,
            args.bootstrap_ci,
        )

    publication_summary = None
    if args.run_publication_outputs:
        publication_summary = evaluate_publication_classification(
            df,
            args.label_col,
            args.id_col,
            classification_dir,
            args.train_test_repeats,
            args.test_size,
            args.use_robust,
            args.bootstrap_repeats,
            args.bootstrap_ci,
            parse_csv_list(args.publication_models),
        )

    confounder_summary = None
    if args.run_confounder_adjustment:
        confounder_summary = evaluate_confounder_adjustment(
            df,
            features,
            args.label_col,
            args.id_col,
            args.age_col,
            args.education_col,
            args.sex_col,
            confounder_dir,
            args.train_test_repeats,
            args.test_size,
            args.use_robust,
            args.bootstrap_repeats,
            args.bootstrap_ci,
            parse_csv_list(args.publication_models),
            args.auroc_stability_margin,
        )

    nested_feature_selection_summary = None
    if args.run_nested_feature_selection:
        nested_feature_selection_summary = evaluate_nested_feature_selection_cv(
            df,
            args.label_col,
            args.id_col,
            args.nested_candidate_set,
            args.nested_custom_features,
            args.nested_select_k,
            nested_fs_dir,
            args.n_splits,
            args.n_repeats,
            args.use_robust,
            args.bootstrap_repeats,
            args.bootstrap_ci,
            parse_csv_list(args.publication_models),
        )
        if args.promote_nested_selection:
            promote_nested_selection_outputs(nested_fs_dir, classification_dir)
            publication_summary = nested_feature_selection_summary.copy()
            publication_summary.insert(0, "feature_set", "Nested-selection CV")

    all_models = make_models(use_robust=args.use_robust)
    roc_models = {name: all_models[name] for name in parse_csv_list(args.roc_models) if name in all_models}
    plot_oof_roc(roc_models, X, y, ids, out_dir, args.n_splits, args.n_repeats)

    if args.run_shap:
        run_shap_tables(
            parse_csv_list(args.shap_models),
            X,
            y,
            features,
            out_dir,
            args.n_splits,
            args.max_shap_display,
            args.use_robust,
        )

    regression_summary = None
    if args.run_regression:
        regression_dir.mkdir(parents=True, exist_ok=True)
        reg_dataset, X_reg, y_reg, reg_ids = prepare_regression_xy(df, features, args.regression_target, args.id_col)
        reg_cols = ([args.id_col] if args.id_col in reg_dataset.columns else []) + [args.regression_target] + features
        reg_dataset[reg_cols].to_csv(regression_dir / "regression_dataset.csv", index=False)
        regression_summary = evaluate_regression_models(
            X_reg,
            y_reg,
            regression_dir,
            args.n_splits,
            args.n_repeats,
            args.use_robust,
            args.bootstrap_repeats,
            args.bootstrap_ci,
        )
        plot_regression_metric_bars(regression_summary, regression_dir)

    summary = {
        "input": str(input_path),
        "sheet_name": args.sheet_name,
        "out_dir": str(out_dir),
        "n_rows": int(len(y)),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "feature_set": args.feature_set,
        "features": features,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "run_train_test_split": args.run_train_test_split,
        "run_publication_outputs": args.run_publication_outputs,
        "run_confounder_adjustment": args.run_confounder_adjustment,
        "run_nested_feature_selection": args.run_nested_feature_selection,
        "promote_nested_selection": args.promote_nested_selection,
        "main_classification_outputs": "nested_feature_selection_cv"
        if args.run_nested_feature_selection and args.promote_nested_selection
        else "fixed_panel_publication_outputs",
        "publication_models": parse_csv_list(args.publication_models),
        "nested_candidate_set": args.nested_candidate_set,
        "nested_select_k": args.nested_select_k,
        "age_col": args.age_col,
        "education_col": args.education_col,
        "sex_col": args.sex_col,
        "auroc_stability_margin": args.auroc_stability_margin,
        "train_test_repeats": args.train_test_repeats,
        "test_size": args.test_size,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_ci": args.bootstrap_ci,
        "run_shap": args.run_shap,
        "run_regression": args.run_regression,
        "regression_target": args.regression_target,
        "best_by_auc": results.iloc[0].to_dict(),
        "best_train_test_by_auc": train_test_summary.iloc[0].to_dict() if train_test_summary is not None else None,
        "best_publication_by_auc": publication_summary.iloc[0].to_dict() if publication_summary is not None else None,
        "best_confounder_adjusted_by_auc": confounder_summary.iloc[0].to_dict() if confounder_summary is not None else None,
        "best_nested_feature_selection_by_auc": nested_feature_selection_summary.iloc[0].to_dict()
        if nested_feature_selection_summary is not None
        else None,
        "best_regression_by_r2": regression_summary.iloc[0].to_dict() if regression_summary is not None else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Paper experiment outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
