#!/usr/bin/env python3
"""Full-cohort within-task domain ablation analysis.

This script evaluates the contribution of demographics, gait and speech feature
domains using the same leakage-controlled repeated-CV protocol as the manuscript
reproduction code. Feature ranking and top-k selection are repeated inside each
outer training fold only.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adualnet_mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "adualnet_xdg_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from paper_experiment_reproduction import (
    RANDOM_STATE,
    bootstrap_mean_ci,
    make_publication_classification_models,
    parse_csv_list,
    predict_probability,
    read_table,
    safe_fit,
    select_top_k_features_train_only,
    summarize_repeated_metrics,
)


DOMAIN_SPECS: Dict[str, List[str]] = {
    "demographics": ["age", "sex", "edu"],
    "gait": [
        "walking_speed",
        "cadence",
        "average_step_length",
        "average_stride_length",
        "average_step_time",
        "average_stride_time",
        "step_length_variability",
        "stride_length_variability",
        "step_time_variability",
        "stride_time_variability",
        "average_step_assymetry",
        "average_stride_assymetry",
        "stance_times",
        "swing_times",
        "double_support_times",
    ],
    "speech": [
        "Timecount_DT",
        "Num_DT",
        "Correct_Num_DT",
        "Num_20s_DT",
        "Correct_Num_20s_DT",
        "Correct_persec_DT",
        "ACC_DT",
        "Timecount_DT_1stNum",
    ],
}

ALIASES: Dict[str, List[str]] = {
    "sex": ["sex", "gender"],
    "edu": ["edu", "education", "Education"],
}

FEATURE_SETS: Dict[str, List[str]] = {
    "demographics-only": ["demographics"],
    "gait-only": ["gait"],
    "speech-only": ["speech"],
    "gait + speech": ["gait", "speech"],
    "gait + speech + demographics": ["gait", "speech", "demographics"],
}

FEATURE_SET_ORDER = list(FEATURE_SETS)
MODEL_ORDER = ["AdaBoost", "RF", "SVM (RBF)", "DT"]

PAIRED_COMPARISONS = [
    ("full model vs gait + speech", "gait + speech + demographics", "gait + speech"),
    ("full model vs demographics-only", "gait + speech + demographics", "demographics-only"),
    ("gait + speech vs demographics-only", "gait + speech", "demographics-only"),
    ("speech-only vs gait-only", "speech-only", "gait-only"),
]

DOMAIN_COLORS = {
    "demographics-only": "#7A869A",
    "gait-only": "#59A14F",
    "speech-only": "#F28E2B",
    "gait + speech": "#4C78A8",
    "gait + speech + demographics": "#8E6C8A",
}

MODEL_COLORS = {
    "AdaBoost": "#4C78A8",
    "RF": "#59A14F",
    "SVM (RBF)": "#F28E2B",
    "DT": "#9C755F",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def match_requested_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    rows: List[Dict[str, object]] = []
    domain_to_actual: Dict[str, List[str]] = {}
    missing: List[str] = []
    lower_to_actual = {str(c).lower(): str(c) for c in df.columns}

    for domain, requested_features in DOMAIN_SPECS.items():
        actuals: List[str] = []
        for requested in requested_features:
            aliases = ALIASES.get(requested, [requested])
            matched = None
            for alias in aliases:
                if alias in df.columns:
                    matched = alias
                    break
                if alias.lower() in lower_to_actual:
                    matched = lower_to_actual[alias.lower()]
                    break
            if matched:
                actuals.append(matched)
            else:
                missing.append(f"{domain}:{requested}")
            rows.append(
                {
                    "domain": domain,
                    "requested_feature": requested,
                    "matched_column": matched or "",
                    "status": "matched" if matched else "missing",
                    "aliases_checked": ",".join(aliases),
                }
            )
        domain_to_actual[domain] = actuals
    return pd.DataFrame(rows), domain_to_actual, missing


def build_feature_sets(domain_to_actual: Dict[str, List[str]]) -> Dict[str, List[str]]:
    feature_sets: Dict[str, List[str]] = {}
    for feature_set, domains in FEATURE_SETS.items():
        features: List[str] = []
        for domain in domains:
            features.extend(domain_to_actual[domain])
        deduped: List[str] = []
        for feature in features:
            if feature not in deduped:
                deduped.append(feature)
        feature_sets[feature_set] = deduped
    return feature_sets


def parse_label_series(df: pd.DataFrame, label_col: str) -> pd.Series:
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")
    y = pd.to_numeric(df[label_col], errors="coerce")
    unique = sorted(set(y.loc[y.notna()].astype(int).tolist()))
    if unique != [0, 1]:
        raise ValueError(f"Classification label must be binary 0/1; got {unique}")
    return y


def prepare_full_cohort(
    df: pd.DataFrame,
    label_col: str,
    id_col: str,
    feature_sets: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    y = parse_label_series(df, label_col)
    cohort = df.loc[y.notna()].copy()
    cohort[label_col] = y.loc[y.notna()].astype(int)
    if id_col not in cohort.columns:
        cohort[id_col] = [str(i) for i in cohort.index]
    for features in feature_sets.values():
        for col in features:
            cohort[col] = pd.to_numeric(cohort[col], errors="coerce")
    return cohort, cohort[label_col].astype(int), cohort[id_col].astype(str).tolist()


def evaluate_domain_ablation(
    cohort: pd.DataFrame,
    y: pd.Series,
    ids: Sequence[str],
    feature_sets: Dict[str, List[str]],
    n_splits: int,
    n_repeats: int,
    select_k_cap: int,
    requested_models: Sequence[str],
    bootstrap_repeats: int,
    bootstrap_ci: float,
    use_robust: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_models = make_publication_classification_models(use_robust=use_robust)
    model_names = [m for m in requested_models if m in all_models]
    if not model_names:
        raise ValueError("No valid model names were requested.")

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    y_np = y.to_numpy(dtype=int)
    split_indices = list(splitter.split(np.zeros((len(y_np), 1)), y_np))

    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []

    for feature_set in FEATURE_SET_ORDER:
        features = feature_sets[feature_set]
        if not features:
            raise ValueError(f"No available features for feature set: {feature_set}")
        X = cohort[features].to_numpy(dtype=float)
        k = min(select_k_cap, len(features))
        for fold_id, (tr_idx, te_idx) in enumerate(split_indices, start=1):
            selected_idx, rank_df = select_top_k_features_train_only(X[tr_idx], y_np[tr_idx], features, k)
            selected_features = [features[i] for i in selected_idx]
            selection_rows.append(
                {
                    "feature_set": feature_set,
                    "fold": fold_id,
                    "n_available_features": len(features),
                    "selected_k": len(selected_features),
                    "selected_features": ",".join(selected_features),
                }
            )
            rank_out = rank_df.copy()
            rank_out.insert(0, "feature_set", feature_set)
            rank_out.insert(1, "fold", fold_id)
            rank_out.insert(2, "selected", rank_out["feature"].isin(selected_features))
            ranking_rows.extend(rank_out.to_dict("records"))

            X_tr = X[tr_idx][:, selected_idx]
            X_te = X[te_idx][:, selected_idx]
            y_tr, y_te = y_np[tr_idx], y_np[te_idx]
            for model_name in model_names:
                est = safe_fit(clone(all_models[model_name]), X_tr, y_tr)
                y_prob = predict_probability(est, X_te)
                y_pred = (y_prob >= 0.5).astype(int)
                metric_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold_id,
                        "auroc": roc_auc_score(y_te, y_prob),
                        "acc": accuracy_score(y_te, y_pred),
                        "precision": precision_score(y_te, y_pred, zero_division=0),
                        "recall": recall_score(y_te, y_pred, zero_division=0),
                        "f1": f1_score(y_te, y_pred, zero_division=0),
                        "selected_k": len(selected_features),
                        "selected_features": ",".join(selected_features),
                    }
                )
                for local_idx, case_idx in enumerate(te_idx):
                    prediction_rows.append(
                        {
                            "feature_set": feature_set,
                            "model": model_name,
                            "fold": fold_id,
                            "ID": ids[case_idx],
                            "y_true": int(y_te[local_idx]),
                            "y_prob": float(y_prob[local_idx]),
                            "y_pred": int(y_pred[local_idx]),
                            "selected_k": len(selected_features),
                            "selected_features": ",".join(selected_features),
                        }
                    )

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected_by_fold = pd.DataFrame(selection_rows)
    rankings = pd.DataFrame(ranking_rows)
    performance = summarize_repeated_metrics(
        fold_metrics,
        metrics=["auroc", "acc", "precision", "recall", "f1"],
        sort_cols=["feature_set", "mean_auroc"],
        ascending=[True, False],
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_ci=bootstrap_ci,
        group_cols=("feature_set", "model"),
    )
    return performance, fold_metrics, predictions, selected_by_fold, rankings


def build_confusion_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (feature_set, model), group in predictions.groupby(["feature_set", "model"], sort=False):
        tn, fp, fn, tp = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1]).ravel()
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "n_predictions": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_selected_feature_frequency(
    selected_by_fold: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    n_folds: int,
) -> pd.DataFrame:
    feature_to_domain = {
        row["matched_column"]: row["domain"]
        for _, row in feature_manifest.iterrows()
        if row["status"] == "matched" and row["matched_column"]
    }
    rows: List[Dict[str, object]] = []
    for feature_set, group in selected_by_fold.groupby("feature_set", sort=False):
        all_features = sorted({f for row in group["selected_features"] for f in str(row).split(",") if f})
        for feature in all_features:
            count = sum(feature in str(row).split(",") for row in group["selected_features"])
            rows.append(
                {
                    "feature_set": feature_set,
                    "domain": feature_to_domain.get(feature, ""),
                    "feature": feature,
                    "selected_fold_count": int(count),
                    "selected_fold_rate": count / n_folds,
                    "n_outer_folds": n_folds,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["feature_set", "selected_fold_count", "feature"], ascending=[True, False, True]
    )


def write_feature_manifest(
    feature_manifest: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    cohort: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    used_rows: List[Dict[str, object]] = []
    for _, row in feature_manifest.iterrows():
        matched = row["matched_column"]
        used_in = [fs for fs, feats in feature_sets.items() if matched and matched in feats]
        missing_count = int(cohort[matched].isna().sum()) if matched and matched in cohort.columns else ""
        used_rows.append(
            {
                **row.to_dict(),
                "used_in_feature_sets": ";".join(used_in),
                "missing_count_in_full_cohort": missing_count,
                "n_subjects_full_cohort": len(cohort),
            }
        )
    out = pd.DataFrame(used_rows)
    out.to_csv(out_dir / "domain_ablation_feature_manifest.csv", index=False)
    return out


def build_paired_delta_auroc(
    fold_metrics: pd.DataFrame,
    bootstrap_repeats: int,
    bootstrap_ci: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in MODEL_ORDER:
        model_df = fold_metrics[fold_metrics["model"].eq(model)]
        for comparison, feature_set_a, feature_set_b in PAIRED_COMPARISONS:
            a = model_df[model_df["feature_set"].eq(feature_set_a)][["fold", "auroc"]].rename(
                columns={"auroc": "auroc_a"}
            )
            b = model_df[model_df["feature_set"].eq(feature_set_b)][["fold", "auroc"]].rename(
                columns={"auroc": "auroc_b"}
            )
            merged = a.merge(b, on="fold", how="inner")
            delta = merged["auroc_a"].to_numpy(dtype=float) - merged["auroc_b"].to_numpy(dtype=float)
            lo, hi = bootstrap_mean_ci(
                delta,
                n_bootstrap=bootstrap_repeats,
                ci=bootstrap_ci,
                seed=RANDOM_STATE + len(rows),
            )
            rows.append(
                {
                    "comparison": comparison,
                    "model": model,
                    "feature_set_a": feature_set_a,
                    "feature_set_b": feature_set_b,
                    "mean_auroc_a": float(merged["auroc_a"].mean()),
                    "mean_auroc_b": float(merged["auroc_b"].mean()),
                    "mean_delta_auroc": float(delta.mean()),
                    "std_delta_auroc": float(delta.std(ddof=1)),
                    "ci95_low_delta_auroc": lo,
                    "ci95_high_delta_auroc": hi,
                    "n_paired_folds": int(len(delta)),
                }
            )
    return pd.DataFrame(rows)


def save_pub(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("svg", "png"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"{stem}.{ext}", **kwargs)


def plot_auroc_bar(performance: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(FEATURE_SET_ORDER))
    width = 0.18
    for idx, model in enumerate(MODEL_ORDER):
        sub = performance[performance["model"].eq(model)].set_index("feature_set")
        vals = [sub.loc[fs, "mean_auroc"] if fs in sub.index else np.nan for fs in FEATURE_SET_ORDER]
        yerr_low = [
            sub.loc[fs, "mean_auroc"] - sub.loc[fs, "ci95_low_auroc"] if fs in sub.index else np.nan
            for fs in FEATURE_SET_ORDER
        ]
        yerr_high = [
            sub.loc[fs, "ci95_high_auroc"] - sub.loc[fs, "mean_auroc"] if fs in sub.index else np.nan
            for fs in FEATURE_SET_ORDER
        ]
        offset = (idx - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=MODEL_COLORS.get(model, "#555555"),
            label=model,
            alpha=0.92,
            yerr=[yerr_low, yerr_high],
            error_kw={"elinewidth": 0.8, "capsize": 2},
        )
    ax.set_ylabel("AUROC (95% CI)")
    ax.set_ylim(0.55, 0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_SET_ORDER, rotation=25, ha="right")
    ax.grid(axis="y", color="#E3E8EC", lw=0.6)
    ax.legend(loc="upper left", ncol=4, fontsize=6.2)
    fig.tight_layout()
    save_pub(fig, out_dir, "domain_ablation_auroc_bar")
    plt.close(fig)


def plot_metric_heatmap(performance: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["mean_auroc", "mean_acc", "mean_precision", "mean_recall", "mean_f1"]
    metric_labels = ["AUROC", "ACC", "Precision", "Recall", "F1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.2, 3.35), sharey=True)
    for ax, metric, label in zip(axes, metrics, metric_labels):
        mat = []
        for fs in FEATURE_SET_ORDER:
            row = []
            for model in MODEL_ORDER:
                sub = performance[(performance["feature_set"].eq(fs)) & (performance["model"].eq(model))]
                row.append(float(sub.iloc[0][metric]) if not sub.empty else np.nan)
            mat.append(row)
        arr = np.array(mat)
        im = ax.imshow(arr, vmin=0.55, vmax=0.95, cmap="Blues", aspect="auto")
        ax.set_title(label, fontweight="bold", pad=5)
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right", fontsize=6)
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(FEATURE_SET_ORDER)))
            ax.set_yticklabels(FEATURE_SET_ORDER, fontsize=6.2)
        else:
            ax.set_yticks(np.arange(len(FEATURE_SET_ORDER)))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.subplots_adjust(left=0.18, right=0.88, bottom=0.29, top=0.87, wspace=0.08)
    cax = fig.add_axes([0.905, 0.30, 0.014, 0.53])
    fig.colorbar(im, cax=cax).set_label("Metric value")
    save_pub(fig, out_dir, "domain_ablation_metric_heatmap")
    plt.close(fig)


def plot_roc_curves(predictions: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    curve_rows: List[Dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, model in zip(axes, MODEL_ORDER):
        for feature_set in FEATURE_SET_ORDER:
            group = predictions[(predictions["model"].eq(model)) & (predictions["feature_set"].eq(feature_set))]
            if group.empty:
                continue
            y_true = group["y_true"].to_numpy(dtype=int)
            y_prob = group["y_prob"].to_numpy(dtype=float)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, lw=1.3, color=DOMAIN_COLORS[feature_set], label=f"{feature_set} ({auc:.3f})")
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
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#A6B3BB")
        ax.set_title(model, loc="left", fontweight="bold", pad=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color="#E3E8EC", lw=0.5)
    for ax in axes[2:]:
        ax.set_xlabel("False positive rate")
    for ax in axes[[0, 2]]:
        ax.set_ylabel("True positive rate")
    axes[1].legend(fontsize=5.4, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    save_pub(fig, out_dir, "domain_ablation_roc_curves")
    plt.close(fig)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(out_dir / "domain_ablation_roc_curve_points.csv", index=False)
    return curves


def plot_paired_delta_auroc(delta: pd.DataFrame, out_dir: Path, model: str = "AdaBoost") -> None:
    sub = delta[delta["model"].eq(model)].iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    y = np.arange(len(sub))
    ax.axvline(0, color="#1E252B", lw=0.8)
    ax.hlines(y, sub["ci95_low_delta_auroc"], sub["ci95_high_delta_auroc"], color="#A6B3BB", lw=1.4)
    ax.scatter(sub["mean_delta_auroc"], y, color="#8E6C8A", s=28, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["comparison"], fontsize=6.4)
    ax.set_xlabel(f"Paired Delta AUROC, {model} (95% CI)")
    ax.grid(axis="x", color="#E3E8EC", lw=0.6)
    fig.tight_layout()
    save_pub(fig, out_dir, "domain_ablation_paired_delta_auroc")
    plt.close(fig)


def build_qc_summary(
    cohort: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    performance: pd.DataFrame,
    selected_frequency: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
) -> Dict[str, object]:
    feature_set_n = {}
    all_same_subjects = True
    for fs, features in feature_sets.items():
        nonempty = cohort[features].notna().any(axis=1)
        feature_set_n[fs] = {
            "n_subjects": int(len(cohort)),
            "n_subjects_with_at_least_one_nonmissing_feature": int(nonempty.sum()),
            "n_available_features": len(features),
            "selected_k": min(14, len(features)),
        }
        if int(nonempty.sum()) != len(cohort):
            all_same_subjects = False

    def auroc(feature_set: str, model: str = "AdaBoost") -> float:
        sub = performance[(performance["feature_set"].eq(feature_set)) & (performance["model"].eq(model))]
        return float(sub.iloc[0]["mean_auroc"]) if not sub.empty else float("nan")

    demo = auroc("demographics-only")
    gait = auroc("gait-only")
    speech = auroc("speech-only")
    gait_speech = auroc("gait + speech")
    full = auroc("gait + speech + demographics")
    edu_rows = selected_frequency[selected_frequency["feature"].isin(["edu", "education", "Education"])]
    return {
        "same_subjects_all_feature_sets": all_same_subjects,
        "n_subjects_full_cohort": int(len(cohort)),
        "feature_set_sample_sizes": feature_set_n,
        "same_cv_splits_all_feature_sets": True,
        "cv_splits": {"n_splits": n_splits, "n_repeats": n_repeats, "n_outer_folds": n_splits * n_repeats},
        "feature_selection_nested_in_training_fold": True,
        "test_fold_role": "Held-out folds are used only for final prediction and metric calculation.",
        "adaBoost_auroc": {
            "demographics_only": demo,
            "gait_only": gait,
            "speech_only": speech,
            "gait_speech": gait_speech,
            "gait_speech_demographics": full,
            "gait_speech_minus_gait": gait_speech - gait,
            "gait_speech_minus_speech": gait_speech - speech,
            "full_minus_gait_speech": full - gait_speech,
        },
        "demographics_only_auroc_high_threshold_0.80": demo >= 0.80,
        "gait_speech_better_than_single_domains": gait_speech > max(gait, speech),
        "full_model_better_than_gait_speech": full > gait_speech,
        "education_selection_frequency_rows": edu_rows.sort_values(
            "selected_fold_rate", ascending=False
        ).to_dict("records"),
    }


def write_results_text(qc: Dict[str, object], performance: pd.DataFrame, out_dir: Path) -> str:
    def metric(feature_set: str, model: str = "AdaBoost") -> pd.Series:
        sub = performance[(performance["feature_set"].eq(feature_set)) & (performance["model"].eq(model))]
        return sub.iloc[0]

    demo = metric("demographics-only")
    gait = metric("gait-only")
    speech = metric("speech-only")
    gait_speech = metric("gait + speech")
    full = metric("gait + speech + demographics")
    au = qc["adaBoost_auroc"]

    if au["demographics_only"] >= au["gait_speech"] - 0.02:
        boundary = (
            "Classification performance was strongly influenced by demographic structure, "
            "particularly education, which defines an important boundary for interpretation."
        )
    elif au["gait_speech"] > au["demographics_only"] + 0.02:
        boundary = "Dual-task gait-speech features provided information beyond demographic structure."
    else:
        boundary = (
            "Dual-task gait-speech features and demographic variables showed partially overlapping "
            "discriminative information."
        )
    speech_sentence = (
        "Speech-task performance carried stronger discriminative information than gait features alone."
        if au["speech_only"] > au["gait_only"] + 0.02
        else "Gait-only and speech-only feature sets showed broadly comparable held-out discrimination."
    )
    demo_add = (
        "Adding demographic variables to gait and speech improved held-out discrimination, indicating additional "
        "explanatory information but also an important interpretive boundary."
        if au["full_minus_gait_speech"] > 0.02
        else "Adding demographic variables to gait and speech produced only a small change in held-out discrimination."
    )
    text = (
        "Full-cohort within-task domain ablation showed that the contribution of dual-task TUG-DT feature domains "
        "varied across modalities. Using the same repeated stratified five-fold cross-validation framework with "
        "feature selection nested within each training fold, AdaBoost achieved AUROCs of "
        f"{demo['mean_auroc']:.3f} for demographics-only, {gait['mean_auroc']:.3f} for gait-only, "
        f"{speech['mean_auroc']:.3f} for speech-only, {gait_speech['mean_auroc']:.3f} for gait plus speech and "
        f"{full['mean_auroc']:.3f} for gait, speech and demographics. {speech_sentence} {demo_add} {boundary} "
        "These analyses were performed within the same dual-task TUG-DT cohort and should be interpreted as domain "
        "contribution analyses rather than external validation."
    )
    (out_dir / "domain_ablation_results_text.md").write_text(text + "\n", encoding="utf-8")
    return text


def write_process_notes(
    docs_dir: Path,
    qc: Dict[str, object],
    results_text: str,
    input_path: Path,
    sheet_name: str,
) -> None:
    lines = [
        "# Full-cohort within-task domain ablation process notes",
        "",
        "## Analysis contract",
        "",
        "This analysis compares feature domains within the same dual-task TUG-DT cohort. It is not external validation and not task-only validation.",
        "",
        "## Input source",
        "",
        f"- Input table: `{input_path}`",
        f"- Sheet name: `{sheet_name}`",
        "",
        "## Leakage-control checks",
        "",
        f"- Same subjects across feature sets: {qc['same_subjects_all_feature_sets']}",
        f"- Same CV splits across feature sets: {qc['same_cv_splits_all_feature_sets']}",
        "- Feature ranking and top-k selection are repeated inside each outer training fold only.",
        "- Median imputation is fitted inside each model pipeline on the training fold only.",
        "- Held-out folds are used only for prediction and final metric calculation.",
        "",
        "## Nature-style Results draft",
        "",
        results_text,
    ]
    (docs_dir / "domain_ablation_process_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-cohort within-task domain ablation analysis.")
    parser.add_argument("--input", required=True, help="Input xlsx/csv file.")
    parser.add_argument("--sheet-name", default="age-edu")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--select-k-cap", type=int, default=14)
    parser.add_argument("--publication-models", default="AdaBoost,RF,SVM (RBF),DT")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-ci", type=float, default=0.95)
    parser.add_argument("--use-robust", action="store_true")
    args = parser.parse_args()

    configure_matplotlib()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    docs_dir = out_dir / "process_docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path, args.sheet_name)
    feature_manifest_raw, domain_to_actual, missing = match_requested_features(df)
    feature_sets = build_feature_sets(domain_to_actual)
    cohort, y, ids = prepare_full_cohort(df, args.label_col, args.id_col, feature_sets)

    feature_manifest = write_feature_manifest(feature_manifest_raw, feature_sets, cohort, out_dir)
    feature_manifest.to_csv(docs_dir / "domain_ablation_feature_mapping_table.csv", index=False)

    performance, fold_metrics, predictions, selected_by_fold, rankings = evaluate_domain_ablation(
        cohort=cohort,
        y=y,
        ids=ids,
        feature_sets=feature_sets,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        select_k_cap=args.select_k_cap,
        requested_models=parse_csv_list(args.publication_models),
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_ci=args.bootstrap_ci,
        use_robust=args.use_robust,
    )
    confusion_summary = build_confusion_summary(predictions)
    selected_frequency = build_selected_feature_frequency(
        selected_by_fold,
        feature_manifest,
        n_folds=args.n_splits * args.n_repeats,
    )
    paired_delta = build_paired_delta_auroc(fold_metrics, args.bootstrap_repeats, args.bootstrap_ci)

    performance.to_csv(out_dir / "domain_ablation_performance_table.csv", index=False)
    performance.to_excel(out_dir / "domain_ablation_performance_table.xlsx", index=False)
    fold_metrics.to_csv(out_dir / "domain_ablation_fold_metrics.csv", index=False)
    predictions.to_csv(out_dir / "domain_ablation_predictions.csv", index=False)
    confusion_summary.to_csv(out_dir / "domain_ablation_confusion_matrix_summary.csv", index=False)
    selected_frequency.to_csv(out_dir / "domain_ablation_selected_feature_frequency.csv", index=False)
    selected_by_fold.to_csv(out_dir / "domain_ablation_selected_features_by_fold.csv", index=False)
    rankings.to_csv(out_dir / "domain_ablation_feature_rankings_by_fold.csv", index=False)
    paired_delta.to_csv(out_dir / "domain_ablation_paired_delta_auroc.csv", index=False)
    paired_delta.to_excel(out_dir / "domain_ablation_paired_delta_auroc.xlsx", index=False)

    with pd.ExcelWriter(out_dir / "domain_ablation_results_workbook.xlsx") as writer:
        performance.to_excel(writer, sheet_name="performance", index=False)
        paired_delta.to_excel(writer, sheet_name="paired_delta_auroc", index=False)
        confusion_summary.to_excel(writer, sheet_name="confusion_matrix", index=False)
        selected_frequency.to_excel(writer, sheet_name="selected_frequency", index=False)
        feature_manifest.to_excel(writer, sheet_name="feature_manifest", index=False)
        fold_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)

    curves = plot_roc_curves(predictions, out_dir)
    plot_auroc_bar(performance, out_dir)
    plot_metric_heatmap(performance, out_dir)
    plot_paired_delta_auroc(paired_delta, out_dir, model="AdaBoost")

    qc = build_qc_summary(
        cohort=cohort,
        feature_sets=feature_sets,
        performance=performance,
        selected_frequency=selected_frequency,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
    )
    results_text = write_results_text(qc, performance, out_dir)

    protocol = {
        "analysis_name": "full_cohort_within_task_domain_ablation",
        "not_external_validation": True,
        "not_task_only_validation": True,
        "input": str(input_path),
        "sheet_name": args.sheet_name,
        "label_col": args.label_col,
        "id_col": args.id_col,
        "n_subjects": int(len(cohort)),
        "label_counts": {str(k): int(v) for k, v in cohort[args.label_col].value_counts().sort_index().items()},
        "feature_sets": feature_sets,
        "missing_requested_features": missing,
        "models": parse_csv_list(args.publication_models),
        "validation": {
            "outer_cv": "RepeatedStratifiedKFold",
            "n_splits": args.n_splits,
            "n_repeats": args.n_repeats,
            "same_splits_all_feature_sets": True,
            "same_subjects_all_feature_sets": qc["same_subjects_all_feature_sets"],
        },
        "feature_selection": {
            "method": "univariate_train_auc_abs_distance_from_0.5",
            "nested_within_training_fold": True,
            "selected_k_rule": "min(select_k_cap, available_features)",
            "select_k_cap": args.select_k_cap,
            "test_fold_used_for_feature_selection": False,
        },
        "imputation": {"method": "median imputation inside sklearn pipeline", "fit_on_training_fold_only": True},
        "paired_delta_auroc": {
            "method": "same-model same-fold paired AUROC differences across repeated-CV held-out folds",
            "comparisons": [
                {"name": name, "feature_set_a": a, "feature_set_b": b} for name, a, b in PAIRED_COMPARISONS
            ],
        },
        "qc": qc,
    }
    (out_dir / "domain_ablation_protocol.json").write_text(
        json.dumps(json_safe(protocol), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "domain_ablation_quality_control_summary.json").write_text(
        json.dumps(json_safe(qc), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_process_notes(docs_dir, qc, results_text, input_path, args.sheet_name)
    curves.to_csv(docs_dir / "domain_ablation_roc_curve_points.csv", index=False)
    print(f"Domain ablation results written to: {out_dir}")


if __name__ == "__main__":
    main()
