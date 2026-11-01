#!/usr/bin/env python3
"""Tune top-k inside each outer training fold using inner CV."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adualnet_mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "adualnet_xdg_cache"))

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from paper_experiment_reproduction import (  # noqa: E402
    RANDOM_STATE,
    make_publication_classification_models,
    parse_csv_list,
    predict_probability,
    prepare_xy,
    read_table,
    safe_fit,
    select_candidate_features,
    select_top_k_features_train_only,
    summarize_repeated_metrics,
    write_publication_calibration_curves,
    write_publication_confusion_matrices,
    write_publication_roc_curves,
)
from run_nested_k_sensitivity import dataframe_to_markdown, parse_int_list  # noqa: E402


PALETTE = {
    "AdaBoost": "#4C78A8",
    "RF": "#59A14F",
    "SVM (RBF)": "#F28E2B",
    "DT": "#9C755F",
}


def parse_rule_list(value: str) -> List[str]:
    rules = [x.strip() for x in value.split(",") if x.strip()]
    valid = {"max_auc", "within_margin"}
    unknown = sorted(set(rules) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown selection rule(s): {', '.join(unknown)}")
    return rules


def choose_k(inner_scores: pd.DataFrame, rule: str, margin: float) -> int:
    ranked = inner_scores.sort_values(["mean_inner_auroc", "k"], ascending=[False, True]).copy()
    if rule == "max_auc":
        return int(ranked.iloc[0]["k"])
    if rule == "within_margin":
        best = float(ranked.iloc[0]["mean_inner_auroc"])
        eligible = inner_scores[inner_scores["mean_inner_auroc"] >= best - margin].sort_values("k")
        return int(eligible.iloc[0]["k"])
    raise ValueError(f"Unsupported selection rule: {rule}")


def tune_k_inner_cv(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    feature_names: Sequence[str],
    model_proto: object,
    model_name: str,
    k_values: Sequence[int],
    inner_splits: int,
    outer_fold: int,
) -> pd.DataFrame:
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE + outer_fold)
    rows: List[Dict[str, object]] = []
    for k in k_values:
        aucs: List[float] = []
        for inner_fold, (tr_idx, va_idx) in enumerate(inner.split(X_outer_train, y_outer_train), start=1):
            selected_idx, _ = select_top_k_features_train_only(
                X_outer_train[tr_idx],
                y_outer_train[tr_idx],
                feature_names,
                k,
            )
            est = safe_fit(clone(model_proto), X_outer_train[tr_idx][:, selected_idx], y_outer_train[tr_idx])
            y_prob = predict_probability(est, X_outer_train[va_idx][:, selected_idx])
            aucs.append(float(roc_auc_score(y_outer_train[va_idx], y_prob)))
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "k": k,
                    "inner_fold": inner_fold,
                    "inner_auroc": aucs[-1],
                }
            )
    fold_scores = pd.DataFrame(rows)
    summary = (
        fold_scores.groupby(["outer_fold", "model", "k"], as_index=False)
        .agg(
            mean_inner_auroc=("inner_auroc", "mean"),
            std_inner_auroc=("inner_auroc", "std"),
            n_inner_folds=("inner_auroc", "size"),
        )
        .sort_values(["outer_fold", "model", "k"])
    )
    summary.attrs["inner_fold_scores"] = fold_scores
    return summary


def plot_selected_k_frequency(freq: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
        }
    )
    rules = list(freq["selection_rule"].drop_duplicates())
    models = list(freq["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(rules), figsize=(3.6 * max(1, len(rules)), 2.8), sharey=True)
    if len(rules) == 1:
        axes = [axes]
    for ax, rule in zip(axes, rules):
        sub = freq[freq["selection_rule"].eq(rule)]
        k_values = sorted(sub["selected_k"].unique())
        x = np.arange(len(k_values))
        width = 0.18
        for offset_idx, model in enumerate(models):
            model_sub = sub[sub["model"].eq(model)].set_index("selected_k")
            values = [float(model_sub.loc[k, "selected_k_rate"]) if k in model_sub.index else 0.0 for k in k_values]
            offset = (offset_idx - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=PALETTE.get(model, "#333333"),
                label=model,
                alpha=0.92,
            )
        label = "Max inner AUROC" if rule == "max_auc" else "Within 0.01 of best"
        ax.set_title(label, loc="left", fontweight="bold", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel("Selected k")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#E3E8EC", lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Outer folds selecting k (%)")
    axes[-1].legend(frameon=False, fontsize=6.2, loc="upper right")
    fig.tight_layout()
    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"innercv_selected_k_frequency.{ext}", **kwargs)
    plt.close(fig)


def write_summary_report(performance: pd.DataFrame, freq: pd.DataFrame, out_dir: Path, margin: float) -> None:
    perf_cols = [
        "selection_rule",
        "model",
        "mean_auroc",
        "ci95_low_auroc",
        "ci95_high_auroc",
        "mean_acc",
        "mean_precision",
        "mean_recall",
        "mean_f1",
    ]
    top_freq = freq.sort_values(["selection_rule", "model", "selected_k_rate"], ascending=[True, True, False])
    top_freq = top_freq.groupby(["selection_rule", "model"], as_index=False).head(1)
    freq_cols = ["selection_rule", "model", "selected_k", "selected_k_count", "selected_k_rate"]
    lines = [
        "# Inner-CV top-k tuning analysis",
        "",
        (
            "This analysis treats the number of selected features as a hyperparameter. "
            "For each outer training fold, each candidate k is evaluated by inner cross-validation; "
            "the selected k is then refit on the full outer training fold before evaluating the outer held-out fold."
        ),
        "",
        f"`max_auc` chooses the k with the highest mean inner-CV AUROC. `within_margin` chooses the smallest k within {margin:.3f} AUROC of the best inner-CV k.",
        "",
        "## Outer-fold performance after inner-CV k tuning",
        "",
        dataframe_to_markdown(performance[perf_cols], perf_cols),
        "",
        "## Most frequently selected k",
        "",
        dataframe_to_markdown(top_freq[freq_cols], freq_cols),
        "",
        "## Interpretation guardrail",
        "",
        "- Use `max_auc` when claiming pure performance-based hyperparameter tuning.",
        "- Use `within_margin` only when the manuscript pre-specifies a parsimony rule.",
        "- Do not claim k=14 was tuned as optimal unless the corresponding rule actually selects k=14 frequently enough to support that claim.",
    ]
    (out_dir / "innercv_k_tuning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune nested top-k inside each outer training fold.")
    parser.add_argument("--input", required=True, help="Input xlsx/csv file.")
    parser.add_argument("--sheet-name", default="info")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--k-values", type=parse_int_list, default=parse_int_list("5,10,14,20"))
    parser.add_argument("--selection-rules", type=parse_rule_list, default=parse_rule_list("max_auc,within_margin"))
    parser.add_argument("--within-margin", type=float, default=0.01)
    parser.add_argument("--nested-candidate-set", default="script_all")
    parser.add_argument("--nested-custom-features", default="")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--publication-models", default="AdaBoost,RF,SVM (RBF),DT")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-ci", type=float, default=0.95)
    parser.add_argument("--use-robust", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(Path(args.input).expanduser().resolve(), args.sheet_name)
    candidate_features = select_candidate_features(df, args.nested_candidate_set, args.nested_custom_features)
    dataset, X, y, ids = prepare_xy(df, candidate_features, args.label_col, args.id_col)
    dataset_cols = ([args.id_col] if args.id_col in dataset.columns else []) + [args.label_col] + candidate_features
    dataset[dataset_cols].to_csv(out_dir / "innercv_k_tuning_dataset.csv", index=False)

    all_models = make_publication_classification_models(use_robust=args.use_robust)
    model_names = [name for name in parse_csv_list(args.publication_models) if name in all_models]
    if not model_names:
        raise ValueError("No valid publication models selected.")

    outer = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=RANDOM_STATE)
    inner_score_rows: List[pd.DataFrame] = []
    inner_fold_score_rows: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []

    for outer_fold, (tr_idx, te_idx) in enumerate(outer.split(X, y), start=1):
        X_tr_outer, X_te_outer = X[tr_idx], X[te_idx]
        y_tr_outer, y_te_outer = y[tr_idx], y[te_idx]
        for model_name in model_names:
            proto = all_models[model_name]
            k_summary = tune_k_inner_cv(
                X_tr_outer,
                y_tr_outer,
                candidate_features,
                proto,
                model_name,
                args.k_values,
                args.inner_splits,
                outer_fold,
            )
            inner_fold_scores = k_summary.attrs.pop("inner_fold_scores")
            inner_score_rows.append(k_summary)
            inner_fold_score_rows.append(inner_fold_scores)

            for rule in args.selection_rules:
                selected_k = choose_k(k_summary, rule, args.within_margin)
                selected_idx, rank_df = select_top_k_features_train_only(
                    X_tr_outer,
                    y_tr_outer,
                    candidate_features,
                    selected_k,
                )
                selected_features = [candidate_features[i] for i in selected_idx]
                est = safe_fit(clone(proto), X_tr_outer[:, selected_idx], y_tr_outer)
                y_prob = predict_probability(est, X_te_outer[:, selected_idx])
                y_pred = (y_prob >= 0.5).astype(int)
                metric_rows.append(
                    {
                        "selection_rule": rule,
                        "feature_set": "Inner-CV tuned top-k",
                        "model": model_name,
                        "fold": outer_fold,
                        "selected_k": selected_k,
                        "auroc": roc_auc_score(y_te_outer, y_prob),
                        "acc": accuracy_score(y_te_outer, y_pred),
                        "precision": precision_score(y_te_outer, y_pred, zero_division=0),
                        "recall": recall_score(y_te_outer, y_pred, zero_division=0),
                        "f1": f1_score(y_te_outer, y_pred, zero_division=0),
                        "selected_features": ",".join(selected_features),
                    }
                )
                selected_rows.append(
                    {
                        "selection_rule": rule,
                        "model": model_name,
                        "fold": outer_fold,
                        "selected_k": selected_k,
                        "selected_features": ",".join(selected_features),
                    }
                )
                rank_out = rank_df.copy()
                rank_out.insert(0, "selection_rule", rule)
                rank_out.insert(1, "model", model_name)
                rank_out.insert(2, "fold", outer_fold)
                rank_out.insert(3, "selected_k", selected_k)
                rank_out.insert(4, "selected", rank_out["feature"].isin(selected_features))
                ranking_rows.extend(rank_out.to_dict("records"))
                for local_idx, case_idx in enumerate(te_idx):
                    prediction_rows.append(
                        {
                            "selection_rule": rule,
                            "feature_set": "Inner-CV tuned top-k",
                            "model": model_name,
                            "fold": outer_fold,
                            "selected_k": selected_k,
                            "ID": ids[case_idx],
                            "y_true": int(y_te_outer[local_idx]),
                            "y_prob": float(y_prob[local_idx]),
                            "y_pred": int(y_pred[local_idx]),
                            "selected_features": ",".join(selected_features),
                        }
                    )
        if outer_fold % 5 == 0:
            print(f"Completed outer fold {outer_fold}", flush=True)

    inner_scores = pd.concat(inner_score_rows, ignore_index=True)
    inner_fold_scores = pd.concat(inner_fold_score_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected = pd.DataFrame(selected_rows)
    rankings = pd.DataFrame(ranking_rows)

    performance = summarize_repeated_metrics(
        fold_metrics,
        metrics=["auroc", "acc", "precision", "recall", "f1"],
        sort_cols=["selection_rule", "mean_auroc"],
        ascending=[True, False],
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_ci=args.bootstrap_ci,
        group_cols=("selection_rule", "model"),
    )

    freq = (
        selected.groupby(["selection_rule", "model", "selected_k"], as_index=False)
        .agg(selected_k_count=("selected_k", "size"))
        .sort_values(["selection_rule", "model", "selected_k"])
    )
    denominator = args.n_splits * args.n_repeats
    freq["selected_k_rate"] = freq["selected_k_count"] / denominator

    inner_scores.to_csv(out_dir / "innercv_k_scores_by_outer_fold.csv", index=False)
    inner_fold_scores.to_csv(out_dir / "innercv_k_scores_by_inner_fold.csv", index=False)
    fold_metrics.to_csv(out_dir / "innercv_tuned_k_fold_metrics.csv", index=False)
    performance.to_csv(out_dir / "innercv_tuned_k_performance.csv", index=False)
    predictions.to_csv(out_dir / "innercv_tuned_k_predictions.csv", index=False)
    selected.to_csv(out_dir / "innercv_tuned_k_selected_by_fold.csv", index=False)
    rankings.to_csv(out_dir / "innercv_tuned_k_feature_rankings_by_fold.csv", index=False)
    freq.to_csv(out_dir / "innercv_selected_k_frequency.csv", index=False)

    with pd.ExcelWriter(out_dir / "innercv_k_tuning_results.xlsx") as writer:
        performance.to_excel(writer, sheet_name="performance", index=False)
        freq.to_excel(writer, sheet_name="selected_k_frequency", index=False)
        fold_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)
        inner_scores.to_excel(writer, sheet_name="inner_scores", index=False)
        selected.to_excel(writer, sheet_name="selected_by_fold", index=False)

    for rule in args.selection_rules:
        rule_dir = out_dir / f"publication_outputs_{rule}"
        rule_dir.mkdir(exist_ok=True)
        write_publication_confusion_matrices(predictions[predictions["selection_rule"].eq(rule)], rule_dir)
        write_publication_roc_curves(predictions[predictions["selection_rule"].eq(rule)], rule_dir)
        write_publication_calibration_curves(predictions[predictions["selection_rule"].eq(rule)], rule_dir)

    protocol = {
        "purpose": "Nested top-k hyperparameter tuning using inner CV within each outer training fold.",
        "leakage_control": (
            "Candidate k values are evaluated only inside the outer training fold. "
            "The outer held-out fold is used once after k selection, feature ranking, and model fitting."
        ),
        "k_values": args.k_values,
        "selection_rules": args.selection_rules,
        "within_margin": args.within_margin,
        "candidate_set": args.nested_candidate_set,
        "n_splits_outer": args.n_splits,
        "n_repeats_outer": args.n_repeats,
        "inner_splits": args.inner_splits,
        "models": model_names,
    }
    (out_dir / "innercv_k_tuning_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_selected_k_frequency(freq, out_dir)
    write_summary_report(performance, freq, out_dir, args.within_margin)
    print(f"Inner-CV top-k tuning results written to: {out_dir}")


if __name__ == "__main__":
    main()
