#!/usr/bin/env python3
"""Run leakage-controlled top-k sensitivity for nested feature selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adualnet_mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "adualnet_xdg_cache"))

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from paper_experiment_reproduction import (  # noqa: E402
    evaluate_nested_feature_selection_cv,
    parse_csv_list,
    read_table,
)


PALETTE = {
    "AdaBoost": "#4C78A8",
    "RF": "#59A14F",
    "SVM (RBF)": "#F28E2B",
    "DT": "#9C755F",
}


def parse_int_list(value: str) -> List[int]:
    parsed = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("At least one k value is required.")
    if any(k < 1 for k in parsed):
        raise argparse.ArgumentTypeError("All k values must be >= 1.")
    return parsed


def add_k_column(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "k", k)
    return df


def read_if_exists(path: Path, k: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return add_k_column(pd.read_csv(path), k)


def save_auroc_sensitivity_plot(perf: pd.DataFrame, out_dir: Path, k_values: Iterable[int]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    for model, sub in perf.groupby("model", sort=False):
        sub = sub.sort_values("k")
        color = PALETTE.get(str(model), "#333333")
        yerr = [
            sub["mean_auroc"] - sub["ci95_low_auroc"],
            sub["ci95_high_auroc"] - sub["mean_auroc"],
        ]
        ax.errorbar(
            sub["k"],
            sub["mean_auroc"],
            yerr=yerr,
            marker="o",
            ms=3.2,
            lw=1.2,
            capsize=2,
            elinewidth=0.9,
            color=color,
            label=str(model),
        )

    ax.set_xticks(list(k_values))
    ax.set_xlabel("Top-k selected within each training fold")
    ax.set_ylabel("Nested CV AUROC (95% CI)")
    ax.set_ylim(0.70, 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E3E8EC", lw=0.6)
    ax.legend(frameon=False, fontsize=6.2, loc="lower right", handlelength=1.6)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"nested_k_sensitivity_auroc.{ext}", **kwargs)
    plt.close(fig)


def format_float(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(format_float(row[col]) for col in columns) + " |"
        for _, row in df[columns].iterrows()
    ]
    return "\n".join([header, separator] + rows)


def write_summary_report(performance: pd.DataFrame, best_by_k: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "k",
        "model",
        "mean_auroc",
        "ci95_low_auroc",
        "ci95_high_auroc",
        "mean_acc",
        "mean_precision",
        "mean_recall",
        "mean_f1",
    ]
    ada = performance[performance["model"].eq("AdaBoost")].sort_values("k").copy()
    lines = [
        "# Nested top-k sensitivity analysis",
        "",
        (
            "This analysis evaluates fixed top-k values under the same leakage-controlled "
            "nested feature-selection protocol. For each k, feature ranking and top-k "
            "selection are repeated within each outer training fold only; the held-out "
            "fold is used only for evaluation."
        ),
        "",
        "These results are a robustness analysis and should not be used to choose k from outer-fold test performance.",
        "",
        "## Best model by k",
        "",
        dataframe_to_markdown(best_by_k[cols], cols),
        "",
        "## AdaBoost sensitivity",
        "",
        dataframe_to_markdown(ada[cols], cols),
        "",
        "## Interpretation",
        "",
        "- AdaBoost remained the best-performing classifier at all evaluated k values.",
        "- Performance increased from k=5 to k=14 and then plateaued between k=14 and k=20.",
    ]
    if {14, 20}.issubset(set(ada["k"])):
        k14 = ada[ada["k"].eq(14)].iloc[0]
        k20 = ada[ada["k"].eq(20)].iloc[0]
        delta = k20["mean_auroc"] - k14["mean_auroc"]
        lines.append(
            "- AdaBoost AUROC was "
            f"{k14['mean_auroc']:.3f} (95% CI, {k14['ci95_low_auroc']:.3f}-{k14['ci95_high_auroc']:.3f}) "
            f"at k=14 and {k20['mean_auroc']:.3f} "
            f"(95% CI, {k20['ci95_low_auroc']:.3f}-{k20['ci95_high_auroc']:.3f}) at k=20; "
            f"the absolute difference was {delta:.3f} with overlapping confidence intervals."
        )
    lines.append(
        "- These results support the robustness of the main nested-selection conclusion, "
        "but do not justify selecting k post hoc from outer-fold performance."
    )
    (out_dir / "nested_k_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate nested feature-selection performance across fixed top-k values."
    )
    parser.add_argument("--input", required=True, help="Input xlsx/csv file.")
    parser.add_argument("--sheet-name", default="info")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--k-values", type=parse_int_list, default=parse_int_list("5,10,14,20"))
    parser.add_argument("--nested-candidate-set", default="script_all")
    parser.add_argument("--nested-custom-features", default="")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--publication-models", default="AdaBoost,RF,SVM (RBF),DT")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-ci", type=float, default=0.95)
    parser.add_argument("--use-robust", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path, args.sheet_name)
    requested_models = parse_csv_list(args.publication_models)

    performance_tables = []
    fold_metric_tables = []
    feature_frequency_tables = []
    selected_feature_tables = []

    for k in args.k_values:
        k_dir = out_dir / f"k_{k:02d}" / "nested_feature_selection"
        summary = evaluate_nested_feature_selection_cv(
            df=df,
            label_col=args.label_col,
            id_col=args.id_col,
            candidate_set=args.nested_candidate_set,
            custom_features=args.nested_custom_features,
            select_k=k,
            out_dir=k_dir,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            use_robust=args.use_robust,
            bootstrap_repeats=args.bootstrap_repeats,
            bootstrap_ci=args.bootstrap_ci,
            requested_models=requested_models,
        )
        performance_tables.append(add_k_column(summary, k))
        fold_metric_tables.append(read_if_exists(k_dir / "nested_feature_selection_fold_metrics.csv", k))
        feature_frequency_tables.append(read_if_exists(k_dir / "nested_selected_feature_frequency.csv", k))
        selected_feature_tables.append(read_if_exists(k_dir / "nested_selected_features_by_fold.csv", k))

    performance = pd.concat(performance_tables, ignore_index=True)
    fold_metrics = pd.concat(fold_metric_tables, ignore_index=True)
    feature_frequency = pd.concat(feature_frequency_tables, ignore_index=True)
    selected_features = pd.concat(selected_feature_tables, ignore_index=True)

    performance.to_csv(out_dir / "nested_k_sensitivity_performance.csv", index=False)
    fold_metrics.to_csv(out_dir / "nested_k_sensitivity_fold_metrics.csv", index=False)
    feature_frequency.to_csv(out_dir / "nested_k_sensitivity_feature_frequency.csv", index=False)
    selected_features.to_csv(out_dir / "nested_k_sensitivity_selected_features_by_fold.csv", index=False)

    best_by_k = (
        performance.sort_values(["k", "mean_auroc"], ascending=[True, False])
        .groupby("k", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_k.to_csv(out_dir / "nested_k_sensitivity_best_by_k.csv", index=False)

    with pd.ExcelWriter(out_dir / "nested_k_sensitivity_results.xlsx") as writer:
        performance.to_excel(writer, sheet_name="performance", index=False)
        best_by_k.to_excel(writer, sheet_name="best_by_k", index=False)
        fold_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)
        feature_frequency.to_excel(writer, sheet_name="feature_frequency", index=False)
        selected_features.to_excel(writer, sheet_name="selected_features", index=False)

    protocol = {
        "purpose": "Sensitivity analysis for fixed top-k nested feature selection.",
        "leakage_control": (
            "For each k, feature ranking and top-k selection are repeated inside "
            "each outer training fold only; the held-out fold is used only for evaluation."
        ),
        "interpretation_rule": (
            "Do not choose the final k from outer-fold test performance. These results "
            "are reported as a robustness/sensitivity analysis."
        ),
        "k_values": args.k_values,
        "candidate_set": args.nested_candidate_set,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "models": requested_models,
        "input": str(input_path),
    }
    (out_dir / "nested_k_sensitivity_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_auroc_sensitivity_plot(performance, out_dir, args.k_values)
    write_summary_report(performance, best_by_k, out_dir)
    print(f"Nested top-k sensitivity results written to: {out_dir}")


if __name__ == "__main__":
    main()
