#!/usr/bin/env python3
"""Run paper-level experiment reproduction from config.ini."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import SRC_DIR, get_path, load_config, run_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-level repeated-CV, ROC, and SHAP reproduction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    input_path = get_path(cfg, base, "paper_experiment", "input_table")
    out_dir = get_path(cfg, base, "paper_experiment", "out_dir")
    if not input_path.exists() and not args.dry_run:
        raise SystemExit(f"Paper experiment input not found: {input_path}")

    cmd = [
        sys.executable,
        str(SRC_DIR / "paper_experiment_reproduction.py"),
        "--input",
        str(input_path),
        "--sheet-name",
        cfg.get("paper_experiment", "sheet_name", fallback="age-edu"),
        "--out-dir",
        str(out_dir),
        "--label-col",
        cfg.get("paper_experiment", "label_col", fallback="label"),
        "--id-col",
        cfg.get("paper_experiment", "id_col", fallback="ID"),
        "--feature-set",
        cfg.get("paper_experiment", "feature_set", fallback="nestedselected"),
        "--custom-features",
        cfg.get("paper_experiment", "custom_features", fallback=""),
        "--n-splits",
        cfg.get("paper_experiment", "n_splits", fallback="5"),
        "--n-repeats",
        cfg.get("paper_experiment", "n_repeats", fallback="10"),
        "--publication-models",
        cfg.get("paper_experiment", "publication_models", fallback="AdaBoost,RF,SVM (RBF),DT"),
        "--nested-candidate-set",
        cfg.get("paper_experiment", "nested_candidate_set", fallback="script_all"),
        "--nested-custom-features",
        cfg.get("paper_experiment", "nested_custom_features", fallback=""),
        "--nested-select-k",
        cfg.get("paper_experiment", "nested_select_k", fallback="14"),
        "--train-test-repeats",
        cfg.get("paper_experiment", "train_test_repeats", fallback="100"),
        "--test-size",
        cfg.get("paper_experiment", "test_size", fallback="0.2"),
        "--bootstrap-repeats",
        cfg.get("paper_experiment", "bootstrap_repeats", fallback="2000"),
        "--bootstrap-ci",
        cfg.get("paper_experiment", "bootstrap_ci", fallback="0.95"),
        "--age-col",
        cfg.get("paper_experiment", "age_col", fallback="age"),
        "--education-col",
        cfg.get("paper_experiment", "education_col", fallback="edu"),
        "--sex-col",
        cfg.get("paper_experiment", "sex_col", fallback=""),
        "--auroc-stability-margin",
        cfg.get("paper_experiment", "auroc_stability_margin", fallback="0.02"),
        "--regression-target",
        cfg.get("paper_experiment", "regression_target", fallback="MoCA"),
        "--shap-models",
        cfg.get("paper_experiment", "shap_models", fallback="GBDT,Random Forest,Logistic Regression"),
        "--roc-models",
        cfg.get("paper_experiment", "roc_models", fallback="GBDT,Random Forest,Logistic Regression,SVM (RBF)"),
        "--max-shap-display",
        cfg.get("paper_experiment", "max_shap_display", fallback="20"),
    ]
    if cfg.getboolean("paper_experiment", "use_robust", fallback=False):
        cmd.append("--use-robust")
    cmd.append(
        "--run-train-test-split"
        if cfg.getboolean("paper_experiment", "run_train_test_split", fallback=True)
        else "--no-run-train-test-split"
    )
    cmd.append(
        "--run-publication-outputs"
        if cfg.getboolean("paper_experiment", "run_publication_outputs", fallback=True)
        else "--no-run-publication-outputs"
    )
    cmd.append(
        "--run-confounder-adjustment"
        if cfg.getboolean("paper_experiment", "run_confounder_adjustment", fallback=True)
        else "--no-run-confounder-adjustment"
    )
    cmd.append(
        "--run-nested-feature-selection"
        if cfg.getboolean("paper_experiment", "run_nested_feature_selection", fallback=True)
        else "--no-run-nested-feature-selection"
    )
    cmd.append(
        "--promote-nested-selection"
        if cfg.getboolean("paper_experiment", "promote_nested_selection", fallback=True)
        else "--no-promote-nested-selection"
    )
    cmd.append("--run-shap" if cfg.getboolean("paper_experiment", "run_shap", fallback=True) else "--no-run-shap")
    cmd.append(
        "--run-regression"
        if cfg.getboolean("paper_experiment", "run_regression", fallback=True)
        else "--no-run-regression"
    )
    run_command(cmd, args.dry_run)

    table2_input = out_dir / "classification" / "model_performance_table.csv"
    table2_prefix = out_dir / "classification" / "table2_classification_performance"
    table2_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "make_table2_classification.py"),
        "--input",
        str(table2_input),
        "--out-prefix",
        str(table2_prefix),
        "--source-note",
        (
            "Repeated stratified five-fold cross-validation with ten repeats; "
            "feature selection nested within each outer training fold; bootstrap 95% confidence intervals."
        ),
    ]
    if table2_input.exists() or args.dry_run:
        run_command(table2_cmd, args.dry_run)
    else:
        print(f"Skipping Table 2 generation because classification output was not found: {table2_input}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
