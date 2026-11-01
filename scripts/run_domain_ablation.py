#!/usr/bin/env python3
"""Run full-cohort within-task domain ablation from config.ini."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import SRC_DIR, get_path, load_config, run_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-cohort within-task domain ablation analysis.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    input_path = get_path(cfg, base, "domain_ablation", "input_table")
    out_dir = get_path(cfg, base, "domain_ablation", "out_dir")
    if not input_path.exists() and not args.dry_run:
        raise SystemExit(f"Domain ablation input not found: {input_path}")

    cmd = [
        sys.executable,
        str(SRC_DIR / "domain_ablation_analysis.py"),
        "--input",
        str(input_path),
        "--sheet-name",
        cfg.get("domain_ablation", "sheet_name", fallback="age-edu"),
        "--out-dir",
        str(out_dir),
        "--label-col",
        cfg.get("domain_ablation", "label_col", fallback="label"),
        "--id-col",
        cfg.get("domain_ablation", "id_col", fallback="ID"),
        "--n-splits",
        cfg.get("domain_ablation", "n_splits", fallback="5"),
        "--n-repeats",
        cfg.get("domain_ablation", "n_repeats", fallback="10"),
        "--select-k-cap",
        cfg.get("domain_ablation", "select_k_cap", fallback="14"),
        "--publication-models",
        cfg.get("domain_ablation", "publication_models", fallback="AdaBoost,RF,SVM (RBF),DT"),
        "--bootstrap-repeats",
        cfg.get("domain_ablation", "bootstrap_repeats", fallback="2000"),
        "--bootstrap-ci",
        cfg.get("domain_ablation", "bootstrap_ci", fallback="0.95"),
    ]
    if cfg.getboolean("domain_ablation", "use_robust", fallback=False):
        cmd.append("--use-robust")
    run_command(cmd, args.dry_run)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
