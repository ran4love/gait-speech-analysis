#!/usr/bin/env python3
"""Generate manuscript figures from public reproduction outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import SRC_DIR, get_path, load_config, run_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures from existing result tables.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    domain_results_dir = get_path(cfg, base, "figures", "domain_results_dir")
    paper_results_dir = get_path(cfg, base, "figures", "paper_results_dir")
    out_dir = get_path(cfg, base, "figures", "out_dir")

    if not args.dry_run:
        for label, path in [
            ("domain_results_dir", domain_results_dir),
            ("paper_results_dir", paper_results_dir),
        ]:
            if not path.exists():
                raise SystemExit(f"Figure input {label} not found: {path}")

    cmd = [
        sys.executable,
        str(SRC_DIR / "figure_model_contribution_interpretability.py"),
        "--domain-results-dir",
        str(domain_results_dir),
        "--paper-results-dir",
        str(paper_results_dir),
        "--out-dir",
        str(out_dir),
        "--figure-stem",
        cfg.get("figures", "fig4_stem", fallback="fig4_model_contribution_interpretability"),
        "--model",
        cfg.get("figures", "domain_model", fallback="AdaBoost"),
        "--bootstrap-repeats",
        cfg.get("figures", "bootstrap_repeats", fallback="5000"),
        "--bootstrap-ci",
        cfg.get("figures", "bootstrap_ci", fallback="0.95"),
    ]
    run_command(cmd, args.dry_run)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
