#!/usr/bin/env python3
"""Config wrapper for Python-only statistical analysis."""

from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "statistical_analysis_python.py"


def get_path(cfg: configparser.ConfigParser, base: Path, section: str, key: str) -> Path:
    raw = cfg.get(section, key).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python-only statistical analysis from config.ini.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = Path(args.config).expanduser().resolve()
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(config, encoding="utf-8")
    base = config.parent

    if not cfg.has_section("statistics"):
        print("No [statistics] section in config; skipping statistical analysis.")
        return

    input_table = get_path(cfg, base, "statistics", "input_table")
    out_dir = get_path(cfg, base, "statistics", "out_dir")
    if not input_table.exists():
        print(f"Statistics input not found; skipping: {input_table}")
        return

    cmd = [
        sys.executable,
        str(SRC),
        "--input",
        str(input_table),
        "--sheet-name",
        cfg.get("statistics", "sheet_name", fallback="info"),
        "--out-dir",
        str(out_dir),
        "--label-col",
        cfg.get("statistics", "label_col", fallback="label"),
        "--id-col",
        cfg.get("statistics", "id_col", fallback="ID"),
        "--group-col",
        cfg.get("statistics", "group_col", fallback="label"),
        "--group-map",
        cfg.get("statistics", "group_map", fallback="0:CU,1:CI"),
        "--group-order",
        cfg.get("statistics", "group_order", fallback=""),
        "--baseline-continuous",
        cfg.get("statistics", "baseline_continuous", fallback="MMSE,MoCA,edu,age"),
        "--categorical-vars",
        cfg.get("statistics", "categorical_vars", fallback="gender"),
        "--feature-cols",
        cfg.get("statistics", "feature_cols", fallback=""),
        "--correlation-cols",
        cfg.get("statistics", "correlation_cols", fallback=""),
        "--alpha",
        cfg.get("statistics", "alpha", fallback="0.05"),
        "--outlier-thr",
        cfg.get("statistics", "outlier_thr", fallback="0.02"),
        "--top-boxplot-features",
        cfg.get("statistics", "top_boxplot_features", fallback="12"),
        "--max-corr-vars",
        cfg.get("statistics", "max_corr_vars", fallback="12"),
    ]
    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
