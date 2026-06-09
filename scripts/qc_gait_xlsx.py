#!/usr/bin/env python3
"""Build a physics-informed QC report for check_previous-format gait xlsx files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import SRC_DIR, get_path, load_config, run_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gait xlsx QC.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    input_xlsx = get_path(cfg, base, "gait", "gait_xlsx_for_qc")
    output_xlsx = get_path(cfg, base, "gait", "gait_qc_xlsx")
    if not input_xlsx.exists() and not args.dry_run:
        raise SystemExit(f"QC input xlsx not found: {input_xlsx}")

    cmd = [
        sys.executable,
        str(SRC_DIR / "qc_check_previous_2026.py"),
        "--input-xlsx",
        str(input_xlsx),
        "--output-xlsx",
        str(output_xlsx),
    ]
    run_command(cmd, args.dry_run)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
