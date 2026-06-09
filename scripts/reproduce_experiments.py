#!/usr/bin/env python3
"""Reproduce the current gait-speech joint analysis from extracted features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import SRC_DIR, get_list, get_path, load_config, run_command


def split_paths(raw_items: list[str], base: Path) -> list[Path]:
    paths = []
    for item in raw_items:
        p = Path(item).expanduser()
        paths.append(p if p.is_absolute() else (base / p).resolve())
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manuscript experiment reproduction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    demo_xlsx = get_path(cfg, base, "experiment", "demo_xlsx")
    gait_xlsx = split_paths(get_list(cfg, "experiment", "gait_xlsx"), base)
    task8_json = get_path(cfg, base, "experiment", "task8_json")
    task9_json = get_path(cfg, base, "experiment", "task9_json")
    out_dir = get_path(cfg, base, "experiment", "out_dir")

    missing = [p for p in [demo_xlsx, task8_json, task9_json, *gait_xlsx] if not p.exists()]
    if missing and not args.dry_run:
        raise SystemExit("Missing experiment input files:\n" + "\n".join(str(p) for p in missing))

    cmd = [
        sys.executable,
        str(SRC_DIR / "legacy_exploratory" / "analyze_dual_task_cost_gait_speech_from_json.py"),
        "--demo-xlsx",
        str(demo_xlsx),
        "--gait-xlsx",
        *[str(p) for p in gait_xlsx],
        "--task8-json",
        str(task8_json),
        "--task9-json",
        str(task9_json),
        "--out-dir",
        str(out_dir),
        "--target",
        cfg.get("experiment", "target", fallback="MoCA"),
        "--max-selected-features",
        cfg.get("experiment", "max_selected_features", fallback="3"),
        "--top-candidates",
        cfg.get("experiment", "top_candidates", fallback="6"),
    ]
    run_command(cmd, args.dry_run)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
