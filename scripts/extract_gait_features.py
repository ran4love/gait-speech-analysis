#!/usr/bin/env python3
"""Extract gait features from MMPose keypoint JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import SRC_DIR, ensure_parent, get_path, load_config, run_command


def add_common_gait_args(cmd: list[str], cfg, section: str = "gait", include_stride_time: bool = False) -> list[str]:
    for option, key in [
        ("--frame-rate", "frame_rate"),
        ("--min-keypoint-score", "min_keypoint_score"),
        ("--smooth-window", "smooth_window"),
        ("--interp-max-gap", "interp_max_gap"),
    ]:
        value = cfg.get(section, key, fallback="").strip()
        if value:
            cmd.extend([option, value])
    if include_stride_time:
        value = cfg.get(section, "min_stride_time_sec", fallback="").strip()
        if value:
            cmd.extend(["--min-stride-time-sec", value])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gait features from MMPose JSON.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["all", "without-ds", "ds"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    keypoint_json_dir = get_path(cfg, base, "gait", "keypoint_json_dir")
    without_ds_json = get_path(cfg, base, "gait", "gait_without_ds_json")
    ds_json = get_path(cfg, base, "gait", "gait_ds_json")
    gait_xlsx = get_path(cfg, base, "gait", "gait_features_xlsx")
    calib = cfg.get("gait", "camera_calib_npz", fallback="").strip()
    calib_path = get_path(cfg, base, "gait", "camera_calib_npz") if calib else None

    if not keypoint_json_dir.exists() and not args.dry_run:
        raise SystemExit(f"Keypoint JSON directory not found: {keypoint_json_dir}")

    gaitfeature = SRC_DIR / "gaitfeature.py"

    if args.mode in {"all", "without-ds"}:
        ensure_parent(without_ds_json)
        cmd = [
            sys.executable,
            str(gaitfeature),
            "without_ds_json",
            "--json-directory",
            str(keypoint_json_dir),
            "--output-json-path",
            str(without_ds_json),
            "--distance-meters",
            cfg.get("gait", "distance_meters", fallback="6"),
        ]
        add_common_gait_args(cmd, cfg, include_stride_time=True)
        if calib_path:
            cmd.extend(["--camera-calib-npz", str(calib_path)])
        run_command(cmd, args.dry_run)

    if args.mode in {"all", "ds"}:
        ensure_parent(ds_json)
        cmd = [
            sys.executable,
            str(gaitfeature),
            "ds",
            "--keypoints-directory",
            str(keypoint_json_dir),
            "--new-features-file",
            str(ds_json),
            "--group",
            cfg.get("gait", "group", fallback="AD"),
        ]
        add_common_gait_args(cmd, cfg)
        if calib_path:
            cmd.extend(["--camera-calib-npz", str(calib_path)])
        run_command(cmd, args.dry_run)

    if args.mode == "all" and cfg.getboolean("gait", "write_check_previous_xlsx", fallback=True):
        ensure_parent(gait_xlsx)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "merge_gait_json_to_xlsx.py"),
            "--without-ds-json",
            str(without_ds_json),
            "--ds-json",
            str(ds_json),
            "--output-xlsx",
            str(gait_xlsx),
        ]
        run_command(cmd, args.dry_run)

    print(f"Gait feature outputs: {without_ds_json}, {ds_json}, {gait_xlsx}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
