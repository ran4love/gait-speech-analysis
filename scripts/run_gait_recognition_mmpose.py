#!/usr/bin/env python3
"""Run MMPose recognition through a user-supplied command template."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from common import get_list, get_path, load_config, run_command, run_shell


def format_template(template: str, **values: Path) -> str:
    mapping = {k: str(v) for k, v in values.items()}
    mapping.update({f"{k}_q": shlex.quote(str(v)) for k, v in values.items()})
    return template.format(**mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gait video-to-keypoint recognition with MMPose.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    run_mmpose = cfg.getboolean("gait", "run_mmpose", fallback=False)
    video_dir = get_path(cfg, base, "gait", "video_dir")
    keypoint_json_dir = get_path(cfg, base, "gait", "keypoint_json_dir")
    template = cfg.get("gait", "mmpose_command_template", fallback="").strip()
    use_shell = cfg.getboolean("gait", "mmpose_use_shell", fallback=False)

    if not run_mmpose:
        print("run_mmpose = no; skipping MMPose recognition.")
        print(f"Expected keypoint JSON directory: {keypoint_json_dir}")
        return

    if not template:
        raise SystemExit(
            "run_mmpose = yes but mmpose_command_template is empty. "
            "Edit config.example.ini or copy it to config.ini and set the MMPose command."
        )
    if not video_dir.exists():
        raise SystemExit(f"Video directory not found: {video_dir}")

    keypoint_json_dir.mkdir(parents=True, exist_ok=True)
    exts = {x.lower() for x in get_list(cfg, "gait", "video_extensions", ".mp4,.mov,.avi,.mkv")}
    videos = [p for p in sorted(video_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

    if "{video}" in template or "{video_q}" in template:
        if not videos:
            raise SystemExit(f"No videos found in {video_dir}")
        for video in videos:
            command = format_template(template, video=video, video_dir=video_dir, keypoint_json_dir=keypoint_json_dir)
            run_shell(command, dry_run=args.dry_run) if use_shell else run_command(shlex.split(command), args.dry_run)
    else:
        command = format_template(template, video_dir=video_dir, keypoint_json_dir=keypoint_json_dir)
        run_shell(command, dry_run=args.dry_run) if use_shell else run_command(shlex.split(command), args.dry_run)

    print(f"MMPose output directory: {keypoint_json_dir}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
