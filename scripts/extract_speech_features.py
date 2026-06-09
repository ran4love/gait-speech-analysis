#!/usr/bin/env python3
"""Extract Serial-7 speech features from WhisperX JSON outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import SRC_DIR, ensure_parent, get_path, load_config, run_command


def run_task(cfg, base: Path, task: str, dry_run: bool) -> None:
    asr_dir = get_path(cfg, base, "speech", f"{task}_asr_dir")
    out_json = get_path(cfg, base, "speech", f"{task}_features_json")
    audio_dir = get_path(cfg, base, "speech", f"{task}_audio_dir")
    if not asr_dir.exists():
        print(f"{task}: ASR JSON directory not found, skipping: {asr_dir}")
        return
    ensure_parent(out_json)
    cmd = [
        sys.executable,
        str(SRC_DIR / "speech_feature_extract.py"),
        "--json-dir",
        str(asr_dir),
        "--out-json",
        str(out_json),
        "--first-five",
        cfg.get("speech", "first_five", fallback="93,86,79,72,65"),
    ]
    if audio_dir.exists():
        cmd.extend(["--audio-dir", str(audio_dir)])
    run_command(cmd, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract speech features from WhisperX JSON.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    run_task(cfg, base, "task8", args.dry_run)
    run_task(cfg, base, "task9", args.dry_run)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
