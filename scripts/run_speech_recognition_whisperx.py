#!/usr/bin/env python3
"""Run WhisperX ASR for TASK08 and TASK09 speech recordings."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from common import SRC_DIR, get_path, load_config, run_command


def run_task(cfg, base: Path, task: str, preset: str, dry_run: bool) -> None:
    audio_dir = get_path(cfg, base, "speech", f"{task}_audio_dir")
    asr_dir = get_path(cfg, base, "speech", f"{task}_asr_dir")
    if not audio_dir.exists():
        print(f"{task}: audio directory not found, skipping: {audio_dir}")
        return
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        print(f"{task}: no .wav files found, skipping: {audio_dir}")
        return
    asr_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SRC_DIR / "whisperx_recognize.py"),
        "--input-dir",
        str(audio_dir),
        "--output-dir",
        str(asr_dir),
        "--model-name",
        cfg.get("speech", "model_name", fallback="large-v2"),
        "--language",
        cfg.get("speech", "language", fallback="zh"),
        "--compute-type",
        cfg.get("speech", "compute_type", fallback="float16"),
        "--preset",
        preset,
    ]
    run_command(cmd, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WhisperX ASR for speech tasks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    if not cfg.getboolean("speech", "run_whisperx", fallback=False):
        print("run_whisperx = no; skipping WhisperX recognition.")
        return
    if not args.dry_run and importlib.util.find_spec("whisperx") is None:
        raise SystemExit("whisperx is not installed in this Python environment. See docs/environment_notes.md.")

    run_task(cfg, base, "task8", "st", args.dry_run)
    run_task(cfg, base, "task9", "dt", args.dry_run)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
