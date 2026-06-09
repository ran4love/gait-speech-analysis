#!/usr/bin/env python3
"""One-command runner for the public gait-speech workflow."""

from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent

VALIDATE_STEP = "validate"
STEP_ORDER = [
    "gait-recognition",
    "gait-features",
    "speech-recognition",
    "speech-features",
    "experiments",
    "paper-experiments",
]

STEP_SCRIPT = {
    "validate": ROOT / "scripts" / "validate_inputs.py",
    "gait-recognition": ROOT / "scripts" / "run_gait_recognition_mmpose.py",
    "gait-features": ROOT / "scripts" / "extract_gait_features.py",
    "speech-recognition": ROOT / "scripts" / "run_speech_recognition_whisperx.py",
    "speech-features": ROOT / "scripts" / "extract_speech_features.py",
    "experiments": ROOT / "scripts" / "reproduce_experiments.py",
    "paper-experiments": ROOT / "scripts" / "reproduce_paper_experiments.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_status_path(config: Path) -> Path:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(config, encoding="utf-8")
        output_raw = parser.get("project", "output_dir", fallback="results").strip() or "results"
    except Exception:
        output_raw = "results"
    output_dir = Path(output_raw).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "step_status.json"


def classify_status(return_code: int, output: str, dry_run: bool) -> str:
    if dry_run:
        return "dry_run"
    if return_code != 0:
        return "failed"
    lower = output.lower()
    if "skipping" in lower and "paper_public_code/src/" not in output and "merge_gait_json_to_xlsx.py" not in output:
        return "skipped"
    return "success"


def update_step_status(status_path: Path, step: str, record: dict) -> None:
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    payload.setdefault("steps", {})
    payload["steps"][step] = record
    payload["last_step"] = step
    payload["updated_at"] = utc_now()
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_step(step: str, config: Path, dry_run: bool, status_path: Path) -> None:
    cmd = [sys.executable, str(STEP_SCRIPT[step]), "--config", str(config)]
    if dry_run:
        cmd.append("--dry-run")
    print(f"\n=== {step} ===", flush=True)
    print(" ".join(cmd), flush=True)
    started = utc_now()
    t0 = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    ended = utc_now()
    status = classify_status(proc.returncode, output, dry_run)
    skip_messages = [line for line in output.splitlines() if "skipping" in line.lower()]
    record = {
        "status": status,
        "return_code": proc.returncode,
        "dry_run": dry_run,
        "started_at": started,
        "ended_at": ended,
        "duration_sec": round(time.time() - t0, 3),
        "command": cmd,
        "skip_messages": skip_messages,
        "status_file": str(status_path),
    }
    if proc.returncode != 0:
        record["error_tail"] = "\n".join(output.splitlines()[-20:])
    update_step_status(status_path, step, record)
    print(f"Step status: {status} (status file: {status_path})", flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MMPose/WhisperX recognition, feature extraction, and experiment reproduction."
    )
    parser.add_argument("--config", default=str(ROOT / "config.example.ini"), help="Path to config ini.")
    parser.add_argument(
        "--step",
        action="append",
        choices=["all", VALIDATE_STEP] + STEP_ORDER,
        default=None,
        help="Workflow step to run. Use --step validate for checks only. Use multiple --step values or --step all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running heavy steps.")
    args = parser.parse_args()

    config = Path(args.config).expanduser().resolve()
    if not config.exists():
        raise SystemExit(f"Config file not found: {config}")

    requested_steps = args.step or ["all"]
    requested = STEP_ORDER if "all" in requested_steps else requested_steps
    status_path = resolve_status_path(config)
    for step in requested:
        run_step(step, config, args.dry_run, status_path)


if __name__ == "__main__":
    main()
