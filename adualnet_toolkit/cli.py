"""CLI with lazy imports, a synthetic walkthrough and the local browser app."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ADualNet local research workbench")
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)
    serve = subs.add_parser("serve", help="Open the local browser interface")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--workspace", default=str(ROOT / "local_workspace"))
    serve.add_argument("--recognition-config", default=str(ROOT / ("recognition.local.json" if (ROOT / "recognition.local.json").exists() else "recognition.example.json")))
    for name in ["video", "video-demo"]:
        video = subs.add_parser(name, help="Single-video recognition/review preparation" if name == "video" else "Synthetic video review demonstration (no neural inference)")
        video.add_argument("--out", required=True)
        video.add_argument("--recognition-config", default=str(ROOT / "recognition.example.json"))
        if name == "video":
            video.add_argument("--input", required=True)
    demo = subs.add_parser("demo", help="Run an entirely synthetic CPU demonstration")
    demo.add_argument("--out", default=None)
    demo.add_argument("--bootstrap", type=int, default=200)
    analyze = subs.add_parser("analyze", help="Analyze your own versioned feature table")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--sheet", default="0")
    analyze.add_argument("--measurement-version", required=True)
    analyze.add_argument("--outcome-definition", required=True)
    analyze.add_argument("--folds", type=int, default=5)
    analyze.add_argument("--repeats", type=int, default=10)
    analyze.add_argument("--bootstrap", type=int, default=2000)
    analyze.add_argument("--models", default="AdaBoost,RF,SVM (RBF),DT")
    validate = subs.add_parser("validate", help="Check a feature table without training")
    validate.add_argument("--input", required=True)
    validate.add_argument("--sheet", default="0")
    validate.add_argument("--folds", type=int, default=5)
    validate.add_argument("--out", default=None)
    contract = subs.add_parser("contract-check", help="Read-only CSV measurement-profile preflight; never trains")
    contract.add_argument("--input", required=True)
    contract.add_argument("--profile", required=True, choices=["toolkit_table_v03", "video_review_v1", "historical_2606_primary", "temporal_audit_20261031"])
    contract.add_argument("--out", default=None)
    review = subs.add_parser("reviewed-speech", help="Extract counts from manually reviewed response annotations")
    review.add_argument("--tasks", required=True)
    review.add_argument("--responses", required=True)
    review.add_argument("--out", required=True)
    pipeline = subs.add_parser("pipeline", help="Run an existing recognition/extraction wrapper with local config")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--step", required=True, choices=["gait-recognition", "gait-features", "speech-recognition", "speech-features"])
    pipeline.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "contract-check":
            from .contracts import inspect_csv
            result = inspect_csv(args.input, args.profile)
            text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
            if args.out:
                path = Path(args.out)
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(text + "\n")
            print(text)
            raise SystemExit(0 if result["schema_passed"] else 2)
        elif args.command == "serve":
            from .web import serve_local
            serve_local(Path(args.workspace), args.port, args.recognition_config)
        elif args.command in ["video", "video-demo"]:
            from .video import load_profile, make_review_demo, recognize_video
            profile = load_profile(args.recognition_config)
            out = Path(args.out).resolve()
            if (out / "state.json").exists():
                raise ValueError("Choose a new video job directory; existing runs are preserved.")
            if args.command == "video":
                recognize_video(Path(args.input).resolve(), out, profile)
            else:
                make_review_demo(out, profile)
        elif args.command == "pipeline":
            cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--config", str(Path(args.config).resolve()), "--step", args.step]
            if args.dry_run:
                cmd.append("--dry-run")
            raise SystemExit(subprocess.run(cmd).returncode)
        elif args.command == "reviewed-speech":
            from .speech import extract_reviewed_speech
            extract_reviewed_speech(args.tasks, args.responses, args.out)
        else:
            from .analysis import load_table, make_demo, prepare_table, run_analysis, write_json
            if args.command == "validate":
                _, _, report = prepare_table(load_table(args.input, args.sheet), args.folds)
                if args.out:
                    write_json(args.out, report)
                print(json.dumps(report, indent=2, ensure_ascii=False))
                raise SystemExit(0 if report["passed"] else 2)
            if args.command == "demo":
                out = Path(args.out) if args.out else ROOT / "local_workspace" / "runs" / f"demo_{timestamp()}"
                input_dir = ROOT / "local_workspace" / "inputs"
                input_dir.mkdir(parents=True, exist_ok=True)
                path = input_dir / f"synthetic_{timestamp()}.csv"
                make_demo().to_csv(path, index=False)
                report = run_analysis(path, out, n_splits=3, n_repeats=2, bootstrap=args.bootstrap,
                                      measurement_version="synthetic_walkthrough_v1", synthetic=True,
                                      outcome_definition="Artificial binary target (0/1), generated for software demonstration")
            else:
                report = run_analysis(args.input, args.out, sheet=args.sheet, n_splits=args.folds, n_repeats=args.repeats,
                                      bootstrap=args.bootstrap, models=tuple(args.models.split(",")),
                                      measurement_version=args.measurement_version, outcome_definition=args.outcome_definition)
            print(f"[DONE] Open {report.resolve()}")
    except (ValueError, OSError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
