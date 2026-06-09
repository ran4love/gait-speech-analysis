#!/usr/bin/env python3
"""Validate public workflow config, paths, dependencies, and table columns."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from common import get_list, get_path, load_config


SCRIPT_ALL_FEATURES = [
    "age",
    "edu",
    "ACC_DT",
    "Correct_persec_DT",
    "Correct_Num_DT",
    "Num_DT",
    "Timecount_DT",
    "Timecount_DT_1stNum",
    "Num_20s_DT",
    "Correct_Num_20s_DT",
    "average_step_length",
    "average_stride_length",
    "step_length_variability",
    "stride_length_variability",
    "average_step_time",
    "average_stride_time",
    "step_time_variability",
    "stride_time_variability",
    "average_step_assymetry",
    "average_stride_assymetry",
    "walking_speed",
    "cadence",
    "stance_times",
    "swing_times",
    "double_support_times",
]

NESTED_SELECTED_FEATURES = [
    "ACC_DT",
    "Correct_Num_20s_DT",
    "Correct_Num_DT",
    "Correct_persec_DT",
    "Num_20s_DT",
    "Num_DT",
    "Timecount_DT_1stNum",
    "age",
    "edu",
    "swing_times",
    "average_step_assymetry",
    "average_stride_assymetry",
    "stance_times",
    "average_stride_length",
]

ADUALSELECTED_FEATURES = [
    "Timecount_DT",
    "average_step_assymetry",
    "stance_times",
    "swing_times",
    "walking_speed",
    "Num_DT",
    "Correct_Num_DT",
    "Num_20s_DT",
    "Correct_Num_20s_DT",
    "Correct_persec_DT",
    "ACC_DT",
    "Timecount_DT_1stNum",
    "edu",
    "age",
]

GAIT_FEATURES = [
    "average_step_length",
    "average_stride_length",
    "step_length_variability",
    "stride_length_variability",
    "average_step_time",
    "average_stride_time",
    "step_time_variability",
    "stride_time_variability",
    "average_step_assymetry",
    "average_stride_assymetry",
    "walking_speed",
    "cadence",
    "stance_times",
    "swing_times",
    "double_support_times",
]


class Validator:
    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []

    def add(self, status: str, check: str, detail: str) -> None:
        self.rows.append({"status": status, "check": check, "detail": detail})

    def ok(self, check: str, detail: str) -> None:
        self.add("pass", check, detail)

    def warn(self, check: str, detail: str) -> None:
        self.add("warn", check, detail)

    def fail(self, check: str, detail: str) -> None:
        self.add("fail", check, detail)

    @property
    def n_fail(self) -> int:
        return sum(1 for row in self.rows if row["status"] == "fail")

    @property
    def n_warn(self) -> int:
        return sum(1 for row in self.rows if row["status"] == "warn")


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.replace("\n", ",").replace(";", ",").split(",") if x.strip()]


def feature_set_columns(feature_set: str, custom_features: str) -> List[str]:
    if feature_set == "script_all":
        return SCRIPT_ALL_FEATURES
    if feature_set == "nestedselected":
        return NESTED_SELECTED_FEATURES
    if feature_set == "adualselected":
        return ADUALSELECTED_FEATURES
    if feature_set == "custom":
        return parse_csv_list(custom_features)
    raise ValueError(f"Unknown feature set: {feature_set}")


def module_exists(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def check_modules(v: Validator, modules: Sequence[tuple[str, str, bool]]) -> None:
    for package_name, import_name, required in modules:
        exists = module_exists(import_name)
        if exists:
            v.ok(f"dependency:{package_name}", f"Python import `{import_name}` is available.")
        elif required:
            v.fail(f"dependency:{package_name}", f"Missing Python package/import `{import_name}`.")
        else:
            v.warn(f"dependency:{package_name}", f"Optional package/import `{import_name}` is missing.")


def read_headers(path: Path, sheet_name: str = "") -> tuple[List[str], str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=sep)
            return [str(x).strip() for x in next(reader, [])], ""
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        if not module_exists("openpyxl"):
            raise RuntimeError("openpyxl is required to inspect xlsx headers.")
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        selected_sheet = sheet_name if sheet_name and sheet_name in wb.sheetnames else wb.sheetnames[0]
        ws = wb[selected_sheet]
        row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
        return [str(x).strip() for x in row if x is not None and str(x).strip()], selected_sheet
    raise RuntimeError(f"Unsupported table format: {path.suffix}")


def check_path(v: Validator, label: str, path: Path, required: bool = True) -> None:
    if path.exists():
        v.ok(f"path:{label}", str(path))
    elif required:
        v.fail(f"path:{label}", f"Missing: {path}")
    else:
        v.warn(f"path:{label}", f"Optional path not found: {path}")


def check_columns(
    v: Validator,
    label: str,
    path: Path,
    required_columns: Iterable[str],
    sheet_name: str = "",
) -> None:
    if not path.exists():
        v.fail(f"columns:{label}", f"Cannot inspect columns because file is missing: {path}")
        return
    try:
        headers, selected_sheet = read_headers(path, sheet_name)
    except Exception as exc:
        v.fail(f"columns:{label}", f"Could not read header row from {path}: {exc}")
        return
    missing = [col for col in dict.fromkeys(required_columns) if col and col not in headers]
    where = f" sheet `{selected_sheet}`" if selected_sheet else ""
    if missing:
        v.fail(f"columns:{label}", f"Missing columns in {path}{where}: {missing}")
    else:
        v.ok(f"columns:{label}", f"All required columns present in {path}{where}.")


def check_json_file(v: Validator, label: str, path: Path, required: bool = True) -> None:
    if not path.exists():
        check_path(v, label, path, required=required)
        return
    try:
        json.loads(path.read_text(encoding="utf-8"))
        v.ok(f"json:{label}", f"Valid JSON: {path}")
    except Exception as exc:
        v.fail(f"json:{label}", f"Invalid JSON {path}: {exc}")


def check_writable_dir(v: Validator, label: str, path: Path, dry_run: bool) -> None:
    if dry_run:
        v.ok(f"output:{label}", f"Dry-run mode: would check/create {path}")
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".write_test_", dir=path, delete=True) as f:
            f.write(b"ok")
        v.ok(f"output:{label}", f"Writable output directory: {path}")
    except Exception as exc:
        v.fail(f"output:{label}", f"Output directory is not writable: {path}; {exc}")


def check_gait(v: Validator, cfg, base: Path) -> None:
    run_mmpose = cfg.getboolean("gait", "run_mmpose", fallback=False)
    video_dir = get_path(cfg, base, "gait", "video_dir")
    keypoint_dir = get_path(cfg, base, "gait", "keypoint_json_dir")
    template = cfg.get("gait", "mmpose_command_template", fallback="").strip()

    if run_mmpose:
        check_path(v, "gait.video_dir", video_dir, required=True)
        if not template:
            v.fail("config:gait.mmpose_command_template", "run_mmpose = yes but mmpose_command_template is empty.")
        else:
            v.ok("config:gait.mmpose_command_template", template)
    else:
        v.ok("config:gait.run_mmpose", "run_mmpose = no; validation expects existing keypoint JSON for feature extraction.")
        check_path(v, "gait.keypoint_json_dir", keypoint_dir, required=False)

    if keypoint_dir.exists():
        json_count = len(list(keypoint_dir.glob("*.json")))
        if json_count:
            v.ok("input:gait_keypoint_json", f"Found {json_count} JSON files in {keypoint_dir}.")
        else:
            v.warn("input:gait_keypoint_json", f"Directory exists but contains no .json files: {keypoint_dir}")


def check_speech(v: Validator, cfg, base: Path) -> None:
    run_whisperx = cfg.getboolean("speech", "run_whisperx", fallback=False)
    if run_whisperx:
        if module_exists("whisperx"):
            v.ok("dependency:whisperx", "whisperx is available.")
        else:
            v.fail("dependency:whisperx", "run_whisperx = yes but whisperx is not installed.")
    else:
        v.ok("config:speech.run_whisperx", "run_whisperx = no; validation expects existing ASR JSON for speech feature extraction.")

    for task in ("task8", "task9"):
        audio_dir = get_path(cfg, base, "speech", f"{task}_audio_dir")
        asr_dir = get_path(cfg, base, "speech", f"{task}_asr_dir")
        if run_whisperx:
            check_path(v, f"speech.{task}_audio_dir", audio_dir, required=True)
        check_path(v, f"speech.{task}_asr_dir", asr_dir, required=False)


def check_experiment(v: Validator, cfg, base: Path) -> None:
    demo_xlsx = get_path(cfg, base, "experiment", "demo_xlsx")
    gait_paths = []
    for item in get_list(cfg, "experiment", "gait_xlsx", fallback=""):
        p = Path(os.path.expanduser(os.path.expandvars(item)))
        gait_paths.append(p if p.is_absolute() else (base / p).resolve())
    task8_json = get_path(cfg, base, "experiment", "task8_json")
    task9_json = get_path(cfg, base, "experiment", "task9_json")

    check_path(v, "experiment.demo_xlsx", demo_xlsx, required=False)
    if demo_xlsx.exists():
        check_columns(v, "experiment.demo_xlsx", demo_xlsx, ["ID", "label", "MoCA", "MMSE"])
    for idx, path in enumerate(gait_paths, start=1):
        check_path(v, f"experiment.gait_xlsx[{idx}]", path, required=False)
        if path.exists():
            check_columns(v, f"experiment.gait_xlsx[{idx}]", path, ["ID", *GAIT_FEATURES])
    check_json_file(v, "experiment.task8_json", task8_json, required=False)
    check_json_file(v, "experiment.task9_json", task9_json, required=False)


def check_paper_experiment(v: Validator, cfg, base: Path) -> None:
    input_table = get_path(cfg, base, "paper_experiment", "input_table")
    sheet_name = cfg.get("paper_experiment", "sheet_name", fallback="age-edu")
    label_col = cfg.get("paper_experiment", "label_col", fallback="label")
    id_col = cfg.get("paper_experiment", "id_col", fallback="ID")
    feature_set = cfg.get("paper_experiment", "feature_set", fallback="nestedselected")
    custom_features = cfg.get("paper_experiment", "custom_features", fallback="")

    check_path(v, "paper_experiment.input_table", input_table, required=False)
    required = [id_col, label_col, *feature_set_columns(feature_set, custom_features)]

    if cfg.getboolean("paper_experiment", "run_publication_outputs", fallback=True):
        required.extend(SCRIPT_ALL_FEATURES)
        required.extend(NESTED_SELECTED_FEATURES)

    if cfg.getboolean("paper_experiment", "run_nested_feature_selection", fallback=True):
        candidate_set = cfg.get("paper_experiment", "nested_candidate_set", fallback="script_all")
        nested_custom = cfg.get("paper_experiment", "nested_custom_features", fallback="")
        required.extend(feature_set_columns(candidate_set, nested_custom))
        select_k = int(cfg.get("paper_experiment", "nested_select_k", fallback="14"))
        if select_k < 1:
            v.fail("config:paper_experiment.nested_select_k", "nested_select_k must be >= 1.")
        else:
            v.ok("config:paper_experiment.nested_select_k", str(select_k))

    if cfg.getboolean("paper_experiment", "promote_nested_selection", fallback=True) and not cfg.getboolean(
        "paper_experiment", "run_nested_feature_selection", fallback=True
    ):
        v.fail(
            "config:paper_experiment.promote_nested_selection",
            "promote_nested_selection = yes requires run_nested_feature_selection = yes.",
        )

    if cfg.getboolean("paper_experiment", "run_regression", fallback=True):
        required.append(cfg.get("paper_experiment", "regression_target", fallback="MoCA"))

    if cfg.getboolean("paper_experiment", "run_confounder_adjustment", fallback=True):
        required.append(cfg.get("paper_experiment", "age_col", fallback="age"))
        required.append(cfg.get("paper_experiment", "education_col", fallback="edu"))
        sex_col = cfg.get("paper_experiment", "sex_col", fallback="").strip()
        if sex_col:
            required.append(sex_col)

    if input_table.exists():
        check_columns(v, "paper_experiment.input_table", input_table, required, sheet_name=sheet_name)
    else:
        v.warn(
            "template:paper_modeling_dataset",
            "Use data/templates/paper_modeling_dataset_template.xlsx as the required column template.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate public gait-speech workflow inputs and environment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Do not create output directories during validation.")
    args = parser.parse_args()

    cfg, base = load_config(args.config)
    v = Validator()

    output_dir = get_path(cfg, base, "project", "output_dir", fallback="results")
    paper_out = get_path(cfg, base, "paper_experiment", "out_dir", fallback="results/paper_experiments")
    experiment_out = get_path(cfg, base, "experiment", "out_dir", fallback="results/experiments")
    gait_out = get_path(cfg, base, "gait", "gait_features_xlsx", fallback="results/gait_features/gait_features_check_previous_format.xlsx").parent
    speech_out = get_path(cfg, base, "speech", "task8_features_json", fallback="results/speech_features/task8_features.json").parent

    check_modules(
        v,
        [
            ("numpy", "numpy", True),
            ("opencv-python", "cv2", False),
            ("matplotlib", "matplotlib", True),
            ("pandas", "pandas", True),
            ("openpyxl", "openpyxl", True),
            ("scikit-learn", "sklearn", True),
            ("shap", "shap", cfg.getboolean("paper_experiment", "run_shap", fallback=True)),
        ],
    )
    check_gait(v, cfg, base)
    check_speech(v, cfg, base)
    check_experiment(v, cfg, base)
    check_paper_experiment(v, cfg, base)

    for label, path in [
        ("project.output_dir", output_dir),
        ("gait_features", gait_out),
        ("speech_features", speech_out),
        ("experiments", experiment_out),
        ("paper_experiments", paper_out),
    ]:
        check_writable_dir(v, label, path, args.dry_run)

    report = {
        "config": str(Path(args.config).expanduser().resolve()),
        "base_dir": str(base),
        "summary": {"fail": v.n_fail, "warn": v.n_warn, "pass": sum(1 for row in v.rows if row["status"] == "pass")},
        "checks": v.rows,
    }
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "validation_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        report_path = None

    for row in v.rows:
        print(f"[{row['status'].upper()}] {row['check']}: {row['detail']}")
    print(f"Validation summary: {report['summary']}")
    if report_path:
        print(f"Validation report written to: {report_path}")

    raise SystemExit(1 if v.n_fail else 0)


if __name__ == "__main__":
    main()
