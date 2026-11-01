"""Dependency-light read-only measurement-profile checks, without model imports."""

import csv
import json
import math
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "docs/workbench_measurement_profiles.json"
MISSING = {"", "nan", "na", "n/a", "null", "none"}


def registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def missing(value):
    return value is None or str(value).strip().lower() in MISSING


def inspect_rows(rows, headers, profile_name):
    reg = registry()
    if profile_name not in reg["profiles"]:
        raise ValueError("Unknown measurement profile")
    profile = reg["profiles"][profile_name]
    required = profile["metadata"] + profile["demographics"] + profile["gait"] + profile["speech"]
    errors, warnings = [], []
    if not rows:
        errors.append("Empty table")
    if len(headers) != len(set(headers)):
        errors.append("Duplicate headers")
    absent = [k for k in required if k not in headers]
    if absent:
        errors.append("Missing required columns: " + ", ".join(absent))
    if any(None in row for row in rows):
        errors.append("CSV row has more fields than the header")
    ids = ["" if row.get("ID") is None else str(row.get("ID", "")).strip() for row in rows]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        errors.append("Missing or duplicate participant IDs; resolve repeated recordings explicitly")
    numeric = profile["demographics"] + profile["gait"] + profile["speech"]
    if "label" in profile["metadata"]:
        numeric += ["label"]
    unavailable = []
    for field in numeric:
        values = [row.get(field) for row in rows]
        if field in headers and rows and all(missing(v) for v in values):
            unavailable.append(field)
            if profile_name != "video_review_v1":
                errors.append(field + ": entirely unavailable; do not fabricate or fill a whole undefined field")
        for value in values:
            if missing(value):
                if field == "label":
                    errors.append("Outcome missing")
                continue
            try:
                number = float(value)
                if not math.isfinite(number) or (field == "label" and number not in (0, 1)):
                    raise ValueError()
            except (ValueError, TypeError):
                errors.append(field + ": invalid numeric value")
                break
    for field in profile.get("structurally_unavailable", []):
        if any(not missing(row.get(field)) for row in rows):
            errors.append(field + ": undefined in this video profile; nonmissing values need a separate validated definition")
    video_version = reg["profiles"]["video_review_v1"]["known_measurement_version"]
    versions = sorted({str(r.get("measurement_version", "")).strip() for r in rows if not missing(r.get("measurement_version"))})
    if profile_name == "video_review_v1" and versions != [video_version]:
        errors.append("Video export must retain its known measurement-version identifier")
    if profile_name != "video_review_v1" and video_version in versions:
        errors.append("Known single-video export is not equivalent to the cohort-table profile; no automatic bridge or phase imputation")
    if profile_name == "video_review_v1":
        for r in rows:
            if str(r.get("Num_DT", "")).strip() in {"0", "0.0"} and (not missing(r.get("ACC_DT")) or not missing(r.get("Timecount_DT_1stNum"))):
                errors.append("Video zero-response accuracy and latency must remain missing")
                break
        warnings.append("Recording-level review export: demographics/outcomes are not generated; structural phase fields remain unavailable")
    if len(versions) > 1:
        errors.append("Mixed measurement versions require a separately documented equivalence review")
    warnings.append("A schema pass does not certify units, source identity, correctness labels, physical accuracy or study admission")
    if profile_name == "temporal_audit_20261031":
        warnings.append("This profile documents a separate panel; it does not add a workbench training entrypoint")
    return {"schema_passed": not errors, "profile": profile_name, "row_count": len(rows),
            "expected_predictors": len(profile["demographics"])+len(profile["gait"])+len(profile["speech"]),
            "errors": list(dict.fromkeys(errors)), "warnings": warnings,
            "entirely_unavailable_fields": unavailable,
            "missing_columns": absent, "observed_measurement_versions": versions,
            "measurement_equivalence_certified": False, "modeling_authorized": False,
            "classifier_imported_fitted_or_scored": False, "registry_version": reg["registry_version"]}


def inspect_csv(path, profile):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        return inspect_rows(list(reader), headers, profile)
