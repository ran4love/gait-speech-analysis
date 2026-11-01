"""Prospective response counts from human-reviewed annotations, without snapping."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def extract_reviewed_speech(tasks_path, responses_path, out_path):
    tasks = pd.read_csv(tasks_path, dtype={"ID": str})
    responses = pd.read_csv(responses_path, dtype={"ID": str})
    for name, frame, required in [("tasks", tasks, ["ID", "task_start_s", "task_end_s"]),
                                  ("responses", responses, ["ID", "response_onset_s", "response_text", "is_correct"])]:
        missing = [col for col in required if col not in frame]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
        if frame.ID.isna().any() or frame.ID.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}: empty ID")
    if tasks.empty or tasks.ID.duplicated().any() or not set(responses.ID).issubset(set(tasks.ID)):
        raise ValueError("Use unique task IDs and match all response IDs to the task manifest.")
    for col in ["task_start_s", "task_end_s"]:
        tasks[col] = pd.to_numeric(tasks[col], errors="raise")
    if not np.isfinite(tasks[["task_start_s", "task_end_s"]]).all().all() or (tasks.task_start_s < 0).any() or (tasks.task_end_s <= tasks.task_start_s).any():
        raise ValueError("Every task needs finite nonnegative start and a later end.")
    responses["response_onset_s"] = pd.to_numeric(responses.response_onset_s, errors="raise")
    responses["is_correct"] = pd.to_numeric(responses.is_correct, errors="raise")
    if not np.isfinite(responses.response_onset_s).all() or not responses.is_correct.isin([0, 1]).all():
        raise ValueError("Response onset must be finite; is_correct must be reviewed 0/1.")
    if responses.response_text.isna().any() or responses.response_text.astype(str).str.strip().eq("").any():
        raise ValueError("Every response annotation needs response_text for review provenance.")
    rows = []
    for row in tasks.itertuples():
        sub = responses[responses.ID.eq(row.ID)].sort_values("response_onset_s")
        if ((sub.response_onset_s < row.task_start_s) | (sub.response_onset_s >= row.task_end_s)).any():
            raise ValueError("A response lies outside its task [start, end) window; review the annotation.")
        duration = float(row.task_end_s-row.task_start_s)
        early = sub[sub.response_onset_s < min(row.task_end_s, row.task_start_s + 20)]
        total, correct = len(sub), int(sub.is_correct.sum())
        rows.append({"ID": row.ID, "Timecount_DT": duration, "Num_DT": total, "Correct_Num_DT": correct,
                     "Num_20s_DT": len(early), "Correct_Num_20s_DT": int(early.is_correct.sum()),
                     "Correct_persec_DT": correct/duration, "ACC_DT": correct/total if total else np.nan,
                     "Timecount_DT_1stNum": float(sub.response_onset_s.iloc[0]-row.task_start_s) if total else np.nan})
    out = Path(out_path)
    if out.exists():
        raise ValueError("Choose a new output file; annotations/features are not overwritten.")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".protocol.json").write_text(json.dumps({
        "definition": "reviewed_response_onset_counts_v1", "task_window": "[start, end)",
        "first20s_window": "[start, min(start+20, end))", "correctness": "reviewer-provided is_correct",
        "no_response_accuracy_and_latency": "missing", "sequence_snapping": False,
        "latency_offset": 0, "count_floor": False, "historical_equivalence_certified": False,
    }, indent=2), encoding="utf-8")
    print(f"[DONE] Reviewed speech features: {out}")
