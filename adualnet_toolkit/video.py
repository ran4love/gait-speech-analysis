"""Single-video recognition, immutable human reviews and versioned feature exports."""

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "adualnet_toolkit/recognition_worker.py"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def dump(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def state(job, status, message, **extra):
    dump(job / "state.json", {"status": status, "message": message, "toolkit_version": __version__,
                             "updated_at": datetime.now(timezone.utc).isoformat(), **extra})


def load_profile(path=None):
    path = Path(path or ROOT / "recognition.example.json").resolve()
    profile = read(path)
    mode = profile.get("execution", {}).get("mode", "local")
    if mode not in ("local", "ssh"):
        raise ValueError("execution.mode must be local or ssh.")
    for section in ["mmpose", "whisper"]:
        executable = profile[section]["python"]
        if executable == "{current_python}":
            if mode == "ssh":
                raise ValueError("SSH mode requires explicit Ubuntu Python paths for both recognition backends.")
            profile[section]["python"] = sys.executable
        elif mode == "local" and not shutil.which(executable):
            profile[section]["python"] = str((path.parent / executable).resolve()) if not Path(executable).is_absolute() else executable
    for name in ["pose2d", "pose2d_weights", "det_model", "det_weights"]:
        raw = profile["mmpose"].get(name, "")
        if mode == "local" and raw and (path.parent / raw).is_file():
            profile["mmpose"][name] = str((path.parent / raw).resolve())
    fps = profile.get("analysis_fps", 30)
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 10 <= fps <= 60:
        raise ValueError("analysis_fps must be finite and between 10 and 60.")
    if not 0 <= float(profile["mmpose"]["bbox_thr"]) <= 1:
        raise ValueError("bbox_thr must be between 0 and 1.")
    if profile["whisper"]["backend"] not in ("faster-whisper", "whisperx"):
        raise ValueError("Choose faster-whisper or whisperx as the speech backend.")
    if profile["mmpose"].get("det_weights") and not profile["mmpose"].get("det_model"):
        raise ValueError("Set det_model together with det_weights; MMPose defaults can otherwise ignore the requested detector weights.")
    return profile


def preflight(profile):
    checks = []
    for key in ["ffmpeg", "ffprobe"]:
        command = profile.get(key, key)
        checks.append({"component": key, "available": bool(shutil.which(command)), "message": command})
    if profile.get("execution", {}).get("mode") == "ssh":
        from .remote_video import remote_checks
        checks.extend(remote_checks(profile))
        return {"ready": all(c["available"] for c in checks), "checks": checks,
                "note": "Local FFmpeg + Ubuntu pretrained inference; videos/audio are transferred to the configured host."}
    for section, module in [("mmpose", "mmpose"), ("whisper", "faster_whisper" if profile["whisper"]["backend"] == "faster-whisper" else "whisperx")]:
        try:
            result = subprocess.run([profile[section]["python"], str(WORKER), "check", "--module", module],
                                    capture_output=True, text=True, timeout=20)
            info = json.loads(result.stdout.strip())
            available = result.returncode == 0 and info["available"]
            message = "Python {} / {}".format(info["python"], module)
        except Exception as exc:
            available, message = False, str(exc)
        checks.append({"component": section, "available": available, "message": message})
    return {"ready": all(c["available"] for c in checks), "checks": checks,
            "note": "Dependency presence only; model weights, CUDA and native-library compatibility are checked during inference."}


def command(cmd):
    print("[RUN] {}".format(Path(str(cmd[0])).name), flush=True)
    # Child output goes directly to the job log. Timed heartbeats keep downloads
    # and native model initialization observable without buffering all output.
    process = subprocess.Popen([str(c) for c in cmd])
    try:
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print("[INFO] Still running: {}".format(Path(str(cmd[0])).name), flush=True)
    except BaseException:
        process.terminate()
        raise
    if code != 0:
        raise RuntimeError("{} exited with code {}; inspect the local run log.".format(Path(str(cmd[0])).name, code))


def probe(video, profile):
    result = subprocess.run([profile.get("ffprobe", "ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
                            capture_output=True, text=True, timeout=60, check=True)
    info = json.loads(result.stdout)
    video_streams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError("No video stream found.")
    duration = float(info["format"]["duration"])
    if not math.isfinite(duration) or not 0 < duration <= float(profile.get("max_duration_s", 600)):
        raise ValueError("Video duration is invalid or exceeds the configured maximum.")
    return info


def normalize_media(source, job, profile):
    original = probe(source, profile)
    if not any(s.get("codec_type") == "audio" for s in original["streams"]):
        raise ValueError("The video has no audio stream. A video with synchronized task audio is required.")
    ffmpeg = profile.get("ffmpeg", "ffmpeg")
    fps = profile["analysis_fps"]
    command([ffmpeg, "-nostdin", "-v", "error", "-n", "-i", source, "-map", "0:v:0", "-map", "0:a:0",
             "-vf", "fps={},scale=trunc(iw/2)*2:trunc(ih/2)*2".format(fps), "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-c:a", "aac", "-af", "aresample=async=1:first_pts=0", "-movflags", "+faststart", job / "review.mp4"])
    command([ffmpeg, "-nostdin", "-v", "error", "-n", "-i", job / "review.mp4", "-vn", "-ac", "1", "-ar", "16000",
             "-af", "aresample=async=1:first_pts=0", job / "audio.wav"])
    normalized = probe(job / "review.mp4", profile)
    stream = next(s for s in normalized["streams"] if s.get("codec_type") == "video")
    meta = {"width": stream["width"], "height": stream["height"], "fps": fps,
            "duration_s": float(normalized["format"]["duration"]), "synthetic": False,
            "timeline": "normalized review.mp4 timeline; both recognition backends use this media",
            "original_sha256": sha(source), "review_video_sha256": sha(job / "review.mp4"),
            "original_probe": original, "normalized_probe": normalized}
    dump(job / "media.json", meta)
    return meta


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return overlap / max(area_a + area_b - overlap, 1e-9)


def assign_tracks(frames):
    """Conservative IoU association; fragmentation/switches require human review."""
    live, next_id, totals = {}, 1, {}
    for index, frame in enumerate(frames):
        if frame.get("frame_index") != index:
            raise ValueError("Pose frames must preserve every decoded frame in order.")
        live = {k: v for k, v in live.items() if index-v[0] <= 10}
        used = set()
        for instance in frame["instances"]:
            points = instance.get("keypoints", [])
            if len(points) != 17:
                raise ValueError("This review pipeline requires a COCO-17 body model.")
            if any(len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point) for point in points):
                raise ValueError("Pose coordinates must be finite COCO-17 x/y pairs.")
            scores = instance.get("keypoint_scores", [])
            # SimCC scores are not probabilities and may exceed 1. Preserve
            # model output, rather than clipping or rejecting valid RTMPose data.
            if len(scores) != 17 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in scores):
                raise ValueError("Pose scores must be 17 finite nonnegative values (not calibrated probabilities).")
            bbox = instance.get("bbox")
            if bbox and isinstance(bbox[0], list):
                bbox = bbox[0]
            if not bbox or len(bbox) != 4:
                bbox = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
            candidates = [(iou(bbox, info[1]), track) for track, info in live.items() if track not in used]
            best = max(candidates, default=(0, None))
            track = best[1] if best[0] >= .3 else next_id
            if track == next_id:
                next_id += 1
            used.add(track)
            live[track] = (index, bbox)
            instance["bbox"] = bbox
            instance["track_id"] = track
            totals[track] = totals.get(track, 0) + 1
    return [{"track_id": k, "frame_count": v} for k, v in sorted(totals.items())]


def draft_review(job, pose_data, transcript, meta):
    tracks = assign_tracks(pose_data["frames"])
    if not tracks:
        raise ValueError("No COCO-17 person track found; review detection settings.")
    dump(job / "poses.json", pose_data)
    responses = []
    for segment in transcript.get("segments", []):
        candidates = segment.get("words") or [{"word": segment.get("text", ""), "start": segment.get("start", 0)}]
        for word in candidates:
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            onset = word.get("start")
            responses.append({"response_onset_s": onset if onset is not None else segment.get("start", 0), "response_text": text,
                              "include": bool(re.search(r"[0-9零〇一二两三四五六七八九十百千]", text)), "is_correct": None})
    review = {"revision": 0, "reviewer": "", "participant_id": "", "task_start_s": 0,
              "task_end_s": meta["duration_s"], "gait_start_s": 0, "gait_end_s": len(pose_data["frames"])/meta["fps"],
              "distance_m": None, "track_id": max(tracks, key=lambda t: t["frame_count"])["track_id"],
              "keypoint_edits": [], "frame_overrides": [], "excluded_frames": [], "responses": responses,
              "scoring_rule": "", "gait_reviewed": False, "speech_reviewed": False, "notes": ""}
    dump(job / "draft.json", review)
    dump(job / "tracks.json", tracks)


def recognize_video(source, job, profile):
    job = Path(job)
    job.mkdir(parents=True, exist_ok=True)
    try:
        state(job, "running", "检查识别环境")
        checks = preflight(profile)
        dump(job / "recognition_preflight.json", checks)
        if not checks["ready"]:
            raise ValueError("Recognition dependencies unavailable: " + ", ".join(c["component"] for c in checks["checks"] if not c["available"]))
        dump(job / "recognition_config.json", profile)
        shutil.copyfile(WORKER, job / "recognition_worker.py")
        state(job, "running", "准备同步视频与音频")
        meta = normalize_media(source, job, profile)
        if profile.get("execution", {}).get("mode") == "ssh":
            from .remote_video import infer_remote
            state(job, "running", "正在传输视频并在 Ubuntu 执行 MMPose / Whisper 推理")
            infer_remote(job, profile)
        else:
            state(job, "running", "MMPose 姿态识别")
            command([profile["mmpose"]["python"], job / "recognition_worker.py", "mmpose", "--input", job / "review.mp4", "--config", job / "recognition_config.json", "--out", job / "pose_raw.json"])
            state(job, "running", "Whisper 语音转录")
            command([profile["whisper"]["python"], job / "recognition_worker.py", "whisper", "--input", job / "audio.wav", "--config", job / "recognition_config.json", "--out", job / "transcript_raw.json"])
        data = read(job / "pose_raw.json")
        if abs(float(data["fps"])-meta["fps"]) > .01:
            raise ValueError("Recognition frame rate differs from the normalized media.")
        if abs(len(data["frames"])/meta["fps"]-meta["duration_s"]) > .25:
            raise ValueError("Pose output does not cover the complete normalized video.")
        draft_review(job, data, read(job / "transcript_raw.json"), meta)
        state(job, "awaiting_review", "识别完成，请确认参与者、任务时间、关键点与回答", revision=0)
    except Exception as exc:
        state(job, "failed", str(exc))
        (job / "FAILED_STEP_README.txt").write_text(str(exc) + "\nInspect recognition_preflight.json and run.log. Original video/partial recognition artifacts are retained.\n", encoding="utf-8")
        raise


def make_review_demo(job, profile):
    """Artificial media plus explicit fixture annotations, never neural inference."""
    job = Path(job)
    job.mkdir(parents=True, exist_ok=True)
    try:
        state(job, "running", "生成纯合成核查演示（未运行神经识别）")
        command([profile.get("ffmpeg", "ffmpeg"), "-nostdin", "-v", "error", "-n", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=12",
                 "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=12", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", job / "review.mp4"])
        meta = {"width": 640, "height": 360, "fps": 30, "duration_s": 12., "synthetic": True,
                "recognition": "synthetic fixtures, no MMPose/Whisper inference"}
        dump(job / "media.json", meta)
        frames = []
        for i in range(360):
            cx = 280 + 50 * math.sin(i/70)
            points = [[cx+(k % 2)*25, 50+k*15] for k in range(17)]
            points[15] = [cx-20+18*math.sin(i/5), 300+8*math.cos(i/5)]
            points[16] = [cx+20-18*math.sin(i/5), 300-8*math.cos(i/5)]
            frames.append({"frame_index": i, "time_s": i/30, "instances": [{"keypoints": points, "keypoint_scores": [.95]*17, "bbox": [cx-60, 40, cx+70, 320]}]})
        pose_data = {"frames": frames, "fps": 30, "provenance": {"synthetic": True}}
        transcript = {"segments": [{"start": t, "end": t+.4, "text": word, "words": [{"word": word, "start": t, "end": t+.4}]} for t, word in [(2., "93"), (4., "86"), (6., "78"), (8., "71")]], "synthetic": True}
        dump(job / "pose_raw.json", pose_data)
        dump(job / "transcript_raw.json", transcript)
        draft_review(job, pose_data, transcript, meta)
        state(job, "awaiting_review", "合成演示：人工确认流程可操作，识别结果为演示注释", revision=0)
    except Exception as exc:
        state(job, "failed", str(exc))
        raise


def validate_review(review, meta, poses):
    result = copy.deepcopy(review)
    if not isinstance(result.get("responses"), list) or not isinstance(result.get("keypoint_edits"), list) or not isinstance(result.get("frame_overrides"), list) or not isinstance(result.get("excluded_frames"), list):
        raise ValueError("Response, keypoint, track and excluded-frame annotations must be lists.")
    def number(key, lower, upper):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError("{} must be a finite number in [{}, {}]".format(key, lower, upper))
        return value
    number("task_start_s", 0, meta["duration_s"])
    number("task_end_s", 0, meta["duration_s"]+.001)
    number("gait_start_s", 0, len(poses["frames"])/meta["fps"])
    number("gait_end_s", 0, len(poses["frames"])/meta["fps"]+.001)
    if result["task_end_s"] <= result["task_start_s"] or result["gait_end_s"] <= result["gait_start_s"]:
        raise ValueError("Task/gait end must be later than start.")
    if result.get("distance_m") is not None:
        number("distance_m", .01, 1000)
    known = {instance["track_id"] for frame in poses["frames"] for instance in frame["instances"]}
    if type(result.get("track_id")) is not int or result["track_id"] not in known:
        raise ValueError("Select an existing person track.")
    for field in ["gait_reviewed", "speech_reviewed"]:
        if type(result.get(field)) is not bool:
            raise ValueError("Review confirmation must be true/false.")
    seen = set()
    for row in result.get("responses", []):
        if type(row.get("include")) is not bool:
            raise ValueError("Each response needs an include flag.")
        onset = row.get("response_onset_s")
        if isinstance(onset, bool) or not isinstance(onset, (int, float)) or not math.isfinite(onset) or not 0 <= onset <= meta["duration_s"]:
            raise ValueError("Response onset must lie on the video timeline.")
        if row["include"]:
            if not result["task_start_s"] <= onset < result["task_end_s"]:
                raise ValueError("Included response lies outside the task [start,end) interval.")
            if not str(row.get("response_text", "")).strip():
                raise ValueError("Included responses need reviewed text.")
            pair = (onset, row["response_text"].strip())
            if pair in seen:
                raise ValueError("Duplicate included response/time; review the response annotations.")
            seen.add(pair)
            if result["speech_reviewed"] and (type(row.get("is_correct")) is not int or row["is_correct"] not in (0, 1)):
                raise ValueError("Confirm every included response's correctness (0/1); no automatic scoring is used.")
    frame_count = len(poses["frames"])
    excluded = result.get("excluded_frames", [])
    if any(type(i) is not int or not 0 <= i < frame_count for i in excluded):
        raise ValueError("Excluded frame indices are invalid.")
    overrides = result.get("frame_overrides", [])
    for edit in overrides + result.get("keypoint_edits", []):
        i, track = edit.get("frame_index"), edit.get("track_id")
        if type(i) is not int or not 0 <= i < frame_count or type(track) is not int:
            raise ValueError("Invalid keypoint/track edit index.")
        if not any(inst["track_id"] == track for inst in poses["frames"][i]["instances"]):
            raise ValueError("Edited track must exist in the selected frame.")
        if "joint" in edit:
            if type(edit["joint"]) is not int or not 0 <= edit["joint"] < 17:
                raise ValueError("Joint index must be 0-16.")
            for axis, maximum in [("x", meta["width"]), ("y", meta["height"])]:
                value = edit.get(axis)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < maximum:
                    raise ValueError("Keypoint correction is outside the normalized video.")
    if result["gait_reviewed"] or result["speech_reviewed"]:
        if not str(result.get("reviewer", "")).strip() or not str(result.get("participant_id", "")).strip():
            raise ValueError("Enter a local participant ID and reviewer name/code.")
    if result["speech_reviewed"] and not str(result.get("scoring_rule", "")).strip():
        raise ValueError("Record the correctness/scoring rule before confirming speech.")
    return result


def save_review(job, supplied):
    job = Path(job)
    current = read(job / "state.json")
    if current["status"] not in ["awaiting_review", "reviewed", "features_ready"]:
        raise ValueError("Recognition must complete before review.")
    revision = current.get("revision", 0)
    if supplied.get("revision") != revision:
        raise ValueError("This review is stale. Reload the saved version before editing.")
    review = validate_review(supplied, read(job / "media.json"), read(job / "poses.json"))
    review["revision"] = revision + 1
    review["saved_at"] = datetime.now(timezone.utc).isoformat()
    folder = job / "reviews"
    folder.mkdir(exist_ok=True)
    path = folder / "review_{:04d}.json".format(review["revision"])
    with path.open("x", encoding="utf-8") as stream:
        json.dump(review, stream, ensure_ascii=False, indent=2, allow_nan=False)
    both = review["gait_reviewed"] and review["speech_reviewed"]
    state(job, "reviewed" if both else "awaiting_review", "人工核查已保存", revision=review["revision"])
    return review


def current_review(job):
    revision = read(job / "state.json").get("revision", 0)
    return read(job / "reviews" / "review_{:04d}.json".format(revision)) if revision else read(job / "draft.json")


def selected_frames(poses, review, fps):
    edits = {(r["frame_index"], r["track_id"], r["joint"]): r for r in review.get("keypoint_edits", [])}
    overrides = {r["frame_index"]: r["track_id"] for r in review.get("frame_overrides", [])}
    excluded = set(review.get("excluded_frames", []))
    selected, missing, run, max_gap = [], 0, 0, 0
    for frame in poses["frames"]:
        i = frame["frame_index"]
        if not review["gait_start_s"] <= i/fps < review["gait_end_s"]:
            continue
        track = overrides.get(i, review["track_id"])
        candidate = next((copy.deepcopy(v) for v in frame["instances"] if v["track_id"] == track), None)
        if i in excluded:
            candidate = None
        if candidate:
            for joint in range(17):
                edit = edits.get((i, track, joint))
                if edit:
                    candidate["keypoints"][joint] = [edit["x"], edit["y"]]
                    candidate.setdefault("keypoint_scores", [0.]*17)[joint] = 1.
            scores = candidate.get("keypoint_scores", [0.]*17)
            valid = len(scores) == 17 and min(scores[15], scores[16]) >= .2
        else:
            valid = False
        if not valid:
            missing += 1
            run += 1
            max_gap = max(max_gap, run)
        else:
            run = 0
        selected.append({"frame_index": i, "instances": [candidate] if candidate else []})
    return selected, {"frames": len(selected), "raw_bilateral_missing": missing,
                      "raw_bilateral_coverage": 1-missing/len(selected) if selected else 0,
                      "longest_gap_frames": max_gap}


def export_features(job):
    import numpy as np
    import pandas as pd
    from .speech import extract_reviewed_speech
    job = Path(job)
    review = current_review(job)
    if not review["gait_reviewed"] or not review["speech_reviewed"]:
        raise ValueError("Both gait and speech need explicit human confirmation before feature export.")
    meta, poses = read(job / "media.json"), read(job / "poses.json")
    revision = review["revision"]
    out = job / "features" / "revision_{:04d}".format(revision)
    if out.exists():
        if (out / "COMPLETE.json").is_file():
            return out
        raise ValueError("An incomplete export exists for this revision; save a new review revision before retry.")
    out.mkdir(parents=True)
    tasks = pd.DataFrame([{"ID": review["participant_id"], "task_start_s": review["task_start_s"], "task_end_s": review["task_end_s"]}])
    tasks.to_csv(out / "reviewed_tasks.csv", index=False)
    responses = [{"ID": review["participant_id"], **{k: r[k] for k in ["response_onset_s", "response_text", "is_correct"]}} for r in review["responses"] if r["include"]]
    pd.DataFrame(responses, columns=["ID", "response_onset_s", "response_text", "is_correct"]).to_csv(out / "reviewed_responses.csv", index=False)
    extract_reviewed_speech(out / "reviewed_tasks.csv", out / "reviewed_responses.csv", out / "speech_features.csv")
    selected, quality = selected_frames(poses, review, meta["fps"])
    sys.path.insert(0, str(ROOT / "src"))
    import gaitfeature
    gait = {}
    warnings = ["Experimental ankle-event intervals and image-space excursions, not validated heel-strike/foot-contact measurements.",
                "Step length/time are stride estimates divided by two by construction; step/stride asymmetry are not independent measurements."]
    if len(selected) < 3 or quality["raw_bilateral_coverage"] < .8 or quality["longest_gap_frames"] > 10:
        warnings.append("gait_extraction_blocked_by_track_qc: coverage<0.8, gap>10 frames or insufficient frames")
    else:
        # Keep original frame slots, including missing frames. Never concatenate
        # separated recordings or remove elapsed time when excluding bad frames.
        gait = gaitfeature.calculate_gait_parameters(selected, meta["fps"],
                    distance_meters=review["distance_m"] if review["distance_m"] is not None else float("nan"))
        warnings.extend(gait.get("qc_fail_reasons", []))
    if review["distance_m"] is None:
        warnings.append("route distance not provided: walking_speed left missing")
    rows = []
    for name in ["average_step_length", "average_stride_length", "step_length_variability", "stride_length_variability",
                 "average_step_time", "average_stride_time", "step_time_variability", "stride_time_variability",
                 "average_step_assymetry", "average_stride_assymetry", "walking_speed", "cadence"]:
        value = clean(gait.get(name))
        unit = "image pixels" if "length" in name else "algorithm-defined %" if "assymetry" in name else "nominal m/s over selected window" if name == "walking_speed" else "steps/min" if name == "cadence" else "s"
        rows.append({"domain": "gait", "feature": name, "value": value, "unit": unit,
                     "status": "available_algorithm_estimate" if value is not None else "missing_or_qc_failed"})
    for name in ["stance_times", "swing_times", "double_support_times"]:
        rows.append({"domain": "gait", "feature": name, "value": None, "unit": "unavailable",
                     "status": "foot-contact phase definition not established for this video pipeline; not fabricated"})
    speech = pd.read_csv(out / "speech_features.csv").iloc[0]
    for name in ["Timecount_DT", "Num_DT", "Correct_Num_DT", "Num_20s_DT", "Correct_Num_20s_DT", "Correct_persec_DT", "ACC_DT", "Timecount_DT_1stNum"]:
        unit = "s" if "Timecount" in name else "proportion" if name == "ACC_DT" else "correct/s" if name == "Correct_persec_DT" else "count"
        rows.append({"domain": "speech", "feature": name, "value": clean(float(speech[name])), "unit": unit, "status": "human-reviewed response annotations"})
    # Missing contact phases stay visible; the export is never auto-fed into a
    # classifier or silently treated as the historical study measurement panel.
    values = pd.DataFrame(rows)
    values.to_csv(out / "feature_values.csv", index=False)
    wide = {"ID": review["participant_id"], "measurement_version": "video_review_frameinterval_v1+reviewed_response_onset_counts_v1",
            "review_revision": revision, "synthetic": meta["synthetic"], **{r["feature"]: r["value"] for r in rows}}
    pd.DataFrame([wide]).to_csv(out / "reviewed_features.csv", index=False)
    with pd.ExcelWriter(out / "reviewed_features.xlsx") as writer:
        pd.DataFrame([wide]).to_excel(writer, sheet_name="Features", index=False)
        values.to_excel(writer, sheet_name="Definitions_and_QC", index=False)
        pd.DataFrame([quality]).to_excel(writer, sheet_name="Track_QC", index=False)
        pd.DataFrame({"warning": warnings + ["Foot-contact phase metrics remain missing; manual review does not establish physiological validity."]}).to_excel(writer, sheet_name="Review_notes", index=False)
    dump(out / "gait_qc.json", {"track_quality": quality, "algorithm_outputs": gait, "warnings": warnings})
    dump(out / "selected_poses.json", {"frames": selected, "fps": meta["fps"]})
    dump(out / "provenance.json", {"toolkit_version": __version__, "synthetic": meta["synthetic"],
         "review_sha256": sha(job / "reviews" / "review_{:04d}.json".format(revision)),
         "pose_raw_sha256": sha(job / "pose_raw.json"), "transcript_raw_sha256": sha(job / "transcript_raw.json"),
         "gait_extractor_sha256": sha(ROOT / "src/gaitfeature.py"), "scoring_rule": review["scoring_rule"],
         "video_adapter_sha256": sha(Path(__file__)),
         "recognition_worker_sha256": sha(job / "recognition_worker.py") if (job / "recognition_worker.py").is_file() else None,
         "measurement_version": wide["measurement_version"], "phase_metrics": "missing: no validated video contact-event definition",
         "historical_equivalence_certified": False, "classifier_fitted_or_scored": False})
    dump(out / "COMPLETE.json", {"review_revision": revision, "status": "features_exported", "synthetic": meta["synthetic"]})
    state(job, "features_ready", "已按当前人工确认版本生成特征和 QC", revision=revision)
    return out
