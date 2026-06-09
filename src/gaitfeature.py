#!/usr/bin/env python3
"""Merged gait feature extraction script.

This file combines:
1) gait_features_extraction_without_ds_new.py
2) gait_features_extraction_ds.py

Correction rules (physical consistency):
1) Step/stride time are estimated from frame intervals, not from pixel length / m/s.
2) Cadence (steps/min) = 120 / stride_time (stride contains two steps).
3) Walking speed can be estimated from total frames and known walking distance.
"""

import argparse
import json
import math
import os
import subprocess

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None


# -----------------------------
# Part A: without_ds workflow
# -----------------------------

DEFAULT_WITHOUT_DS_VIDEO_DIRECTORY = r"K:\0.database\PKUh\VIDEO\video-new"
DEFAULT_WITHOUT_DS_JSON_DIRECTORY = r"K:\0.database\PKUh\VIDEO\pred"
DEFAULT_WITHOUT_DS_OUTPUT_JSON_PATH = r"K:\0.database\PKUh\VIDEO"
DEFAULT_WITHOUT_DS_DISTANCE_METERS = 6
DEFAULT_MIN_KEYPOINT_SCORE = 0.2
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_INTERP_MAX_GAP = 10
DEFAULT_MIN_STRIDE_TIME_SEC = 0.4


def array_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def array_mul(a, scalar):
    return [x * scalar for x in a]


def array_norm(a):
    return math.sqrt(sum(x * x for x in a))


def mean(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def std(values):
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def near_zero(value, eps=1e-12):
    return value is not None and not math.isnan(value) and abs(value) <= eps


def unique_preserve_order(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def safe_ratio(numerator, denominator):
    if not denominator:
        return float("nan")
    return numerator / denominator


def percentile(values, p):
    if not values:
        return float("nan")
    arr = sorted(values)
    if p <= 0:
        return arr[0]
    if p >= 100:
        return arr[-1]
    k = (len(arr) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return arr[f]
    return arr[f] * (c - k) + arr[c] * (k - f)


def moving_average_optional(values, window):
    if window <= 1:
        return values[:]
    n = len(values)
    half = window // 2
    out = []
    for i in range(n):
        if values[i] is None:
            out.append(None)
            continue
        s = max(0, i - half)
        e = min(n, i + half + 1)
        win = [x for x in values[s:e] if x is not None]
        out.append(sum(win) / len(win) if win else None)
    return out


def interpolate_nones(values, max_gap=DEFAULT_INTERP_MAX_GAP):
    out = values[:]
    valid_idx = [i for i, v in enumerate(out) if v is not None]
    if not valid_idx:
        return out
    for start, end in zip(valid_idx, valid_idx[1:]):
        gap = end - start - 1
        if gap <= 0:
            continue
        if gap > max_gap:
            continue
        v0 = out[start]
        v1 = out[end]
        for i in range(start + 1, end):
            t = (i - start) / (end - start)
            out[i] = v0 + (v1 - v0) * t
    return out


def load_camera_calibration(camera_calib_npz):
    if not camera_calib_npz:
        return None
    if np is None:
        raise RuntimeError("NumPy is required to load camera calibration.")
    if cv2 is None:
        raise RuntimeError("OpenCV is required to apply camera calibration.")
    if not os.path.isfile(camera_calib_npz):
        raise FileNotFoundError(f"Calibration file not found: {camera_calib_npz}")

    z = np.load(camera_calib_npz, allow_pickle=True)
    keys = set(z.files)

    K = None
    for k in ("K", "camera_matrix", "mtx"):
        if k in keys:
            K = np.array(z[k], dtype=np.float64)
            break
    if K is None:
        raise ValueError(f"No camera matrix key found in {camera_calib_npz}.")

    D = None
    for k in ("D", "dist_coeffs", "distCoeffs", "dist"):
        if k in keys:
            D = np.array(z[k], dtype=np.float64)
            break
    if D is None:
        raise ValueError(f"No distortion coefficients key found in {camera_calib_npz}.")

    model = "standard"
    if "model" in keys:
        model_raw = z["model"]
        if isinstance(model_raw, np.ndarray) and model_raw.shape == ():
            model_raw = model_raw.item()
        model = str(model_raw)
    elif D.size == 4:
        model = "fisheye"

    return {
        "K": K,
        "D": D.reshape(-1, 1) if D.ndim == 1 else D,
        "model": model.lower(),
        "path": camera_calib_npz,
    }


def undistort_tracks(xs, ys, camera_calib):
    if camera_calib is None:
        return xs, ys
    idx = [i for i in range(len(xs)) if xs[i] is not None and ys[i] is not None]
    if not idx:
        return xs, ys

    pts = np.array([[[float(xs[i]), float(ys[i])]] for i in idx], dtype=np.float64)
    K = camera_calib["K"]
    D = camera_calib["D"]
    model = camera_calib.get("model", "standard")

    if model == "fisheye":
        und = cv2.fisheye.undistortPoints(pts, K, D, P=K)
    else:
        und = cv2.undistortPoints(pts, K, D, P=K)

    out_x = xs[:]
    out_y = ys[:]
    for j, i in enumerate(idx):
        out_x[i] = float(und[j, 0, 0])
        out_y[i] = float(und[j, 0, 1])
    return out_x, out_y


def extract_heel_tracks(
    data,
    min_keypoint_score=DEFAULT_MIN_KEYPOINT_SCORE,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    interp_max_gap=DEFAULT_INTERP_MAX_GAP,
    camera_calib=None,
):
    n = len(data)
    left_x = [None] * n
    left_y = [None] * n
    right_x = [None] * n
    right_y = [None] * n
    raw_left_detected_count = 0
    raw_right_detected_count = 0

    for idx, frame in enumerate(data):
        instances = frame.get("instances") or []
        if not instances:
            continue
        inst0 = instances[0]
        keypoints = inst0.get("keypoints")
        if not keypoints or len(keypoints) < 17:
            continue
        scores = inst0.get("keypoint_scores")

        left_score = 1.0
        right_score = 1.0
        if isinstance(scores, list) and len(scores) >= 17:
            left_score = float(scores[15])
            right_score = float(scores[16])

        if left_score >= min_keypoint_score:
            left_x[idx] = float(keypoints[15][0])
            left_y[idx] = float(keypoints[15][1])
            raw_left_detected_count += 1
        if right_score >= min_keypoint_score:
            right_x[idx] = float(keypoints[16][0])
            right_y[idx] = float(keypoints[16][1])
            raw_right_detected_count += 1

    raw_bilateral_detected_count = 0
    for idx in range(n):
        if (
            left_x[idx] is not None
            and left_y[idx] is not None
            and right_x[idx] is not None
            and right_y[idx] is not None
        ):
            raw_bilateral_detected_count += 1

    # Apply lens distortion correction before interpolation/smoothing.
    left_x, left_y = undistort_tracks(left_x, left_y, camera_calib)
    right_x, right_y = undistort_tracks(right_x, right_y, camera_calib)

    left_x = interpolate_nones(left_x, max_gap=interp_max_gap)
    left_y = interpolate_nones(left_y, max_gap=interp_max_gap)
    right_x = interpolate_nones(right_x, max_gap=interp_max_gap)
    right_y = interpolate_nones(right_y, max_gap=interp_max_gap)

    left_x = moving_average_optional(left_x, smooth_window)
    left_y = moving_average_optional(left_y, smooth_window)
    right_x = moving_average_optional(right_x, smooth_window)
    right_y = moving_average_optional(right_y, smooth_window)

    tracks = []
    for idx in range(n):
        if (
            left_x[idx] is None
            or left_y[idx] is None
            or right_x[idx] is None
            or right_y[idx] is None
        ):
            continue
        tracks.append((idx, [left_x[idx], left_y[idx]], [right_x[idx], right_y[idx]]))
    usable_frame_count = len(tracks)
    track_quality = {
        "total_frame_count": n,
        "raw_left_detected_frame_count": raw_left_detected_count,
        "raw_right_detected_frame_count": raw_right_detected_count,
        "raw_bilateral_detected_frame_count": raw_bilateral_detected_count,
        "usable_frame_count": usable_frame_count,
        "usable_frame_ratio": safe_ratio(usable_frame_count, n),
        "heel_track_missing_ratio": safe_ratio(n - raw_bilateral_detected_count, n),
    }
    return tracks, track_quality


def estimate_event_thresholds(tracks):
    if len(tracks) < 3:
        return 570.0, 10.0, 8.0

    all_y = []
    delta_y = []
    delta_x = []
    for i, (_, left_heel, right_heel) in enumerate(tracks):
        all_y.extend([left_heel[1], right_heel[1]])
        if i > 0:
            _, left_prev, _ = tracks[i - 1]
            delta_y.append(abs(left_heel[1] - left_prev[1]))
            delta_x.append(abs(left_heel[0] - left_prev[0]))

    ground_threshold = percentile(all_y, 65)
    tolerance = max(8.0, percentile(delta_y, 65)) if delta_y else 10.0
    min_x_movement = max(8.0, percentile(delta_x, 70)) if delta_x else 8.0
    return ground_threshold, tolerance, min_x_movement


def get_frame_rate(video_path):
    """Get video frame rate with ffprobe/ffmpeg/OpenCV fallback."""
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(
            ffprobe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            check=False,
        )
        raw = result.stdout.strip()
        if raw and "/" in raw:
            num, den = raw.split("/", 1)
            num = float(num)
            den = float(den)
            if den != 0:
                return num / den
        elif raw:
            return float(raw)
    except (FileNotFoundError, ValueError, ZeroDivisionError):
        pass

    ffmpeg_cmd = ["ffmpeg", "-i", video_path]
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            check=False,
        )
        lines = result.stdout.split("\n")
        for line in lines:
            if "fps" not in line:
                continue
            parts = line.split(",")
            for part in parts:
                if "fps" not in part:
                    continue
                token = part.strip().split(" ")[0]
                try:
                    return float(token)
                except ValueError:
                    continue
    except FileNotFoundError:
        pass

    try:
        if cv2 is not None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                return float(fps)
    except Exception:
        pass

    return None


def load_json_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_distance_hor(p1, p2):
    return abs(p1[0] - p2[0])


def calculate_step_length(left_heel_prev, right_heel_prev, left_heel_curr, right_heel_curr):
    step_length_right = calculate_distance(left_heel_prev, right_heel_curr)
    step_length_left = calculate_distance(right_heel_prev, left_heel_curr)
    return step_length_left, step_length_right


def calculate_stride_length(left_heel_prev, right_heel_prev, left_heel_curr, right_heel_curr):
    stride_length_left = calculate_distance(left_heel_prev, left_heel_curr)
    stride_length_right = calculate_distance(right_heel_prev, right_heel_curr)
    return stride_length_left, stride_length_right


def calculate_cadence(stride_time):
    # Stride time is one full gait cycle (two steps), so cadence is 120/Tstride.
    return 120 / stride_time if stride_time and stride_time > 0 else float("nan")


def summarize_interval_metrics(values, metric_name):
    average_value = mean(values)
    qc_fail_reasons = []
    if not values:
        qc_fail_reasons.append(f"no_{metric_name}_samples")
        return average_value, float("nan"), qc_fail_reasons
    if len(values) < 2:
        qc_fail_reasons.append(f"insufficient_{metric_name}_samples_for_variability")
        return average_value, float("nan"), qc_fail_reasons

    variability = std(values)
    if near_zero(variability):
        qc_fail_reasons.append(f"{metric_name}_variability_zero_degenerate")
        return average_value, float("nan"), qc_fail_reasons
    return average_value, variability, qc_fail_reasons


def calculate_step_time(step_times):
    return summarize_interval_metrics(step_times, "step_time")


def calculate_stride_time(stride_times):
    return summarize_interval_metrics(stride_times, "stride_time")


def calculate_step_length_variability(step_lengths):
    return std(step_lengths)


def calculate_stride_length_variability(stride_lengths):
    return std(stride_lengths)


def calculate_asymmetry(left_value, right_value):
    average = (left_value + right_value) / 2
    return abs(left_value - right_value) / average * 100 if average != 0 else 0


def calculate_walking_speed(video_path, distance_meters):
    from moviepy.editor import VideoFileClip

    try:
        with VideoFileClip(video_path) as clip:
            total_seconds = clip.duration
    except Exception as exc:
        print(f"Unable to load video file: {exc}")
        return None

    if total_seconds > 0:
        walking_speed = distance_meters / total_seconds
        return walking_speed
    raise ValueError("Total time from video must be greater than 0")


def calculate_walking_speed_from_frames(frame_count, frame_rate, distance_meters):
    if not frame_rate or frame_rate <= 0 or frame_count <= 0:
        return float("nan")
    total_seconds = frame_count / frame_rate
    if total_seconds <= 0:
        return float("nan")
    return distance_meters / total_seconds


def has_full_stride_occurred(last_heel, current_heel, ground_threshold=570, tolerance=10, min_x_movement=8):
    if current_heel[1] >= ground_threshold:
        if abs(current_heel[1] - last_heel[1]) <= tolerance:
            if abs(current_heel[0] - last_heel[0]) >= min_x_movement:
                return True
    return False


def calculate_step_and_stride_lengths(
    data,
    frame_rate,
    min_stride_time_sec=DEFAULT_MIN_STRIDE_TIME_SEC,
    min_keypoint_score=DEFAULT_MIN_KEYPOINT_SCORE,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    interp_max_gap=DEFAULT_INTERP_MAX_GAP,
    camera_calib=None,
):
    step_lengths = []
    stride_lengths = []
    step_times = []
    stride_times = []
    step_asymmetry = []
    stride_asymmetry = []

    tracks, track_quality = extract_heel_tracks(
        data,
        min_keypoint_score=min_keypoint_score,
        smooth_window=smooth_window,
        interp_max_gap=interp_max_gap,
        camera_calib=camera_calib,
    )
    if len(tracks) < 3:
        return (
            step_lengths,
            stride_lengths,
            step_times,
            stride_times,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            step_asymmetry,
            stride_asymmetry,
            ["insufficient_track_frames"],
            track_quality,
        )

    ground_threshold, tolerance, min_x_movement = estimate_event_thresholds(tracks)

    last_left_heel = None
    last_right_heel = None
    raw_stride_event_count = 0
    accepted_stride_event_count = 0
    suppressed_stride_event_count = 0

    accumulated_left_stride_distance = 0
    accumulated_right_stride_distance = 0
    last_stride_frame_idx = None
    min_stride_interval_frames = 1
    if frame_rate and frame_rate > 0:
        min_stride_interval_frames = max(1, int(round(min_stride_time_sec * frame_rate)))

    for j in range(1, len(tracks)):
        frame_idx_prev, left_heel_prev, right_heel_prev = tracks[j - 1]
        frame_idx_curr, left_heel_curr, right_heel_curr = tracks[j]

        if last_left_heel is None:
            last_left_heel = left_heel_prev
            last_right_heel = right_heel_prev

        accumulated_left_stride_distance += calculate_distance_hor(last_left_heel, left_heel_curr)
        accumulated_right_stride_distance += calculate_distance_hor(last_right_heel, right_heel_curr)

        stride_event = has_full_stride_occurred(
            last_left_heel,
            left_heel_curr,
            ground_threshold=ground_threshold,
            tolerance=tolerance,
            min_x_movement=min_x_movement,
        )
        if stride_event:
            raw_stride_event_count += 1
        if stride_event and last_stride_frame_idx is not None:
            if (frame_idx_curr - last_stride_frame_idx) < min_stride_interval_frames:
                stride_event = False
                suppressed_stride_event_count += 1

        if stride_event:
            accepted_stride_event_count += 1
            stride_length_left = accumulated_left_stride_distance
            stride_length_right = accumulated_right_stride_distance
            step_length_left = stride_length_left / 2.0
            step_length_right = stride_length_right / 2.0

            # Use symmetric left/right definitions to avoid artificial asymmetry inflation.
            stride_lengths.append((stride_length_left + stride_length_right) / 2.0)
            step_lengths.append((step_length_left + step_length_right) / 2.0)
            stride_asymmetry.append(calculate_asymmetry(stride_length_left, stride_length_right))
            step_asymmetry.append(calculate_asymmetry(step_length_left, step_length_right))

            if last_stride_frame_idx is not None and frame_rate and frame_rate > 0:
                stride_time = (frame_idx_curr - last_stride_frame_idx) / frame_rate
                if stride_time > 0:
                    stride_times.append(stride_time)
                    step_times.append(stride_time / 2.0)
            last_stride_frame_idx = frame_idx_curr

            accumulated_left_stride_distance = 0
            accumulated_right_stride_distance = 0

        last_left_heel = left_heel_curr
        last_right_heel = right_heel_curr

    avg_stride_time, stride_time_variability, stride_time_qc_reasons = calculate_stride_time(stride_times)
    avg_step_time, step_time_variability, step_time_qc_reasons = calculate_step_time(step_times)
    qc_fail_reasons = unique_preserve_order(step_time_qc_reasons + stride_time_qc_reasons)
    event_detection_success_flag = len(step_times) >= 2 and len(stride_times) >= 2
    if not event_detection_success_flag:
        qc_fail_reasons.append("event_detection_insufficient_intervals")
    qc_fail_reasons = unique_preserve_order(qc_fail_reasons)

    return (
        step_lengths,
        stride_lengths,
        step_times,
        stride_times,
        avg_step_time,
        step_time_variability,
        avg_stride_time,
        stride_time_variability,
        step_asymmetry,
        stride_asymmetry,
        qc_fail_reasons,
        {
            **track_quality,
            "event_detection_success_flag": event_detection_success_flag,
            "event_ground_threshold": ground_threshold,
            "event_tolerance": tolerance,
            "event_min_x_movement": min_x_movement,
            "event_min_stride_time_sec": min_stride_time_sec,
            "event_min_stride_interval_frames": min_stride_interval_frames,
            "raw_stride_event_count": raw_stride_event_count,
            "accepted_stride_event_count": accepted_stride_event_count,
            "suppressed_stride_event_count": suppressed_stride_event_count,
        },
    )


def calculate_gait_parameters(
    data,
    frame_rate,
    video_path=None,
    distance_meters=DEFAULT_WITHOUT_DS_DISTANCE_METERS,
    min_stride_time_sec=DEFAULT_MIN_STRIDE_TIME_SEC,
    min_keypoint_score=DEFAULT_MIN_KEYPOINT_SCORE,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    interp_max_gap=DEFAULT_INTERP_MAX_GAP,
    camera_calib=None,
):
    if video_path:
        walking_speed = calculate_walking_speed(video_path, distance_meters)
    else:
        walking_speed = calculate_walking_speed_from_frames(len(data), frame_rate, distance_meters)
    (
        step_lengths,
        stride_lengths,
        step_times,
        stride_times,
        avg_step_time,
        step_time_variability,
        avg_stride_time,
        stride_time_variability,
        step_asymmetry,
        stride_asymmetry,
        qc_fail_reasons,
        extraction_quality,
    ) = calculate_step_and_stride_lengths(
        data,
        frame_rate,
        min_stride_time_sec=min_stride_time_sec,
        min_keypoint_score=min_keypoint_score,
        smooth_window=smooth_window,
        interp_max_gap=interp_max_gap,
        camera_calib=camera_calib,
    )

    return compile_results(
        step_lengths,
        stride_lengths,
        step_times,
        stride_times,
        avg_step_time,
        step_time_variability,
        avg_stride_time,
        stride_time_variability,
        step_asymmetry,
        stride_asymmetry,
        walking_speed,
        qc_fail_reasons=qc_fail_reasons,
        extraction_quality=extraction_quality,
    )


def compile_results(
    step_lengths,
    stride_lengths,
    step_times,
    stride_times,
    avg_step_time,
    step_time_variability,
    avg_stride_time,
    stride_time_variability,
    step_asymmetry,
    stride_asymmetry,
    walking_speed,
    qc_fail_reasons=None,
    extraction_quality=None,
):
    qc_fail_reasons = list(qc_fail_reasons or [])
    if len(step_times) < 2 and "insufficient_step_time_samples_for_variability" not in qc_fail_reasons:
        qc_fail_reasons.append("insufficient_step_time_samples_for_variability")
    if len(stride_times) < 2 and "insufficient_stride_time_samples_for_variability" not in qc_fail_reasons:
        qc_fail_reasons.append("insufficient_stride_time_samples_for_variability")
    if near_zero(step_time_variability):
        step_time_variability = float("nan")
        qc_fail_reasons.append("step_time_variability_zero_degenerate")
    if near_zero(stride_time_variability):
        stride_time_variability = float("nan")
        qc_fail_reasons.append("stride_time_variability_zero_degenerate")

    cadence = calculate_cadence(avg_stride_time)
    if math.isnan(cadence):
        qc_fail_reasons.append("invalid_cadence_from_stride_time")
    qc_fail_reasons = unique_preserve_order(qc_fail_reasons)

    return {
        "average_step_length": mean(step_lengths),
        "average_stride_length": mean(stride_lengths),
        "step_length_variability": std(step_lengths),
        "stride_length_variability": std(stride_lengths),
        "average_step_time": avg_step_time,
        "average_stride_time": avg_stride_time,
        "step_time_variability": step_time_variability,
        "stride_time_variability": stride_time_variability,
        "average_step_assymetry": mean(step_asymmetry),
        "average_stride_assymetry": mean(stride_asymmetry),
        "walking_speed": walking_speed,
        "cadence": cadence,
        "time_estimation_method": "frame_interval",
        "step_count_estimate": len(step_times),
        "stride_count_estimate": len(stride_times),
        "qc_status": "qc_fail" if qc_fail_reasons else "ok",
        "qc_fail_reasons": qc_fail_reasons,
        "usable_frame_ratio": (extraction_quality or {}).get("usable_frame_ratio"),
        "heel_track_missing_ratio": (extraction_quality or {}).get("heel_track_missing_ratio"),
        "event_detection_success_flag": (extraction_quality or {}).get("event_detection_success_flag", False),
        "total_frame_count": (extraction_quality or {}).get("total_frame_count"),
        "usable_frame_count": (extraction_quality or {}).get("usable_frame_count"),
        "raw_bilateral_detected_frame_count": (extraction_quality or {}).get("raw_bilateral_detected_frame_count"),
        "event_ground_threshold": (extraction_quality or {}).get("event_ground_threshold"),
        "event_tolerance": (extraction_quality or {}).get("event_tolerance"),
        "event_min_x_movement": (extraction_quality or {}).get("event_min_x_movement"),
        "event_min_stride_time_sec": (extraction_quality or {}).get("event_min_stride_time_sec"),
        "event_min_stride_interval_frames": (extraction_quality or {}).get("event_min_stride_interval_frames"),
        "raw_stride_event_count": (extraction_quality or {}).get("raw_stride_event_count"),
        "accepted_stride_event_count": (extraction_quality or {}).get("accepted_stride_event_count"),
        "suppressed_stride_event_count": (extraction_quality or {}).get("suppressed_stride_event_count"),
    }


def process_videos(
    video_directory,
    json_directory,
    output_json_path,
    distance_meters=DEFAULT_WITHOUT_DS_DISTANCE_METERS,
    min_stride_time_sec=DEFAULT_MIN_STRIDE_TIME_SEC,
    min_keypoint_score=DEFAULT_MIN_KEYPOINT_SCORE,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    interp_max_gap=DEFAULT_INTERP_MAX_GAP,
    camera_calib_npz=None,
):
    camera_calib = load_camera_calibration(camera_calib_npz)
    results = {}
    for filename in os.listdir(video_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_directory, filename)
            frame_rate = get_frame_rate(video_path)
            json_filename = filename.replace(".mp4", ".json")
            json_path = os.path.join(json_directory, json_filename)

            data = load_json_data(json_path)
            gait_parameters = calculate_gait_parameters(
                data,
                frame_rate,
                video_path,
                distance_meters=distance_meters,
                min_stride_time_sec=min_stride_time_sec,
                min_keypoint_score=min_keypoint_score,
                smooth_window=smooth_window,
                interp_max_gap=interp_max_gap,
                camera_calib=camera_calib,
            )
            results[filename] = gait_parameters

    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)


def process_json_directory_without_ds(
    json_directory,
    output_json_path,
    frame_rate,
    distance_meters=DEFAULT_WITHOUT_DS_DISTANCE_METERS,
    min_stride_time_sec=DEFAULT_MIN_STRIDE_TIME_SEC,
    min_keypoint_score=DEFAULT_MIN_KEYPOINT_SCORE,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    interp_max_gap=DEFAULT_INTERP_MAX_GAP,
    camera_calib_npz=None,
):
    camera_calib = load_camera_calibration(camera_calib_npz)
    results = {}
    for filename in os.listdir(json_directory):
        if not filename.endswith(".json"):
            continue
        json_path = os.path.join(json_directory, filename)
        data = load_json_data(json_path)
        gait_parameters = calculate_gait_parameters(
            data,
            frame_rate,
            video_path=None,
            distance_meters=distance_meters,
            min_stride_time_sec=min_stride_time_sec,
            min_keypoint_score=min_keypoint_score,
            smooth_window=smooth_window,
            interp_max_gap=interp_max_gap,
            camera_calib=camera_calib,
        )
        results[filename.replace(".json", "")] = gait_parameters

    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)


# -----------------------------
# Part B: ds workflow
# -----------------------------

DEFAULT_DS_FRAME_RATE = 30
DEFAULT_DS_GROUP = "HC"
DEFAULT_DS_KEYPOINTS_DIRECTORY = r"D:\是大学啦\AD\dual_task_gait\OutputResults\HC_keypoints"
DEFAULT_DS_NEW_FEATURES_FILE = r"D:\是大学啦\AD\dual_task_gait\OutputResults\HC_gait_features_ds.json"


def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def save_data(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
        print("Data saved to", file_path)


def calculate_velocity(positions, frame_rate):
    velocities = []
    for i in range(1, len(positions)):
        velocity = array_mul(array_sub(positions[i], positions[i - 1]), frame_rate)
        velocities.append(velocity)
    return velocities


def calculate_acceleration(velocities, frame_rate):
    accelerations = []
    for i in range(1, len(velocities)):
        acceleration = array_mul(array_sub(velocities[i], velocities[i - 1]), frame_rate)
        accelerations.append(acceleration)
    return accelerations


def select_base_thresholds(group):
    if group == "AD":
        return 1.138, 1062.0
    return 1.089, 1295.0


def classify_gait_phases(left_heel_positions, right_heel_positions, frame_rate, group):
    left_velocities = calculate_velocity(left_heel_positions, frame_rate)
    right_velocities = calculate_velocity(right_heel_positions, frame_rate)

    left_accelerations = calculate_acceleration(left_velocities, frame_rate)
    right_accelerations = calculate_acceleration(right_velocities, frame_rate)

    stance_times = 0
    swing_times = 0
    double_support_times = 0

    n = min(len(left_velocities), len(right_velocities), len(left_accelerations), len(right_accelerations))
    if n <= 0 or not frame_rate or frame_rate <= 0:
        return {
            "stance_times": float("nan"),
            "swing_times": float("nan"),
            "double_support_times": float("nan"),
            "qc_status": "qc_fail",
            "qc_fail_reasons": ["insufficient_velocity_acceleration_samples"],
            "event_detection_success_flag": False,
        }

    base_v_threshold, base_a_threshold = select_base_thresholds(group)

    pair_velocity_max = []
    pair_accel_max = []
    for i in range(n):
        pair_velocity_max.append(max(array_norm(left_velocities[i]), array_norm(right_velocities[i])))
        pair_accel_max.append(max(array_norm(left_accelerations[i]), array_norm(right_accelerations[i])))

    # Adapt thresholds to current data scale (different camera/viewpoint/noise profiles).
    target_stance_ratio = 62.0 if group == "AD" else 58.0
    velocity_threshold = max(base_v_threshold, percentile(pair_velocity_max, target_stance_ratio))

    stance_idx = [i for i, v in enumerate(pair_velocity_max) if v < velocity_threshold]
    if stance_idx:
        stance_accel = [pair_accel_max[i] for i in stance_idx]
        acceleration_threshold = max(base_a_threshold, percentile(stance_accel, 30.0))
    else:
        acceleration_threshold = base_a_threshold

    for i in range(n):
        if array_norm(left_velocities[i]) < velocity_threshold and array_norm(right_velocities[i]) < velocity_threshold:
            stance_times += 1 / frame_rate
            if array_norm(left_accelerations[i]) < acceleration_threshold and array_norm(right_accelerations[i]) < acceleration_threshold:
                double_support_times += 1 / frame_rate
        else:
            swing_times += 1 / frame_rate

    return {
        "stance_times": stance_times,
        "swing_times": swing_times,
        "double_support_times": double_support_times,
        "qc_status": "ok",
        "qc_fail_reasons": [],
        "event_detection_success_flag": True,
    }


def update_and_save_new_features(
    keypoints_directory,
    new_features_file,
    frame_rate,
    group=DEFAULT_DS_GROUP,
    min_keypoint_score=DEFAULT_MIN_KEYPOINT_SCORE,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    interp_max_gap=DEFAULT_INTERP_MAX_GAP,
    camera_calib_npz=None,
):
    camera_calib = load_camera_calibration(camera_calib_npz)
    new_features_data = {}

    for filename in os.listdir(keypoints_directory):
        if filename.endswith(".json"):
            keypoints_file = os.path.join(keypoints_directory, filename)
            with open(keypoints_file, "r") as file:
                data = json.load(file)

            tracks, track_quality = extract_heel_tracks(
                data,
                min_keypoint_score=min_keypoint_score,
                smooth_window=smooth_window,
                interp_max_gap=interp_max_gap,
                camera_calib=camera_calib,
            )
            left_heel_positions = [t[1] for t in tracks]
            right_heel_positions = [t[2] for t in tracks]

            results = classify_gait_phases(left_heel_positions, right_heel_positions, frame_rate, group)
            results["usable_frame_ratio"] = track_quality.get("usable_frame_ratio")
            results["heel_track_missing_ratio"] = track_quality.get("heel_track_missing_ratio")
            results["total_frame_count"] = track_quality.get("total_frame_count")
            results["usable_frame_count"] = track_quality.get("usable_frame_count")
            results["raw_bilateral_detected_frame_count"] = track_quality.get("raw_bilateral_detected_frame_count")

            file_key = os.path.splitext(filename)[0]
            new_features_data[file_key] = results

    save_data(new_features_data, new_features_file)


def plot_heel_positions(keypoints_directory):
    import matplotlib.pyplot as plt

    left_heel_y_positions = []
    right_heel_y_positions = []

    for filename in os.listdir(keypoints_directory):
        if filename.endswith(".json"):
            keypoints_file = os.path.join(keypoints_directory, filename)
            data = load_data(keypoints_file)

            for frame in data:
                left_heel_y = frame["instances"][0]["keypoints"][15][1]
                right_heel_y = frame["instances"][0]["keypoints"][16][1]
                left_heel_y_positions.append(left_heel_y)
                right_heel_y_positions.append(right_heel_y)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        range(len(left_heel_y_positions)),
        left_heel_y_positions,
        c="blue",
        label="Left Heel Y Position",
        alpha=0.5,
    )
    plt.title("Heel Y Positions Over Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()


def build_parser():
    parser = argparse.ArgumentParser(description="Merged gait feature extraction tool.")
    subparsers = parser.add_subparsers(dest="command")

    without_ds = subparsers.add_parser("without_ds", help="Run non-dual-task gait feature extraction.")
    without_ds.add_argument(
        "--video-directory",
        default=DEFAULT_WITHOUT_DS_VIDEO_DIRECTORY,
        help="Directory containing input mp4 files.",
    )
    without_ds.add_argument(
        "--json-directory",
        default=DEFAULT_WITHOUT_DS_JSON_DIRECTORY,
        help="Directory containing keypoint json files.",
    )
    without_ds.add_argument(
        "--output-json-path",
        default=DEFAULT_WITHOUT_DS_OUTPUT_JSON_PATH,
        help="Output json path.",
    )
    without_ds.add_argument(
        "--distance-meters",
        type=float,
        default=DEFAULT_WITHOUT_DS_DISTANCE_METERS,
        help="Walking distance in meters used for speed calculation.",
    )
    without_ds.add_argument(
        "--min-stride-time-sec",
        type=float,
        default=DEFAULT_MIN_STRIDE_TIME_SEC,
        help="Minimum allowed stride interval in seconds for accepting consecutive gait events.",
    )
    without_ds.add_argument(
        "--min-keypoint-score",
        type=float,
        default=DEFAULT_MIN_KEYPOINT_SCORE,
        help="Minimum keypoint confidence for heel tracking.",
    )
    without_ds.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Smoothing window size for heel trajectories.",
    )
    without_ds.add_argument(
        "--interp-max-gap",
        type=int,
        default=DEFAULT_INTERP_MAX_GAP,
        help="Max gap (frames) for linear interpolation of low-confidence points.",
    )
    without_ds.add_argument(
        "--camera-calib-npz",
        default=None,
        help="Optional camera calibration .npz file for keypoint undistortion.",
    )

    without_ds_json = subparsers.add_parser(
        "without_ds_json",
        help="Run non-dual-task extraction from keypoint json directory only.",
    )
    without_ds_json.add_argument(
        "--json-directory",
        default=DEFAULT_WITHOUT_DS_JSON_DIRECTORY,
        help="Directory containing keypoint json files.",
    )
    without_ds_json.add_argument(
        "--output-json-path",
        default=DEFAULT_WITHOUT_DS_OUTPUT_JSON_PATH,
        help="Output json path.",
    )
    without_ds_json.add_argument(
        "--frame-rate",
        type=float,
        default=DEFAULT_DS_FRAME_RATE,
        help="Frame rate used to estimate temporal gait metrics.",
    )
    without_ds_json.add_argument(
        "--distance-meters",
        type=float,
        default=DEFAULT_WITHOUT_DS_DISTANCE_METERS,
        help="Walking distance in meters used for speed estimation.",
    )
    without_ds_json.add_argument(
        "--min-stride-time-sec",
        type=float,
        default=DEFAULT_MIN_STRIDE_TIME_SEC,
        help="Minimum allowed stride interval in seconds for accepting consecutive gait events.",
    )
    without_ds_json.add_argument(
        "--min-keypoint-score",
        type=float,
        default=DEFAULT_MIN_KEYPOINT_SCORE,
        help="Minimum keypoint confidence for heel tracking.",
    )
    without_ds_json.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Smoothing window size for heel trajectories.",
    )
    without_ds_json.add_argument(
        "--interp-max-gap",
        type=int,
        default=DEFAULT_INTERP_MAX_GAP,
        help="Max gap (frames) for linear interpolation of low-confidence points.",
    )
    without_ds_json.add_argument(
        "--camera-calib-npz",
        default=None,
        help="Optional camera calibration .npz file for keypoint undistortion.",
    )

    ds = subparsers.add_parser("ds", help="Run dual-task gait phase feature extraction.")
    ds.add_argument(
        "--keypoints-directory",
        default=DEFAULT_DS_KEYPOINTS_DIRECTORY,
        help="Directory containing keypoint json files.",
    )
    ds.add_argument(
        "--new-features-file",
        default=DEFAULT_DS_NEW_FEATURES_FILE,
        help="Output feature json file.",
    )
    ds.add_argument(
        "--frame-rate",
        type=float,
        default=DEFAULT_DS_FRAME_RATE,
        help="Frame rate used for velocity/acceleration calculation.",
    )
    ds.add_argument(
        "--group",
        default=DEFAULT_DS_GROUP,
        help="Group for threshold selection. Use AD or HC.",
    )
    ds.add_argument(
        "--min-keypoint-score",
        type=float,
        default=DEFAULT_MIN_KEYPOINT_SCORE,
        help="Minimum keypoint confidence for heel tracking.",
    )
    ds.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Smoothing window size for heel trajectories.",
    )
    ds.add_argument(
        "--interp-max-gap",
        type=int,
        default=DEFAULT_INTERP_MAX_GAP,
        help="Max gap (frames) for linear interpolation of low-confidence points.",
    )
    ds.add_argument(
        "--camera-calib-npz",
        default=None,
        help="Optional camera calibration .npz file for keypoint undistortion.",
    )

    plot = subparsers.add_parser("plot", help="Plot heel y positions from keypoint json files.")
    plot.add_argument(
        "--keypoints-directory",
        default=DEFAULT_DS_KEYPOINTS_DIRECTORY,
        help="Directory containing keypoint json files.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "without_ds":
        process_videos(
            args.video_directory,
            args.json_directory,
            args.output_json_path,
            distance_meters=args.distance_meters,
            min_stride_time_sec=args.min_stride_time_sec,
            min_keypoint_score=args.min_keypoint_score,
            smooth_window=args.smooth_window,
            interp_max_gap=args.interp_max_gap,
            camera_calib_npz=args.camera_calib_npz,
        )
    elif args.command == "without_ds_json":
        process_json_directory_without_ds(
            args.json_directory,
            args.output_json_path,
            args.frame_rate,
            distance_meters=args.distance_meters,
            min_stride_time_sec=args.min_stride_time_sec,
            min_keypoint_score=args.min_keypoint_score,
            smooth_window=args.smooth_window,
            interp_max_gap=args.interp_max_gap,
            camera_calib_npz=args.camera_calib_npz,
        )
    elif args.command == "ds":
        update_and_save_new_features(
            args.keypoints_directory,
            args.new_features_file,
            args.frame_rate,
            group=args.group,
            min_keypoint_score=args.min_keypoint_score,
            smooth_window=args.smooth_window,
            interp_max_gap=args.interp_max_gap,
            camera_calib_npz=args.camera_calib_npz,
        )
    elif args.command == "plot":
        plot_heel_positions(args.keypoints_directory)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
