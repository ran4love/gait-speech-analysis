#!/usr/bin/env python3
"""Build public fixtures from mathematical coordinates, never from local study data."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gaitfeature


def finite(value):
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if hasattr(value, "item"):
        return finite(value.item())
    if isinstance(value, float):
        return round(value, 10) if math.isfinite(value) else None
    return value


def make_fixture():
    frames = []
    for i in range(360):
        cx = 280 + 50 * math.sin(i / 70)
        points = [[cx + (k % 2) * 25, 50 + k * 15] for k in range(17)]
        # Illustrative excursions exercise the unchanged image-space event detector.
        # They are not simulated physiological ground truth.
        phase = 2 * math.pi * i / 30
        points[15] = [cx - 20 + 75 * math.sin(phase), 300 + 8 * math.cos(phase)]
        points[16] = [cx + 20 - 75 * math.sin(phase), 300 - 8 * math.cos(phase)]
        points = [[round(v, 6) for v in point] for point in points]
        frames.append({"frame_index": i, "instances": [{"keypoints": points, "keypoint_scores": [.95] * 17}]})
    result = gaitfeature.calculate_gait_parameters(frames, 30, distance_meters=float("nan"))
    return finite({
        "version": "public_synthetic_review_v1", "synthetic": True,
        "id": "SYNTHETIC_PUBLIC_001", "width": 640, "height": 360, "fps": 30, "duration_s": 12,
        "frames": frames, "gait_reference": result,
        "raw_transcript": "93 86 78 71",
        "responses": [{"onset": t, "text": word, "include": True, "correct": None}
                      for t, word in [(2, "93"), (4, "86"), (6, "78"), (8, "71")]],
        "provenance": {
            "origin": "Mathematical coordinates; scripted transcript; no human recordings or cohort rows",
            "neural_inference_performed": False,
            "gait_computation": "Precomputed with existing gaitfeature.calculate_gait_parameters; fixed 0-12s window",
            "gait_extractor_sha256": hashlib.sha256((ROOT / "src/gaitfeature.py").read_bytes()).hexdigest(),
            "frames_sha256": hashlib.sha256(json.dumps(frames, separators=(",", ":")).encode()).hexdigest(),
            "speech_computation": "Live human-reviewed response counts; same definitions as reviewed_response_onset_counts_v1",
            "historical_equivalence_certified": False, "clinical_validity_established": False,
            "contact_phases": "missing: no validated foot-contact definition", "metric_spatial_calibration": False,
        },
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "public_demo/assets/fixture.js")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = "/* Generated synthetic fixture; no study data. */\nwindow.PUBLIC_FIXTURE=" + json.dumps(make_fixture(), ensure_ascii=True, separators=(",", ":"), allow_nan=False) + ";\n"
    if args.check:
        if not args.out.exists() or args.out.read_text() != payload:
            raise SystemExit("Public fixture differs from deterministic generator. Rebuild and review before publishing.")
        print("[DONE] Synthetic public fixture verified")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print("[DONE] Generated mathematical public fixture:", args.out)


if __name__ == "__main__":
    main()
