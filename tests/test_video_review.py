"""Review gates, immutable annotations, timeline integrity and seekable media."""

import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from adualnet_toolkit.video import (
    ROOT, assign_tracks, current_review, draft_review, dump, export_features, load_profile, read, save_review,
    selected_frames, sha, state, validate_review,
)
from adualnet_toolkit.video_routes import VideoService, send_media


def fixture(job):
    meta = {"width": 640, "height": 360, "fps": 30, "duration_s": 2., "synthetic": True}
    poses = {"frames": [{"frame_index": i, "time_s": i/30, "instances": [{"bbox": [100, 30, 300, 340],
               "keypoints": [[200., 50.+15*k] for k in range(17)], "keypoint_scores": [.95]*17}]} for i in range(60)]}
    transcript = {"segments": [{"start": .5, "end": .9, "text": "93", "words": [{"word": "93", "start": .5, "end": .9}]}]}
    dump(job / "media.json", meta)
    dump(job / "pose_raw.json", poses)
    dump(job / "transcript_raw.json", transcript)
    draft_review(job, poses, transcript, meta)
    state(job, "awaiting_review", "synthetic test", revision=0)
    return meta, read(job / "poses.json")


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.job = Path(self.temp.name)
        self.meta, self.poses = fixture(self.job)
        self.review = current_review(self.job)

    def confirmed(self):
        review = copy.deepcopy(self.review)
        review.update(participant_id="SYNTHETIC", reviewer="test", scoring_rule="Synthetic test annotation",
                      gait_reviewed=True, speech_reviewed=True)
        review["responses"][0]["is_correct"] = 0
        return review

    def test_cannot_export_before_both_manual_confirmations(self):
        with self.assertRaisesRegex(ValueError, "explicit human"):
            export_features(self.job)
        review = self.confirmed()
        review["speech_reviewed"] = False
        save_review(self.job, review)
        with self.assertRaisesRegex(ValueError, "explicit human"):
            export_features(self.job)

    def test_included_answers_require_manual_correctness(self):
        review = self.confirmed()
        review["responses"][0]["is_correct"] = None
        with self.assertRaisesRegex(ValueError, "correctness"):
            save_review(self.job, review)

    def test_out_of_window_and_invalid_coordinates_rejected(self):
        review = self.confirmed()
        review["task_start_s"] = 1.
        with self.assertRaisesRegex(ValueError, "outside the task"):
            save_review(self.job, review)
        review = self.confirmed()
        review["keypoint_edits"] = [{"frame_index": 0, "track_id": 1, "joint": 15, "x": 10000, "y": 10}]
        with self.assertRaisesRegex(ValueError, "outside the normalized"):
            save_review(self.job, review)

    def test_excluded_frames_keep_elapsed_time_and_flag_long_gaps(self):
        review = self.confirmed()
        review["excluded_frames"] = list(range(10, 30))
        frames, qc = selected_frames(self.poses, review, 30)
        self.assertEqual(len(frames), 60)
        self.assertEqual(qc["longest_gap_frames"], 20)
        self.assertEqual(frames[30]["frame_index"], 30)

    def test_export_preserves_raws_versions_and_recorded_errors(self):
        import pandas as pd
        before = sha(self.job / "pose_raw.json")
        saved = save_review(self.job, self.confirmed())
        out = export_features(self.job)
        self.assertTrue((out / "reviewed_features.xlsx").exists())
        features = pd.read_csv(out / "reviewed_features.csv").iloc[0]
        self.assertEqual(features.Correct_Num_DT, 0)
        self.assertEqual(features.Num_DT, 1)
        self.assertTrue(pd.isna(features.stance_times))
        self.assertEqual(before, sha(self.job / "pose_raw.json"))
        saved["speech_reviewed"] = False
        next_review = save_review(self.job, saved)
        self.assertEqual(next_review["revision"], 2)
        self.assertTrue((out / "COMPLETE.json").exists())
        with self.assertRaisesRegex(ValueError, "stale"):
            save_review(self.job, saved)
        with self.assertRaisesRegex(ValueError, "explicit human"):
            export_features(self.job)

    def test_joint_correction_is_applied_only_to_reviewed_instance(self):
        review = self.confirmed()
        review["keypoint_edits"] = [{"frame_index": 4, "track_id": 1, "joint": 15, "x": 12, "y": 34}]
        frames, _ = selected_frames(self.poses, review, 30)
        self.assertEqual(frames[4]["instances"][0]["keypoints"][15], [12, 34])
        self.assertNotEqual(self.poses["frames"][4]["instances"][0]["keypoints"][15], [12, 34])

    def test_simcc_scores_are_not_clipped_to_probability_range(self):
        frames = copy.deepcopy(self.poses["frames"])
        frames[0]["instances"][0]["keypoint_scores"][0] = 1.034
        self.assertTrue(assign_tracks(frames))
        self.assertEqual(frames[0]["instances"][0]["keypoint_scores"][0], 1.034)
        frames[0]["instances"][0]["keypoint_scores"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            assign_tracks(frames)


class MediaTests(unittest.TestCase):
    def test_detector_weights_cannot_be_silently_overridden_by_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = read(ROOT / "recognition.example.json")
            config["mmpose"]["det_weights"] = "custom_detector.pth"
            path = Path(tmp) / "recognition.json"
            dump(path, config)
            with self.assertRaisesRegex(ValueError, "Set det_model"):
                load_profile(path)

    def test_byte_range_for_video_seeking(self):
        class Handler:
            headers = {"Range": "bytes=2-5"}
            wfile = io.BytesIO()
            def send_response(self, code):
                self.code = code
            def send_header(self, k, v):
                pass
            def end_headers(self):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.mp4"
            path.write_bytes(b"0123456789")
            handler = Handler()
            send_media(handler, path)
            self.assertEqual(handler.code, 206)
            self.assertEqual(handler.wfile.getvalue(), b"2345")

    def test_local_media_routes_reject_arbitrary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = VideoService(tmp)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                service.folder("../secret")

    def test_browser_inference_requires_processing_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = VideoService(tmp)
            identifier = service.upload(io.BytesIO(b"fixture"), 7, ".mp4")["job_id"]
            with self.assertRaisesRegex(ValueError, "Confirm authorized"):
                service.post_api("/api/video/" + identifier + "/start", {})


if __name__ == "__main__":
    unittest.main()
