"""Public artifact and privacy checks; only mathematically generated fixtures."""

import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public_demo"
spec = importlib.util.spec_from_file_location("public_fixture_builder", ROOT / "scripts/build_public_demo_fixture.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PublicDemoTests(unittest.TestCase):
    def fixture(self):
        return json.loads((PUBLIC / "assets/fixture.js").read_text().split("window.PUBLIC_FIXTURE=", 1)[1].strip()[:-1])

    def test_fixture_origin_and_full_precision_source_guard(self):
        fixture = self.fixture()
        self.assertTrue(fixture["synthetic"])
        self.assertEqual(fixture["id"], "SYNTHETIC_PUBLIC_001")
        self.assertFalse(fixture["provenance"]["neural_inference_performed"])
        self.assertEqual(fixture, builder.make_fixture())

    def test_all_pose_frames_and_unreviewed_answers_retained(self):
        fixture = self.fixture()
        self.assertEqual(len(fixture["frames"]), 360)
        self.assertTrue(all(len(f["instances"][0]["keypoints"]) == 17 for f in fixture["frames"]))
        self.assertTrue(all(r["correct"] is None for r in fixture["responses"]))
        self.assertEqual(fixture["responses"][2]["text"], "78")

    def test_no_contact_phase_or_metric_distance_fabrication(self):
        gait = self.fixture()["gait_reference"]
        self.assertIsNone(gait["walking_speed"])
        for name in ["stance_times", "swing_times", "double_support_times"]:
            self.assertIsNone(gait.get(name))

    def test_gait_reference_exercises_existing_detector_without_hiding_qc(self):
        gait = self.fixture()["gait_reference"]
        self.assertGreater(gait["stride_count_estimate"], 2)
        self.assertIsNotNone(gait["average_stride_time"])
        self.assertAlmostEqual(gait["average_stride_time"], 2 * gait["average_step_time"])
        self.assertIn("qc_fail_reasons", gait)

    def test_public_ui_has_no_file_inputs_or_application_network_clients(self):
        html = (PUBLIC / "index.html").read_text()
        self.assertNotRegex(html, r'type=["\']file["\']')
        self.assertIn("connect-src 'none'", html)
        self.assertIn("form-action 'none'", html)
        for file in (PUBLIC / "assets").glob("*.js"):
            code = file.read_text()
            self.assertNotRegex(code, r"\bfetch\s*\(|XMLHttpRequest|sendBeacon\s*\(|new WebSocket|serviceWorker")

    def test_public_directory_is_explicit_and_contains_no_private_routes(self):
        expected = {"index.html", ".nojekyll", "assets/demo.css", "assets/demo.js", "assets/core.js", "assets/xlsx.js", "assets/fixture.js"}
        self.assertEqual({p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob("*") if p.is_file()}, expected)
        for name in expected:
            data = (PUBLIC / name).read_text()
            self.assertNotRegex(data, r"/Users/|/home/|100\.99\.|10\.20\.|127\.0\.0\.1|recognition\.local\.json|local_workspace/")
            self.assertNotIn("__LOCAL_TOKEN__", data)

    def test_asset_paths_work_on_a_github_project_subpath(self):
        html = (PUBLIC / "index.html").read_text()
        paths = re.findall(r'(?:src|href)="(\./assets/[^"]+)"', html)
        self.assertEqual(len(paths), 5)
        self.assertTrue(all((PUBLIC / p.split("?", 1)[0]).is_file() for p in paths))
        self.assertTrue(all("?v=" in p for p in paths))

    def test_mit_license_is_present_and_not_a_pending_grant(self):
        self.assertIn("MIT License", (ROOT / "LICENSE").read_text())
        self.assertIn("Permission is hereby granted", (ROOT / "LICENSE").read_text())
        self.assertNotIn("not yet selected", (ROOT / "LICENSE_NOTICE.md").read_text())

    def test_response_action_labels_do_not_break_into_vertical_letters(self):
        css = (PUBLIC / "assets/demo.css").read_text()
        rule = re.search(r"\.row-actions button\{([^}]+)\}", css).group(1)
        self.assertIn("white-space:nowrap", rule)
        self.assertIn("flex-shrink:0", rule)


if __name__ == "__main__":
    unittest.main()
