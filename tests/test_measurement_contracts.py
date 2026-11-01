"""Measurement-name compatibility is not measurement equivalence; no model fits."""

import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from adualnet_toolkit.contracts import inspect_csv, inspect_rows, registry


class ContractTests(unittest.TestCase):
    def video(self):
        p = registry()["profiles"]["video_review_v1"]
        row = {k: "1" for group in ["metadata", "gait", "speech"] for k in p[group]}
        row.update(ID="SYNTHETIC_CONTRACT", measurement_version=p["known_measurement_version"], synthetic="True")
        for name in p["structurally_unavailable"]:
            row[name] = ""
        return row

    def test_profile_counts_and_no_equivalence_claim(self):
        profiles = registry()["profiles"]
        for name, n in [("toolkit_table_v03", 26), ("historical_2606_primary", 25), ("temporal_audit_20261031", 15)]:
            p = profiles[name]
            self.assertEqual(sum(len(p[k]) for k in ["demographics", "gait", "speech"]), n)
        self.assertFalse(registry()["cross_profile_equivalence"])

    def test_video_missing_contact_fields_are_valid_but_not_authorized(self):
        row = self.video()
        report = inspect_rows([row], list(row), "video_review_v1")
        self.assertTrue(report["schema_passed"])
        self.assertFalse(report["modeling_authorized"])
        self.assertIn("stance_times", report["entirely_unavailable_fields"])

    def test_fabricated_zero_contact_phase_rejected(self):
        row = self.video()
        row["stance_times"] = "0"
        self.assertFalse(inspect_rows([row], list(row), "video_review_v1")["schema_passed"])

    def test_video_export_not_a_cohort_table_even_if_columns_filled(self):
        row = self.video()
        p = registry()["profiles"]["toolkit_table_v03"]
        for group in ["metadata", "gait", "speech", "demographics"]:
            for key in p[group]:
                if key not in row or row[key] == "":
                    row[key] = "0"
        errors = inspect_rows([row], list(row), "toolkit_table_v03")["errors"]
        self.assertTrue(any("not equivalent" in e for e in errors))

    def test_zero_response_missingness_not_changed_to_accuracy_zero(self):
        row = self.video()
        row.update(Num_DT="0", ACC_DT="", Timecount_DT_1stNum="")
        self.assertTrue(inspect_rows([row], list(row), "video_review_v1")["schema_passed"])
        row["ACC_DT"] = "0"
        self.assertFalse(inspect_rows([row], list(row), "video_review_v1")["schema_passed"])

    def test_mixed_versions_missing_ids_and_nonfinite_values_rejected(self):
        a, b = self.video(), self.video()
        b.update(ID="SYNTHETIC_B", measurement_version="other_version")
        self.assertFalse(inspect_rows([a, b], list(a), "video_review_v1")["schema_passed"])
        a["ID"] = None
        a["Num_DT"] = "inf"
        self.assertFalse(inspect_rows([a], list(a), "video_review_v1")["schema_passed"])

    def test_empty_duplicate_and_input_preservation(self):
        self.assertFalse(inspect_rows([], [], "toolkit_table_v03")["schema_passed"])
        row = self.video()
        self.assertFalse(inspect_rows([row], list(row)+["ID"], "video_review_v1")["schema_passed"])
        before = copy.deepcopy(row)
        inspect_rows([row], list(row), "video_review_v1")
        self.assertEqual(row, before)

    def test_cli_is_read_only_and_does_not_import_sklearn(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.csv"
            row = self.video()
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            before = path.read_bytes()
            code = ("import sys\nfrom adualnet_toolkit.cli import main\n"
                    "try:\n main(['contract-check','--input',sys.argv[1],'--profile','video_review_v1'])\n"
                    "except SystemExit as e:\n assert e.code==0\n"
                    "assert 'sklearn' not in sys.modules\n")
            result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["schema_passed"])
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(inspect_csv(path, "video_review_v1")["schema_passed"])


if __name__ == "__main__":
    unittest.main()
