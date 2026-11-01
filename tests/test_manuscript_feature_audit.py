import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit", ROOT / "scripts/audit_manuscript_feature_table.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONTRACT = json.loads((ROOT / "docs/adualnet_2606_feature_contract.json").read_text())


class FeatureAuditTests(unittest.TestCase):
    def row(self):
        r = dict.fromkeys(CONTRACT["columns_in_original_order"], 0)
        r.update(ID="synthetic-only", walking_speed=0.5, average_step_length=50,
                 average_stride_length=100, step_length_variability=2,
                 stride_length_variability=4, average_step_time=100,
                 average_stride_time=200, step_time_variability=4,
                 stride_time_variability=8, cadence=0.3, Timecount_DT=10,
                 Timecount_DT_1stNum=1, Num_DT=5, Correct_Num_DT=4,
                 Num_20s_DT=5, Correct_Num_20s_DT=4)
        return r

    def test_contract_counts(self):
        self.assertEqual(len(CONTRACT["fields"]), 31)
        self.assertEqual(len(CONTRACT["primary_predictors"]), 25)
        self.assertNotIn("MoCA", CONTRACT["primary_predictors"])
        self.assertNotIn("label", CONTRACT["primary_predictors"])
        self.assertTrue(CONTRACT["author_confirmation"]["uniform_thresholds_confirmed_by_author"])

    def test_legacy_signature_never_certifies_physical_units(self):
        report = MODULE.inspect_rows([self.row()], CONTRACT)
        self.assertTrue(all(n == 1 for n in report["legacy_formula_match_counts"].values()))
        self.assertFalse(report["publication_ready"])
        self.assertEqual(report["speech_flags_without_identifiers"], [])

    def test_missing_column_and_empty_table(self):
        row = self.row()
        del row["walking_speed"]
        self.assertEqual(MODULE.inspect_rows([row], CONTRACT)["missing_columns"], ["walking_speed"])
        self.assertFalse(MODULE.inspect_rows([], CONTRACT)["publication_ready"])

    def test_zero_denominator_and_flagged_counts(self):
        row = self.row()
        row.update(walking_speed=0, average_stride_time=0, Num_20s_DT=3)
        report = MODULE.inspect_rows([row], CONTRACT)
        self.assertEqual(report["legacy_formula_match_counts"]["legacy_cadence"], 0)
        self.assertIn("within20s_window_disagrees_with_full_task", report["speech_flags_without_identifiers"][0]["flags"])

    def test_duplicate_keys(self):
        self.assertEqual(MODULE.inspect_rows([self.row(), self.row()], CONTRACT)["duplicate_identifier_count"], 1)


if __name__ == "__main__":
    unittest.main()
