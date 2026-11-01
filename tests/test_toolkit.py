"""Scientific invariants and public-interface failure cases; synthetic inputs only."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from adualnet_toolkit.analysis import load_table, make_demo, prepare_table, run_analysis
from adualnet_toolkit.speech import extract_reviewed_speech
from adualnet_toolkit.uncertainty import bootstrap_weights, repeat_metrics, summarize_predictions


class InputTests(unittest.TestCase):
    def test_duplicate_file_headers_are_not_silently_renamed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.csv"
            path.write_text("ID,label,age,age\nsynthetic,0,65,66\n")
            with self.assertRaisesRegex(ValueError, "Duplicate CSV"):
                load_table(path)

    def test_fractional_labels_are_rejected_before_integer_cast(self):
        df = make_demo(40)
        df["label"] = df.label.astype(float)
        df.loc[0, "label"] = .25
        self.assertFalse(prepare_table(df, 2)[2]["passed"])

    def test_duplicate_participants_and_ambiguous_aliases_rejected(self):
        df = make_demo(40)
        df.loc[1, "ID"] = df.loc[0, "ID"]
        df["gender"] = df.sex
        errors = prepare_table(df, 2)[2]["errors"]
        self.assertTrue(any("one row per participant" in e for e in errors))
        self.assertTrue(any("ambiguous" in e for e in errors))

    def test_mapping_missingness_and_unused_outcomes(self):
        df = make_demo(40).rename(columns={"sex": "gender", "edu": "education"})
        df["MoCA"] = 20
        df.loc[0, "walking_speed"] = np.nan
        features, _, audit = prepare_table(df, 2)
        self.assertTrue(audit["passed"])
        self.assertNotIn("MoCA", features)
        self.assertIn("MoCA", audit["unused_columns"])
        self.assertEqual(len(features), 40)

    def test_infinite_and_missing_features_fail(self):
        df = make_demo(40).drop(columns="cadence")
        df.loc[0, "age"] = np.inf
        errors = prepare_table(df, 2)[2]["errors"]
        self.assertTrue(any("cadence" in e for e in errors))
        self.assertTrue(any("infinite" in e for e in errors))


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 0, 1, 1, 0])
        self.p = np.array([[.1, .5, .5, .8, .7, .2], [.3, .6, .2, .6, .4, .6]])
        self.w = bootstrap_weights(self.y, 40, 42)

    def test_fast_weighted_auc_matches_sklearn_with_ties(self):
        _, _, boot = summarize_predictions(self.y, self.p, self.w)
        for i, w in enumerate(self.w):
            expected = np.mean([roc_auc_score(self.y, p, sample_weight=w) for p in self.p])
            self.assertAlmostEqual(boot[i, 0], expected, places=12)
            np.testing.assert_allclose(boot[i], repeat_metrics(self.y, self.p, w).mean(axis=0), atol=1e-12)

    def test_duplicating_predictions_does_not_narrow_ci(self):
        a, _, ba = summarize_predictions(self.y, self.p, self.w)
        b, _, bb = summarize_predictions(self.y, np.repeat(self.p, 5, axis=0), self.w)
        np.testing.assert_allclose(ba, bb, atol=1e-12)
        self.assertAlmostEqual(a["ci95_low_auroc"], b["ci95_low_auroc"])

    def test_identical_model_paired_difference_is_zero(self):
        _, _, a = summarize_predictions(self.y, self.p, self.w)
        _, _, b = summarize_predictions(self.y, self.p.copy(), self.w)
        np.testing.assert_array_equal(a - b, np.zeros_like(a))


class ReviewedSpeechTests(unittest.TestCase):
    def test_preserves_reviewed_errors_and_zero_response_missingness(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame({"ID": ["S1", "S2"], "task_start_s": [3, 3], "task_end_s": [28, 28]}).to_csv(d / "tasks.csv", index=False)
            pd.DataFrame({"ID": ["S1", "S1", "S1"], "response_onset_s": [4, 12, 23],
                          "response_text": ["93", "wrong answer", "79"], "is_correct": [1, 0, 1]}).to_csv(d / "responses.csv", index=False)
            extract_reviewed_speech(d / "tasks.csv", d / "responses.csv", d / "features.csv")
            result = pd.read_csv(d / "features.csv")
            self.assertEqual(result.loc[0, "Num_DT"], 3)
            self.assertEqual(result.loc[0, "Correct_Num_DT"], 2)
            self.assertEqual(result.loc[0, "Num_20s_DT"], 2)
            self.assertEqual(result.loc[0, "Timecount_DT_1stNum"], 1)
            self.assertTrue(pd.isna(result.loc[1, "ACC_DT"]))


class IntegrationTests(unittest.TestCase):
    def test_synthetic_run_subject_splits_and_private_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame = make_demo(40)
            frame.to_csv(tmp / "input.csv", index=False)
            out = tmp / "run"
            run_analysis(tmp / "input.csv", out, n_splits=2, n_repeats=2, bootstrap=20, models=("DT",),
                         measurement_version="synthetic-test-v1", outcome_definition="Artificial binary outcome",
                         synthetic=True, log=lambda msg: None)
            predictions = pd.read_csv(out / "predictions_no_id.csv")
            sizes = predictions.groupby(["feature_set", "model", "repeat", "anon_subject_id"]).size()
            self.assertTrue(sizes.eq(1).all())
            self.assertTrue(predictions.groupby("feature_set").size().eq(80).all())
            splits = pd.read_csv(out / "cv_splits_no_id.csv")
            for _, g in splits.groupby(["repeat", "fold"]):
                self.assertFalse(set(g[g.role.eq("train")].anon_subject_id) & set(g[g.role.eq("test")].anon_subject_id))
            self.assertIn("SYNTHETIC DEMONSTRATION", (out / "report.html").read_text())
            for file in out.glob("*"):
                if file.suffix in (".csv", ".json", ".html"):
                    self.assertNotIn("SYNTHETIC_0001", file.read_text())
            with self.assertRaisesRegex(ValueError, "already contains"):
                run_analysis(tmp / "input.csv", out)


if __name__ == "__main__":
    unittest.main()
