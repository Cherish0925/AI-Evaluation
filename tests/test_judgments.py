from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ai_eval.judgments import calibrate


class JudgeCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "judgments.jsonl"

    def write(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return self.path

    def test_confusion_kappa_and_per_class_preserve_orientation(self):
        values = [("pass", "pass"), ("pass", "fail"), ("fail", "pass"), ("fail", "fail")]
        result = calibrate(self.write([{"id": str(i), "reference": ref, "judge": judge}
                                      for i, (ref, judge) in enumerate(values)]),
                           labels=["pass", "fail", "abstain"], resamples=100)
        self.assertEqual(result["agreement"], .5)
        self.assertEqual(result["kappa"], 0)
        self.assertEqual(result["confusion"]["rows"], [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        self.assertEqual(result["per_class"][0]["precision"], .5)
        self.assertEqual(result["per_class"][1]["recall"], .5)
        self.assertIsNone(result["per_class"][2]["precision"])
        self.assertIsNone(result["per_class"][2]["recall"])

    def test_unavailable_labels_are_not_mismatches_or_implicit_classes(self):
        rows = [{"id": "a", "reference": "pass", "judge": "pass"},
                {"id": "b", "reference": "fail", "judge": None},
                {"id": "c", "judge": "fail"}, {"id": "d"}]
        result = calibrate(self.write(rows), resamples=100)
        self.assertEqual(result["agreement"], 1)
        self.assertEqual(result["coverage"]["paired_fraction"], .25)
        self.assertEqual(result["coverage"]["missing_reference"], 2)
        self.assertEqual(result["coverage"]["missing_judge"], 2)
        self.assertEqual(result["coverage"]["missing_both"], 1)
        self.assertEqual(result["confusion"]["labels"], ["fail", "pass"])
        self.assertEqual(sum(sum(row) for row in result["confusion"]["rows"]), 1)

    def test_group_bootstrap_retains_large_clusters(self):
        rows = [{"id": str(i), "group": "large", "reference": "yes", "judge": "yes"} for i in range(9)]
        rows.append({"id": "last", "group": "small", "reference": "yes", "judge": "no"})
        result = calibrate(self.write(rows), resamples=400, seed=19)
        self.assertEqual(result["agreement"], .9)
        self.assertEqual(result["interval"]["independent_groups"], 2)
        self.assertEqual((result["interval"]["lower"], result["interval"]["upper"]), (0, 1))
        reordered = calibrate(self.write(list(reversed(rows))), resamples=400, seed=19)
        self.assertEqual(result["interval"], reordered["interval"])
        self.assertNotEqual(result["source"]["sha256"], reordered["source"]["sha256"])

    def test_single_group_and_degenerate_marginals_do_not_invent_estimates(self):
        result = calibrate(self.write([{"id": str(i), "specimen": "shared",
                                      "reference": "yes", "judge": "yes"} for i in range(3)]),
                           group_key="specimen", resamples=100)
        self.assertEqual(result["agreement"], 1)
        self.assertIsNone(result["interval"])
        self.assertEqual(result["interval_reason"], "fewer_than_two_independent_groups")
        self.assertIsNone(result["kappa"])
        self.assertEqual(result["kappa_reason"], "degenerate_paired_marginals")

    def test_no_paired_labels_preserves_coverage_and_null_estimates(self):
        result = calibrate(self.write([{"id": "a", "reference": "pass"},
                                      {"id": "b", "judge": "fail"}]), resamples=100)
        self.assertIsNone(result["agreement"])
        self.assertIsNone(result["kappa"])
        self.assertIsNone(result["interval"])
        self.assertEqual(result["coverage"]["paired_samples"], 0)
        self.assertEqual(result["coverage"]["total_samples"], 2)
        json.dumps(result, allow_nan=False)

    def test_negative_kappa_and_unequal_class_counts(self):
        rows = [{"id": str(i), "reference": ref, "judge": judge} for i, (ref, judge) in
                enumerate([("yes", "no"), ("yes", "no"), ("yes", "no"), ("no", "yes")])]
        result = calibrate(self.write(rows), labels=["yes", "no"], resamples=100)
        self.assertEqual(result["confusion"]["rows"], [[0, 3], [1, 0]])
        self.assertEqual(result["expected_agreement"], .375)
        self.assertEqual(result["kappa"], -.6)

    def test_duplicate_ids_unknown_classes_and_nonfinite_json_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate judgment id"):
            calibrate(self.write([{"id": "same"}, {"id": "same"}]))
        with self.assertRaisesRegex(ValueError, "absent from declared labels"):
            calibrate(self.write([{"id": "a", "reference": "other"}]), labels=["yes", "no"])
        for text in ['{"id":"a","id":"b"}', '{"id":"a","metadata":{"n":NaN}}',
                     '{"id":"a","metadata":{"n":1e999}}']:
            self.path.write_text(text)
            with self.assertRaises(ValueError):
                calibrate(self.path)

    def test_schema_settings_and_empty_data(self):
        for row in [{"id": 2}, {"id": "a", "reference": True}, {"id": "a", "group": None},
                    {"id": "a", "slices": {"complex": {}}}, {"id": "a", "metadata": []}]:
            with self.assertRaises(ValueError):
                calibrate(self.write([row]))
        path = self.write([{"id": "a", "reference": "yes", "judge": "yes"}])
        for args in [{"labels": []}, {"labels": ["yes", "yes"]}, {"confidence": 1},
                     {"seed": True}, {"resamples": 0}, {"group_key": ""}]:
            with self.assertRaises(ValueError):
                calibrate(path, **args)
        with self.assertRaisesRegex(ValueError, "no records"):
            calibrate(self.write([]))


if __name__ == "__main__":
    unittest.main()
