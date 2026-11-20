import math
import unittest
from unittest.mock import Mock, patch

from ai_eval.metrics import score


def measure(name, output, target, **params):
    return score({"scorer": name, "params": params}, output, target)


class MetricsTests(unittest.TestCase):
    def test_exact_match_is_structural_and_normalization_is_explicit(self):
        self.assertEqual(measure("exact_match", {"a": [1], "b": 2}, {"b": 2, "a": [1]}), 1)
        self.assertEqual(measure("exact_match", True, 1), 0)
        self.assertEqual(measure("exact_match", " Answer ", "answer"), 0)
        self.assertEqual(measure("exact_match", " Answer ", "answer", strip=True, casefold=True), 1)

    def test_numeric_tolerance_is_relative_to_reference(self):
        self.assertEqual(measure("numeric", 110, 100, rel_tol=.1), 1)
        self.assertEqual(measure("numeric", 111, 100, rel_tol=.1), 0)
        self.assertEqual(measure("numeric", 1e-9, 0), 1)
        self.assertEqual(measure("numeric", 1e308, -1e308, rel_tol=1.8), 0)
        self.assertEqual(measure("numeric", 1e308, -1e308, rel_tol=2), 1)
        with self.assertRaises(ValueError):
            measure("numeric", "1", 1)
        with self.assertRaises(ValueError):
            measure("numeric", 1, 1, abs_tol=-1)

    def test_set_and_multiset_f1_have_different_denominators(self):
        self.assertEqual(measure("set_f1", ["a", "a", "b"], ["a"]), 2 / 3)
        self.assertEqual(measure("token_f1", "a a b", "a"), .5)
        self.assertEqual(measure("set_f1", [], []), 1)
        self.assertEqual(measure("token_f1", "", "a"), 0)
        self.assertEqual(measure("token_f1", "a,", "a"), 0)

    def test_retrieval_denominators_and_duplicate_ranking(self):
        self.assertEqual(measure("recall_at_k", ["a", "x", "b"], ["a", "b"], k=2), .5)
        with self.assertRaises(ValueError):
            measure("recall_at_k", ["a", "a"], ["a", "b"], k=2)
        with self.assertRaises(ValueError):
            measure("recall_at_k", [], [], k=1)
        with self.assertRaises(ValueError):
            measure("recall_at_k", ["a"], ["a"], k=True)

    def test_ndcg_uses_graded_ideal_ranking_and_discount(self):
        expected = (3 + 7 / math.log2(3)) / (7 + 3 / math.log2(3))
        self.assertAlmostEqual(measure("ndcg_at_k", ["b", "a"], {"a": 3, "b": 2}, k=2), expected)
        self.assertEqual(measure("ndcg_at_k", ["a", "b"], {"a": 3, "b": 2}, k=2), 1)
        self.assertEqual(measure("ndcg_at_k", ["x"], {"a": 1}, k=1), 0)
        with self.assertRaises(ValueError):
            measure("ndcg_at_k", ["a"], {"a": 0}, k=1)

    def test_probabilities_are_supplied_not_guessed_or_renormalized(self):
        self.assertAlmostEqual(measure("brier", .8, 1), .04)
        self.assertAlmostEqual(measure("brier", {"a": .8, "b": .2}, "a"), .08)
        self.assertAlmostEqual(measure("log_loss", .8, 1), -math.log(.8))
        self.assertAlmostEqual(measure("log_loss", .8, 0), -math.log(.2))
        self.assertAlmostEqual(measure("log_loss", {"a": .8, "b": .2}, "a"), -math.log(.8))
        for output, target in (("likely", 1), (2, 1), ({"a": .8, "b": .3}, "a"), (True, 1)):
            with self.subTest(output=output), self.assertRaises(ValueError):
                measure("brier", output, target)

    def test_infinite_log_loss_needs_explicit_cap(self):
        with self.assertRaises(ValueError):
            measure("log_loss", 0, 1)
        self.assertAlmostEqual(measure("log_loss", 0, 1, epsilon=1e-6), -math.log(1e-6))
        self.assertEqual(measure("log_loss", 1, 1), 0)

    def test_scorer_plugin_and_nonfinite_result(self):
        point = Mock()
        point.load.return_value = lambda output, target, scale: scale * (output - target)
        with patch("ai_eval.metrics.entry_points", return_value=[point]):
            self.assertEqual(measure("custom", 5, 3, scale=2), 4)
        point.load.return_value = lambda *args: float("nan")
        with patch("ai_eval.metrics.entry_points", return_value=[point]), self.assertRaises(ValueError):
            measure("custom", None, None)
        with patch("ai_eval.metrics.entry_points", return_value=[point, point]), self.assertRaises(ValueError):
            measure("custom", None, None)


if __name__ == "__main__":
    unittest.main()
