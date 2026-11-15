import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_routing_replay import bit_metrics, validate_coverage


class RoutingEvidenceTests(unittest.TestCase):
    def test_full_logit_gate_detects_signed_zero_and_invalid_numbers(self):
        self.assertEqual(bit_metrics([0.0], [-0.0])["different_bits"], 1)
        self.assertEqual(bit_metrics([1.0], [1.0])["different_bits"], 0)
        for bad in ([float("nan")], [float("inf")], [1.0, 2.0]):
            with self.assertRaises(ValueError):
                bit_metrics(bad, [1.0])

    def test_requires_each_position_and_layer_once(self):
        routes = [{"position": p, "layer": layer, "logits_scores_and_weights_exact": True}
                  for p in range(2) for layer in range(40)]
        validate_coverage([0, 1], routes)
        for bad in (routes[:-1], routes[:-1] + [routes[0]], routes + [routes[0]]):
            with self.assertRaises(ValueError):
                validate_coverage([0, 1], bad)
        with self.assertRaises(ValueError):
            validate_coverage([], [])
        routes[0]["logits_scores_and_weights_exact"] = False
        with self.assertRaises(ValueError):
            validate_coverage([0, 1], routes)
