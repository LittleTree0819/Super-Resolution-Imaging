from __future__ import annotations

import unittest

import numpy as np

from super_resolution_imaging.training.evaluation import confidence_interval


class EvaluationTests(unittest.TestCase):
    def test_confidence_interval_is_deterministic_and_contains_mean(self) -> None:
        values = np.asarray([-1.0, 0.0, 1.0, 2.0])
        first = confidence_interval(values, repetitions=1000)
        second = confidence_interval(values, repetitions=1000)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], values.mean())
        self.assertGreaterEqual(first[1], values.mean())


if __name__ == "__main__":
    unittest.main()
