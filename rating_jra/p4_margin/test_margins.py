from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from margins import multiplier_arrays, sequential_margins
from rating_data import prepare_input
from test_rating import race_rows


class MarginTest(unittest.TestCase):
    def test_stage_margins_ties_missing_and_negative(self):
        finish = np.array([1, 2, 3, 4, 5, 6])
        times = np.array([100, 100.2, 100.5, 101, 102, 103], dtype=float)
        deltas, statuses = sequential_margins(finish, times, 5)
        np.testing.assert_allclose(deltas, [0.2, 0.3, 0.5, 1.0, 1.0])
        self.assertTrue(np.all(statuses == 1))

        finish = np.array([1, 1, 2, 3, 4, 5, 6])
        times = np.array([100, 100, 100.2, 100.5, 101, 102, 103], dtype=float)
        deltas, _ = sequential_margins(finish, times, 5)
        np.testing.assert_allclose(deltas, [0.0, 0.2, 0.3, 0.5, 1.0])

        deltas, statuses = sequential_margins(np.array([1, 2, 0]), np.array([100, 100.2, np.nan]), 2)
        self.assertAlmostEqual(deltas[0], 0.2)
        self.assertEqual(statuses[1], 0)

        _, statuses = sequential_margins(np.array([1, 2, 3]), np.array([100, np.nan, 101]), 2)
        self.assertTrue(np.all(statuses[:2] == 2))

        deltas, statuses = sequential_margins(np.array([1, 2, 3]), np.array([100.5, 100, 101]), 2)
        self.assertEqual(deltas[0], 0.0)
        self.assertEqual(statuses[0], 3)

    def test_multiplier_normalization_caps_and_neutral_missing(self):
        raw = pd.DataFrame(race_rows(1, "2020-01-01", [1, 2, 3, 4], [1, 2, 3, 4], condition_code=701))
        prepared = prepare_input(raw)
        margins = pd.DataFrame([{
            "race_id": 1, "distance": 2000,
            "stage_1": 0.5, "stage_2": 1.0, "stage_3": np.nan, "stage_4": np.nan, "stage_5": np.nan,
        }])
        stages, race = multiplier_arrays(margins, prepared, {"method": "A", "gamma": 1.0, "m_max": 2.0, "normalize": False})
        self.assertEqual(stages[0, 0], 1.5)
        self.assertEqual(stages[0, 2], 1.0)
        self.assertTrue(np.all(race == 1))
        stages, _ = multiplier_arrays(margins, prepared, {"method": "A", "gamma": 1.0, "m_max": 2.0, "normalize": True})
        self.assertEqual(stages[0, 0], 1.25)
        _, race = multiplier_arrays(margins, prepared, {"method": "B", "gamma": 1.0, "m_max": 2.0, "m_min": 0.5, "normalize": False})
        self.assertEqual(race[0], 1.625)
        _, race = multiplier_arrays(margins, prepared, {"method": "B", "gamma": 1.0, "m_max": 1.5, "m_min": 0.5, "normalize": False})
        self.assertEqual(race[0], 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
