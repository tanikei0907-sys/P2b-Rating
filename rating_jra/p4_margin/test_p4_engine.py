from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from p4_engine import P4Engine
from rating_data import prepare_input
from rating_engine import run_engine
from rating_initialization import InitialRule
from test_rating import BASE_CONFIG, race_rows, sample_raw, sorted_snapshots

STANDARDIZATION = [{"center": 0.0, "scale": 1.0} for _ in range(4)]


def run_p4(raw, method="A", stages=None, scales=None, rule=None, pedigree=None):
    prepared = prepare_input(raw, allow_future=True)
    n = len(prepared.races)
    if stages is None:
        stages = np.ones((n, 5))
    if scales is None:
        scales = np.ones(n)
    if rule is None:
        rule = InitialRule.p0_like(1, BASE_CONFIG.var0)
    if pedigree is None:
        pedigree = pd.DataFrame({"horse_id": prepared.horse_ids, "sire_id": np.nan, "bms_id": np.nan})
    engine = P4Engine(prepared, BASE_CONFIG, rule, pedigree, STANDARDIZATION, method, np.asarray(stages, dtype=float), np.asarray(scales, dtype=float), True)
    return prepared, engine, engine.run()


class P4EngineTest(unittest.TestCase):
    def test_unit_multipliers_match_frozen_update(self):
        expected = run_engine(prepare_input(sample_raw()), [BASE_CONFIG], record=True, initial_rule=InitialRule.p0_like(1, BASE_CONFIG.var0))
        for method in ["A", "B"]:
            _, _, actual = run_p4(sample_raw(), method=method)
            pd.testing.assert_frame_equal(sorted_snapshots(actual), sorted_snapshots(expected), check_exact=True)
            np.testing.assert_array_equal(actual.nll, expected.nll)

    def test_uniform_stage_multiplier_equals_race_scaling(self):
        _, _, stages = run_p4(sample_raw(), method="A", stages=np.full((4, 5), 2.0))
        _, _, race = run_p4(sample_raw(), method="B", scales=np.full(4, 2.0))
        columns = ["mu_pre", "var_pre_eff", "mu_post", "var_post", "m_pre"]
        np.testing.assert_allclose(stages.snapshots[columns], race.snapshots[columns], atol=1e-12, rtol=0)
        np.testing.assert_allclose(stages.nll, race.nll, atol=0, rtol=0)

    def test_margin_changes_updates_but_not_observed_nll_formula(self):
        raw = pd.DataFrame(race_rows(9001, "2020-01-01", [1, 2, 3, 4], [1, 2, 3, 4], condition_code=701))
        _, _, base = run_p4(raw)
        multipliers = np.array([[1.0, 2.0, 3.0, 1.0, 1.0]])
        _, _, modulated = run_p4(raw, stages=multipliers)
        self.assertAlmostEqual(base.nll[0], modulated.nll[0], places=12)
        self.assertFalse(np.allclose(base.snapshots["mu_post"], modulated.snapshots["mu_post"]))

    def test_same_day_pedigree_history_is_deferred(self):
        raw = pd.DataFrame(
            race_rows(10001, "2015-01-01", [4, 3, 2, 1], [1, 2, 3, 4], condition_code=701)
            + race_rows(10002, "2015-01-01", [8, 7, 6, 5], [1, 2, 3, 4], condition_code=701)
            + race_rows(10003, "2015-01-02", [10, 9], [1, 2], condition_code=701)
        )
        pedigree = pd.DataFrame({"horse_id": np.arange(1, 11), "sire_id": 123, "bms_id": 456})
        rule = InitialRule(
            b=np.array([[0.05, 0.0, -0.05]]), var0_d=np.array([[1.0, 1.0, 1.0]]),
            beta=np.array([[0.1, 0.2, 0.05, -0.1]]),
        )
        _, engine, result = run_p4(raw, stages=np.full((3, 5), 1.2), rule=rule, pedigree=pedigree)
        self.assertTrue(result.snapshots[result.snapshots["race_id"].isin([10001, 10002])]["N_sire"].eq(0).all())
        self.assertTrue(result.snapshots[result.snapshots["race_id"].eq(10003)]["N_sire"].eq(8).all())
        self.assertEqual(len(engine.history), 10)

    def test_search_mode_matches_recording_nll(self):
        prepared, _, recorded = run_p4(sample_raw(), stages=np.full((4, 5), 1.2))
        pedigree = pd.DataFrame({"horse_id": prepared.horse_ids, "sire_id": np.nan, "bms_id": np.nan})
        engine = P4Engine(
            prepared, BASE_CONFIG, InitialRule.p0_like(1, BASE_CONFIG.var0), pedigree, STANDARDIZATION,
            "A", np.full((4, 5), 1.2), np.ones(4), False,
        )
        searched = engine.run()
        self.assertEqual(searched.nll_count, int(np.isfinite(recorded.nll).sum()))
        self.assertAlmostEqual(searched.nll_sum[0], float(np.nansum(recorded.nll)), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
