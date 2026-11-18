from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import P0Config, file_sha256
from rating_data import load_or_prepare, prepare_input
from rating_engine import run_engine
from rating_initialization import FeatureBlock, InitialRule
from rating_pedigree import PedigreeBase, aggregate_side, build_or_load_features
from test_rating import BASE_CONFIG, race_rows, sample_raw

DIAGNOSTICS = {}


class Stage3Tests(unittest.TestCase):
    def test_01_p1_initialization_and_existing_horse(self):
        rule = InitialRule(
            b=np.array([[0.1, 0.2, 0.3]]),
            var0_d=np.array([[0.5, 0.75, 1.25]]),
            beta=np.zeros((1, 4)),
        )
        result = run_engine(prepare_input(sample_raw()), [BASE_CONFIG], record=True, initial_rule=rule, model_name="P1")
        first = result.snapshots[result.snapshots["n_graph_starts"].eq(0)].sort_values("horse_id")
        np.testing.assert_allclose(first["mu0"], [0.2, 0.2, 0.2, 0.1, 0.1], atol=1e-12)
        np.testing.assert_allclose(first["var_pre_persistent"], [0.75, 0.75, 0.75, 0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(first["var_pre_eff"], [0.75, 0.75, 0.75, 0.5, 0.5], atol=1e-12)
        early_raw = pd.DataFrame(race_rows(6001, "2012-05-01", [1, 2], [1, 2], condition_code=703))
        early = run_engine(prepare_input(early_raw), [BASE_CONFIG], record=True, initial_rule=rule, model_name="P1")
        self.assertTrue((early.snapshots["debut_type_code"] == 4).all())
        self.assertTrue((early.snapshots["mu0"] == 0).all())
        self.assertTrue((early.snapshots["var_pre_persistent"] == BASE_CONFIG.var0).all())
        self.assertTrue((early.state["var_cap"] == BASE_CONFIG.var0).all())
        DIAGNOSTICS["p1_initialization"] = "PASS"

    def test_02_p2_initial_features_and_status(self):
        prepared = prepare_input(sample_raw())
        provider = {}
        for race in prepared.races:
            provider[race.race_index] = FeatureBlock(
                X=np.tile(np.array([0.5, 0.25, 0.0, 0.0]), (race.n, 1)),
                N_sire=np.full(race.n, 10, dtype=np.int32),
                N_bms=np.full(race.n, 20, dtype=np.int32),
                sire_status=np.ones(race.n, dtype=np.int8),
                bms_status=np.ones(race.n, dtype=np.int8),
            )
        rule = InitialRule(
            b=np.array([[0.1, 0.2, 0.3]]),
            var0_d=np.array([[0.5, 0.75, 1.25]]),
            beta=np.array([[0.2, 0.1, 0.0, 0.0]]),
        )
        result = run_engine(prepared, [BASE_CONFIG], record=True, initial_rule=rule, feature_provider=provider, model_name="P2a")
        first = result.snapshots[result.snapshots["n_graph_starts"].eq(0)].sort_values("horse_id")
        np.testing.assert_allclose(first["mu0"], [0.325, 0.325, 0.325, 0.225, 0.225], atol=1e-12)
        self.assertTrue((first["N_sire"] == 10).all())
        self.assertTrue((first["N_bms"] == 20).all())
        self.assertTrue((first["sire_status_code"] == 1).all())
        DIAGNOSTICS["p2_initialization"] = "PASS"

    def test_03_asof_excludes_same_day_and_distinguishes_status(self):
        history = pd.DataFrame({
            "date": [100, 200, 300, 350],
            "sire_id": [11, 11, 11, 11],
            "r_win": [1.0, 2.0, 3.0, 100.0],
            "r_top3": [1.0, 2.0, 3.0, 100.0],
        })
        candidates = pd.DataFrame({
            "race_index": [0, 1, 2, 3, 4, 5],
            "date": [350, 350, 400, 350, 350, 350],
            "debut_type_code": [4, 1, 1, 1, 1, 4],
            "sire_id": [11, 11, 11, 22, np.nan, 11],
        })
        N, sum_win, sum_top3, status = aggregate_side(candidates, history, "sire")
        np.testing.assert_array_equal(N, [0, 3, 4, 0, 0, 0])
        np.testing.assert_allclose(sum_win, [0.0, 6.0, 106.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(sum_top3, [0.0, 6.0, 106.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(status, [4, 1, 1, 2, 3, 4])
        DIAGNOSTICS["asof_same_day_and_status"] = "PASS"

    def test_04_standardization_and_zero_information(self):
        candidates = pd.DataFrame({
            "race_index": [0, 0, 1, 2],
            "race_date": pd.to_datetime(["2018-01-01", "2019-01-01", "2021-01-01", "2018-02-01"]),
            "horse_id": [1, 2, 3, 4],
            "debut_type_code": [1, 1, 1, 1],
            "n_graph_starts": [0, 0, 0, 0],
            "condition_code": [701, 701, 701, 701],
            "N_sire": [10, 20, 0, 5],
            "N_bms": [0, 0, 0, 0],
            "sire_sum_win": [2.0, 2.0, 0.0, 1.0],
            "sire_sum_top3": [0.0, 0.0, 0.0, 0.0],
            "bms_sum_win": [0.0, 0.0, 0.0, 0.0],
            "bms_sum_top3": [0.0, 0.0, 0.0, 0.0],
            "sire_status": [1, 1, 2, 3],
            "bms_status": [2, 2, 2, 2],
        })
        base = PedigreeBase(candidates=candidates, history=pd.DataFrame(), cache_key="unit_test")
        features = build_or_load_features(base, 20, use_cache=False)
        block = features.provider[0]
        np.testing.assert_allclose(block.X[:, 0], [1.0, -1.0], atol=1e-12)
        self.assertTrue((block.X[:, 1:] == 0).all())
        self.assertEqual(features.constants[0]["standardization_horses"], 2)
        self.assertEqual(features.constants[1]["scale"], 1.0)
        validation_block = features.provider[1]
        self.assertTrue((validation_block.X == 0).all())
        missing_block = features.provider[2]
        self.assertTrue((missing_block.X == 0).all())
        DIAGNOSTICS["standardization"] = "PASS"

    def test_05_p0_like_rule_and_actual_p0_invariance(self):
        prepared_small = prepare_input(sample_raw())
        rule = InitialRule.p0_like(1, BASE_CONFIG.var0)
        baseline = run_engine(prepared_small, [BASE_CONFIG], record=True)
        with_rule = run_engine(prepared_small, [BASE_CONFIG], record=True, initial_rule=rule)
        columns = ["race_id", "horse_id", "mu_pre", "var_pre_eff", "m_pre", "mu_post", "var_post"]
        pd.testing.assert_frame_equal(
            baseline.snapshots[columns].sort_values(["race_id", "horse_id"]).reset_index(drop=True),
            with_rule.snapshots[columns].sort_values(["race_id", "horse_id"]).reset_index(drop=True),
            check_exact=True,
        )
        config_path = Path("stage2_output/config_p0.yaml")
        self.assertEqual(file_sha256(config_path), "a06b15ef4e9affe345807b500a6b46fa769a6e1939ad0f309805d4a639a303c8")
        if not Path("stage2_input/BaoZ_2_snapshot.mdb").exists():
            self.skipTest("実DBスナップショットは公開対象外のため、この検証はローカル環境でのみ実行します")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        prepared = load_or_prepare(Path("stage2_input/BaoZ_2_snapshot.mdb"))
        masks = prepared.evaluation_masks()
        p0 = P0Config(**config["p0"])
        result = run_engine(prepared, [p0], evaluation_mask=masks["tuning_E"])
        actual = float(result.nll_sum[0] / result.nll_count)
        self.assertAlmostEqual(actual, 2.370406040511258, places=12)
        DIAGNOSTICS["actual_p0_nll_invariance"] = actual


def tearDownModule():
    Path("stage3_test_diagnostics.json").write_text(json.dumps(DIAGNOSTICS, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
