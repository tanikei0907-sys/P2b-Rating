from __future__ import annotations

import copy
import itertools
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import (
    P0Config,
    WEIGHTS,
    diagonal_update,
    harville_inclusion_probabilities,
    pl_race_single,
    pl_single_order,
)
from rating_data import prepare_input
from rating_engine import run_engine

DIAGNOSTICS = {}
BASE_CONFIG = P0Config(var0=1.0, q=0.004, kappa=0.7, beta_kg=0.0, weight="W_share")


def race_rows(
    race_id: int,
    date: str,
    horse_ids: list[int],
    finish: list[int],
    weights: list[float] | None = None,
    abnormal: list[int] | None = None,
    affiliations: list[int] | None = None,
    condition_code: int = 703,
    organizer_code: int = 1,
    surface_code: int = 0,
    venue_code: int = 5,
    venue_name: str = "東京",
    race_data_created: str = "1900-01-01",
):
    weights = weights or [54.0] * len(horse_ids)
    abnormal = abnormal or [0] * len(horse_ids)
    affiliations = affiliations or [1 if i % 2 == 0 else 2 for i in range(len(horse_ids))]
    rows = []
    for index, horse_id in enumerate(horse_ids):
        rows.append({
            "race_id": race_id,
            "race_date": date,
            "organizer_code": organizer_code,
            "surface_code": surface_code,
            "venue_code": venue_code,
            "venue_name": venue_name,
            "distance": 1600,
            "condition_code": condition_code,
            "n_registered": len(horse_ids),
            "horse_id": horse_id,
            "horse_number": index + 1,
            "weight_kg": weights[index],
            "finish_position": finish[index],
            "abnormal_code": abnormal[index],
            "affiliation_code": affiliations[index],
            "result_version": "unit_test_v1",
            "race_data_created": race_data_created,
        })
    return rows


def sample_raw() -> pd.DataFrame:
    rows = []
    rows += race_rows(1001, "2020-01-01", [1, 2, 3], [1, 2, 0], [54, 55, 56], [0, 0, 4])
    rows += race_rows(1002, "2020-01-01", [4, 5], [1, 2], [53, 54], condition_code=701)
    rows += race_rows(1003, "2020-01-02", [1, 2, 4], [1, 2, 3], [54, 54, 55])
    rows += race_rows(1004, "2020-02-01", [1, 3, 5], [2, 1, 3], [54, 56, 53])
    return pd.DataFrame(rows)


def sorted_snapshots(result):
    columns = [
        "race_id", "race_date", "horse_id", "mu_pre", "var_pre_persistent", "var_pre_eff", "m_pre",
        "days_since_start", "n_graph_starts", "mu_post", "var_post", "finish_position",
    ]
    return result.snapshots[columns].sort_values(["race_id", "horse_id"]).reset_index(drop=True)


class RatingTests(unittest.TestCase):
    def test_01_gradient_information_weights_and_full_update(self):
        scores = np.array([0.2, -0.1, 0.3, 0.05])
        order = np.array([0, 1, 2], dtype=np.int16)
        unranked = np.array([3], dtype=np.int16)
        weights = WEIGHTS["W_share"]
        ll, gradient, diagonal, information = pl_single_order(scores, order, unranked, 3, weights, True)
        epsilon = 1.0e-6
        numeric_gradient = np.zeros(4)
        numeric_information = np.zeros((4, 4))
        for index in range(4):
            plus = scores.copy()
            minus = scores.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            ll_plus = pl_single_order(plus, order, unranked, 3, weights)[0]
            ll_minus = pl_single_order(minus, order, unranked, 3, weights)[0]
            numeric_gradient[index] = (ll_plus - ll_minus) / (2 * epsilon)
            gradient_plus = pl_single_order(plus, order, unranked, 3, weights)[1]
            gradient_minus = pl_single_order(minus, order, unranked, 3, weights)[1]
            numeric_information[index] = -(gradient_plus - gradient_minus) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numeric_gradient, atol=2e-8, rtol=1e-7)
        np.testing.assert_allclose(diagonal, np.diag(information), atol=1e-12)
        np.testing.assert_allclose(information, numeric_information, atol=2e-7, rtol=1e-6)
        for weight in WEIGHTS.values():
            self.assertEqual(weight.shape, (5,))
            self.assertTrue(np.all(weight >= 0))
            self.assertAlmostEqual(float(weight.sum()), 1.0)
        mu = np.array([0.1, -0.2, 0.0, 0.3])
        var = np.array([1.0, 0.8, 1.2, 0.9])
        finish = np.array([1, 2, 3, 0])
        diag = diagonal_update(mu, var, np.array([54, 54, 55, 56]), order[None, :], unranked, finish, BASE_CONFIG)
        full = diagonal_update(mu, var, np.array([54, 54, 55, 56]), order[None, :], unranked, finish, BASE_CONFIG, full_information=True)
        DIAGNOSTICS["diagonal_vs_full"] = {
            "mu_diagonal": diag["mu_post"].tolist(),
            "mu_full": full["mu_post"].tolist(),
            "max_abs_mu_difference": float(np.max(np.abs(diag["mu_post"] - full["mu_post"]))),
            "var_diagonal": diag["var_post"].tolist(),
            "var_full": full["var_post"].tolist(),
        }

    def test_02_synthetic_recovery_rest_small_fields_and_ties(self):
        rng = np.random.default_rng(982451653)
        horse_count = 60
        true_mu = rng.normal(0, 1, horse_count)
        rows = []
        race_count = 1200
        tie_races = 0
        small_races = 0
        for race_index in range(race_count):
            field = 2 if race_index % 17 == 0 else int(rng.integers(5, 9))
            small_races += int(field == 2)
            horses = rng.choice(horse_count, size=field, replace=False)
            remaining = list(horses)
            order = []
            while remaining:
                values = np.exp(true_mu[remaining] - np.max(true_mu[remaining]))
                selected = int(rng.choice(len(remaining), p=values / values.sum()))
                order.append(remaining.pop(selected))
            local_order = [list(horses).index(horse) for horse in order]
            finish = np.empty(field, dtype=int)
            finish[local_order] = np.arange(1, field + 1)
            if field >= 4 and race_index % 23 == 0:
                finish[local_order[0]] = 1
                finish[local_order[1]] = 1
                finish[local_order[2:]] = np.arange(2, field)
                tie_races += 1
            date = pd.Timestamp("2012-01-01") + pd.Timedelta(days=int(race_index * 2 + race_index % 11))
            rows += race_rows(
                500000 + race_index,
                date.strftime("%Y-%m-%d"),
                [int(1000 + horse) for horse in horses],
                finish.tolist(),
                rng.uniform(50, 58, field).tolist(),
            )
        raw = pd.DataFrame(rows)
        correlations = []
        variance_ratios = []
        for count in [250, 600, 1200]:
            race_ids = [500000 + index for index in range(count)]
            prepared = prepare_input(raw[raw["race_id"].isin(race_ids)].copy())
            result = run_engine(prepared, [BASE_CONFIG])
            state = result.state.sort_values("horse_id")
            correlation = float(np.corrcoef(state["mu"], true_mu)[0, 1])
            correlations.append(correlation)
            mse = float(np.mean((state["mu"].to_numpy() - true_mu) ** 2))
            variance_ratios.append(float(state["var"].mean() / mse))
        DIAGNOSTICS["synthetic_recovery"] = {
            "race_counts": [250, 600, 1200],
            "correlations": correlations,
            "posterior_var_over_squared_error": variance_ratios,
            "small_field_races": small_races,
            "tie_races": tie_races,
            "includes_multiweek_rest": True,
        }
        self.assertGreater(correlations[-1], 0.55)
        self.assertGreater(correlations[-1], correlations[0])
        self.assertLess(abs(correlations[-1] - correlations[-2]), 0.1)
        self.assertTrue(np.all(np.isfinite(variance_ratios)))

    def test_03_truncation_invariance(self):
        raw = sample_raw()
        full = run_engine(prepare_input(raw), [BASE_CONFIG], record=True)
        cutoff = pd.Timestamp("2020-01-02")
        truncated_raw = raw[pd.to_datetime(raw["race_date"]) < cutoff].copy()
        truncated = run_engine(prepare_input(truncated_raw), [BASE_CONFIG], record=True)
        expected = sorted_snapshots(full)
        expected = expected[pd.to_datetime(expected["race_date"]) < cutoff].reset_index(drop=True)
        actual = sorted_snapshots(truncated).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)

    def test_04_future_result_modification(self):
        raw = sample_raw()
        baseline = run_engine(prepare_input(raw), [BASE_CONFIG], record=True)
        modified_raw = raw.copy()
        mask = modified_raw["race_id"].eq(1003)
        modified_raw.loc[mask, "finish_position"] = [3, 2, 1]
        modified = run_engine(prepare_input(modified_raw), [BASE_CONFIG], record=True)
        pre_columns = [column for column in sorted_snapshots(baseline).columns if column not in {"mu_post", "var_post", "finish_position"}]
        left = sorted_snapshots(baseline)
        right = sorted_snapshots(modified)
        left = left[left["race_id"] <= 1003].reset_index(drop=True)[pre_columns]
        right = right[right["race_id"] <= 1003].reset_index(drop=True)[pre_columns]
        pd.testing.assert_frame_equal(left, right, check_exact=True)

    def test_05_same_day_order_invariance(self):
        prepared = prepare_input(sample_raw())
        reversed_prepared = copy.deepcopy(prepared)
        reversed_prepared.races.reverse()
        baseline = run_engine(prepared, [BASE_CONFIG], record=True)
        reversed_result = run_engine(reversed_prepared, [BASE_CONFIG], record=True)
        pd.testing.assert_frame_equal(sorted_snapshots(baseline), sorted_snapshots(reversed_result), check_exact=True)
        pd.testing.assert_frame_equal(
            baseline.state.sort_values("horse_id").reset_index(drop=True),
            reversed_result.state.sort_values("horse_id").reset_index(drop=True),
            check_exact=True,
        )

    def test_06_shift_invariance(self):
        mu = np.array([0.2, -0.1, 0.4])
        var = np.array([1.0, 1.0, 1.0])
        orders = np.array([[0, 1]], dtype=np.int16)
        unranked = np.array([2], dtype=np.int16)
        finish = np.array([1, 2, 0])
        base = diagonal_update(mu, var, np.array([54, 54, 54]), orders, unranked, finish, BASE_CONFIG)
        shifted = diagonal_update(mu + 4.0, var, np.array([54, 54, 54]), orders, unranked, finish, BASE_CONFIG)
        np.testing.assert_allclose(
            base["mu_post"][:, None] - base["mu_post"][None, :],
            shifted["mu_post"][:, None] - shifted["mu_post"][None, :],
            atol=1e-12,
        )

    def test_07_ties_and_no_rank(self):
        raw = pd.DataFrame(race_rows(2001, "2020-01-01", [1, 2, 3], [1, 1, 2], [54, 54, 54]))
        prepared = prepare_input(raw)
        result = run_engine(prepared, [BASE_CONFIG], record=True)
        race = prepared.races[0]
        self.assertEqual(len(race.orders), 2)
        manual = diagonal_update(
            np.zeros(3), np.ones(3), np.array([54, 54, 54]), race.orders,
            race.unranked, race.finish, BASE_CONFIG,
        )
        actual = result.state.sort_values("horse_id")
        np.testing.assert_allclose(actual["mu"], manual["mu_post"], atol=1e-12)
        np.testing.assert_allclose(actual["var"], manual["var_post"], atol=1e-12)
        no_rank_raw = pd.DataFrame(race_rows(2002, "2020-01-02", [1, 2, 3], [1, 2, 0], [54, 54, 54], [0, 0, 4]))
        no_rank_result = run_engine(prepare_input(no_rank_raw), [BASE_CONFIG], record=True)
        third = no_rank_result.snapshots[no_rank_result.snapshots["horse_id"].eq(3)].iloc[0]
        self.assertEqual(third["mu_pre"], third["mu_post"])
        self.assertEqual(third["var_pre_persistent"], third["var_post"])
        third_state = no_rank_result.state[no_rank_result.state["horse_id"].eq(3)].iloc[0]
        self.assertEqual(third_state["n_graph_starts"], 1)
        self.assertEqual(third_state["last_start_date"], race_rows(1, "2020-01-02", [1], [1])[0]["race_date"] and 18263)
        renamed_raw = raw.copy()
        renamed_raw["horse_id"] = renamed_raw["horse_id"].map({1: 30, 2: 10, 3: 20})
        renamed_raw["horse_number"] = [3, 1, 2]
        renamed = run_engine(prepare_input(renamed_raw), [BASE_CONFIG], record=True)
        left = result.state.set_index("horse_id")[["mu", "var"]]
        right = renamed.state.set_index("horse_id")[["mu", "var"]]
        for original, renamed_id in {1: 30, 2: 10, 3: 20}.items():
            np.testing.assert_allclose(left.loc[original], right.loc[renamed_id], atol=1e-12)

    def test_08_non_jra_exclusion(self):
        rows = race_rows(3001, "2020-01-01", [1, 99, 2], [1, 2, 3], [54, 60, 55], affiliations=[1, 3, 2])
        with_non_jra = pd.DataFrame(rows)
        without_non_jra = pd.DataFrame(race_rows(3001, "2020-01-01", [1, 2], [1, 3], [54, 55], affiliations=[1, 2]))
        without_non_jra["n_registered"] = 3
        left = run_engine(prepare_input(with_non_jra), [BASE_CONFIG], record=True)
        right = run_engine(prepare_input(without_non_jra), [BASE_CONFIG], record=True)
        columns = ["race_id", "horse_id", "mu_pre", "var_pre_eff", "m_pre", "mu_post", "var_post", "finish_position"]
        pd.testing.assert_frame_equal(
            left.snapshots[columns].sort_values("horse_id").reset_index(drop=True),
            right.snapshots[columns].sort_values("horse_id").reset_index(drop=True),
            check_exact=True,
        )
        snapshots = left.snapshots.sort_values("horse_id")
        self.assertEqual(snapshots["n_field_retained"].iloc[0], 2)
        self.assertEqual(snapshots["n_removed_nonJRA"].iloc[0], 1)
        np.testing.assert_allclose(snapshots["finish_position"], [1, 2])
        config = P0Config(1.0, 0.004, 0.7, 0.03, "W_share")
        weighted = run_engine(prepare_input(with_non_jra), [config], record=True)
        np.testing.assert_allclose(weighted.snapshots.sort_values("horse_id")["m_pre"], [0.015, -0.015], atol=1e-12)

    def test_09_excluded_data_not_used(self):
        base = sample_raw()
        extras = []
        extras += race_rows(9001, "2020-01-03", [1, 2], [1, 2], organizer_code=2)
        extras += race_rows(9002, "2020-01-03", [1, 2], [1, 2], surface_code=2)
        extras += race_rows(9003, "2020-01-03", [1, 2], [1, 2], venue_code=101, venue_name="海外")
        extras += race_rows(9004, "2020-01-03", [1, 2], [1, 2], organizer_code=2, venue_name="帯広")
        combined = pd.concat([base, pd.DataFrame(extras)], ignore_index=True)
        left = run_engine(prepare_input(base), [BASE_CONFIG], record=True)
        right = run_engine(prepare_input(combined), [BASE_CONFIG], record=True)
        pd.testing.assert_frame_equal(sorted_snapshots(left), sorted_snapshots(right), check_exact=True)
        changed = combined.copy()
        changed.loc[changed["race_id"].isin([9001, 9002, 9003, 9004]), "finish_position"] = 99
        changed_result = run_engine(prepare_input(changed), [BASE_CONFIG], record=True)
        pd.testing.assert_frame_equal(sorted_snapshots(left), sorted_snapshots(changed_result), check_exact=True)

    def test_10_variance_and_initialization(self):
        raw = pd.DataFrame(race_rows(4001, "2020-01-01", [1, 2], [1, 2], [54, 54]))
        first = run_engine(prepare_input(raw), [BASE_CONFIG], record=True)
        self.assertTrue((first.snapshots["days_since_start"] == 0).all())
        self.assertTrue((first.snapshots["var_pre_eff"] == 1.0).all())
        rest_raw = pd.concat([raw, pd.DataFrame(race_rows(4002, "2021-01-01", [1, 2], [1, 2], [54, 54]))])
        rest = run_engine(prepare_input(rest_raw), [BASE_CONFIG], record=True)
        second = rest.snapshots[rest.snapshots["race_id"].eq(4002)]
        self.assertTrue((second["var_pre_persistent"] < second["var_pre_eff"]).all())
        self.assertTrue((second["var_pre_eff"] <= 1.0).all())
        self.assertTrue((second["days_since_start"] == 366).all())
        self.assertTrue((second["var_post"] <= second["var_pre_eff"]).all())
        from rating_core import effective_variance
        np.testing.assert_allclose(effective_variance(np.array([0.2]), 0.01, np.array([3000]), 0.7), [0.7])
        np.testing.assert_allclose(effective_variance(np.array([0.2]), 0.01, np.array([3000]), 1.3), [1.2])

    def test_11_lock(self):
        raw = pd.DataFrame(race_rows(5001, "2025-01-01", [1, 2], [1, 2]))
        with self.assertRaisesRegex(ValueError, "2025年以降"):
            prepare_input(raw)

    def test_12_result_version_and_race_date_order(self):
        raw = sample_raw()
        duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "重複"):
            prepare_input(duplicated)
        changed = raw.copy()
        changed["race_data_created"] = changed["race_data_created"].astype(str) + "_ignored"
        left = run_engine(prepare_input(raw), [BASE_CONFIG], record=True)
        right = run_engine(prepare_input(changed), [BASE_CONFIG], record=True)
        pd.testing.assert_frame_equal(sorted_snapshots(left), sorted_snapshots(right), check_exact=True)
        self.assertTrue((left.snapshots["race_date"].is_monotonic_increasing))
        before = sorted_snapshots(left)
        rerun = run_engine(prepare_input(changed), [BASE_CONFIG], record=True)
        pd.testing.assert_frame_equal(before, sorted_snapshots(rerun), check_exact=True)

    def test_13_harville_probabilities(self):
        scores = np.array([0.2, -0.1, 0.3, 0.05])
        probabilities = harville_inclusion_probabilities(scores, 3)
        top_three = probabilities.sum(axis=1)
        self.assertAlmostEqual(float(top_three.sum()), 3.0, places=12)
        brute = np.zeros(4)
        for order in itertools.permutations(range(4)):
            probability = 1.0
            remaining = list(order)
            for selected in order:
                denominator = sum(np.exp(scores[index]) for index in remaining)
                probability *= np.exp(scores[selected]) / denominator
                remaining.remove(selected)
            brute[list(order[:3])] += probability
        np.testing.assert_allclose(top_three, brute, atol=1e-12)


def tearDownModule():
    Path("stage2_test_diagnostics.json").write_text(json.dumps(DIAGNOSTICS, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
