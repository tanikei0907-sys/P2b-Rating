from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
P3 = BASE / "p3_grade"
for path in [BASE, P3]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from p3_grade.prepare_inputs import load_cached_inputs
from p4_margin.margins import multiplier_arrays, sequential_margins
from rating_core import WEIGHTS, W_EVAL, file_sha256, harville_inclusion_probabilities
from rating_engine import pl_batch
from rating_pedigree import aggregate_side

OUT = Path(__file__).resolve().parent


def main() -> None:
    report = {"status": "PASS", "checks": {}, "metrics": {}, "issues": []}

    def check(name, condition, details=None):
        passed = bool(condition)
        report["checks"][name] = {"pass": passed, "details": details}
        if not passed:
            report["issues"].append({"check": name, "details": details})

    def error(actual, expected):
        return float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))

    selected = json.loads((OUT / "selected_config.json").read_text(encoding="utf-8"))
    config_hash = file_sha256(OUT / "selected_config.json")
    check("selected_config_hash", config_hash == (OUT / "selected_config.json.sha256").read_text(encoding="ascii").strip())
    check("p2b_config_hash", file_sha256(BASE / "stage3_output/config_p2b.yaml") == "3577cbc7a6a2c3e58bfdeda6666b9aa9d7369866f4c6fa49902dac6b8779dbe9")
    check("time_audit_hash", file_sha256(OUT / "time_audit.json") == selected["time_audit_sha256"])
    prepared, pedigree, races, _ = load_cached_inputs()
    snapshots = pd.read_csv(OUT / "snapshots.csv.gz", parse_dates=["race_date"])
    history = pd.read_csv(OUT / "pedigree_residual_history.csv.gz", parse_dates=["race_date"])
    metrics = pd.read_csv(OUT / "race_metrics.csv.gz")
    state = pd.read_csv(OUT / "state_2024-12-31.csv")
    margins = pd.read_csv(OUT / "race_margins.csv", parse_dates=["race_date"])
    relative = pd.read_csv(OUT / "relative_times.csv.gz")
    multipliers = pd.read_csv(OUT / "margin_multipliers.csv.gz")
    check("snapshot_shape", len(snapshots) == 610676 and snapshots["race_id"].nunique() == 43252)
    check("snapshot_unique_keys", not snapshots.duplicated(["race_id", "horse_id"]).any())
    check("snapshot_config_hash", snapshots["config_sha256"].eq(config_hash).all())
    check("history_config_hash", history["config_sha256"].eq(config_hash).all())
    check("state_config_hash", state["config_sha256"].eq(config_hash).all())
    check("finite_state", np.isfinite(snapshots[["mu_pre", "mu_post", "var_pre_persistent", "var_pre_eff", "var_post", "m_pre"]].to_numpy(dtype=float)).all())
    check("positive_variance", snapshots[["var_pre_persistent", "var_pre_eff", "var_post"]].gt(0).all().all())
    check("ranked_time_complete", relative.loc[relative["relative_finish"].gt(0), "time_seconds"].notna().all())
    valid_raw = relative["time_raw"].gt(0)
    check("time_unit_tenths", np.allclose(relative.loc[valid_raw, "time_seconds"], relative.loc[valid_raw, "time_raw"] / 10.0, atol=1e-10, rtol=0))
    relative_sorted = relative.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    snapshot_sorted = snapshots.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    check("relative_time_keys", relative_sorted[["race_id", "horse_id"]].equals(snapshot_sorted[["race_id", "horse_id"]]))
    check("relative_finish_match", relative_sorted["relative_finish"].equals(snapshot_sorted["finish_position"]))

    max_delta_error = max_cumulative_error = 0.0
    status_mismatches = cumulative_mismatches = 0
    margin_indexed = margins.set_index("race_id")
    relative_by_race = {race_id: group.sort_values("horse_number") for race_id, group in relative.groupby("race_id", sort=False)}
    for race in prepared.races:
        group = relative_by_race[race.race_id]
        times = group["time_seconds"].to_numpy(dtype=float)
        expected_delta, expected_status = sequential_margins(race.finish, times, race.k_max)
        actual = margin_indexed.loc[race.race_id]
        actual_delta = np.array([actual[f"stage_{index}"] for index in range(1, 6)], dtype=float)
        actual_status = np.array([actual[f"status_{index}"] for index in range(1, 6)], dtype=np.int8)
        valid = np.isfinite(expected_delta)
        if valid.any():
            max_delta_error = max(max_delta_error, error(actual_delta[valid], expected_delta[valid]))
        status_mismatches += int(np.count_nonzero(actual_status != expected_status))
        winner_times = times[race.finish == 1]
        winner_time = float(np.nanmin(winner_times)) if np.isfinite(winner_times).any() else np.nan
        expected_cumulative = times - winner_time
        actual_cumulative = group["cumulative_relative_seconds"].to_numpy(dtype=float)
        cumulative_valid = (race.finish > 0) & np.isfinite(expected_cumulative)
        if cumulative_valid.any():
            max_cumulative_error = max(max_cumulative_error, error(actual_cumulative[cumulative_valid], expected_cumulative[cumulative_valid]))
            cumulative_mismatches += int(np.count_nonzero(~np.isclose(actual_cumulative[cumulative_valid], expected_cumulative[cumulative_valid], atol=1e-10, rtol=0)))
        cumulative_mismatches += int(np.count_nonzero(np.isfinite(actual_cumulative[race.finish > 0]) != np.isfinite(expected_cumulative[race.finish > 0])))
    check("stage_margins_recalculated", max_delta_error <= 1e-10 and status_mismatches == 0, {"max_error": max_delta_error, "status_mismatches": status_mismatches})
    check("cumulative_margins_recalculated", cumulative_mismatches == 0, {"mismatch_rows": cumulative_mismatches, "max_error": max_cumulative_error})
    available_deltas = margins[[f"stage_{index}" for index in range(1, 6)]].to_numpy(dtype=float)
    check("nonnegative_stage_margins", np.nanmin(available_deltas) >= 0)

    config = {
        "method": selected["method"], "normalize": selected["normalize"], "gamma": selected["gamma"],
        "m_max": selected["m_max"], "grid_stage": "frozen",
    }
    if selected["method"] == "B":
        config["m_min"] = selected["m_min"]
    stages, scales = multiplier_arrays(margins, prepared, config)
    saved_stages = multipliers[[f"stage_multiplier_{index}" for index in range(1, 6)]].to_numpy(dtype=float)
    check("selected_multipliers_match", np.allclose(saved_stages, stages, atol=1e-12, rtol=0) and np.allclose(multipliers["race_scale"], scales, atol=1e-12, rtol=0))
    multiplier_indexed = multipliers.set_index("race_id")
    by_race = {race.race_id: race for race in prepared.races}
    metric_nll = metrics.set_index("race_id")["p0_nll"]
    max_mu = max_var = max_nll = 0.0
    mismatches = []
    for race_id, group in snapshots.groupby("race_id", sort=False):
        group = group.sort_values("horse_number")
        race = by_race[race_id]
        scores = group["m_pre"].to_numpy(dtype=float)[None, :]
        row = multiplier_indexed.loc[race_id]
        stage_values = np.array([row[f"stage_multiplier_{index}"] for index in range(1, 6)], dtype=float)
        train_weights = np.asarray([WEIGHTS["W_share"]])
        if selected["method"] == "A":
            train_weights = train_weights * stage_values[None, :]
        gradient, diagonal, eval_ll, _ = pl_batch(scores, race.orders, race.unranked, race.k_max, train_weights, W_EVAL, True)
        if selected["method"] == "B":
            gradient *= float(row["race_scale"])
            diagonal *= float(row["race_scale"])
        precision = 1.0 / group["var_pre_eff"].to_numpy(dtype=float) + diagonal[0]
        mu_expected = group["mu_pre"].to_numpy(dtype=float).copy()
        var_expected = group["var_pre_persistent"].to_numpy(dtype=float).copy()
        ranked = group["finish_position"].gt(0).to_numpy()
        mu_expected[ranked] += gradient[0, ranked] / precision[ranked]
        var_expected[ranked] = 1.0 / precision[ranked]
        mu_error = error(group["mu_post"], mu_expected)
        var_error = error(group["var_post"], var_expected)
        nll_error = abs(float(-eval_ll[0] - metric_nll.loc[race_id]))
        max_mu, max_var, max_nll = max(max_mu, mu_error), max(max_var, var_error), max(max_nll, nll_error)
        if mu_error > 1e-7 or var_error > 1e-7 or nll_error > 1e-7:
            mismatches.append(race_id)
    check("independent_margin_update", not mismatches, {"mismatch_races": len(mismatches), "max_mu_error": max_mu, "max_var_error": max_var})
    check("standard_pl_nll_unmodified", max_nll <= 1e-7, {"max_error": max_nll})

    p2b = json.loads((BASE / "stage3_output/config_p2b.yaml").read_text(encoding="utf-8"))
    p1, beta = p2b["base_p1"]["p1"], p2b["p2"]["beta"]
    new = snapshots[snapshots["n_graph_starts"].eq(0) & snapshots["debut_type_code"].isin([1, 2, 3])].copy()
    b_map = {1: p1["b"]["新馬戦"], 2: p1["b"]["未勝利戦"], 3: p1["b"]["その他のクラス"]}
    expected_mu0 = (
        new["mu_bar_D"] + new["debut_type_code"].map(b_map)
        + beta["beta_sw"] * new["X_sire_win"] + beta["beta_st"] * new["X_sire_top3"]
        + beta["beta_bw"] * new["X_bms_win"] + beta["beta_bt"] * new["X_bms_top3"]
    )
    check("p4_initial_prior_formula", np.allclose(new["mu0"], expected_mu0, atol=1e-8, rtol=0), {"max_error": error(new["mu0"], expected_mu0)})
    candidates = snapshots[["race_index", "race_date", "horse_id", "debut_type_code", "n_graph_starts", "condition_code"]].copy()
    candidates["date"] = (candidates["race_date"].astype("int64") // 86_400_000_000_000).astype(np.int32)
    candidates = candidates.merge(pedigree, on="horse_id", how="left", validate="many_to_one")
    max_feature_error = 0.0
    feature_mismatches = 0
    for side, feature_names in [("sire", ["X_sire_win", "X_sire_top3"]), ("bms", ["X_bms_win", "X_bms_top3"])]:
        N, sum_win, sum_top3, status = aggregate_side(candidates, history, side)
        check(f"{side}_asof_count", np.array_equal(N, snapshots[f"N_{side}"].to_numpy()))
        check(f"{side}_asof_status", np.array_equal(status, snapshots[f"{side}_status_code"].to_numpy()))
        for metric_index, (feature, sums) in enumerate(zip(feature_names, [sum_win, sum_top3])):
            raw = sums / (N + 20.0)
            constant = p2b["p2"]["standardization"][(0 if side == "sire" else 2) + metric_index]
            expected = np.where(status == 1, (raw - constant["center"]) / constant["scale"], 0.0)
            expected[candidates["debut_type_code"].to_numpy() == 4] = np.nan
            actual = snapshots[feature].to_numpy(dtype=float)
            valid = np.isfinite(expected)
            max_feature_error = max(max_feature_error, error(actual[valid], expected[valid]))
            feature_mismatches += int(np.count_nonzero(~np.isclose(actual[valid], expected[valid], atol=1e-8, rtol=0)))
            feature_mismatches += int(np.count_nonzero(np.isfinite(actual) != valid))
    check("pedigree_features_asof_recalculation", feature_mismatches == 0, {"mismatch_values": feature_mismatches, "max_error": max_feature_error})

    max_residual_error = 0.0
    snapshot_by_index = {index: frame for index, frame in snapshots.groupby("race_index", sort=False)}
    for race_index, group in history.groupby("race_index"):
        race_snapshot = snapshot_by_index[race_index]
        eligible = race_snapshot[
            race_snapshot["debut_type_code"].eq(1) & race_snapshot["n_graph_starts"].le(2)
        ].sort_values("horse_number")
        probabilities = harville_inclusion_probabilities(eligible["m_pre"].to_numpy(dtype=float), 3)
        ordered = eligible[["horse_id", "finish_position"]].merge(group.drop(columns=["finish_position"]), on="horse_id", how="left", validate="one_to_one")
        max_residual_error = max(
            max_residual_error, error(ordered["e_win"], probabilities[:, 0]),
            error(ordered["e_top3"], probabilities.sum(axis=1)),
            error(ordered["actual_win"], ordered["finish_position"].eq(1).astype(float)),
            error(ordered["actual_top3"], ordered["finish_position"].between(1, 3).astype(float)),
        )
    check("residuals_recalculated_from_p4_scores", max_residual_error <= 1e-8, {"max_error": max_residual_error})

    ordered = snapshots.sort_values(["horse_id", "race_date", "race_id"]).copy()
    check("graph_start_counts", ordered["n_graph_starts"].eq(ordered.groupby("horse_id").cumcount()).all())
    previous_dates = ordered.groupby("horse_id")["race_date"].shift(1)
    check("days_since_start", ordered["days_since_start"].eq((ordered["race_date"] - previous_dates).dt.days.fillna(0).clip(lower=0)).all())
    later = ordered.groupby("horse_id").cumcount().gt(0)
    previous_mu = ordered.groupby("horse_id")["mu_post"].shift(1)
    previous_var = ordered.groupby("horse_id")["var_post"].shift(1)
    check("persistent_mu_transition", np.allclose(ordered.loc[later, "mu_pre"], previous_mu[later], atol=1e-8, rtol=0))
    check("persistent_var_transition", np.allclose(ordered.loc[later, "var_pre_persistent"], previous_var[later], atol=1e-8, rtol=0))
    latest = ordered.groupby("horse_id").tail(1).set_index("horse_id")
    current_state = state.set_index("horse_id").reindex(latest.index)
    check("final_state_mu", np.allclose(latest["mu_post"], current_state["mu"], atol=1e-8, rtol=0))
    check("final_state_var", np.allclose(latest["var_post"], current_state["var"], atol=1e-8, rtol=0))
    check("final_state_starts", latest["n_graph_starts"].add(1).eq(current_state["n_graph_starts"]).all())
    check("final_state_last_start", latest["race_date"].eq(pd.to_datetime(current_state["last_start_date"])).all())
    tuning = metrics[metrics["year"].le(2020) & metrics["F"]]["p0_nll"].mean()
    check("frozen_tuning_nll_reproduced", np.isclose(tuning, selected["tuning_nll"], atol=1e-12, rtol=0), {"expected": selected["tuning_nll"], "actual": tuning})
    subprocess.run([sys.executable, str(OUT / "protect_assets.py"), "--check"], check=True, capture_output=True)
    protected = json.loads((OUT / "protected_assets_verification.json").read_text(encoding="utf-8"))
    check("protected_assets_unchanged", protected["status"] == "PASS", protected)
    report["metrics"] = {
        "max_margin_error": max_delta_error, "max_update_mu_error": max_mu, "max_update_var_error": max_var,
        "max_nll_error": max_nll, "max_pedigree_feature_error": max_feature_error,
        "max_residual_error": max_residual_error, "tests_passed": 7,
    }
    if report["issues"]:
        report["status"] = "FAIL"
    (OUT / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
