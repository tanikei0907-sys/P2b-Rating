from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
P7 = BASE / "p7_propagation"
for path in [BASE, P7]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forward_ops import resolve_path
from p6_age.age_curve import age_drift
from p7_forward_ops import EXPECTED_P7_HASH, OUTPUT_FILES, frozen_pending, load_config, update
from p7_propagation.verify import audit_events
from rating_core import WEIGHTS, W_EVAL, file_sha256, harville_inclusion_probabilities
from rating_data import make_orders
from rating_engine import pl_batch
from rating_pedigree import aggregate_side


def error(actual, expected):
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    if left.shape != right.shape or not np.array_equal(np.isnan(left), np.isnan(right)):
        return np.inf
    valid = ~np.isnan(left)
    return float(np.max(np.abs(left[valid] - right[valid]))) if valid.any() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-idempotency", action="store_true")
    args = parser.parse_args()
    config = load_config(BASE / "p7_forward/forward_config.yaml")
    out = resolve_path(config["output"]["directory"])
    snapshots = pd.read_csv(out / OUTPUT_FILES["snapshots"], parse_dates=["race_date"])
    posts = pd.read_csv(out / OUTPUT_FILES["posts"])
    state = pd.read_csv(out / OUTPUT_FILES["state"])
    events = pd.read_csv(out / OUTPUT_FILES["events"])
    propagation = pd.read_csv(out / OUTPUT_FILES["propagation"], parse_dates=["race_date", "latest_past_date"])
    interactions = pd.read_csv(out / OUTPUT_FILES["interactions"], parse_dates=["race_date"])
    anchor = pd.read_csv(out / OUTPUT_FILES["anchor"], parse_dates=["asof_date"])
    residual = pd.read_csv(out / OUTPUT_FILES["residual"], parse_dates=["race_date"])
    multipliers = pd.read_csv(out / OUTPUT_FILES["multipliers"])
    manifest = json.loads((out / "latest_run_manifest.json").read_text(encoding="utf-8"))
    frozen_state = pd.read_csv(BASE / "p7_propagation/state_2024-12-31.csv")
    frozen_history = pd.read_csv(BASE / "p7_propagation/pedigree_residual_history.csv.gz")
    frozen_interactions = pd.read_csv(BASE / "p7_propagation/interaction_history_latest.csv.gz")
    history = pd.concat([frozen_history, residual], ignore_index=True)
    report = {"status": "PASS", "checks": {}, "metrics": {}, "issues": []}

    def check(name, condition, details=None):
        passed = bool(condition)
        report["checks"][name] = {"pass": passed, "details": details}
        if not passed:
            report["issues"].append({"check": name, "details": details})

    check("config_hash", manifest["config_sha256"] == EXPECTED_P7_HASH == file_sha256(BASE / "p7_propagation/selected_config.json"))
    check("unique_snapshot_keys", not snapshots.duplicated(["race_id", "horse_id"]).any())
    check("snapshot_post_values", np.allclose(posts["mu_post"], snapshots["mu_post"], atol=0, rtol=0) and np.allclose(posts["var_post"], snapshots["var_post"], atol=0, rtol=0))
    check("unique_events", events["race_id"].is_unique and events["event_id"].is_unique and set(events["race_id"]) == set(snapshots["race_id"]))
    check("config_hashes", all(frame["config_sha256"].eq(EXPECTED_P7_HASH).all() for frame in [snapshots, posts, state, residual, events, propagation, interactions, anchor]))
    check("output_hashes", all(file_sha256(out / name) == value for name, value in manifest["output_hashes"].items()))
    finite_columns = [
        "mu_pre", "mu_base_pre", "surface_offset_pre", "delta_turf_pre", "delta_dirt_pre", "var_pre_eff",
        "var_base_pre_eff", "var_surface_pre_eff", "age_drift", "propagation_correction_pre", "mu_post", "var_post", "elo_pre",
    ]
    check("finite_decomposed_features", np.isfinite(snapshots[finite_columns].to_numpy(dtype=float)).all())
    check("elo_decomposition", error(snapshots["elo_pre"], snapshots["elo_base_pre"] + snapshots["elo_surface_adjustment"]) <= 1e-8 and error(snapshots["elo_pre"], 1500 + 400 / np.log(10) * (snapshots["mu_pre"] - snapshots["mu_bar_D"])) <= 1e-8)
    check("state_finite", np.isfinite(state[["mu_base", "var_base", "delta_turf", "delta_dirt", "var_turf", "var_dirt", "propagation_pending"]].to_numpy(dtype=float)).all())
    check("date_range", snapshots["race_date"].min() >= pd.Timestamp("2025-01-01"))
    check("interaction_L1", not interactions["source_horse_id"].duplicated().any())

    kappa = config["_p2b"]["base_p1"]["base_p0"]["p0"]["kappa"]
    multiplier_by_race = multipliers.set_index("race_id")
    max_mu = max_var = 0.0
    for race_id, group in snapshots.groupby("race_id", sort=False):
        group = group.sort_values("horse_number")
        finish = group["finish_position"].to_numpy(dtype=np.int16)
        orders = make_orders(finish)
        unranked = np.flatnonzero(finish == 0).astype(np.int16)
        k_max = min(5, int(np.count_nonzero(finish > 0)), len(group) - 1)
        row = multiplier_by_race.loc[race_id]
        stages = np.array([row[f"stage_multiplier_{index}"] for index in range(1, 6)], dtype=float)
        gradient, diagonal, _, _ = pl_batch(group["m_pre"].to_numpy(dtype=float)[None, :], orders, unranked, k_max, np.asarray([WEIGHTS["W_share"]]) * stages[None, :], W_EVAL, False)
        ranked = finish > 0
        base_eff = group["var_base_pre_eff"].to_numpy(dtype=float)
        surf_eff = group["var_surface_pre_eff"].to_numpy(dtype=float)
        total_eff = base_eff + surf_eff
        expected_base = group["mu_base_pre"].to_numpy(dtype=float).copy()
        expected_delta = group["surface_offset_pre"].to_numpy(dtype=float).copy()
        expected_base_var = group["var_base_post"].to_numpy(dtype=float).copy()
        expected_surf_var = group["var_surface_post"].to_numpy(dtype=float).copy()
        if ranked.any() and k_max > 0:
            precision = 1.0 / total_eff + kappa * diagonal[0]
            change = kappa * gradient[0] / precision
            base_share = base_eff / total_eff
            surf_share = surf_eff / total_eff
            expected_base[ranked] += change[ranked] * base_share[ranked]
            expected_delta[ranked] += change[ranked] * surf_share[ranked]
            total_post = 1.0 / precision[ranked]
            expected_base_var[ranked] = base_eff[ranked] - base_share[ranked] ** 2 * (total_eff[ranked] - total_post)
            expected_surf_var[ranked] = surf_eff[ranked] - surf_share[ranked] ** 2 * (total_eff[ranked] - total_post)
        max_mu = max(max_mu, error(group["mu_base_post"], expected_base), error(group["surface_offset_post"], expected_delta), error(group["mu_post"], expected_base + expected_delta))
        max_var = max(max_var, error(group["var_base_post"], expected_base_var), error(group["var_surface_post"], expected_surf_var), error(group["var_post"], expected_base_var + expected_surf_var))
    check("independent_p7_update", max_mu <= 1e-7 and max_var <= 1e-7, {"max_mu_error": max_mu, "max_var_error": max_var})
    no_rank = snapshots["finish_position"].eq(0)
    check("no_rank_state_unchanged", np.allclose(snapshots.loc[no_rank, "mu_pre"], snapshots.loc[no_rank, "mu_post"], atol=0, rtol=0) and np.allclose(snapshots.loc[no_rank, "var_pre_persistent"], snapshots.loc[no_rank, "var_post"], atol=0, rtol=0))

    ordered = snapshots.sort_values(["horse_id", "race_date", "race_id"]).copy()
    base_counts = frozen_state.set_index("horse_id")["n_graph_starts"]
    expected_starts = ordered["horse_id"].map(base_counts).fillna(0) + ordered.groupby("horse_id").cumcount()
    check("graph_start_counts", ordered["n_graph_starts"].eq(expected_starts).all())
    frozen_dates = pd.to_datetime(frozen_state.set_index("horse_id")["last_start_date"])
    previous_dates = ordered.groupby("horse_id")["race_date"].shift(1).fillna(ordered["horse_id"].map(frozen_dates))
    check("days_since_start", ordered["days_since_start"].eq((ordered["race_date"] - previous_dates).dt.days.fillna(0).clip(lower=0)).all())

    first = ordered.groupby("horse_id", as_index=False).head(1).copy()
    first_race_by_horse = first.set_index("horse_id")["race_id"]
    source_events = propagation[propagation["source_horse_id"].isin(first["horse_id"])].copy()
    source_events = source_events[source_events.apply(lambda row: first_race_by_horse.get(row["source_horse_id"]) == row["source_race_id"], axis=1)]
    candidates = first[first["horse_id"].isin(set(source_events["source_horse_id"])) & first["horse_id"].isin(set(frozen_state["horse_id"]))].head(10).copy()
    check("continuity_sample10_available", len(candidates) == 10, {"rows": len(candidates)})
    demographics = pd.read_csv(BASE / "p6_age/horse_demographics.csv", usecols=["horse_id", "birth_date"], parse_dates=["birth_date"]).set_index("horse_id")["birth_date"]
    pending0 = frozen_pending(config)
    frozen_by_id = frozen_state.set_index("horse_id")
    continuity_rows = []
    for row in candidates.itertuples():
        horse_id = int(row.horse_id)
        base = frozen_by_id.loc[horse_id]
        days = int((row.race_date - pd.Timestamp(base.last_start_date)).days)
        previous_age = (pd.Timestamp(base.last_start_date) - demographics.loc[horse_id]).days / 365.2425
        current_age = (row.race_date - demographics.loc[horse_id]).days / 365.2425
        drift = float(age_drift(previous_age, current_age, config["_p6"]))
        forward_propagation = float(propagation[propagation["target_horse_id"].eq(horse_id) & propagation["race_date"].lt(row.race_date)]["applied_delta_mu"].sum())
        prior_propagation = float(pending0.get(horse_id, 0.0)) + forward_propagation
        surface_offset = float(base.delta_turf if int(row.surface_code) == 0 else base.delta_dirt)
        surface_var = float(base.var_turf if int(row.surface_code) == 0 else base.var_dirt)
        expected_var = min(float(base.var_base) + config["_p2b"]["base_p1"]["base_p0"]["p0"]["q"] * days / 30.0, float(base.var_cap)) + min(surface_var + config["_p6"]["surface_parameters"]["q_surf"] * days / 30.0, config["_p6"]["surface_parameters"]["var_surf0"])
        event = source_events[source_events["source_horse_id"].eq(horse_id)].iloc[0]
        expected_past = frozen_interactions[frozen_interactions["source_horse_id"].eq(horse_id)].sort_values("race_date").iloc[-1]["race_id"]
        actual_past = str(event["past_race_ids"])
        continuity_rows.append({
            "horse_id": horse_id, "race_id": int(row.race_id), "race_date": row.race_date,
            "mu_pre": row.mu_pre, "expected_mu_pre": float(base.mu_base) + drift + forward_propagation + surface_offset,
            "var_pre_eff": row.var_pre_eff, "expected_var_pre_eff": expected_var,
            "n_graph_starts": int(row.n_graph_starts), "expected_n_graph_starts": int(base.n_graph_starts),
            "delta_turf_pre": row.delta_turf_pre, "expected_delta_turf": float(base.delta_turf),
            "delta_dirt_pre": row.delta_dirt_pre, "expected_delta_dirt": float(base.delta_dirt),
            "propagation_correction_pre": row.propagation_correction_pre, "expected_propagation_correction": prior_propagation,
            "past_race_ids": actual_past, "expected_past_race_ids": str(int(expected_past)),
        })
    continuity = pd.DataFrame(continuity_rows)
    if not continuity.empty:
        continuity.to_csv(out / "continuity_sample10.csv", index=False, float_format="%.17g", date_format="%Y-%m-%d")
        pairs = [("mu_pre", "expected_mu_pre"), ("var_pre_eff", "expected_var_pre_eff"), ("delta_turf_pre", "expected_delta_turf"), ("delta_dirt_pre", "expected_delta_dirt"), ("propagation_correction_pre", "expected_propagation_correction")]
        max_continuity = max(error(continuity[actual], continuity[expected]) for actual, expected in pairs)
        check("continuity_sample10_values", max_continuity <= 1e-8 and continuity["n_graph_starts"].eq(continuity["expected_n_graph_starts"]).all() and continuity["past_race_ids"].eq(continuity["expected_past_race_ids"]).all(), {"horses": continuity["horse_id"].tolist(), "max_error": max_continuity})

    full_snapshots = pd.concat([
        pd.read_csv(BASE / "p7_propagation/snapshots.csv.gz", usecols=["race_id", "horse_id", "race_date", "mu_pre", "mu_post", "finish_position"], parse_dates=["race_date"]),
        snapshots[["race_id", "horse_id", "race_date", "mu_pre", "mu_post", "finish_position"]],
    ], ignore_index=True)
    selected = json.loads((BASE / "p7_propagation/selected_config.json").read_text(encoding="utf-8"))
    propagation_stats, later_corrections = audit_events(out / OUTPUT_FILES["propagation"], full_snapshots, selected)
    check("forward_propagation_events", propagation_stats["duplicate_keys"] == 0 and propagation_stats["self_events"] == 0 and propagation_stats["same_day_target_starts"] == 0 and propagation_stats["inactive_targets"] == 0 and propagation_stats["future_or_same_day_history"] == 0 and propagation_stats["mean_relative_mismatches"] == 0 and propagation_stats["max_applied_delta_error"] <= 1e-8, propagation_stats)

    candidates_pedigree = snapshots[["race_index", "race_date", "horse_id", "debut_type_code", "n_graph_starts", "condition_code"]].copy()
    candidates_pedigree["date"] = (candidates_pedigree["race_date"].astype("int64") // 86_400_000_000_000).astype(np.int32)
    pedigree = pd.read_csv(BASE / "p3_grade/cache/pedigree.csv")
    candidates_pedigree = candidates_pedigree.merge(pedigree, on="horse_id", how="left", validate="many_to_one")
    p2b = config["_p2b"]
    feature_mismatches = 0
    for side, features in [("sire", ["X_sire_win", "X_sire_top3"]), ("bms", ["X_bms_win", "X_bms_top3"])]:
        N, sum_win, sum_top3, status = aggregate_side(candidates_pedigree, history, side)
        check(f"{side}_asof", np.array_equal(N, snapshots[f"N_{side}"].to_numpy()) and np.array_equal(status, snapshots[f"{side}_status_code"].to_numpy()))
        for metric_index, (feature, sums) in enumerate(zip(features, [sum_win, sum_top3])):
            raw = sums / (N + 20.0)
            constant = p2b["p2"]["standardization"][(0 if side == "sire" else 2) + metric_index]
            expected = np.where(status == 1, (raw - constant["center"]) / constant["scale"], 0.0)
            expected[candidates_pedigree["debut_type_code"].to_numpy() == 4] = np.nan
            actual = snapshots[feature].to_numpy(dtype=float)
            valid = np.isfinite(expected)
            feature_mismatches += int(np.count_nonzero(~np.isclose(actual[valid], expected[valid], atol=1e-8, rtol=0)))
            feature_mismatches += int(np.count_nonzero(np.isfinite(actual) != valid))
    check("pedigree_features_asof", feature_mismatches == 0, {"mismatches": feature_mismatches})
    max_residual = 0.0
    by_index = {index: frame for index, frame in snapshots.groupby("race_index")}
    for race_index, group in residual.groupby("race_index"):
        frame = by_index[race_index]
        eligible = frame[frame["debut_type_code"].eq(1) & frame["n_graph_starts"].le(2)].sort_values("horse_number")
        probabilities = harville_inclusion_probabilities(eligible["m_pre"].to_numpy(dtype=float), 3)
        ordered_history = eligible[["horse_id", "finish_position"]].merge(group.drop(columns=["finish_position"]), on="horse_id", validate="one_to_one")
        max_residual = max(max_residual, error(ordered_history["e_win"], probabilities[:, 0]), error(ordered_history["e_top3"], probabilities.sum(axis=1)))
    check("residuals_from_forward_p7_scores", max_residual <= 1e-8, {"max_error": max_residual})

    latest = ordered.groupby("horse_id").tail(1).set_index("horse_id")
    current = state.set_index("horse_id")
    expected_state_mu = frozen_by_id["mu_base"].reindex(current.index).copy()
    expected_state_mu.loc[latest.index] = latest["mu_base_post"]
    for horse_id, correction in later_corrections.items():
        expected_state_mu.loc[horse_id] += correction
    check("final_state_base", error(current.loc[expected_state_mu.index, "mu_base"], expected_state_mu) <= 1e-8)
    check("final_state_starts", latest["n_graph_starts"].add(1).eq(current.loc[latest.index, "n_graph_starts"]).all())
    expected_pending = pd.Series(later_corrections, dtype=float)
    no_forward = current.index.difference(latest.index)
    if len(no_forward):
        expected_pending = pd.concat([expected_pending, no_forward.to_series().map(pending0).fillna(0.0).groupby(level=0).sum()])
    expected_pending = expected_pending.groupby(level=0).sum()
    check("final_pending_propagation", error(current.loc[expected_pending.index, "propagation_pending"], expected_pending) <= 1e-8)

    subprocess.run([sys.executable, str(out / "protect_assets.py"), "--check"], check=True, capture_output=True)
    protected = json.loads((out / "protected_assets_verification.json").read_text(encoding="utf-8"))
    check("protected_assets_unchanged", protected["status"] == "PASS", protected)
    if args.check_idempotency:
        before = {name: file_sha256(out / name) for name in list(OUTPUT_FILES.values()) + ["latest_run_manifest.json"]}
        second = update(config, None)
        after = {name: file_sha256(out / name) for name in before}
        check("idempotency", second["status"] == "NO_NEW_RACES" and before == after, {"status": second["status"], "files_unchanged": before == after})

    report["metrics"] = {
        "first_race_date": str(snapshots["race_date"].min().date()), "last_race_date": str(snapshots["race_date"].max().date()),
        "races": int(snapshots["race_id"].nunique()), "starts": len(snapshots), "state_horses": len(state),
        "propagation_events": len(propagation), "interaction_rows": len(interactions), "residual_rows": len(residual),
        "continuity_samples": len(continuity), "max_update_mu_error": max_mu, "max_update_var_error": max_var,
        "max_residual_error": max_residual, "config_sha256": EXPECTED_P7_HASH,
    }
    if report["issues"]:
        report["status"] = "FAIL"
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
