from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from forward_ops import EXPECTED_P2B_HASH, OUTPUT_FILES, copy_database, load_config, load_events
from rating_core import WEIGHTS, file_sha256
from rating_data import prepare_input
from viewer_data import cache_paths, cleanup_access_lock_files, load_viewer_data, source_fingerprint

BASE = Path(__file__).resolve().parent
SCALE = 400 / np.log(10)
TOLERANCE = 5.0e-9


def mismatch_details(mask: pd.Series, limit: int = 5) -> dict:
    values = mask[mask].index.tolist()[:limit]
    return {"mismatch_rows": int(mask.sum()), "sample_indices": values}


def main() -> None:
    config = load_config(BASE / "forward_config.yaml")
    report = {"status": "PASS", "checks": {}, "metrics": {}, "issues": []}

    def check(name, condition, details=None):
        passed = bool(condition)
        report["checks"][name] = {"pass": passed, "details": details}
        if not passed:
            report["issues"].append({"check": name, "details": details})

    def close_enough(actual, expected, atol=TOLERANCE):
        return np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=atol, rtol=0)

    def absolute_error(actual, expected):
        return float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))

    out = BASE / "p2b_forward"
    snapshots = pd.read_csv(out / OUTPUT_FILES["snapshots"], parse_dates=["race_date"])
    posts = pd.read_csv(out / OUTPUT_FILES["posts"], parse_dates=["race_date"])
    events = load_events(out)
    state = pd.read_csv(out / OUTPUT_FILES["state"])
    anchor = pd.read_csv(out / OUTPUT_FILES["anchor"], parse_dates=["asof_date"])
    residual = pd.read_csv(out / OUTPUT_FILES["residual"], parse_dates=["race_date"])
    p0_snapshots = pd.read_csv(out / OUTPUT_FILES["p0_snapshots"], parse_dates=["race_date"])
    p0_state = pd.read_csv(out / OUTPUT_FILES["p0_state"])
    keys = ["race_id", "horse_id"]
    model = config["_model"]
    p1 = model["base_p1"]["p1"]
    selected = model["base_p1"]["base_p0"]["p0"]
    beta = model["p2"]["beta"]
    cap_by_debut = {1: p1["var0_d"]["新馬戦"], 2: p1["var0_d"]["未勝利戦"], 3: p1["var0_d"]["その他のクラス"], 4: 2.0}
    b_map = {1: p1["b"]["新馬戦"], 2: p1["b"]["未勝利戦"], 3: p1["b"]["その他のクラス"]}

    check("config_hash", file_sha256(BASE / "stage3_output/config_p2b.yaml") == EXPECTED_P2B_HASH)
    check("date_range", snapshots["race_date"].ge("2025-01-01").all(), {"min": str(snapshots["race_date"].min().date()), "max": str(snapshots["race_date"].max().date())})
    check("race_dates_sorted", snapshots["race_date"].is_monotonic_increasing)
    check("snapshot_post_keys", snapshots[keys].reset_index(drop=True).equals(posts[keys].reset_index(drop=True)))
    for name, frame in [("snapshot", snapshots), ("post", posts), ("p0_snapshot", p0_snapshots)]:
        check(f"{name}_unique_keys", not frame.duplicated(keys).any(), int(frame.duplicated(keys).sum()))
    check("event_unique_race_ids", events["race_id"].is_unique and events["event_id"].is_unique, len(events))
    check("event_snapshot_race_match", set(events["race_id"]) == set(snapshots["race_id"]), {"events": len(events), "snapshot_races": snapshots["race_id"].nunique()})
    check("p2b_prior_type", snapshots["prior_type"].eq("P2b").all())
    check("p0_separate_prior_type", p0_snapshots["prior_type"].eq("P0").all() and p0_state["prior_type"].eq("P0").all())
    check("p2b_and_p0_keys", snapshots[keys].reset_index(drop=True).equals(p0_snapshots[keys].reset_index(drop=True)))
    check("positive_variances", snapshots["var_pre_persistent"].gt(0).all() and snapshots["var_pre_eff"].gt(0).all() and snapshots["var_post"].gt(0).all())
    check("debut_types", snapshots["debut_type_code"].isin([1, 2, 3, 4]).all(), snapshots["debut_type_code"].value_counts().to_dict())
    caps = snapshots["debut_type_code"].map(cap_by_debut)
    check("persistent_variance_cap", (snapshots["var_pre_persistent"] <= caps + TOLERANCE).all(), mismatch_details(snapshots["var_pre_persistent"] > caps + TOLERANCE))
    check("effective_variance_not_reduced_by_layoff", (snapshots["var_pre_eff"] + TOLERANCE >= snapshots["var_pre_persistent"]).all())
    expected_eff = np.minimum(snapshots["var_pre_persistent"] + selected["q"] * snapshots["days_since_start"].clip(lower=0) / 30.0, caps)
    check("effective_variance_formula", close_enough(snapshots["var_pre_eff"], expected_eff), {"max_error": absolute_error(snapshots["var_pre_eff"], expected_eff)})
    check("effective_variance_cap", (snapshots["var_pre_eff"] <= caps + TOLERANCE).all(), mismatch_details(snapshots["var_pre_eff"] > caps + TOLERANCE))
    check("elo_pre_formula", close_enough(snapshots["elo_pre"], 1500 + SCALE * (snapshots["mu_pre"] - snapshots["mu_bar_D"])))
    check("post_values_match", close_enough(posts["mu_post"], snapshots["mu_post"]) and close_enough(posts["var_post"], snapshots["var_post"]))
    check("m_pre_formula", close_enough(snapshots["m_pre"], snapshots["mu_pre"] - selected["beta_kg"] * (snapshots["weight_kg"] - snapshots.groupby("race_id")["weight_kg"].transform("mean"))))
    check("finite_values", np.isfinite(snapshots[["mu_pre", "mu_post", "var_pre_persistent", "var_pre_eff", "m_pre", "elo_pre", "days_since_start", "n_graph_starts", "mu_bar_D"]].to_numpy(dtype=float)).all())
    check("no_unexplained_rank_zero", snapshots.loc[snapshots["finish_position"].eq(0), "abnormal_code"].isin([4, 5]).all())
    check("no_rank_abnormal_zero_conflicts", not snapshots["abnormal_code"].isin([4, 5]).mul(snapshots["finish_position"].gt(0)).any())
    check("no_same_horse_same_day", not snapshots.groupby(["horse_id", "race_date"])["race_id"].nunique().gt(1).any())
    check("at_least_two_candidates", snapshots.groupby("race_id").size().ge(2).all())
    check("snapshot_state_config_hash", snapshots["config_sha256"].eq(EXPECTED_P2B_HASH).all() and state["config_sha256"].eq(EXPECTED_P2B_HASH).all())
    check("event_config_hash", events["config_sha256"].eq(EXPECTED_P2B_HASH).all())
    check("anchor_unique_dates", anchor["asof_date"].is_unique and anchor["asof_date"].is_monotonic_increasing)
    check("anchor_snapshot_date_match", set(anchor["asof_date"]) == set(snapshots["race_date"]))
    check("anchor_values_match", close_enough(snapshots["mu_bar_D"], snapshots["race_date"].map(anchor.set_index("asof_date")["mu_bar_D"])))
    check("residual_first_three", residual["n_graph_starts"].max() <= 2 and residual["n_graph_starts"].min() >= 0)
    check("state_unique_horses", state["horse_id"].is_unique and p0_state["horse_id"].is_unique)

    with tempfile.TemporaryDirectory() as temporary:
        db_copy = copy_database(Path(config["database"]["path"]), Path(temporary), config["database"].get("linked_tables", {}))
        current_db_hash = file_sha256(db_copy)
        source = load_raw_from_db_for_audit(db_copy)
        cleanup_access_lock_files(temporary)
    source["race_date"] = pd.to_datetime(source["race_date"])
    check("source_unique_runner_keys", not source.duplicated(keys).any())
    source_races = set(source["race_id"])
    processed_races = set(events["race_id"])
    unprocessed = source[~source["race_id"].isin(processed_races)]
    check("processed_races_still_in_source", processed_races <= source_races, {"missing_races": len(processed_races - source_races)})
    processed_source = source[source["race_id"].isin(processed_races)]
    expected_counts = processed_source[processed_source["affiliation_code"].isin([1, 2]) & ~processed_source["abnormal_code"].isin([1, 2, 3])].groupby("race_id").size().reindex(sorted(processed_races), fill_value=0)
    actual_counts = snapshots.groupby("race_id").size().reindex(expected_counts.index, fill_value=0)
    check("retained_field_counts", expected_counts.equals(actual_counts), {"mismatch_races": int(expected_counts.ne(actual_counts).sum())})
    non_jra = processed_source[processed_source["affiliation_code"].isin([3, 4])].groupby("race_id").size().reindex(expected_counts.index, fill_value=0)
    actual_removed = snapshots.groupby("race_id")["n_removed_nonJRA"].first().reindex(expected_counts.index, fill_value=0)
    check("non_jra_removed_counts", non_jra.equals(actual_removed), {"mismatch_races": int(non_jra.ne(actual_removed).sum())})
    original = processed_source.groupby("race_id")["n_registered"].first().reindex(expected_counts.index)
    retained_original = snapshots.groupby("race_id")["n_field_original"].first().reindex(expected_counts.index)
    check("original_field_counts", original.equals(retained_original), {"mismatch_races": int(original.ne(retained_original).sum())})
    candidates = processed_source[processed_source["affiliation_code"].isin([1, 2]) & ~processed_source["abnormal_code"].isin([1, 2, 3])].copy()
    candidates["expected_finish"] = 0
    for _, indices in candidates.groupby("race_id").groups.items():
        positive = sorted(candidates.loc[indices, "finish_position"].loc[lambda value: value.gt(0)].unique())
        mapping = {rank: index + 1 for index, rank in enumerate(positive)}
        candidates.loc[indices, "expected_finish"] = candidates.loc[indices, "finish_position"].map(mapping).fillna(0).astype(int)
    expected_finish = candidates.set_index(keys)["expected_finish"]
    actual_finish = snapshots.set_index(keys)["finish_position"]
    check("relative_finish_ranks", expected_finish.reindex(actual_finish.index).equals(actual_finish), {"mismatch_rows": int(expected_finish.reindex(actual_finish.index).ne(actual_finish).sum())})
    check("processed_result_versions_unchanged", all(set(group["result_version"].astype(str)) == {events.set_index("race_id").loc[race_id, "result_version"]} for race_id, group in processed_source.groupby("race_id")))
    for column in ["race_date", "condition_code", "surface_code", "distance"]:
        actual = snapshots.set_index(keys)[column]
        expected = candidates.set_index(keys)[column].reindex(actual.index)
        if column == "race_date":
            expected = pd.to_datetime(expected)
        check(f"source_{column}_match", actual.equals(expected), mismatch_details(actual.ne(expected)))
    actual_numbers = snapshots.set_index(keys)["horse_number"]
    expected_numbers = candidates.set_index(keys)["horse_number"].reindex(actual_numbers.index)
    check("source_horse_numbers_match", actual_numbers.equals(expected_numbers))

    frozen_state = pd.read_csv(BASE / "stage3_output/p2b/state_2024-12-31.csv", usecols=["horse_id", "n_graph_starts", "last_start_date", "mu", "var", "var_cap", "mu0", "debut_type_code"])
    frozen_state["last_start_date"] = pd.to_datetime(frozen_state["last_start_date"])
    base_counts = frozen_state.set_index("horse_id")["n_graph_starts"]
    ordered = snapshots.sort_values(["horse_id", "race_date", "race_id"]).copy()
    ordered["expected_nstarts"] = ordered["horse_id"].map(base_counts).fillna(0) + ordered.groupby("horse_id").cumcount()
    check("graph_start_counts", ordered["n_graph_starts"].eq(ordered["expected_nstarts"]).all(), mismatch_details(ordered["n_graph_starts"].ne(ordered["expected_nstarts"])))
    previous_dates = ordered.groupby("horse_id")["race_date"].shift(1).fillna(ordered["horse_id"].map(frozen_state.set_index("horse_id")["last_start_date"]))
    expected_days = (ordered["race_date"] - previous_dates).dt.days.fillna(0).clip(lower=0)
    check("days_since_start", ordered["days_since_start"].eq(expected_days).all(), mismatch_details(ordered["days_since_start"].ne(expected_days)))
    later = ordered.groupby("horse_id").cumcount().gt(0)
    for column, previous in [("mu_pre", "mu_post"), ("var_pre_persistent", "var_post")]:
        expected = ordered.groupby("horse_id")[previous].shift(1)
        valid = later & expected.notna()
        check(f"persistent_{column}_transition", close_enough(ordered.loc[valid, column], expected[valid]), {"max_error": absolute_error(ordered.loc[valid, column], expected[valid])})
    check("same_day_pre_state_consistent", not ordered[ordered["race_date"].eq(ordered.groupby("horse_id")["race_date"].shift(1))].index.any())
    no_rank = snapshots["abnormal_code"].isin([4, 5]) | snapshots["finish_position"].eq(0)
    check("no_rank_mu_unchanged", close_enough(snapshots.loc[no_rank, "mu_pre"], snapshots.loc[no_rank, "mu_post"]), {"no_rank_rows": int(no_rank.sum())})
    check("no_rank_var_unchanged", close_enough(snapshots.loc[no_rank, "var_pre_persistent"], snapshots.loc[no_rank, "var_post"]))
    ranked = ~no_rank
    check("ranked_var_post_cap", (snapshots.loc[ranked, "var_post"] <= snapshots.loc[ranked, "var_pre_eff"] + TOLERANCE).all())
    check("skip_flag_consistent", snapshots["skip_mu_var_update"].eq(no_rank).all())

    new = snapshots[snapshots["n_graph_starts"].eq(0)].copy()
    check("new_debut_codes", new["debut_type_code"].isin([1, 2, 3]).all())
    expected_mu0 = new["mu_bar_D"] + new["debut_type_code"].map(b_map) + beta["beta_sw"] * new["X_sire_win"] + beta["beta_st"] * new["X_sire_top3"] + beta["beta_bw"] * new["X_bms_win"] + beta["beta_bt"] * new["X_bms_top3"]
    check("p2b_initial_mu0", close_enough(new["mu0"], expected_mu0), {"max_error": absolute_error(new["mu0"], expected_mu0)})
    check("p2b_initial_mu_applied", close_enough(new["mu_pre"], new["mu0"]))
    check("p2b_initial_var", close_enough(new["var_pre_eff"], new["debut_type_code"].map(cap_by_debut)))
    check("pedigree_status_not_all_missing", new["pedigree_status"].ne("ID_MISSING").mean() > 0.9, new["pedigree_status"].value_counts().to_dict())
    new_features = ["X_sire_win", "X_sire_top3", "X_bms_win", "X_bms_top3"]
    check("non_debut_pedigree_not_used", snapshots.loc[snapshots["debut_type_code"].eq(4), new_features].isna().all().all())
    check("new_debut_features_present", new[new_features].notna().all().all())
    expected_status = np.select([(new["sire_status_code"].eq(1) & new["bms_status_code"].eq(1)), (new["sire_status_code"].eq(3) | new["bms_status_code"].eq(3)), (new["sire_status_code"].eq(2) & new["bms_status_code"].eq(2))], ["OK", "ID_MISSING", "NO_HISTORY"], default="PARTIAL_HISTORY")
    check("pedigree_status_mapping", new["pedigree_status"].eq(expected_status).all(), new["pedigree_status"].value_counts().to_dict())

    latest_by_horse = ordered.groupby("horse_id").tail(1).set_index("horse_id")
    state_by_horse = state.set_index("horse_id")
    current_state = state_by_horse.reindex(latest_by_horse.index)
    check("final_state_matches_last_post", close_enough(latest_by_horse["mu_post"], current_state["mu"]) and close_enough(latest_by_horse["var_post"], current_state["var"]))
    check("final_state_nstarts", latest_by_horse["n_graph_starts"].add(1).eq(current_state["n_graph_starts"]).all())
    state_last_dates = pd.to_datetime(current_state["last_start_date"])
    check("final_state_last_start", latest_by_horse["race_date"].eq(state_last_dates).all())
    unchanged_state_ids = state_by_horse.index.difference(latest_by_horse.index)
    frozen_by_horse = frozen_state.set_index("horse_id")
    check("untouched_frozen_state", close_enough(state_by_horse.loc[unchanged_state_ids, "mu"], frozen_by_horse.loc[unchanged_state_ids, "mu"]) and close_enough(state_by_horse.loc[unchanged_state_ids, "var"], frozen_by_horse.loc[unchanged_state_ids, "var"]))
    check("state_var_caps", close_enough(state["var_cap"], state["debut_type_code"].map(cap_by_debut)))

    history = pd.concat([
        pd.read_csv(BASE / "stage3_output/p2b/snapshots.csv.gz", usecols=["race_id", "race_date", "horse_id", "mu_post"]),
        snapshots[["race_id", "race_date", "horse_id", "mu_post"]],
    ], ignore_index=True)
    history["race_date"] = pd.to_datetime(history["race_date"])
    history = history.sort_values(["horse_id", "race_date", "race_id"])
    prior_groups = {day: group for day, group in history.groupby("race_date")}
    anchor_errors = []
    new_errors = []
    for row in anchor.itertuples():
        day = row.asof_date
        active_ids = set()
        for offset in range(1, 366):
            previous_day = day - pd.Timedelta(days=offset)
            if previous_day in prior_groups:
                active_ids.update(prior_groups[previous_day]["horse_id"].tolist())
        latest = history[history["horse_id"].isin(active_ids) & history["race_date"].lt(day)].groupby("horse_id").tail(1)
        if len(latest) == 0:
            latest = history[history["race_date"].lt(day)].groupby("horse_id").tail(1)
        if len(latest) != row.n_active_365d:
            anchor_errors.append({"date": str(day.date()), "kind": "active_count", "expected": len(latest), "actual": row.n_active_365d})
        if len(latest):
            for column, expected in [("mu_bar_D", latest["mu_post"].mean()), ("active_latest_mu_std", latest["mu_post"].std(ddof=0)), ("active_latest_mu_median", latest["mu_post"].median())]:
                if not np.isclose(getattr(row, column), expected, atol=TOLERANCE, rtol=0):
                    anchor_errors.append({"date": str(day.date()), "kind": column, "expected": expected, "actual": getattr(row, column)})
        day_new = new[new["race_date"].eq(day)]
        if len(day_new) != row.n_new_horses:
            new_errors.append({"date": str(day.date()), "kind": "n_new", "expected": len(day_new), "actual": row.n_new_horses})
        if len(day_new) and not np.isclose(day_new["mu0"].mean(), row.mu0_mean, atol=TOLERANCE, rtol=0):
            new_errors.append({"date": str(day.date()), "kind": "mu0_mean", "expected": day_new["mu0"].mean(), "actual": row.mu0_mean})
    check("anchor_active_independent_recalculation", not anchor_errors, {"mismatches": len(anchor_errors), "sample": anchor_errors[:5]})
    check("anchor_new_horse_independent_recalculation", not new_errors, {"mismatches": len(new_errors), "sample": new_errors[:5]})
    check("anchor_threshold_groups", anchor["anchor_group"].eq(np.where(anchor["n_active_365d"].ge(100), "AT_OR_ABOVE_N_MIN", "BELOW_N_MIN")).all())
    delta = anchor["mu_bar_D"].diff()
    check("anchor_daily_delta", close_enough(delta.iloc[1:], anchor["mu_bar_delta_1d"].iloc[1:]))
    check("anchor_crossings_limited", anchor["threshold_crossing"].fillna("").ne("").sum() <= 1, int(anchor["threshold_crossing"].fillna("").ne("").sum()))

    numeric_errors = []
    max_mu_error = 0.0
    max_var_error = 0.0
    prepared = prepare_input(processed_source, source_hash=current_db_hash, allow_future=True)
    processed_by_id = {int(race.race_id): race for race in prepared.races}
    train_weights = np.asarray([WEIGHTS[selected["weight"]]])
    for race_id, group in snapshots.groupby("race_id", sort=False):
        group = group.sort_values("horse_number")
        race = processed_by_id[int(race_id)]
        scores = group["m_pre"].to_numpy(dtype=float)[None, :]
        gradient = np.zeros((1, len(group)), dtype=float)
        diagonal = np.zeros((1, len(group)), dtype=float)
        for order in race.orders:
            for stage in range(race.k_max):
                candidates_in_stage = np.concatenate((order[stage:], race.unranked))
                probabilities = np.exp(scores[0, candidates_in_stage] - np.max(scores[0, candidates_in_stage]))
                probabilities /= probabilities.sum()
                weight = train_weights[0, stage] / len(race.orders)
                gradient[0, candidates_in_stage] -= weight * probabilities
                gradient[0, order[stage]] += weight
                diagonal[0, candidates_in_stage] += weight * probabilities * (1.0 - probabilities)
        precision = 1.0 / group["var_pre_eff"].to_numpy(dtype=float) + selected["kappa"] * diagonal[0]
        mu_expected = group["mu_pre"].to_numpy(dtype=float).copy()
        var_expected = group["var_pre_persistent"].to_numpy(dtype=float).copy()
        ranked_mask = group["finish_position"].gt(0).to_numpy()
        mu_expected[ranked_mask] += selected["kappa"] * gradient[0, ranked_mask] / precision[ranked_mask]
        var_expected[ranked_mask] = 1.0 / precision[ranked_mask]
        mu_error = absolute_error(group["mu_post"], mu_expected)
        var_error = absolute_error(group["var_post"], var_expected)
        max_mu_error = max(max_mu_error, mu_error)
        max_var_error = max(max_var_error, var_error)
        if mu_error > TOLERANCE or var_error > TOLERANCE:
            numeric_errors.append({"race_id": int(race_id), "mu_error": mu_error, "var_error": var_error})
    check("independent_pl_update_recalculation", not numeric_errors, {"mismatch_races": len(numeric_errors), "sample": numeric_errors[:5], "max_mu_error": max_mu_error, "max_var_error": max_var_error})

    frozen_p0_last_index = pd.read_csv(BASE / "stage2_output/snapshots.csv.gz", usecols=["race_index"])["race_index"].max()
    residual_key = residual[["race_index", "horse_id"]].drop_duplicates().copy()
    residual_key["race_index"] -= int(frozen_p0_last_index) + 1
    expected_residual = p0_snapshots[p0_snapshots["debut_type_code"].eq(1) & p0_snapshots["n_graph_starts"].le(2)][["race_index", "horse_id"]].drop_duplicates()
    check("residual_key_coverage", set(map(tuple, residual_key.to_numpy())) == set(map(tuple, expected_residual.to_numpy())), {"actual": len(residual_key), "expected": len(expected_residual)})
    check("residual_unique_races_horses", not residual.duplicated(["race_index", "horse_id"]).any())
    check("residual_ids_nonnegative", residual[["sire_id", "bms_id"]].fillna(0).ge(0).all().all())

    from viewer_data import load_viewer_config
    viewer_config = load_viewer_config(BASE / "viewer_config.yaml")
    viewer, auxiliary, fingerprint = load_viewer_data(viewer_config)
    merged_path, auxiliary_path, fingerprint_path = cache_paths(viewer_config)
    check("viewer_cache_fingerprint_current", json.loads(fingerprint_path.read_text(encoding="utf-8")) == source_fingerprint(viewer_config))
    check("viewer_period", viewer["race_date"].min() >= pd.Timestamp("2017-01-01"), str(viewer["race_date"].min()))
    check("viewer_unique_keys", not viewer.duplicated(["race_id", "horse_id", "record_type"]).any())
    final_viewer = viewer[viewer["record_type"].eq("final")]
    future_viewer = final_viewer[final_viewer["race_date"].ge("2025-01-01")]
    check("viewer_elo_pre", close_enough(final_viewer["elo_pre"], 1500 + SCALE * (final_viewer["mu_pre"] - final_viewer["mu_bar_D"])))
    check("viewer_elo_post", close_enough(final_viewer["elo_post"], 1500 + SCALE * (final_viewer["mu_post"] - final_viewer["mu_bar_D"])))
    check("viewer_elo_delta", close_enough(final_viewer["elo_delta"], final_viewer["elo_post"] - final_viewer["elo_pre"]))
    check("viewer_elo_sd", close_enough(final_viewer["elo_sd"], SCALE * np.sqrt(final_viewer["var_pre_eff"])))
    check("viewer_p0_present", future_viewer["p0_mu_pre"].notna().all(), int(future_viewer["p0_mu_pre"].isna().sum()))
    check("viewer_prior_boost_formula", close_enough(future_viewer["elo_prior_boost"], SCALE * (future_viewer["mu_pre"] - future_viewer["p0_mu_pre"])))
    check("viewer_names", final_viewer["horse_name"].notna().all() and final_viewer["race_name"].notna().all())
    check("viewer_record_types", set(viewer["record_type"].unique()) <= {"final", "predict"}, set(viewer["record_type"].unique()))
    check("viewer_prior_boost_not_constant", future_viewer["elo_prior_boost"].nunique() > 100, int(future_viewer["elo_prior_boost"].nunique()))
    check("viewer_source_values", close_enough(future_viewer.sort_values(keys)["mu_pre"], snapshots.sort_values(keys)["mu_pre"]))
    check("viewer_auxiliary_ids", auxiliary[["sire_id", "bms_id"]].notna().any(axis=1).all())
    for side in ["sire", "bms"]:
        expected_ids = set(residual[f"{side}_id"].dropna().astype("int64")) - {0}
        actual_ids = set(auxiliary.loc[auxiliary["side"].eq(side), f"{side}_id"].dropna().astype("int64"))
        check(f"viewer_auxiliary_{side}_forward_coverage", expected_ids <= actual_ids, {"expected_forward_ids": len(expected_ids), "missing_ids": len(expected_ids - actual_ids)})
    check("viewer_auxiliary_complete_metrics", auxiliary[["offspring", "starts", "offspring_win_within3_rate", "residual_win_mean", "residual_top3_mean"]].notna().all().all())
    no_new_starts = auxiliary["new_horse_starts"].eq(0)
    undefined_rates_valid = all(
        auxiliary.loc[no_new_starts, metric].isna().all()
        and np.isfinite(auxiliary.loc[~no_new_starts, metric]).all()
        for metric in ["new_horse_win_rate", "new_horse_top3_rate"]
    )
    check("viewer_auxiliary_undefined_rates_explicit", undefined_rates_valid, {"pre_2017_debut_rows": int(no_new_starts.sum())})

    report["metrics"] = {
        "forward_rows": len(snapshots),
        "forward_races": snapshots["race_id"].nunique(),
        "first_race_date": str(snapshots["race_date"].min().date()),
        "last_race_date": str(snapshots["race_date"].max().date()),
        "new_starts": len(new),
        "new_horses": new["horse_id"].nunique(),
        "new_status": new["pedigree_status"].value_counts().to_dict(),
        "viewer_rows": len(viewer),
        "viewer_future_rows": len(future_viewer),
        "state_horses": len(state),
        "anchor_days": len(anchor),
        "residual_rows": len(residual),
        "current_db_sha256": current_db_hash,
        "unprocessed_final_races": int(unprocessed["race_id"].nunique()),
        "unprocessed_first_date": str(unprocessed["race_date"].min().date()) if not unprocessed.empty else None,
        "unprocessed_last_date": str(unprocessed["race_date"].max().date()) if not unprocessed.empty else None,
        "max_pl_mu_error": max_mu_error,
        "max_pl_var_error": max_var_error,
        "config_sha256": EXPECTED_P2B_HASH,
    }
    if report["issues"]:
        report["status"] = "FAIL"
    (BASE / "forward_comprehensive_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    gc.collect()


def load_raw_from_db_for_audit(db_path: Path) -> pd.DataFrame:
    from rating_data import load_raw_from_db
    return load_raw_from_db(db_path=db_path, start_date="2025-01-01", end_date=None, final_only=True)


if __name__ == "__main__":
    main()
