from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from forward_ops import copy_database, resolve_path
from p4_forward_ops import EXPECTED_P4_HASH, OUTPUT_FILES, load_config, update
from p4_margin.margins import multiplier_arrays
from rating_core import WEIGHTS, W_EVAL, file_sha256, harville_inclusion_probabilities
from rating_data import make_orders
from rating_engine import pl_batch
from rating_pedigree import aggregate_side, load_pedigree_ids
from viewer_data import cleanup_access_lock_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-idempotency", action="store_true")
    args = parser.parse_args()
    config = load_config(BASE / "p4_margin/forward_config.yaml")
    out = resolve_path(config["output"]["directory"])
    selected = config["_p4"]
    snapshots = pd.read_csv(out / OUTPUT_FILES["snapshots"], parse_dates=["race_date"])
    posts = pd.read_csv(out / OUTPUT_FILES["posts"])
    state = pd.read_csv(out / OUTPUT_FILES["state"])
    events = pd.read_csv(out / OUTPUT_FILES["events"])
    anchor = pd.read_csv(out / OUTPUT_FILES["anchor"], parse_dates=["asof_date"])
    residual = pd.read_csv(out / OUTPUT_FILES["residual"], parse_dates=["race_date"])
    margins = pd.read_csv(out / OUTPUT_FILES["margins"])
    multipliers = pd.read_csv(out / OUTPUT_FILES["multipliers"])
    manifest = json.loads((out / "latest_run_manifest.json").read_text(encoding="utf-8"))
    frozen_state = pd.read_csv(BASE / "p4_margin/state_2024-12-31.csv")
    frozen_history = pd.read_csv(BASE / "p4_margin/pedigree_residual_history.csv.gz")
    history = pd.concat([frozen_history, residual], ignore_index=True)
    report = {"status": "PASS", "checks": {}, "metrics": {}, "issues": []}

    def check(name, condition, details=None):
        passed = bool(condition)
        report["checks"][name] = {"pass": passed, "details": details}
        if not passed:
            report["issues"].append({"check": name, "details": details})

    def error(actual, expected):
        return float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))

    check("config_hash", manifest["config_sha256"] == EXPECTED_P4_HASH == file_sha256(BASE / "p4_margin/selected_config.json"))
    check("unique_snapshot_keys", not snapshots.duplicated(["race_id", "horse_id"]).any())
    check("unique_post_keys", not posts.duplicated(["race_id", "horse_id"]).any())
    check("snapshot_post_values", np.allclose(posts["mu_post"], snapshots["mu_post"], atol=0, rtol=0) and np.allclose(posts["var_post"], snapshots["var_post"], atol=0, rtol=0))
    check("unique_events", events["race_id"].is_unique and events["event_id"].is_unique)
    check("event_snapshot_races", set(events["race_id"]) == set(snapshots["race_id"]))
    check("event_count", len(events) == snapshots["race_id"].nunique())
    check("config_hashes", all(frame["config_sha256"].eq(EXPECTED_P4_HASH).all() for frame in [snapshots, posts, state, residual, events, margins, multipliers]))
    check("output_hashes", all(file_sha256(out / name) == value for name, value in manifest["output_hashes"].items()))
    check("source_hash_consistency", snapshots["source_db_sha256"].eq(manifest["input_db_sha256"]).all() and events["input_db_sha256"].eq(manifest["input_db_sha256"]).all())
    check("finite_values", np.isfinite(snapshots[["mu_pre", "mu_post", "var_pre_persistent", "var_pre_eff", "var_post", "elo_pre", "winner_margin_seconds"]].to_numpy(dtype=float)).all())
    check("elo_formula", np.allclose(snapshots["elo_pre"], 1500 + 400 / np.log(10) * (snapshots["mu_pre"] - snapshots["mu_bar_D"]), atol=1e-10, rtol=0))
    check("date_range", snapshots["race_date"].min() >= pd.Timestamp("2025-01-01"))
    check("residual_first_three", residual["n_graph_starts"].between(0, 2).all() and not residual.duplicated(["race_id", "horse_id"]).any())

    class PreparedMargins:
        def __init__(self, race_ids):
            self.races = [type("RaceRef", (), {"race_id": int(race_id)}) for race_id in race_ids]
    stages, scales = multiplier_arrays(margins, PreparedMargins(margins["race_id"].tolist()), selected)
    ordered_multipliers = multipliers.set_index("race_id").loc[margins["race_id"]].reset_index()
    saved_stages = ordered_multipliers[[f"stage_multiplier_{index}" for index in range(1, 6)]].to_numpy(dtype=float)
    check("margin_multiplier_formula", np.allclose(saved_stages, stages, atol=1e-12, rtol=0) and np.allclose(ordered_multipliers["race_scale"], scales, atol=1e-12, rtol=0))
    check("raw_margin_scale", snapshots["winner_margin_normalized"].mul(snapshots["distance"] / 1000.0).sub(snapshots["winner_margin_seconds"]).abs().max() <= 1e-10)

    multiplier_by_race = multipliers.set_index("race_id")
    metric_max_mu = metric_max_var = metric_max_nll = 0.0
    for race_id, group in snapshots.groupby("race_id", sort=False):
        group = group.sort_values("horse_number")
        finish = group["finish_position"].to_numpy(dtype=np.int16)
        orders = make_orders(finish)
        unranked = np.flatnonzero(finish == 0).astype(np.int16)
        k_max = min(5, int(np.count_nonzero(finish > 0)), len(group) - 1)
        scores = group["m_pre"].to_numpy(dtype=float)[None, :]
        row = multiplier_by_race.loc[race_id]
        stage_values = np.array([row[f"stage_multiplier_{index}"] for index in range(1, 6)], dtype=float)
        train_weights = np.asarray([WEIGHTS["W_share"]]) * stage_values[None, :]
        gradient, diagonal, eval_ll, _ = pl_batch(scores, orders, unranked, k_max, train_weights, W_EVAL, True)
        precision = 1.0 / group["var_pre_eff"].to_numpy(dtype=float) + diagonal[0]
        expected_mu = group["mu_pre"].to_numpy(dtype=float).copy()
        expected_var = group["var_pre_persistent"].to_numpy(dtype=float).copy()
        ranked = finish > 0
        expected_mu[ranked] += gradient[0, ranked] / precision[ranked]
        expected_var[ranked] = 1.0 / precision[ranked]
        metric_max_mu = max(metric_max_mu, error(group["mu_post"], expected_mu))
        metric_max_var = max(metric_max_var, error(group["var_post"], expected_var))
        metric_max_nll = max(metric_max_nll, abs(float(-eval_ll[0])))
    check("independent_margin_update", metric_max_mu <= 1e-8 and metric_max_var <= 1e-8, {"max_mu_error": metric_max_mu, "max_var_error": metric_max_var})
    no_rank = snapshots["finish_position"].eq(0)
    check("no_rank_state_unchanged", np.allclose(snapshots.loc[no_rank, "mu_pre"], snapshots.loc[no_rank, "mu_post"], atol=0, rtol=0) and np.allclose(snapshots.loc[no_rank, "var_pre_persistent"], snapshots.loc[no_rank, "var_post"], atol=0, rtol=0))

    ordered = snapshots.sort_values(["horse_id", "race_date", "race_id"]).copy()
    base_counts = frozen_state.set_index("horse_id")["n_graph_starts"]
    ordered["expected_starts"] = ordered["horse_id"].map(base_counts).fillna(0) + ordered.groupby("horse_id").cumcount()
    check("graph_start_counts", ordered["n_graph_starts"].eq(ordered["expected_starts"]).all())
    frozen_dates = pd.to_datetime(frozen_state.set_index("horse_id")["last_start_date"])
    previous_dates = ordered.groupby("horse_id")["race_date"].shift(1).fillna(ordered["horse_id"].map(frozen_dates))
    check("days_since_start", ordered["days_since_start"].eq((ordered["race_date"] - previous_dates).dt.days.fillna(0).clip(lower=0)).all())
    later = ordered.groupby("horse_id").cumcount().gt(0)
    previous_mu = ordered.groupby("horse_id")["mu_post"].shift(1)
    previous_var = ordered.groupby("horse_id")["var_post"].shift(1)
    check("persistent_mu_transition", np.allclose(ordered.loc[later, "mu_pre"], previous_mu[later], atol=1e-10, rtol=0))
    check("persistent_var_transition", np.allclose(ordered.loc[later, "var_pre_persistent"], previous_var[later], atol=1e-10, rtol=0))
    latest = ordered.groupby("horse_id").tail(1).set_index("horse_id")
    current_state = state.set_index("horse_id").reindex(latest.index)
    check("final_state", np.allclose(latest["mu_post"], current_state["mu"], atol=1e-10, rtol=0) and np.allclose(latest["var_post"], current_state["var"], atol=1e-10, rtol=0))
    check("final_state_starts", latest["n_graph_starts"].add(1).eq(current_state["n_graph_starts"]).all())
    check("final_state_dates", latest["race_date"].eq(pd.to_datetime(current_state["last_start_date"])).all())
    unchanged_ids = state.set_index("horse_id").index.difference(latest.index)
    check("untouched_frozen_state", np.allclose(state.set_index("horse_id").loc[unchanged_ids, "mu"], frozen_state.set_index("horse_id").loc[unchanged_ids, "mu"], atol=1e-10, rtol=0) and np.allclose(state.set_index("horse_id").loc[unchanged_ids, "var"], frozen_state.set_index("horse_id").loc[unchanged_ids, "var"], atol=1e-10, rtol=0))

    work = out / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        db = copy_database(resolve_path(config["database"]["path"]), Path(temporary), config["database"].get("linked_tables", {}))
        try:
            pedigree = load_pedigree_ids(db, fallback_to_horse_master=True)
        finally:
            gc.collect()
            cleanup_access_lock_files(temporary)
    candidates = snapshots[["race_index", "race_date", "horse_id", "debut_type_code", "n_graph_starts", "condition_code"]].copy()
    candidates["date"] = (candidates["race_date"].astype("int64") // 86_400_000_000_000).astype(np.int32)
    candidates = candidates.merge(pedigree, on="horse_id", how="left", validate="many_to_one")
    max_feature_error = 0.0
    feature_mismatches = 0
    p2b = config["_p2b"]
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
    check("pedigree_asof_features", feature_mismatches == 0, {"mismatches": feature_mismatches, "max_error": max_feature_error})
    p1, beta = p2b["base_p1"]["p1"], p2b["p2"]["beta"]
    new = snapshots[snapshots["n_graph_starts"].eq(0)].copy()
    b_map = {1: p1["b"]["新馬戦"], 2: p1["b"]["未勝利戦"], 3: p1["b"]["その他のクラス"]}
    expected_mu0 = new["mu_bar_D"] + new["debut_type_code"].map(b_map) + beta["beta_sw"] * new["X_sire_win"] + beta["beta_st"] * new["X_sire_top3"] + beta["beta_bw"] * new["X_bms_win"] + beta["beta_bt"] * new["X_bms_top3"]
    check("p4_initial_prior", np.allclose(new["mu0"], expected_mu0, atol=1e-8, rtol=0), {"max_error": error(new["mu0"], expected_mu0)})

    max_residual_error = 0.0
    by_index = {index: frame for index, frame in snapshots.groupby("race_index", sort=False)}
    for race_index, group in residual.groupby("race_index"):
        frame = by_index[race_index]
        eligible = frame[frame["debut_type_code"].eq(1) & frame["n_graph_starts"].le(2)].sort_values("horse_number")
        probabilities = harville_inclusion_probabilities(eligible["m_pre"].to_numpy(dtype=float), 3)
        ordered_history = eligible[["horse_id", "finish_position"]].merge(group.drop(columns=["finish_position"]), on="horse_id", validate="one_to_one")
        max_residual_error = max(
            max_residual_error, error(ordered_history["e_win"], probabilities[:, 0]),
            error(ordered_history["e_top3"], probabilities.sum(axis=1)),
            error(ordered_history["actual_win"], ordered_history["finish_position"].eq(1).astype(float)),
            error(ordered_history["actual_top3"], ordered_history["finish_position"].between(1, 3).astype(float)),
        )
    check("p4_residual_history", max_residual_error <= 1e-8, {"max_error": max_residual_error})

    full_history = pd.concat([
        pd.read_csv(BASE / "p4_margin/snapshots.csv.gz", usecols=["race_id", "race_date", "horse_id", "mu_post"]),
        snapshots[["race_id", "race_date", "horse_id", "mu_post"]],
    ], ignore_index=True)
    full_history["race_date"] = pd.to_datetime(full_history["race_date"])
    full_history = full_history.sort_values(["horse_id", "race_date", "race_id"])
    by_date = {day: frame for day, frame in full_history.groupby("race_date")}
    anchor_errors = []
    for row in anchor.itertuples():
        active_ids = set()
        for offset in range(1, 366):
            day = row.asof_date - pd.Timedelta(days=offset)
            if day in by_date:
                active_ids.update(by_date[day]["horse_id"].tolist())
        latest_active = full_history[full_history["horse_id"].isin(active_ids) & full_history["race_date"].lt(row.asof_date)].groupby("horse_id").tail(1)
        if len(latest_active) != row.n_active_365d or not np.isclose(latest_active["mu_post"].mean(), row.mu_bar_D, atol=1e-10, rtol=0):
            anchor_errors.append(str(row.asof_date.date()))
    check("daily_anchor_recalculation", not anchor_errors, {"mismatches": len(anchor_errors), "sample": anchor_errors[:5]})

    subprocess.run([sys.executable, str(BASE / "p4_margin/protect_assets.py"), "--check"], check=True, capture_output=True)
    protected = json.loads((BASE / "p4_margin/protected_assets_verification.json").read_text(encoding="utf-8"))
    check("protected_assets_unchanged", protected["status"] == "PASS", protected)
    report["metrics"] = {
        "first_race_date": str(snapshots["race_date"].min().date()), "last_race_date": str(snapshots["race_date"].max().date()),
        "races": int(snapshots["race_id"].nunique()), "starts": len(snapshots), "state_horses": len(state),
        "anchor_days": len(anchor), "residual_rows": len(residual), "run_id": manifest["run_id"],
        "max_update_mu_error": metric_max_mu, "max_update_var_error": metric_max_var,
        "max_pedigree_feature_error": max_feature_error, "max_residual_error": max_residual_error,
        "config_sha256": EXPECTED_P4_HASH,
    }
    if args.check_idempotency:
        before = {name: file_sha256(out / name) for name in OUTPUT_FILES.values()}
        second = update(config, None)
        after = {name: file_sha256(out / name) for name in OUTPUT_FILES.values()}
        check("idempotency", second["status"] == "NO_NEW_RACES" and before == after, {"status": second["status"], "files_unchanged": before == after})
    if report["issues"]:
        report["status"] = "FAIL"
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
