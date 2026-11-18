from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import W_EVAL, file_sha256, pl_race_single
from rating_data import make_orders

OUT = Path(__file__).with_name("stage2_output")


def main() -> None:
    config_path = OUT / "config_p0.yaml"
    config_hash = file_sha256(config_path)
    assert config_hash == config_path.with_suffix(config_path.suffix + ".sha256").read_text(encoding="ascii").strip()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_VALIDATION_EVALUATION"
    assert file_sha256("stage2_input/BaoZ_2_snapshot.mdb") == config["input_db_sha256"]
    required_snapshot_columns = {
        "race_id", "race_date", "horse_id", "mu_pre", "var_pre_persistent", "var_pre_eff", "m_pre",
        "days_since_start", "n_graph_starts", "prior_type", "mu0", "debut_type", "X_sire", "X_bms",
        "N_sire", "N_bms", "pedigree_status", "n_field_original", "n_field_retained", "n_removed_nonJRA",
    }
    snapshots = pd.read_csv(OUT / "snapshots.csv.gz", parse_dates=["race_date"])
    assert required_snapshot_columns <= set(snapshots.columns)
    assert len(snapshots) == 610676
    assert snapshots["race_id"].nunique() == 43252
    assert snapshots["race_date"].max() < pd.Timestamp("2025-01-01")
    assert snapshots["prior_type"].eq("P0").all()
    assert snapshots["config_sha256"].eq(config_hash).all()
    assert snapshots["mu0"].notna().all()
    assert not snapshots.duplicated(["race_id", "horse_id"]).any()
    same_day = snapshots.groupby(["horse_id", "race_date"])["race_id"].nunique()
    assert not same_day.gt(1).any()
    posts = pd.read_csv(OUT / "post_updates.csv.gz")
    assert len(posts) == len(snapshots)
    assert posts["config_sha256"].eq(config_hash).all()
    state = pd.read_csv(OUT / "state_2024-12-31.csv")
    assert len(state) == 67030
    assert state["config_sha256"].eq(config_hash).all()
    assert state["mu0"].notna().all()
    assert state["n_graph_starts"].gt(0).all()
    assert pd.to_datetime(state["last_start_date"]).max() < pd.Timestamp("2025-01-01")
    events = pd.read_csv(OUT / "result_events.csv", parse_dates=["race_date"])
    assert len(events) == 43252
    assert events["event_id"].is_unique and events["race_id"].is_unique
    assert events["race_date"].is_monotonic_increasing
    assert events["race_date"].max() < pd.Timestamp("2025-01-01")
    daily = pd.read_csv(OUT / "mu_anchor_daily.csv", parse_dates=["asof_date"])
    assert {
        "asof_date", "n_active_365d", "mu_bar_D", "active_latest_mu_std", "active_latest_mu_median",
        "mu_bar_delta_1d", "n_new_horses", "mu0_mean", "anchor_group", "threshold_crossing",
    } <= set(daily.columns)
    assert len(daily) == 1409
    assert daily["asof_date"].max() < pd.Timestamp("2025-01-01")
    ET.parse(OUT / "mu_anchor_timeseries.svg")
    search = pd.read_csv(OUT / "hyperparameter_search.csv")
    assert len(search) == 216
    assert search["eval_pl_nll_tuning_E"].is_monotonic_increasing
    best = search.iloc[0]
    for name in ["var0", "q", "kappa", "beta_kg", "weight"]:
        assert best[name] == config["p0"][name]
    assert len(pd.read_csv(OUT / "hyperparameter_sensitivity.csv")) == 15

    metrics = pd.read_csv(OUT / "race_metrics.csv.gz")
    metrics_by_id = metrics.set_index("race_id")
    sample_race_ids = metrics["race_id"].iloc[::1750].tolist()
    max_p0_error = 0.0
    max_s0_error = 0.0
    for race_id in sample_race_ids:
        group = snapshots[snapshots["race_id"].eq(race_id)]
        finish = group["finish_position"].to_numpy(dtype=np.int16)
        orders = make_orders(finish)
        unranked = np.flatnonzero(finish == 0).astype(np.int16)
        k_max = min(5, int(np.count_nonzero(finish > 0)), len(group) - 1)
        scores = group["m_pre"].to_numpy(dtype=float)
        p0 = -pl_race_single(scores, orders, unranked, k_max, W_EVAL)[0]
        s0 = -pl_race_single(np.zeros(len(group)), orders, unranked, k_max, W_EVAL)[0]
        max_p0_error = max(max_p0_error, abs(p0 - metrics_by_id.loc[race_id, "p0_nll"]))
        max_s0_error = max(max_s0_error, abs(s0 - metrics_by_id.loc[race_id, "s0_nll"]))
    assert max_p0_error < 1.0e-9
    assert max_s0_error < 1.0e-9
    validation_e = metrics[metrics["validation"] & metrics["E"]]
    pooled = pd.read_csv(OUT / "main_comparisons.csv")
    pooled = pooled[(pooled["period"] == "validation") & (pooled["pool"] == "E") & (pooled["year"] == "POOLED")].iloc[0]
    assert np.isclose(validation_e["delta_nll"].mean(), pooled["delta_nll"])
    assert pooled["ci95_upper"] < 0
    assert int(metrics[(metrics["validation"]) & (metrics["E"])].groupby("year")["delta_nll"].mean().lt(0).sum()) == 4
    report = (OUT / "stage2_report.md").read_text(encoding="utf-8")
    assert "段階3の血統事前分布は実装・実行していない" in report
    for name in ["rating_update_for_features.py", "rating_final_eval.py"]:
        assert Path(name).exists()
    verification = {
        "status": "PASS",
        "config_sha256": config_hash,
        "input_db_sha256": config["input_db_sha256"],
        "snapshot_rows": len(snapshots),
        "races": int(snapshots["race_id"].nunique()),
        "state_horses": len(state),
        "result_events": len(events),
        "anchor_days": len(daily),
        "search_configurations": len(search),
        "independent_nll_sample_races": len(sample_race_ids),
        "max_abs_p0_nll_error": max_p0_error,
        "max_abs_s0_nll_error": max_s0_error,
        "validation_E_pooled_delta_nll": float(pooled["delta_nll"]),
        "validation_E_ci95_upper": float(pooled["ci95_upper"]),
        "future_scripts_executed": False,
    }
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
