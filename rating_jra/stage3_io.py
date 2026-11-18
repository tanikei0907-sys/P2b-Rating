from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rating_data import PreparedInput
from rating_engine import EngineResult
from rating_evaluation import DEBUT_TYPE_LABELS, anchor_svg, anchor_transition_summary
from rating_pedigree import STATUS_LABELS


def combined_pedigree_status(sire: pd.Series, bms: pd.Series) -> pd.Series:
    result = pd.Series("OK", index=sire.index, dtype=object)
    result[sire.eq(0) & bms.eq(0)] = "NOT_USED"
    result[sire.eq(4) & bms.eq(4)] = "EXISTING_UNKNOWN"
    result[sire.eq(2) & bms.eq(2)] = "NO_HISTORY"
    result[(sire.eq(2) ^ bms.eq(2)) & ~sire.eq(3) & ~bms.eq(3)] = "PARTIAL_HISTORY"
    result[sire.eq(3) | bms.eq(3)] = "ID_MISSING"
    return result


def save_stage3_model(
    prepared: PreparedInput,
    result: EngineResult,
    model: str,
    config_hash: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = result.snapshots.copy()
    snapshots["prior_type"] = model
    snapshots["debut_type"] = snapshots["debut_type_code"].map(DEBUT_TYPE_LABELS)
    snapshots["X_sire"] = snapshots["X_sire_win"]
    snapshots["X_bms"] = snapshots["X_bms_win"]
    snapshots["pedigree_status"] = combined_pedigree_status(snapshots["sire_status_code"], snapshots["bms_status_code"])
    snapshots["sire_status"] = snapshots["sire_status_code"].map(STATUS_LABELS)
    snapshots["bms_status"] = snapshots["bms_status_code"].map(STATUS_LABELS)
    snapshots["config_sha256"] = config_hash
    leading = [
        "race_id", "race_date", "horse_id", "mu_pre", "var_pre_persistent", "var_pre_eff",
        "m_pre", "days_since_start", "n_graph_starts", "prior_type", "mu0", "debut_type",
        "X_sire", "X_bms", "N_sire", "N_bms", "pedigree_status", "n_field_original",
        "n_field_retained", "n_removed_nonJRA", "X_sire_win", "X_sire_top3", "X_bms_win",
        "X_bms_top3", "sire_status", "bms_status",
    ]
    snapshots = snapshots[leading + [column for column in snapshots.columns if column not in leading]]
    snapshots.to_csv(output_dir / "snapshots.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    snapshots[["race_id", "race_date", "horse_id", "mu_post", "var_post", "config_sha256"]].to_csv(
        output_dir / "post_updates.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d"
    )
    state = result.state.copy()
    initial_mu0 = snapshots.groupby("horse_id")["mu0"].first()
    state["mu0"] = state["horse_id"].map(initial_mu0)
    state["last_start_date"] = pd.to_datetime(state["last_start_date"], unit="D", origin="unix").dt.strftime("%Y-%m-%d")
    state["debut_type"] = state["debut_type_code"].map(DEBUT_TYPE_LABELS)
    state["prior_type"] = model
    state["config_sha256"] = config_hash
    state.to_csv(output_dir / "state_2024-12-31.csv", index=False, float_format="%.12g")
    daily = result.daily.copy()
    daily["asof_date"] = pd.to_datetime(daily["asof_date"], unit="D", origin="unix")
    daily.to_csv(output_dir / "mu_anchor_daily.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d")
    anchor_transition_summary(daily).to_csv(output_dir / "mu_anchor_transition_summary.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d")
    anchor_svg(daily, output_dir / "mu_anchor_timeseries.svg")
    events = pd.DataFrame({
        "update_sequence": np.arange(1, len(prepared.races) + 1),
        "event_id": [f"{model}::{race.race_id}::{prepared.source_hash[:16]}" for race in prepared.races],
        "race_id": [race.race_id for race in prepared.races],
        "race_date": [pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date) for race in prepared.races],
        "result_version": [race.result_version for race in prepared.races],
        "input_db_sha256": prepared.source_hash,
        "config_sha256": config_hash,
    })
    if events["event_id"].duplicated().any() or events["race_id"].duplicated().any():
        raise AssertionError("更新イベントが重複しています")
    events.to_csv(output_dir / "result_events.csv", index=False, date_format="%Y-%m-%d")
