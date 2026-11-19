from __future__ import annotations

import heapq
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from forward_ops import append_frame, copy_database, enrich_snapshots, filter_new_races, read_state, resolve_path, state_frame
from p4_margin.margins import multiplier_arrays, sequential_margins
from p4_margin.p4_engine import P4Engine
from rating_core import P0Config, file_sha256
from rating_data import load_raw_from_db, prepare_input
from rating_pedigree import load_pedigree_ids
from stage3_evaluate import rule_for_p2
from viewer_data import cleanup_access_lock_files

EXPECTED_P4_HASH = "436f08ffb840a60ef33354fc28fa9434d0193a0460c73ba8c8f29d8038a26ba1"
EXPECTED_P2B_HASH = "3577cbc7a6a2c3e58bfdeda6666b9aa9d7369866f4c6fa49902dac6b8779dbe9"
OUTPUT_FILES = {
    "snapshots": "snapshots_2025plus.csv.gz",
    "posts": "post_updates_2025plus.csv.gz",
    "state": "state_latest.csv",
    "residual": "pedigree_residual_history_2025plus.csv.gz",
    "anchor": "mu_anchor_daily_2025plus.csv",
    "events": "result_events_2025plus.csv",
    "margins": "margins_2025plus.csv.gz",
    "multipliers": "margin_multipliers_2025plus.csv.gz",
}


class P4ForwardError(RuntimeError):
    pass


def load_config(path: str | Path = "p4_margin/forward_config.yaml") -> dict:
    config_path = resolve_path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    if model["config_sha256"] != EXPECTED_P4_HASH:
        raise P4ForwardError("forward_configのP4ハッシュ固定値が不正です")
    p4_path = resolve_path(model["config_path"])
    if file_sha256(p4_path) != EXPECTED_P4_HASH:
        raise P4ForwardError("P4 configのハッシュが不一致です")
    p4 = json.loads(p4_path.read_text(encoding="utf-8"))
    if p4["method"] != "A" or p4["normalize"] or p4["gamma"] != 2.0 or p4["m_max"] != 2.0:
        raise P4ForwardError("P4は方式A・生タイム差・γ=2・cap=2に固定されています")
    p2b_path = resolve_path(model["p2b_config_path"])
    if file_sha256(p2b_path) != EXPECTED_P2B_HASH:
        raise P4ForwardError("P2bベースラインconfigのハッシュが不一致です")
    final = json.loads((BASE / "p4_margin/final_model.json").read_text(encoding="utf-8"))
    if final["adopted_model"] != "P4" or final["adopted_config_sha256"] != EXPECTED_P4_HASH:
        raise P4ForwardError("P4が最終採用モデルとして確定されていません")
    config["_p4"] = p4
    config["_p2b"] = json.loads(p2b_path.read_text(encoding="utf-8"))
    return config


def load_raw_with_times(
    db_path: Path,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    final_only: bool = True,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_clause = ""
    if end_date is not None:
        end_clause = f"AND s.[月日] < #{pd.Timestamp(end_date).strftime('%Y-%m-%d')}#"
    final_clause = "AND s.[データ区分] = '7'" if final_only else ""
    sql = f"""
        SELECT s.[競走コード] AS race_id, s.[月日] AS race_date,
               s.[主催者コード] AS organizer_code, s.[トラック種別コード] AS surface_code,
               s.[場コード] AS venue_code, s.[場名] AS venue_name,
               s.[距離] AS distance, s.[競走条件コード] AS condition_code,
               s.[登録頭数] AS n_registered, s.[データ作成年月日] AS race_data_created,
               s.[データ区分] AS result_version, s.[馬場状態コード] AS track_condition_code,
               r.[馬番] AS horse_number, r.[血統登録番号] AS horse_id,
               r.[負担重量] AS weight_raw, r.[確定着順] AS finish_position,
               r.[異常区分コード] AS abnormal_code, r.[タイム] AS time_raw,
               r.[着差] AS stored_margin_raw, rm.[東西所属コード] AS affiliation_code
        FROM ([レースT] AS s INNER JOIN [出走馬T] AS r ON s.[競走コード]=r.[競走コード])
        INNER JOIN [出走馬マスタ] AS rm
          ON r.[競走コード]=rm.[競走コード] AND r.[馬番]=rm.[馬番]
        WHERE s.[月日] >= #{start}# {end_clause} {final_clause}
          AND s.[主催者コード]=1 AND s.[トラック種別コード] IN (0,1)
    """
    connection = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};ReadOnly=1;"
    with pyodbc_connect(connection) as con:
        frame = pd.read_sql(sql, con)
        con.close()
    frame["weight_kg"] = pd.to_numeric(frame["weight_raw"], errors="coerce") / 10.0
    frame.loc[frame["weight_kg"].le(0), "weight_kg"] = np.nan
    frame["result_version"] = "BaoZ_data_division_" + frame["result_version"].astype(str)
    return frame.drop(columns=["weight_raw"])


def pyodbc_connect(connection: str):
    import pyodbc
    return pyodbc.connect(connection, readonly=True, autocommit=True)


def build_margins(raw: pd.DataFrame, prepared) -> pd.DataFrame:
    frame = raw.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"])
    target = frame[
        frame["organizer_code"].eq(1) & frame["surface_code"].isin([0, 1])
        & frame["venue_code"].lt(101)
        & ~frame["venue_name"].fillna("").astype(str).str.contains("帯広", regex=False)
    ]
    candidates = target[target["affiliation_code"].isin([1, 2]) & ~target["abnormal_code"].isin([1, 2, 3])].copy()
    for column in ["race_id", "horse_id", "horse_number"]:
        candidates[column] = pd.to_numeric(candidates[column], errors="raise").astype("int64")
    candidates["time_seconds"] = pd.to_numeric(candidates["time_raw"], errors="coerce").where(pd.to_numeric(candidates["time_raw"], errors="coerce") > 0) / 10.0
    by_race = {race_id: group.sort_values("horse_number") for race_id, group in candidates.groupby("race_id", sort=False)}
    records = []
    for race in prepared.races:
        group = by_race[race.race_id]
        expected_ids = prepared.horse_ids[race.horse_idx]
        if not np.array_equal(group["horse_id"].to_numpy(dtype=np.int64), expected_ids):
            raise P4ForwardError(f"タイムと馬IDの対応が不一致です: {race.race_id}")
        deltas, statuses = sequential_margins(race.finish, group["time_seconds"].to_numpy(dtype=float), race.k_max)
        records.append({
            "race_id": int(race.race_id),
            "race_date": pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date),
            "distance": int(race.distance), "condition_code": int(race.condition_code),
            "track_condition_code": int(group["track_condition_code"].iloc[0]), "k_max": int(race.k_max),
            **{f"stage_{index}": float(deltas[index - 1]) for index in range(1, 6)},
            **{f"status_{index}": int(statuses[index - 1]) for index in range(1, 6)},
        })
    result = pd.DataFrame(records)
    if result["race_id"].duplicated().any():
        raise P4ForwardError("マージンのrace_idが重複しています")
    return result


def load_events(out: Path) -> pd.DataFrame:
    path = out / OUTPUT_FILES["events"]
    if not path.exists():
        return pd.DataFrame(columns=["race_id", "race_date", "result_version", "config_sha256"])
    events = pd.read_csv(path)
    if not events["config_sha256"].eq(EXPECTED_P4_HASH).all() or events["race_id"].duplicated().any():
        raise P4ForwardError("既存P4更新イベントが不正です")
    return events


def load_history(config: dict) -> pd.DataFrame:
    model = config["model"]
    frames = [pd.read_csv(resolve_path(model["frozen_residual_history_path"]))]
    forward = resolve_path(config["output"]["directory"]) / OUTPUT_FILES["residual"]
    if forward.exists():
        frames.append(pd.read_csv(forward))
    history = pd.concat(frames, ignore_index=True)
    if history.duplicated(["race_index", "horse_id"]).any():
        raise P4ForwardError("P4残差履歴に重複があります")
    return history


def initialize_history(engine: P4Engine, history: pd.DataFrame) -> None:
    for side in ["sire", "bms"]:
        valid = history[history[f"{side}_id"].notna() & history[f"{side}_id"].ne(0)]
        grouped = valid.groupby(f"{side}_id").agg(N=("r_win", "size"), sum_win=("r_win", "sum"), sum_top3=("r_top3", "sum"))
        engine.statistics[side][0].update({int(key): int(value) for key, value in grouped["N"].items()})
        engine.statistics[side][1].update({int(key): float(value) for key, value in grouped["sum_win"].items()})
        engine.statistics[side][2].update({int(key): float(value) for key, value in grouped["sum_top3"].items()})


def run_forward(raw, source_hash: str, config: dict, state: pd.DataFrame, history: pd.DataFrame, pedigree: pd.DataFrame, dry_run: bool = False, race_index_start: int = 0):
    prepared = prepare_input(raw, source_hash=source_hash, allow_future=True, allow_unranked=dry_run)
    margins = build_margins(raw, prepared)
    future_ids = prepared.horse_ids.copy()
    union_ids = np.sort(np.union1d(future_ids, state["horse_id"].to_numpy(dtype=np.int64)))
    future_to_union = {int(horse_id): index for index, horse_id in enumerate(union_ids)}
    old_to_union = np.array([future_to_union[int(horse_id)] for horse_id in future_ids], dtype=np.int32)
    state_index = {int(horse_id): index for index, horse_id in enumerate(union_ids)}
    base_counts = np.zeros(len(union_ids), dtype=np.int32)
    for row in state.itertuples():
        base_counts[state_index[int(row.horse_id)]] = int(row.n_graph_starts)
    for race in prepared.races:
        race.horse_idx = old_to_union[race.horse_idx]
        race.pre_nstarts = base_counts[race.horse_idx] + race.pre_nstarts
        if race.condition_code == 701 and np.any(race.pre_nstarts > 0):
            raise P4ForwardError(f"新馬戦に過去グラフ出走馬があります: {race.race_id}")
    prepared.horse_ids = union_ids
    p2b = config["_p2b"]
    p0 = P0Config(**p2b["base_p1"]["base_p0"]["p0"])
    rule = rule_for_p2(p2b)
    stages, scales = multiplier_arrays(margins, prepared, config["_p4"])
    engine = P4Engine(prepared, p0, rule, pedigree, p2b["p2"]["standardization"], "A", stages, scales, True)
    for row in state.itertuples():
        index = state_index[int(row.horse_id)]
        engine.mu[0, index] = float(row.mu)
        engine.var[0, index] = float(row.var)
        engine.var_cap[0, index] = float(row.var_cap)
        engine.seen[index] = True
        engine.last_start[index] = int((pd.Timestamp(row.last_start_date) - pd.Timestamp("1970-01-01")).days)
        engine.n_starts[index] = int(row.n_graph_starts)
        engine.debut_types[index] = int(row.debut_type_code)
        engine.mu0_state[0, index] = float(row.mu0)
        engine.active[index] = True
        heapq.heappush(engine.expiry_heap, (int(engine.last_start[index]) + 365, index, int(engine.last_start[index])))
    initialize_history(engine, history)
    result = engine.run()
    if engine.history.empty:
        engine.history = history.iloc[0:0].copy()
    else:
        engine.history["race_id"] = engine.history["race_index"].map({race.race_index: race.race_id for race in prepared.races})
    if race_index_start:
        result.snapshots["race_index"] += race_index_start
        engine.history["race_index"] += race_index_start
    return prepared, result, engine, margins, stages, scales


def write_outputs(staging: Path, outputs: dict[str, pd.DataFrame]) -> None:
    for key, name in OUTPUT_FILES.items():
        outputs[key].to_csv(
            staging / name, index=False, compression="gzip" if name.endswith(".gz") else None,
            float_format="%.17g", date_format="%Y-%m-%d",
        )


def backup_existing(out: Path, backup_root: Path, run_id: str) -> tuple[Path, set[str]]:
    backup = backup_root / run_id
    backup.mkdir(parents=True, exist_ok=False)
    existing = set()
    for name in OUTPUT_FILES.values():
        path = out / name
        if path.exists():
            shutil.copy2(path, backup / name)
            existing.add(name)
    return backup, existing


def restore_existing(out: Path, backup: Path, existing: set[str]) -> None:
    for name in OUTPUT_FILES.values():
        path = out / name
        archived = backup / name
        if name in existing:
            shutil.copyfile(archived, path)
        elif path.exists():
            path.unlink()


def update(config: dict, end_date: str | None = None, raw_loader=load_raw_with_times) -> dict:
    out = resolve_path(config["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    start_date = config["processing"].get("start_date", "2025-01-01")
    end_date = end_date or config["processing"].get("end_date")
    end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end_date else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(4).hex()
    work_root = resolve_path(config["output"]["temp_directory"])
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as temporary:
        db = copy_database(resolve_path(config["database"]["path"]), Path(temporary), config["database"].get("linked_tables", {}))
        source_hash = file_sha256(db)
        try:
            raw = raw_loader(db, start_date, end_exclusive, True)
            events = load_events(out)
            raw = filter_new_races(raw, events)
            if raw.empty:
                return {"status": "NO_NEW_RACES", "new_races": 0, "run_id": run_id}
            if not events.empty:
                last_processed = pd.to_datetime(events["race_date"]).max()
                if pd.to_datetime(raw["race_date"]).min() <= last_processed:
                    raise P4ForwardError("処理済み日以前のレースが新規に追加されました。単方向更新を停止します")
            state_path = out / OUTPUT_FILES["state"]
            state = read_state(state_path if state_path.exists() else resolve_path(config["model"]["frozen_state_path"]), EXPECTED_P4_HASH)
            history = load_history(config)
            pedigree = load_pedigree_ids(db, fallback_to_horse_master=True)
            snapshots_path = out / OUTPUT_FILES["snapshots"]
            race_index_start = int(pd.read_csv(snapshots_path, usecols=["race_index"])["race_index"].max()) + 1 if snapshots_path.exists() else 0
            prepared, result, engine, margins, stages, scales = run_forward(raw, source_hash, config, state, history, pedigree, race_index_start=race_index_start)
            snapshots = enrich_snapshots(result, engine, "P4", EXPECTED_P4_HASH)
            snapshots["source_db_sha256"] = source_hash
            margin_index = margins.set_index("race_id")
            snapshots["winner_margin_seconds"] = snapshots["race_id"].map(margin_index["stage_1"])
            snapshots["winner_margin_normalized"] = snapshots["winner_margin_seconds"] / (snapshots["distance"] / 1000.0)
            snapshots["race_margin_scale"] = 1.0
            posts = snapshots[["race_id", "race_date", "horse_id", "mu_post", "var_post", "config_sha256", "source_db_sha256"]].copy()
            daily = result.daily.copy()
            daily["asof_date"] = pd.to_datetime(daily["asof_date"], unit="D", origin="unix")
            residual = engine.history.copy()
            residual = residual[residual["race_date"].ge("2025-01-01")].copy()
            residual["config_sha256"] = EXPECTED_P4_HASH
            residual["source_db_sha256"] = source_hash
            now = datetime.now(timezone.utc).isoformat()
            new_events = pd.DataFrame({
                "event_id": [f"P4::{race.race_id}::{source_hash[:16]}" for race in prepared.races],
                "race_id": [race.race_id for race in prepared.races],
                "race_date": [pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date) for race in prepared.races],
                "result_version": [race.result_version for race in prepared.races],
                "processed_at_utc": now, "input_db_sha256": source_hash, "config_sha256": EXPECTED_P4_HASH,
            })
            margins["config_sha256"] = EXPECTED_P4_HASH
            margins["source_db_sha256"] = source_hash
            multipliers = pd.DataFrame({
                "race_id": [race.race_id for race in prepared.races],
                **{f"stage_multiplier_{index}": stages[:, index - 1] for index in range(1, 6)},
                "race_scale": scales, "config_sha256": EXPECTED_P4_HASH, "source_db_sha256": source_hash,
            })
            outputs = {
                "snapshots": append_frame(snapshots_path, snapshots, ["race_id", "horse_id"], ["race_date", "race_id", "horse_id"]),
                "posts": append_frame(out / OUTPUT_FILES["posts"], posts, ["race_id", "horse_id"], ["race_date", "race_id", "horse_id"]),
                "state": state_frame(result, engine, "P4", EXPECTED_P4_HASH),
                "residual": append_frame(out / OUTPUT_FILES["residual"], residual, ["race_id", "horse_id"], ["race_date", "race_id", "horse_id"]),
                "anchor": append_frame(out / OUTPUT_FILES["anchor"], daily, ["asof_date"], ["asof_date"]),
                "events": append_frame(out / OUTPUT_FILES["events"], new_events, ["race_id"], ["race_date", "race_id"]),
                "margins": append_frame(out / OUTPUT_FILES["margins"], margins, ["race_id"], ["race_date", "race_id"]),
                "multipliers": append_frame(out / OUTPUT_FILES["multipliers"], multipliers, ["race_id"], ["race_id"]),
            }
            backup, existing = backup_existing(out, resolve_path(config["output"]["backup_directory"]), run_id)
            staging = Path(temporary) / "staging"
            staging.mkdir()
            try:
                write_outputs(staging, outputs)
                hashes = {name: file_sha256(staging / name) for name in OUTPUT_FILES.values()}
                for name in OUTPUT_FILES.values():
                    os.replace(staging / name, out / name)
            except Exception:
                restore_existing(out, backup, existing)
                raise
        finally:
            import gc
            gc.collect()
            cleanup_access_lock_files(temporary)
    manifest = {
        "status": "UPDATED", "run_id": run_id, "new_races": int(raw["race_id"].nunique()),
        "first_race_date": str(pd.to_datetime(raw["race_date"]).min().date()),
        "last_race_date": str(pd.to_datetime(raw["race_date"]).max().date()),
        "config_sha256": EXPECTED_P4_HASH, "input_db_sha256": source_hash,
        "backup": str(backup), "output_hashes": hashes,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / f"run_manifest_{run_id}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "latest_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def predict_snapshot(config: dict, predict_date: str | None = None, raw_loader=load_raw_with_times) -> dict:
    out = resolve_path(config["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    predict_date = predict_date or config["processing"].get("predict_date")
    start = predict_date or datetime.now().strftime("%Y-%m-%d")
    end_exclusive = (pd.Timestamp(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if predict_date else None
    work_root = resolve_path(config["output"]["temp_directory"])
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as temporary:
        db = copy_database(resolve_path(config["database"]["path"]), Path(temporary), config["database"].get("linked_tables", {}))
        source_hash = file_sha256(db)
        try:
            raw = raw_loader(db, start, end_exclusive, False)
            if raw.empty:
                raise P4ForwardError("予測対象のレースがありません")
            target = predict_date or raw["race_date"].min().strftime("%Y-%m-%d")
            raw = raw[raw["race_date"].eq(target)].copy()
            if raw.empty:
                raise P4ForwardError(f"指定日のレースがありません: {target}")
            events = load_events(out)
            if not events.empty and pd.Timestamp(target) <= pd.to_datetime(events["race_date"]).max():
                raise P4ForwardError("予測日が最新確定日以前です。過去の確定レースへの将来状態の適用は行いません")
            state_path = out / OUTPUT_FILES["state"]
            state = read_state(state_path if state_path.exists() else resolve_path(config["model"]["frozen_state_path"]), EXPECTED_P4_HASH)
            history = load_history(config)
            pedigree = load_pedigree_ids(db, fallback_to_horse_master=True)
            _, result, engine, _, _, _ = run_forward(raw, source_hash, config, state, history, pedigree, dry_run=True)
            snapshots = enrich_snapshots(result, engine, "P4", EXPECTED_P4_HASH)
            snapshots["source_db_sha256"] = source_hash
            snapshots = snapshots.drop(columns=["mu_post", "var_post", "skip_mu_var_update"])
            output = out / f"predict_snapshots_{pd.Timestamp(target).strftime('%Y%m%d')}.csv"
            temporary_output = output.with_suffix(".tmp")
            snapshots.to_csv(temporary_output, index=False, float_format="%.17g", date_format="%Y-%m-%d")
            os.replace(temporary_output, output)
        finally:
            import gc
            gc.collect()
            cleanup_access_lock_files(temporary)
    return {
        "status": "PREDICT_SNAPSHOT_WRITTEN", "race_date": target,
        "races": int(snapshots["race_id"].nunique()), "starts": len(snapshots),
        "output": str(output), "state_files_modified": False,
    }
