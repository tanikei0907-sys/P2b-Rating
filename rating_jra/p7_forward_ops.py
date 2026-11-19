from __future__ import annotations

import heapq
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forward_ops import append_frame, copy_database, enrich_snapshots, filter_new_races, read_state, resolve_path, state_frame
from p4_forward_ops import build_margins, initialize_history, load_raw_with_times, pyodbc_connect
from p4_margin.margins import multiplier_arrays
from p7_forward_engine import P7ProductionEngine, interaction_history_frame
from rating_core import P0Config, file_sha256
from rating_data import prepare_input
from rating_pedigree import load_pedigree_ids
from stage3_evaluate import rule_for_p2
from viewer_data import cleanup_access_lock_files

BASE = Path(__file__).resolve().parent
EXPECTED_P7_HASH = "3246e20ccee1102638213235d6b9e6f373079de6165fda16ddc3ae76d4b9e174"
EXPECTED_P6_HASH = "b433360ecf33afe503b3ca73cac8da57f2c0c3eab30462f538a4dab95867a730"
EXPECTED_P4_HASH = "436f08ffb840a60ef33354fc28fa9434d0193a0460c73ba8c8f29d8038a26ba1"
EXPECTED_P2B_HASH = "3577cbc7a6a2c3e58bfdeda6666b9aa9d7369866f4c6fa49902dac6b8779dbe9"
SCALE = 400 / np.log(10)
OUTPUT_FILES = {
    "snapshots": "snapshots_2025plus.csv.gz",
    "posts": "post_updates_2025plus.csv.gz",
    "state": "state_latest.csv",
    "residual": "pedigree_residual_history_2025plus.csv.gz",
    "anchor": "mu_anchor_daily_2025plus.csv",
    "events": "result_events_2025plus.csv",
    "propagation": "propagation_events_2025plus.csv.gz",
    "interactions": "interaction_history_latest.csv.gz",
    "margins": "margins_2025plus.csv.gz",
    "multipliers": "margin_multipliers_2025plus.csv.gz",
}
PROPAGATION_COLUMNS = [
    "race_date", "source_horse_id", "target_horse_id", "source_race_id", "source_delta_mu",
    "mean_relative_weight", "applied_delta_mu", "past_race_ids", "latest_past_date",
]


class P7ForwardError(RuntimeError):
    pass


def load_config(path: str | Path = "p7_forward/forward_config.yaml") -> dict:
    config_path = resolve_path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    if model["config_sha256"] != EXPECTED_P7_HASH:
        raise P7ForwardError("forward_configのP7ハッシュ固定値が不正です")
    for key, expected in [("p7", EXPECTED_P7_HASH), ("p6", EXPECTED_P6_HASH), ("p4", EXPECTED_P4_HASH), ("p2b", EXPECTED_P2B_HASH)]:
        config_key = "config_path" if key == "p7" else f"{key}_config_path"
        if file_sha256(resolve_path(model[config_key])) != expected:
            raise P7ForwardError(f"{key.upper()} configのハッシュが不一致です")
    p7 = json.loads(resolve_path(model["config_path"]).read_text(encoding="utf-8"))
    propagation = p7["propagation"]
    if propagation != {"L": 1, "eta": 0.07500000000000001, "method": "A", "tau_prop": 30.0}:
        raise P7ForwardError("P7方式Aの凍結パラメータが不一致です")
    final = json.loads((BASE / "p7_propagation/final_model.json").read_text(encoding="utf-8"))
    if final["adopted_model"] != "P7" or final["config_sha256"] != EXPECTED_P7_HASH:
        raise P7ForwardError("P7が最終採用モデルとして確定されていません")
    config["_p7"] = p7
    config["_p6"] = json.loads(resolve_path(model["p6_config_path"]).read_text(encoding="utf-8"))
    config["_p4"] = json.loads(resolve_path(model["p4_config_path"]).read_text(encoding="utf-8"))
    config["_p2b"] = json.loads(resolve_path(model["p2b_config_path"]).read_text(encoding="utf-8"))
    if config["_p4"]["method"] != "A" or config["_p4"]["normalize"] or config["_p4"]["gamma"] != 2.0 or config["_p4"]["m_max"] != 2.0:
        raise P7ForwardError("P4着差倍率の凍結値が不正です")
    return config


def load_events(out: Path) -> pd.DataFrame:
    path = out / OUTPUT_FILES["events"]
    if not path.exists():
        return pd.DataFrame(columns=["race_id", "race_date", "result_version", "config_sha256"])
    events = pd.read_csv(path)
    if not events["config_sha256"].eq(EXPECTED_P7_HASH).all() or events["race_id"].duplicated().any():
        raise P7ForwardError("既存P7更新イベントが不正です")
    return events


def load_history(config: dict) -> pd.DataFrame:
    frames = [pd.read_csv(resolve_path(config["model"]["frozen_residual_history_path"]))]
    forward = resolve_path(config["output"]["directory"]) / OUTPUT_FILES["residual"]
    if forward.exists():
        frames.append(pd.read_csv(forward))
    history = pd.concat(frames, ignore_index=True)
    if history.duplicated(["race_index", "horse_id"]).any():
        raise P7ForwardError("P7残差履歴に重複があります")
    return history


def load_demographics(db: Path, horse_ids: np.ndarray, config: dict) -> pd.DataFrame:
    connection = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db};ReadOnly=1;"
    with pyodbc_connect(connection) as con:
        current = pd.read_sql("SELECT [血統登録番号] AS horse_id,[生年月日] AS birth_date,[性別コード] AS sex_code FROM [競走馬マスタ]", con)
    frozen = pd.read_csv(BASE / "p6_age/horse_demographics.csv", usecols=["horse_id", "birth_date", "sex_code"])
    combined = pd.concat([frozen, current], ignore_index=True)
    combined["horse_id"] = pd.to_numeric(combined["horse_id"], errors="coerce").astype("Int64")
    combined = combined.dropna(subset=["horse_id"]).drop_duplicates("horse_id", keep="first")
    combined["horse_id"] = combined["horse_id"].astype(np.int64)
    result = combined.set_index("horse_id").reindex(horse_ids).reset_index()
    result["birth_date"] = pd.to_datetime(result["birth_date"], errors="coerce")
    missing = result["birth_date"].isna()
    if missing.any():
        years = (result.loc[missing, "horse_id"].astype(np.int64) // 1_000_000).astype(int)
        result.loc[missing, "birth_date"] = pd.to_datetime(dict(year=years, month=1, day=1))
    result["sex_code"] = pd.to_numeric(result["sex_code"], errors="coerce").fillna(0).astype(np.int8)
    if result["birth_date"].isna().any():
        raise P7ForwardError("生年月日を復元できない馬があります")
    return result


def interaction_source(config: dict) -> Path:
    forward = resolve_path(config["output"]["directory"]) / OUTPUT_FILES["interactions"]
    return forward if forward.exists() else resolve_path(config["model"]["frozen_interaction_history_path"])


def restore_interactions(engine: P7ProductionEngine, frame: pd.DataFrame) -> None:
    if not frame.empty and not frame["config_sha256"].eq(EXPECTED_P7_HASH).all():
        raise P7ForwardError("対戦履歴のconfigハッシュが不正です")
    grouped = {}
    for row in frame.itertuples():
        opponents = {int(target): float(weight) for target, weight in json.loads(row.opponents_json).items()}
        grouped.setdefault(int(row.source_horse_id), []).append({
            "race_id": int(row.race_id), "date": int((pd.Timestamp(row.race_date) - pd.Timestamp("1970-01-01")).days),
            "opponents": opponents,
        })
    for source_id, history in grouped.items():
        if source_id not in engine.horse_index:
            raise P7ForwardError(f"対戦履歴の馬IDが状態にありません: {source_id}")
        source = engine.horse_index[source_id]
        converted = []
        for past in sorted(history, key=lambda value: (value["date"], value["race_id"]))[-engine.L:]:
            if past["date"] > engine.last_start[source]:
                raise P7ForwardError("対戦履歴が最終出走日より未来です")
            opponents = {}
            for target_id, weight in past["opponents"].items():
                if target_id not in engine.horse_index:
                    raise P7ForwardError(f"対戦相手の馬IDが状態にありません: {target_id}")
                opponents[engine.horse_index[target_id]] = weight
            converted.append({**past, "opponents": opponents})
        engine.interaction_history[source] = converted


def frozen_pending(config: dict) -> pd.Series:
    if "_frozen_pending" in config:
        return config["_frozen_pending"]
    state = pd.read_csv(resolve_path(config["model"]["frozen_state_path"]), usecols=["horse_id", "last_start_date"])
    last_dates = pd.to_datetime(state["last_start_date"])
    last_map = pd.Series(last_dates.to_numpy(), index=state["horse_id"].to_numpy(dtype=np.int64))
    totals = {}
    path = resolve_path(config["model"]["frozen_propagation_events_path"])
    for chunk in pd.read_csv(path, usecols=["target_horse_id", "race_date", "applied_delta_mu"], chunksize=500000, parse_dates=["race_date"]):
        eligible = chunk[chunk["race_date"].gt(chunk["target_horse_id"].map(last_map))]
        for horse_id, value in eligible.groupby("target_horse_id")["applied_delta_mu"].sum().items():
            totals[int(horse_id)] = totals.get(int(horse_id), 0.0) + float(value)
    result = pd.Series(totals, dtype=float)
    config["_frozen_pending"] = result
    return result


def restore_pending(engine: P7ProductionEngine, state: pd.DataFrame, config: dict) -> None:
    if "propagation_pending" in state.columns:
        values = pd.to_numeric(state["propagation_pending"], errors="raise").to_numpy(dtype=float)
    else:
        values = state["horse_id"].map(frozen_pending(config)).fillna(0.0).to_numpy(dtype=float)
    for horse_id, value in zip(state["horse_id"], values):
        engine.pending_propagation[engine.horse_index[int(horse_id)]] = float(value)


def run_forward(raw, source_hash: str, config: dict, state: pd.DataFrame, history: pd.DataFrame, pedigree: pd.DataFrame, demographics: pd.DataFrame, dry_run: bool = False, race_index_start: int = 0):
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
            raise P7ForwardError(f"新馬戦に過去グラフ出走馬があります: {race.race_id}")
    prepared.horse_ids = union_ids
    p2b = config["_p2b"]
    p0 = P0Config(**p2b["base_p1"]["base_p0"]["p0"])
    rule = rule_for_p2(p2b)
    stages, scales = multiplier_arrays(margins, prepared, config["_p4"])
    engine = P7ProductionEngine(
        prepared, p0, rule, pedigree, p2b["p2"]["standardization"], stages,
        config["_p6"], demographics, config["_p7"]["propagation"], True,
    )
    required = {"mu_base", "var_base", "delta_turf", "delta_dirt", "var_turf", "var_dirt"}
    if not required <= set(state.columns):
        raise P7ForwardError("P7状態の馬場別列が不足しています")
    for row in state.itertuples():
        index = state_index[int(row.horse_id)]
        engine.mu[0, index] = float(row.mu_base)
        engine.var[0, index] = float(row.var_base)
        engine.var_cap[0, index] = float(row.var_cap)
        engine.surface_delta[0, index] = float(row.delta_turf)
        engine.surface_delta[1, index] = float(row.delta_dirt)
        engine.var_surface[0, index] = float(row.var_turf)
        engine.var_surface[1, index] = float(row.var_dirt)
        engine.seen[index] = True
        engine.last_start[index] = int((pd.Timestamp(row.last_start_date) - pd.Timestamp("1970-01-01")).days)
        engine.n_starts[index] = int(row.n_graph_starts)
        engine.debut_types[index] = int(row.debut_type_code)
        engine.mu0_state[0, index] = float(row.mu0)
        engine.active[index] = True
        heapq.heappush(engine.expiry_heap, (int(engine.last_start[index]) + 365, index, int(engine.last_start[index])))
    initialize_history(engine, history)
    restore_interactions(engine, pd.read_csv(interaction_source(config)))
    restore_pending(engine, state, config)
    result = engine.run()
    if engine.history.empty:
        engine.history = history.iloc[0:0].copy()
    else:
        engine.history["race_id"] = engine.history["race_index"].map({race.race_index: race.race_id for race in prepared.races})
    if race_index_start:
        result.snapshots["race_index"] += race_index_start
        if not engine.history.empty:
            engine.history["race_index"] += race_index_start
    return prepared, result, engine, margins, stages, scales


def add_derived_elo(snapshots: pd.DataFrame) -> pd.DataFrame:
    result = snapshots.copy()
    result["elo_base_pre"] = 1500 + SCALE * (result["mu_base_pre"] - result["mu_bar_D"])
    result["elo_surface_adjustment"] = SCALE * result["surface_offset_pre"]
    result["elo_propagation_adjustment"] = SCALE * result["propagation_correction_pre"]
    return result


def write_outputs(staging: Path, outputs: dict[str, pd.DataFrame]) -> None:
    for key, name in OUTPUT_FILES.items():
        outputs[key].to_csv(staging / name, index=False, compression="gzip" if name.endswith(".gz") else None, float_format="%.17g", date_format="%Y-%m-%d")


def backup_existing(out: Path, backup_root: Path, run_id: str) -> tuple[Path, set[str]]:
    backup = backup_root / run_id
    backup.mkdir(parents=True, exist_ok=False)
    existing = set()
    for name in list(OUTPUT_FILES.values()) + ["latest_run_manifest.json"]:
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
    manifest = out / "latest_run_manifest.json"
    archived_manifest = backup / manifest.name
    if manifest.name in existing:
        shutil.copyfile(archived_manifest, manifest)
    elif manifest.exists():
        manifest.unlink()


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
                return {"status": "NO_NEW_RACES", "new_races": 0, "run_id": run_id, "config_sha256": EXPECTED_P7_HASH}
            if not events.empty and pd.to_datetime(raw["race_date"]).min() <= pd.to_datetime(events["race_date"]).max():
                raise P7ForwardError("処理済み日以前のレースが新規に追加されました。単方向更新を停止します")
            state_path = out / OUTPUT_FILES["state"]
            state = read_state(state_path if state_path.exists() else resolve_path(config["model"]["frozen_state_path"]), EXPECTED_P7_HASH)
            history = load_history(config)
            pedigree = load_pedigree_ids(db, fallback_to_horse_master=True)
            demographics = load_demographics(db, np.sort(np.union1d(raw["horse_id"].to_numpy(dtype=np.int64), state["horse_id"].to_numpy(dtype=np.int64))), config)
            snapshots_path = out / OUTPUT_FILES["snapshots"]
            race_index_start = int(pd.read_csv(snapshots_path, usecols=["race_index"])["race_index"].max()) + 1 if snapshots_path.exists() else 0
            prepared, result, engine, margins, stages, scales = run_forward(raw, source_hash, config, state, history, pedigree, demographics, race_index_start=race_index_start)
            snapshots = add_derived_elo(enrich_snapshots(result, engine, "P7", EXPECTED_P7_HASH))
            snapshots["source_db_sha256"] = source_hash
            margin_index = margins.set_index("race_id")
            snapshots["winner_margin_seconds"] = snapshots["race_id"].map(margin_index["stage_1"])
            snapshots["winner_margin_normalized"] = snapshots["winner_margin_seconds"] / (snapshots["distance"] / 1000.0)
            snapshots["race_margin_scale"] = 1.0
            posts = snapshots[["race_id", "race_date", "horse_id", "mu_post", "var_post", "config_sha256", "source_db_sha256"]].copy()
            daily = result.daily.copy()
            daily["asof_date"] = pd.to_datetime(daily["asof_date"], unit="D", origin="unix")
            daily["config_sha256"] = EXPECTED_P7_HASH
            daily["source_db_sha256"] = source_hash
            residual = engine.history.copy()
            residual = residual[residual["race_date"].ge("2025-01-01")].copy()
            residual["config_sha256"] = EXPECTED_P7_HASH
            residual["source_db_sha256"] = source_hash
            propagation = pd.DataFrame(engine.propagation_events, columns=PROPAGATION_COLUMNS)
            propagation["config_sha256"] = EXPECTED_P7_HASH
            propagation["source_db_sha256"] = source_hash
            interactions = interaction_history_frame(engine, EXPECTED_P7_HASH, source_hash)
            now = datetime.now(timezone.utc).isoformat()
            new_events = pd.DataFrame({
                "event_id": [f"P7::{race.race_id}::{source_hash[:16]}" for race in prepared.races],
                "race_id": [race.race_id for race in prepared.races],
                "race_date": [pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date) for race in prepared.races],
                "result_version": [race.result_version for race in prepared.races],
                "processed_at_utc": now, "input_db_sha256": source_hash, "config_sha256": EXPECTED_P7_HASH,
            })
            margins["config_sha256"] = EXPECTED_P7_HASH
            margins["source_db_sha256"] = source_hash
            multipliers = pd.DataFrame({
                "race_id": [race.race_id for race in prepared.races],
                **{f"stage_multiplier_{index}": stages[:, index - 1] for index in range(1, 6)},
                "race_scale": scales, "config_sha256": EXPECTED_P7_HASH, "source_db_sha256": source_hash,
            })
            latest_state = state_frame(result, engine, "P7", EXPECTED_P7_HASH)
            latest_state["propagation_pending"] = engine.pending_propagation
            latest_state["source_db_sha256"] = source_hash
            outputs = {
                "snapshots": append_frame(snapshots_path, snapshots, ["race_id", "horse_id"], ["race_date", "race_id", "horse_id"]),
                "posts": append_frame(out / OUTPUT_FILES["posts"], posts, ["race_id", "horse_id"], ["race_date", "race_id", "horse_id"]),
                "state": latest_state,
                "residual": append_frame(out / OUTPUT_FILES["residual"], residual, ["race_id", "horse_id"], ["race_date", "race_id", "horse_id"]),
                "anchor": append_frame(out / OUTPUT_FILES["anchor"], daily, ["asof_date"], ["asof_date"]),
                "events": append_frame(out / OUTPUT_FILES["events"], new_events, ["race_id"], ["race_date", "race_id"]),
                "propagation": append_frame(out / OUTPUT_FILES["propagation"], propagation, ["race_date", "source_horse_id", "target_horse_id"], ["race_date", "source_race_id", "source_horse_id", "target_horse_id"]),
                "interactions": interactions,
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
        "status": "UPDATED", "model": "P7", "run_id": run_id, "new_races": int(raw["race_id"].nunique()),
        "new_starts": int(len(snapshots)), "first_race_date": str(pd.to_datetime(raw["race_date"]).min().date()),
        "last_race_date": str(pd.to_datetime(raw["race_date"]).max().date()),
        "state_horses": int(len(latest_state)), "interaction_rows": int(len(interactions)),
        "new_propagation_events": int(len(propagation)), "config_sha256": EXPECTED_P7_HASH, "input_db_sha256": source_hash,
        "backup": str(backup), "output_hashes": hashes, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
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
                raise P7ForwardError("予測対象のレースがありません")
            target = predict_date or raw["race_date"].min().strftime("%Y-%m-%d")
            raw = raw[raw["race_date"].eq(target)].copy()
            if raw.empty:
                raise P7ForwardError(f"指定日のレースがありません: {target}")
            events = load_events(out)
            if not events.empty and pd.Timestamp(target) <= pd.to_datetime(events["race_date"]).max():
                raise P7ForwardError("予測日が最新確定日以前です。過去の確定レースへの将来状態の適用は行いません")
            state_path = out / OUTPUT_FILES["state"]
            state = read_state(state_path if state_path.exists() else resolve_path(config["model"]["frozen_state_path"]), EXPECTED_P7_HASH)
            history = load_history(config)
            pedigree = load_pedigree_ids(db, fallback_to_horse_master=True)
            demographics = load_demographics(db, np.sort(np.union1d(raw["horse_id"].to_numpy(dtype=np.int64), state["horse_id"].to_numpy(dtype=np.int64))), config)
            _, result, engine, _, _, _ = run_forward(raw, source_hash, config, state, history, pedigree, demographics, dry_run=True)
            snapshots = add_derived_elo(enrich_snapshots(result, engine, "P7", EXPECTED_P7_HASH))
            snapshots["source_db_sha256"] = source_hash
            snapshots = snapshots.drop(columns=[
                "mu_post", "var_post", "skip_mu_var_update", "mu_base_post", "surface_offset_post", "var_base_post", "var_surface_post",
            ])
            output = out / f"predict_snapshots_{pd.Timestamp(target).strftime('%Y%m%d')}.csv"
            temporary_output = output.with_suffix(".tmp")
            snapshots.to_csv(temporary_output, index=False, float_format="%.17g", date_format="%Y-%m-%d")
            os.replace(temporary_output, output)
        finally:
            import gc
            gc.collect()
            cleanup_access_lock_files(temporary)
    return {
        "status": "PREDICT_SNAPSHOT_WRITTEN", "model": "P7", "race_date": target,
        "races": int(snapshots["race_id"].nunique()), "starts": len(snapshots),
        "output": str(output), "config_sha256": EXPECTED_P7_HASH, "state_files_modified": False,
    }
