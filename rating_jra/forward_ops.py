from __future__ import annotations

import argparse
import base64
import heapq
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import P0Config, file_sha256
from rating_data import load_raw_from_db, prepare_input
from rating_engine import BatchP0Engine
from rating_evaluation import DEBUT_TYPE_LABELS
from rating_pedigree import build_or_load_base, build_or_load_features, load_pedigree_ids, standardization_matches
from stage3_evaluate import rule_for_p2
from stage3_io import combined_pedigree_status

BASE_DIR = Path(__file__).resolve().parent
EXPECTED_P2B_HASH = "3577cbc7a6a2c3e58bfdeda6666b9aa9d7369866f4c6fa49902dac6b8779dbe9"
OUTPUT_FILES = {
    "snapshots": "snapshots_2025plus.csv.gz",
    "posts": "post_updates_2025plus.csv.gz",
    "state": "state_latest.csv",
    "residual": "pedigree_residual_history_2025plus.csv.gz",
    "anchor": "mu_anchor_daily_2025plus.csv",
    "events": "result_events_2025plus.csv",
    "p0_state": "state_p0_residual_latest.csv",
    "p0_snapshots": "p0_residual_snapshots_2025plus.csv.gz",
}


class ForwardError(RuntimeError):
    pass


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = BASE_DIR / path
    if local.exists():
        return local
    project_relative = BASE_DIR.parent / path
    if project_relative.exists():
        return project_relative
    return local


def load_config(path: str | Path) -> dict:
    config_path = resolve_path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    expected = model["config_sha256"]
    if expected != EXPECTED_P2B_HASH:
        raise ForwardError("forward_config.yamlのP2bハッシュ固定値が不正です")
    frozen_path = resolve_path(model["config_path"])
    actual = file_sha256(frozen_path)
    if actual != EXPECTED_P2B_HASH:
        raise ForwardError(f"config_p2b.yamlのハッシュ不一致: {actual}")
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    if payload["p2"]["kappa_s"] != 20:
        raise ForwardError("凍結κが20ではありません")
    if not payload["p2"].get("standardization"):
        raise ForwardError("血統標準化定数がありません")
    p0_path = resolve_path(model["p0_config_path"])
    if file_sha256(p0_path) != model["p0_config_sha256"]:
        raise ForwardError("P0 configのハッシュが一致しません")
    config["_model"] = payload
    config["_p0"] = json.loads(p0_path.read_text(encoding="utf-8"))
    return config


def read_state(path: Path, expected_hash: str) -> pd.DataFrame:
    state = pd.read_csv(path)
    if "config_sha256" in state.columns and not state["config_sha256"].eq(expected_hash).all():
        raise ForwardError(f"状態ファイルのconfigハッシュが不正です: {path}")
    required = {"horse_id", "mu", "var", "var_cap", "mu0", "last_start_date", "n_graph_starts", "debut_type_code"}
    if not required <= set(state.columns):
        raise ForwardError(f"状態ファイルの必須列が不足しています: {path}")
    if state["mu0"].isna().any():
        raise ForwardError(f"状態ファイルのmu0が欠損しています: {path}")
    return state


def copy_and_relink_database(db_path: Path, temp_dir: Path, linked_tables: dict[str, str]) -> Path:
    destination = temp_dir.resolve() / db_path.name
    if not linked_tables:
        return destination
    linked_dir = temp_dir.resolve() / "linked"
    linked_dir.mkdir(exist_ok=True)
    remapped = {}
    for table, source_value in linked_tables.items():
        source = resolve_path(source_value)
        if not source.is_file():
            raise ForwardError(f"リンク先DBがありません: {table}: {source}")
        before = file_sha256(source)
        target = linked_dir / source.name
        shutil.copyfile(source, target)
        if file_sha256(source) != before:
            raise ForwardError(f"リンク先DBコピー中に元DBが変更されました: {table}")
        if file_sha256(target) != before:
            raise ForwardError(f"リンク先DBコピーのハッシュが一致しません: {table}")
        remapped[table] = target.name
    table_json = base64.b64encode(json.dumps(remapped, ensure_ascii=False).encode("utf-8")).decode("ascii")
    script = f"""
$ErrorActionPreference = 'Stop'
$app = New-Object -ComObject Access.Application
$app.Visible = $false
try {{
    $db = $app.DBEngine.OpenDatabase('{destination}', $false, $false)
    $tables = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{table_json}')) | ConvertFrom-Json
    foreach ($tableDef in $db.TableDefs) {{
        $property = $tables.PSObject.Properties[$tableDef.Name]
        if ($null -ne $property) {{
            $tableDef.Connect = ';DATABASE=' + (Join-Path '{linked_dir}' $property.Value)
        }}
    }}
    $db.Close()
}} finally {{
    $app.Quit()
}}
"""
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-EncodedCommand", encoded], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise ForwardError(f"一時DBのリンク付け替えに失敗しました: {result.stdout}{result.stderr}")
    return destination


def copy_database(db_path: Path, temp_dir: Path, linked_tables: dict[str, str] | None = None) -> Path:
    if not db_path.is_file():
        raise ForwardError(f"入力DBがありません: {db_path}")
    before = file_sha256(db_path)
    destination = temp_dir / db_path.name
    shutil.copyfile(db_path, destination)
    after = file_sha256(db_path)
    if before != after:
        raise ForwardError("DBコピー中に元DBが変更されました。再度実行してください")
    if file_sha256(destination) != before:
        raise ForwardError("一時DBコピーのハッシュが一致しません")
    return copy_and_relink_database(destination, temp_dir, linked_tables or {})


def model_rule(config: dict):
    return P0Config(**config["_model"]["base_p1"]["base_p0"]["p0"]), rule_for_p2(config["_model"])


def initialize_forward_states(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = config["model"]
    out = resolve_path(config["output"]["directory"])
    p2b_path = out / OUTPUT_FILES["state"]
    p0_path = out / OUTPUT_FILES["p0_state"]
    if not p2b_path.exists():
        shutil.copyfile(resolve_path(model["frozen_p2b_state_path"]), p2b_path)
    if not p0_path.exists():
        shutil.copyfile(resolve_path(model["frozen_p0_state_path"]), p0_path)
    p2b = read_state(p2b_path, EXPECTED_P2B_HASH)
    p0 = read_state(p0_path, model["p0_config_sha256"])
    return p2b, p0


def prediction_state(config: dict) -> pd.DataFrame:
    path = resolve_path(config["output"]["directory"]) / OUTPUT_FILES["state"]
    if not path.exists():
        path = resolve_path(config["model"]["frozen_p2b_state_path"])
    return read_state(path, EXPECTED_P2B_HASH)


def prediction_p0_state(config: dict) -> pd.DataFrame:
    path = resolve_path(config["output"]["directory"]) / OUTPUT_FILES["p0_state"]
    if not path.exists():
        path = resolve_path(config["model"]["frozen_p0_state_path"])
    return read_state(path, config["model"]["p0_config_sha256"])


def run_continuation(raw, source_hash, selected, state, initial_rule, provider, model_name, config_hash, dry_run=False, race_index_start=0):
    prepared = prepare_input(raw, source_hash=source_hash, allow_future=True, allow_unranked=dry_run)
    for index, race in enumerate(prepared.races):
        race.race_index = race_index_start + index
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
            raise ForwardError(f"新馬戦に過去グラフ出走馬があります: {race.race_id}")
    prepared.horse_ids = union_ids
    engine = BatchP0Engine(prepared, [selected], record=True, initial_rule=initial_rule, feature_provider=provider, dry_run=dry_run)
    engine.model_name = model_name
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
    result = engine.run()
    return prepared, result, engine


def add_elo(snapshots: pd.DataFrame) -> pd.DataFrame:
    result = snapshots.copy()
    result["elo_pre"] = 1500 + (400 / np.log(10)) * (result["mu_pre"] - result["mu_bar_D"])
    return result


def enrich_snapshots(result, engine, model, config_hash):
    snapshots = result.snapshots.copy()
    snapshots["prior_type"] = model
    snapshots["debut_type"] = snapshots["debut_type_code"].map(DEBUT_TYPE_LABELS)
    snapshots["X_sire"] = snapshots["X_sire_win"]
    snapshots["X_bms"] = snapshots["X_bms_win"]
    snapshots["pedigree_status"] = combined_pedigree_status(snapshots["sire_status_code"], snapshots["bms_status_code"])
    snapshots["config_sha256"] = config_hash
    return add_elo(snapshots)


def state_frame(result, engine, model, config_hash):
    state = result.state.copy()
    state["mu0"] = engine.mu0_state[0]
    state["last_start_date"] = pd.to_datetime(state["last_start_date"], unit="D", origin="unix").dt.strftime("%Y-%m-%d")
    state["debut_type"] = state["debut_type_code"].map(DEBUT_TYPE_LABELS)
    state["prior_type"] = model
    state["config_sha256"] = config_hash
    return state


def load_events(out: Path) -> pd.DataFrame:
    path = out / OUTPUT_FILES["events"]
    if not path.exists():
        return pd.DataFrame(columns=["race_id", "race_date", "result_version", "config_sha256"])
    events = pd.read_csv(path)
    if not events["config_sha256"].eq(EXPECTED_P2B_HASH).all():
        raise ForwardError("既存更新イベントのconfigハッシュが不正です")
    if events["race_id"].duplicated().any():
        raise ForwardError("既存更新イベントにrace_idの重複があります")
    return events


def filter_new_races(raw: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return raw
    versions = events.drop_duplicates("race_id").set_index("race_id")["result_version"].to_dict()
    for race_id, group in raw.groupby("race_id"):
        if race_id in versions:
            current = set(group["result_version"].astype(str))
            if current != {versions[race_id]}:
                raise ForwardError(f"処理済みレースの結果版が変わりました: {race_id}")
    return raw[~raw["race_id"].isin(events["race_id"].to_numpy(dtype=np.int64))].copy()


def append_frame(existing_path: Path, new: pd.DataFrame, keys: list[str], sort_columns: list[str]) -> pd.DataFrame:
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        if existing.duplicated(keys).any():
            raise ForwardError(f"既存出力に重複があります: {existing_path.name}")
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new.copy()
    if combined.duplicated(keys).any():
        raise ForwardError(f"追記後のキーが重複しています: {existing_path.name}")
    for column in sort_columns:
        if column in {"race_date", "asof_date"}:
            combined[column] = pd.to_datetime(combined[column])
    return combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def backup_outputs(out: Path, backup_root: Path, run_id: str) -> Path:
    backup_dir = backup_root / run_id
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in list(OUTPUT_FILES.values()) + ["latest_run_manifest.json"]:
        path = out / name
        if path.exists():
            shutil.copy2(path, backup_dir / name)
    return backup_dir


def restore_backup(out: Path, backup_dir: Path) -> None:
    for name in list(OUTPUT_FILES.values()) + ["latest_run_manifest.json"]:
        path = out / name
        backup = backup_dir / name
        if backup.exists():
            shutil.copyfile(backup, path)
        elif path.exists():
            path.unlink()


def write_outputs(staging: Path, outputs: dict[str, pd.DataFrame]) -> None:
    for key, name in OUTPUT_FILES.items():
        frame = outputs[key]
        path = staging / name
        if name.endswith(".gz"):
            frame.to_csv(path, index=False, compression="gzip", float_format="%.17g", date_format="%Y-%m-%d")
        else:
            frame.to_csv(path, index=False, float_format="%.17g", date_format="%Y-%m-%d")


def commit_outputs(staging: Path, out: Path) -> dict[str, str]:
    hashes = {}
    for name in OUTPUT_FILES.values():
        source = staging / name
        if not source.exists():
            raise ForwardError(f"出力が生成されていません: {name}")
        hashes[name] = file_sha256(source)
        os.replace(source, out / name)
    return hashes


def update(config: dict, end_date: str | None, raw_loader=load_raw_from_db) -> dict:
    out = resolve_path(config["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    selected, rule = model_rule(config)
    p0 = P0Config(**config["_p0"]["p0"])
    model = config["model"]
    start_date = config["processing"].get("start_date", "2025-01-01")
    end_date = end_date or config["processing"].get("end_date")
    end_exclusive = None
    if end_date:
        end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    work_root = resolve_path(config["output"]["temp_directory"])
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as temp:
        db_copy = copy_database(resolve_path(config["database"]["path"]), Path(temp), config["database"].get("linked_tables", {}))
        source_hash = file_sha256(db_copy)
        raw = raw_loader(db_path=db_copy, start_date=start_date, end_date=end_exclusive, final_only=True)
        events = load_events(out)
        raw = filter_new_races(raw, events)
        if raw.empty:
            return {"status": "NO_NEW_RACES", "new_races": 0, "run_id": run_id}
        p2b_state, p0_state = initialize_forward_states(config)
        backup_dir = backup_outputs(out, resolve_path(config["output"]["backup_directory"]), run_id)
        try:
            staging = Path(temp) / "staging"
            staging.mkdir()
            previous_p0_path = out / OUTPUT_FILES["p0_snapshots"]
            race_index_start = 0
            if previous_p0_path.exists():
                race_index_start = int(pd.read_csv(previous_p0_path, usecols=["race_index"])["race_index"].max()) + 1
            p0_prepared, p0_result, p0_engine = run_continuation(raw, source_hash, p0, p0_state, None, None, "P0", model["p0_config_sha256"], race_index_start=race_index_start)
            history = pd.read_csv(resolve_path(model["frozen_p0_snapshots_path"]))
            history["race_date"] = pd.to_datetime(history["race_date"])
            offset = int(history["race_index"].max()) + 1
            if previous_p0_path.exists():
                previous_p0 = pd.read_csv(previous_p0_path)
                previous_p0["race_date"] = pd.to_datetime(previous_p0["race_date"])
                previous_p0["race_index"] += offset
                history = pd.concat([history, previous_p0], ignore_index=True)
            future_p0 = p0_result.snapshots.copy()
            future_p0["race_date"] = pd.to_datetime(future_p0["race_date"])
            future_p0["race_index"] += offset
            combined = pd.concat([history, future_p0], ignore_index=True)
            residual_input = Path(temp) / "p0_residual_input.csv.gz"
            combined.to_csv(residual_input, index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
            base = build_or_load_base(residual_input, db_copy, use_cache=False, pedigree_loader=lambda path: load_pedigree_ids(path, fallback_to_horse_master=True))
            features = build_or_load_features(base, 20, use_cache=False)
            if not standardization_matches(features.constants, config["_model"]["p2"]["standardization"]):
                raise ForwardError("血統標準化定数が凍結値と一致しません")
            current_race_indices = set(p0_result.snapshots["race_index"].to_numpy(dtype=np.int64).tolist())
            provider = {int(index - offset): block for index, block in features.provider.items() if int(index - offset) in current_race_indices}
            prepared, result, engine = run_continuation(raw, source_hash, selected, p2b_state, rule, provider, "P2b", EXPECTED_P2B_HASH, race_index_start=race_index_start)
            snapshots = enrich_snapshots(result, engine, "P2b", EXPECTED_P2B_HASH)
            snapshots["source_db_sha256"] = source_hash
            posts = snapshots[["race_id", "race_date", "horse_id", "mu_post", "var_post", "config_sha256", "source_db_sha256"]].copy()
            p0_snapshots = enrich_snapshots(p0_result, p0_engine, "P0", model["p0_config_sha256"])
            p0_snapshots["source_db_sha256"] = source_hash
            new_events = pd.DataFrame({
                "event_id": [f"P2b::{race.race_id}::{source_hash[:16]}" for race in prepared.races],
                "race_id": [race.race_id for race in prepared.races],
                "race_date": [pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date) for race in prepared.races],
                "result_version": [race.result_version for race in prepared.races],
                "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_db_sha256": source_hash,
                "config_sha256": EXPECTED_P2B_HASH,
            })
            daily = result.daily.copy()
            daily["asof_date"] = pd.to_datetime(daily["asof_date"], unit="D", origin="unix")
            residual = base.history[base.history["race_date"] >= pd.Timestamp("2025-01-01")].copy()
            residual["config_sha256"] = model["p0_config_sha256"]
            outputs = {
                "snapshots": append_frame(out / OUTPUT_FILES["snapshots"], snapshots, ["race_id", "horse_id"], ["race_date", "race_id", "horse_id"]),
                "posts": append_frame(out / OUTPUT_FILES["posts"], posts, ["race_id", "horse_id"], ["race_date", "race_id", "horse_id"]),
                "state": state_frame(result, engine, "P2b", EXPECTED_P2B_HASH),
                "residual": residual.sort_values(["race_date", "race_index", "horse_id"]).reset_index(drop=True),
                "anchor": append_frame(out / OUTPUT_FILES["anchor"], daily, ["asof_date"], ["asof_date"]),
                "events": append_frame(out / OUTPUT_FILES["events"], new_events, ["race_id"], ["race_date", "race_id"]),
                "p0_state": state_frame(p0_result, p0_engine, "P0", model["p0_config_sha256"]),
                "p0_snapshots": append_frame(out / OUTPUT_FILES["p0_snapshots"], p0_snapshots, ["race_id", "horse_id"], ["race_date", "race_id", "horse_id"]),
            }
            write_outputs(staging, outputs)
            hashes = commit_outputs(staging, out)
        except Exception:
            restore_backup(out, backup_dir)
            raise
    manifest = {
        "status": "UPDATED",
        "run_id": run_id,
        "new_races": len(raw["race_id"].unique()),
        "first_race_date": str(raw["race_date"].min().date()),
        "last_race_date": str(raw["race_date"].max().date()),
        "config_sha256": EXPECTED_P2B_HASH,
        "backup": str(backup_dir),
        "output_hashes": hashes,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / f"run_manifest_{run_id}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "latest_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def rebuild(config: dict, end_date: str | None, raw_loader=load_raw_from_db) -> dict:
    out = resolve_path(config["output"]["directory"])
    if out.exists():
        archive_root = resolve_path(config["output"].get("archive_directory", "p2b_forward_legacy"))
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = archive_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        shutil.move(str(out), str(archive))
    return update(config, end_date, raw_loader=raw_loader)


def predict_snapshot(config: dict, predict_date: str | None, raw_loader=load_raw_from_db) -> dict:
    selected, rule = model_rule(config)
    out = resolve_path(config["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    predict_date = predict_date or config["processing"].get("predict_date")
    start = predict_date or datetime.now().strftime("%Y-%m-%d")
    end_exclusive = (pd.Timestamp(start) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if predict_date else None
    work_root = resolve_path(config["output"]["temp_directory"])
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as temp:
        db_copy = copy_database(resolve_path(config["database"]["path"]), Path(temp), config["database"].get("linked_tables", {}))
        source_hash = file_sha256(db_copy)
        raw = raw_loader(db_path=db_copy, start_date=start, end_date=end_exclusive, final_only=False)
        if raw.empty:
            raise ForwardError("予測対象のレースがありません")
        if predict_date is None:
            target_date = raw["race_date"].min().strftime("%Y-%m-%d")
            raw = raw[raw["race_date"].eq(target_date)]
        else:
            target_date = pd.Timestamp(predict_date).strftime("%Y-%m-%d")
            raw = raw[raw["race_date"].eq(target_date)]
        if raw.empty:
            raise ForwardError(f"指定日のレースがありません: {target_date}")
        state = prediction_state(config)
        p0_state = prediction_p0_state(config)
        p0_selected = P0Config(**config["_p0"]["p0"])
        p0_forward_path = out / OUTPUT_FILES["p0_snapshots"]
        race_index_start = 0
        if p0_forward_path.exists():
            race_index_start = int(pd.read_csv(p0_forward_path, usecols=["race_index"])["race_index"].max()) + 1
        _, p0_result, _ = run_continuation(raw, source_hash, p0_selected, p0_state, None, None, "P0", config["model"]["p0_config_sha256"], dry_run=True, race_index_start=race_index_start)
        history = pd.read_csv(resolve_path(config["model"]["frozen_p0_snapshots_path"]))
        history["race_date"] = pd.to_datetime(history["race_date"])
        offset = int(history["race_index"].max()) + 1
        if p0_forward_path.exists():
            forward_p0 = pd.read_csv(p0_forward_path)
            forward_p0["race_date"] = pd.to_datetime(forward_p0["race_date"])
            forward_p0["race_index"] += offset
            history = pd.concat([history, forward_p0], ignore_index=True)
        future_p0 = p0_result.snapshots.copy()
        future_p0["race_date"] = pd.to_datetime(future_p0["race_date"])
        future_p0["race_index"] += offset
        combined = pd.concat([history, future_p0], ignore_index=True)
        residual_input = Path(temp) / "p0_residual_input.csv.gz"
        combined.to_csv(residual_input, index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
        base = build_or_load_base(residual_input, db_copy, use_cache=False, pedigree_loader=lambda path: load_pedigree_ids(path, fallback_to_horse_master=True))
        features = build_or_load_features(base, 20, use_cache=False)
        if not standardization_matches(features.constants, config["_model"]["p2"]["standardization"]):
            raise ForwardError("血統標準化定数が凍結値と一致しません")
        current_race_indices = set(p0_result.snapshots["race_index"].to_numpy(dtype=np.int64).tolist())
        provider = {int(index - offset): block for index, block in features.provider.items() if int(index - offset) in current_race_indices}
        _, result, engine = run_continuation(raw, source_hash, selected, state, rule, provider, "P2b", EXPECTED_P2B_HASH, dry_run=True, race_index_start=race_index_start)
        snapshots = enrich_snapshots(result, engine, "P2b", EXPECTED_P2B_HASH)
        snapshots["source_db_sha256"] = source_hash
        snapshots = snapshots.drop(columns=["mu_post", "var_post", "skip_mu_var_update"])
        output_path = out / f"predict_snapshots_{pd.Timestamp(target_date).strftime('%Y%m%d')}.csv"
        temporary = output_path.with_suffix(".tmp")
        snapshots.to_csv(temporary, index=False, float_format="%.17g", date_format="%Y-%m-%d")
        os.replace(temporary, output_path)
    return {
        "status": "PREDICT_SNAPSHOT_WRITTEN",
        "race_date": target_date,
        "races": int(snapshots["race_id"].nunique()),
        "starts": len(snapshots),
        "output": str(output_path),
        "state_files_modified": False,
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["update", "predict_snapshot", "rebuild"])
    parser.add_argument("--config", default="forward_config.yaml")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--config-hash", default=None)
    args = parser.parse_args()
    try:
        if args.config_hash is not None and args.config_hash != EXPECTED_P2B_HASH:
            raise ForwardError("指定されたconfigハッシュが一致しません")
        config = load_config(args.config)
        if args.mode == "update":
            result = update(config, args.end_date)
        elif args.mode == "rebuild":
            result = rebuild(config, args.end_date)
        else:
            result = predict_snapshot(config, args.date)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "ERROR", "mode": args.mode, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main_cli()
