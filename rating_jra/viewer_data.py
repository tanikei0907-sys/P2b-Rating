from __future__ import annotations

import gc
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import file_sha256

BASE_DIR = Path(__file__).resolve().parent
EXPECTED_P2B_HASH = "3577cbc7a6a2c3e58bfdeda6666b9aa9d7369866f4c6fa49902dac6b8779dbe9"
START_DATE = pd.Timestamp("2017-01-01")
SCALE = 400 / np.log(10)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def load_viewer_config(path: str | Path = "viewer_config.yaml") -> dict:
    config_path = resolve_path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = BASE_DIR / "stage3_output/config_p2b.yaml"
    actual = file_sha256(model_path)
    if actual != EXPECTED_P2B_HASH:
        raise ValueError(f"config_p2b.yamlのハッシュが一致しません: {actual}")
    return config


def input_paths(config: dict) -> dict[str, Path]:
    return {
        "snapshots": BASE_DIR / "stage3_output/p2b/snapshots.csv.gz",
        "posts": BASE_DIR / "stage3_output/p2b/post_updates.csv.gz",
        "anchor": BASE_DIR / "stage3_output/p2b/mu_anchor_daily.csv",
        "forward_snapshots": BASE_DIR / "p2b_forward/snapshots_2025plus.csv.gz",
        "forward_posts": BASE_DIR / "p2b_forward/post_updates_2025plus.csv.gz",
        "forward_anchor": BASE_DIR / "p2b_forward/mu_anchor_daily_2025plus.csv",
        "p0_snapshots": BASE_DIR / "stage2_output/snapshots.csv.gz",
        "forward_p0_snapshots": BASE_DIR / "p2b_forward/p0_residual_snapshots_2025plus.csv.gz",
        "sire_auxiliary": BASE_DIR / "stage3_output/sire_auxiliary.csv",
        "bms_auxiliary": BASE_DIR / "stage3_output/bms_auxiliary.csv",
        "residual_history": BASE_DIR / "stage3_output/pedigree_residual_history.csv.gz",
        "forward_residual": BASE_DIR / "p2b_forward/pedigree_residual_history_2025plus.csv.gz",
    }


def source_fingerprint(config: dict) -> dict:
    paths = input_paths(config)
    paths["p2b_config"] = BASE_DIR / "stage3_output/config_p2b.yaml"
    paths["viewer_config"] = resolve_path("viewer_config.yaml")
    paths["viewer_data"] = Path(__file__)
    paths["viewer_app"] = BASE_DIR / "rating_viewer.py"
    paths["predict_snapshots"] = sorted((BASE_DIR / "p2b_forward").glob("predict_snapshots_*.csv"))
    records = {}
    for key, value in paths.items():
        if isinstance(value, list):
            records[key] = [fingerprint_file(path) for path in value]
        elif value.exists():
            records[key] = fingerprint_file(value)
        else:
            records[key] = None
    return records


def fingerprint_file(path: Path) -> dict:
    try:
        display_path = path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        display_path = path.name
    return {"path": display_path, "size": path.stat().st_size, "sha256": file_sha256(path)}


def cache_paths(config: dict) -> tuple[Path, Path, Path]:
    cache_dir = resolve_path(config["paths"]["cache_dir"])
    return (
        cache_dir / "viewer_merged_2017plus.parquet",
        cache_dir / "viewer_auxiliary.parquet",
        cache_dir / "viewer_source_fingerprint.json",
    )


def read_snapshot(path: Path, record_type: str) -> pd.DataFrame:
    columns = [
        "race_id", "race_date", "horse_id", "mu_pre", "var_pre_persistent", "var_pre_eff", "m_pre",
        "days_since_start", "n_graph_starts", "mu0", "debut_type", "n_field_original", "n_field_retained",
        "n_removed_nonJRA", "condition_code", "surface_code", "distance", "mu_bar_D", "mu_post", "var_post",
        "X_sire_win", "X_sire_top3", "X_bms_win", "X_bms_top3", "N_sire", "N_bms", "pedigree_status",
        "sire_status", "bms_status", "config_sha256", "source_db_sha256", "elo_pre", "race_index",
    ]
    frame = pd.read_csv(path, usecols=lambda column: column in columns)
    frame["race_date"] = pd.to_datetime(frame["race_date"])
    frame = frame[frame["race_date"].ge(START_DATE)].copy()
    frame["record_type"] = record_type
    if "mu_post" not in frame.columns:
        frame["mu_post"] = np.nan
    if "var_post" not in frame.columns:
        frame["var_post"] = np.nan
    if "mu_bar_D" not in frame.columns:
        frame["mu_bar_D"] = np.nan
    return merge_snapshot_post(frame, None, None)


def merge_snapshot_post(snapshot: pd.DataFrame, post_path: Path | None, anchor_path: Path | None) -> pd.DataFrame:
    if post_path is not None and post_path.exists():
        posts = pd.read_csv(post_path, usecols=["race_id", "horse_id", "mu_post", "var_post"])
        snapshot = snapshot.drop(columns=["mu_post", "var_post"], errors="ignore").merge(
            posts, on=["race_id", "horse_id"], how="left", validate="one_to_one"
        )
    if anchor_path is not None and anchor_path.exists():
        anchor = pd.read_csv(anchor_path)
        anchor["asof_date"] = pd.to_datetime(anchor["asof_date"])
        extra = anchor[["asof_date", "n_active_365d", "active_latest_mu_std", "active_latest_mu_median"]]
        snapshot = snapshot.merge(extra, left_on="race_date", right_on="asof_date", how="left", validate="many_to_one")
    snapshot["mu_pre"] = pd.to_numeric(snapshot["mu_pre"], errors="coerce")
    snapshot["mu_post"] = pd.to_numeric(snapshot["mu_post"], errors="coerce")
    snapshot["var_pre_eff"] = pd.to_numeric(snapshot["var_pre_eff"], errors="coerce")
    if "elo_pre" not in snapshot.columns:
        snapshot["elo_pre"] = np.nan
    snapshot["elo_pre"] = snapshot["elo_pre"].fillna(1500 + SCALE * (snapshot["mu_pre"] - snapshot["mu_bar_D"]))
    snapshot["elo_post"] = 1500 + SCALE * (snapshot["mu_post"] - snapshot["mu_bar_D"])
    snapshot["elo_delta"] = snapshot["elo_post"] - snapshot["elo_pre"]
    snapshot["elo_sd"] = SCALE * np.sqrt(snapshot["var_pre_eff"])
    return snapshot


def load_p0_prior() -> pd.DataFrame:
    frames = []
    for path in [
        BASE_DIR / "stage2_output/snapshots.csv.gz",
        BASE_DIR / "p2b_forward/p0_residual_snapshots_2025plus.csv.gz",
    ]:
        if path.exists():
            frame = pd.read_csv(path, usecols=["race_id", "race_date", "horse_id", "mu_pre"])
            frame["race_date"] = pd.to_datetime(frame["race_date"])
            frames.append(frame[frame["race_date"].ge(START_DATE)])
    if not frames:
        return pd.DataFrame(columns=["race_id", "horse_id", "p0_mu_pre"])
    combined = pd.concat(frames, ignore_index=True)
    return combined[["race_id", "race_date", "horse_id", "mu_pre"]].rename(columns={"mu_pre": "p0_mu_pre"})


def cleanup_access_lock_files(directory: str | Path) -> None:
    for path in Path(directory).rglob("*.ldb"):
        for _ in range(10):
            try:
                path.unlink()
                break
            except PermissionError:
                time.sleep(0.2)
        else:
            raise RuntimeError(f"Accessロックファイルを削除できません: {path}")


def load_master(config: dict) -> dict[str, pd.DataFrame]:
    from forward_ops import copy_database
    from rating_pedigree import load_pedigree_ids
    linked = config["database"].get("linked_tables", {})
    predict_dates = sorted({
        pd.to_datetime(path.stem.rsplit("_", 1)[-1], format="%Y%m%d").strftime("%Y-%m-%d")
        for path in (BASE_DIR / "p2b_forward").glob("predict_snapshots_*.csv")
    })
    predict_clause = ""
    if predict_dates:
        dates = ", ".join(f"#{date}#" for date in predict_dates)
        predict_clause = f" OR s.[月日] IN ({dates})"
    with tempfile.TemporaryDirectory() as temporary:
        db_copy = copy_database(resolve_path(config["database"]["path"]), Path(temporary), linked)
        connection = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_copy};Mode=Share Deny None;"
        import pyodbc
        try:
            with pyodbc.connect(connection, readonly=True, autocommit=True) as con:
                runners = pd.read_sql(
                f"""
                SELECT s.[競走コード] AS race_id, s.[月日] AS race_date, s.[場名] AS venue_name,
                       s.[競走番号] AS race_number, s.[距離] AS distance,
                       s.[トラック種別コード] AS surface_code, s.[頭数] AS official_field_size,
                       r.[馬番] AS horse_number, r.[血統登録番号] AS horse_id, r.[馬名] AS horse_name,
                       r.[確定着順] AS official_finish, r.[単勝配当] AS win_payout,
                       r.[複勝配当] AS place_payout, r.[異常区分コード] AS abnormal_code,
                       r.[データ区分] AS data_division,
                       m.[競走名本題] AS race_name, m.[競走条件名称] AS condition_name,
                       m.[グレードコード] AS grade_code, m.[競走種別コード] AS race_age_code
                FROM ([レースT] AS s INNER JOIN [出走馬T] AS r ON s.[競走コード]=r.[競走コード])
                LEFT JOIN [レースマスタ] AS m ON s.[競走コード]=m.[競走コード]
                WHERE s.[月日] >= #2017-01-01#
                  AND s.[主催者コード] = 1
                  AND s.[トラック種別コード] IN (0, 1)
                  AND (r.[データ区分] = '7'{predict_clause})
                """,
                con,
            )
            horses = pd.read_sql("SELECT [血統登録番号] AS horse_id,[馬名] AS horse_name FROM [競走馬マスタ]", con)
            breeding = pd.read_sql("SELECT [繁殖登録番号] AS pedigree_id,[馬名] AS pedigree_name FROM [繁殖馬マスタ]", con)
            pedigree = load_pedigree_ids(db_copy, fallback_to_horse_master=True)
            con.close()
        finally:
            gc.collect()
            cleanup_access_lock_files(temporary)
    return {"runners": runners, "horses": horses, "breeding": breeding, "pedigree": pedigree}


def merge_master(merged: pd.DataFrame, master: dict[str, pd.DataFrame]) -> pd.DataFrame:
    runners = master["runners"].copy()
    runners["race_id"] = runners["race_id"].astype("int64")
    runners["horse_id"] = runners["horse_id"].astype("int64")
    runners = runners.rename(columns={
        "race_date": "race_date_master",
        "surface_code": "surface_code_master",
        "distance": "distance_master",
        "condition_code": "condition_code_master",
    })
    if "data_division" not in runners.columns:
        runners["data_division"] = "7"
    runners["_final_result"] = runners["data_division"].astype(str).eq("7")
    runners = runners.sort_values(["race_id", "horse_id", "_final_result", "race_date_master"], kind="stable")
    runners = runners.drop_duplicates(["race_id", "horse_id"], keep="last").drop(columns=["_final_result"])
    merged = merged.merge(runners, on=["race_id", "horse_id"], how="left", validate="many_to_one")
    for source, master_column in [
        ("race_date", "race_date_master"),
        ("surface_code", "surface_code_master"),
        ("distance", "distance_master"),
        ("condition_code", "condition_code_master"),
    ]:
        if master_column in merged.columns:
            merged[source] = merged[source].fillna(merged[master_column]) if source in merged.columns else merged[master_column]
    horses = master["horses"].drop_duplicates("horse_id")
    merged = merged.merge(horses.rename(columns={"horse_name": "master_horse_name"}), on="horse_id", how="left", validate="many_to_one")
    merged["horse_name"] = merged["horse_name"].fillna(merged["master_horse_name"])
    merged = merged.merge(master["pedigree"], on="horse_id", how="left", validate="many_to_one")
    breeding = master["breeding"].drop_duplicates("pedigree_id")
    merged = merged.merge(
        breeding.rename(columns={"pedigree_id": "sire_id", "pedigree_name": "sire_name"}),
        on="sire_id", how="left", validate="many_to_one",
    )
    merged = merged.merge(
        breeding.rename(columns={"pedigree_id": "bms_id", "pedigree_name": "bms_name"}),
        on="bms_id", how="left", validate="many_to_one",
    )
    return merged


def load_auxiliary(master: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from rating_pedigree import pedigree_auxiliary
    residual_paths = [
        BASE_DIR / "stage3_output/pedigree_residual_history.csv.gz",
        BASE_DIR / "p2b_forward/pedigree_residual_history_2025plus.csv.gz",
    ]
    residuals = []
    for path in residual_paths:
        if path.exists():
            residual = pd.read_csv(path, usecols=["race_date", "horse_id", "n_graph_starts", "sire_id", "bms_id", "actual_win", "actual_top3", "r_win", "r_top3"])
            residual["race_date"] = pd.to_datetime(residual["race_date"])
            residuals.append(residual[residual["race_date"].ge(START_DATE)])
    if not residuals:
        return pd.DataFrame()
    residual = pd.concat(residuals, ignore_index=True)
    breeding = master["breeding"].drop_duplicates("pedigree_id")
    frames = []
    for side in ["sire", "bms"]:
        frame = pedigree_auxiliary(residual, side)
        grouped = residual[residual[f"{side}_id"].notna() & residual[f"{side}_id"].ne(0)].groupby(f"{side}_id").agg(
            residual_win_mean=("r_win", "mean"),
            residual_top3_mean=("r_top3", "mean"),
            residual_starts=("r_win", "size"),
        ).reset_index()
        frame = frame.merge(grouped, on=f"{side}_id", how="left", validate="one_to_one")
        frame = frame.merge(breeding, left_on=f"{side}_id", right_on="pedigree_id", how="left", validate="many_to_one")
        frame["side"] = side
        frames.append(frame)
    auxiliary = pd.concat(frames, ignore_index=True)
    auxiliary["pedigree_name"] = auxiliary["pedigree_name"].fillna(auxiliary["sire_id"].fillna(auxiliary["bms_id"]).astype("Int64").astype(str))
    return auxiliary


def build_cache(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = input_paths(config)
    snapshots = merge_snapshot_post(read_snapshot(paths["snapshots"], "final"), paths["posts"], paths["anchor"])
    if paths["forward_snapshots"].exists():
        forward = merge_snapshot_post(read_snapshot(paths["forward_snapshots"], "final"), paths["forward_posts"], paths["forward_anchor"])
        snapshots = pd.concat([snapshots, forward], ignore_index=True)
    master = load_master(config)
    merged = merge_master(snapshots, master)
    p0 = load_p0_prior().rename(columns={"race_date": "p0_race_date"})
    if not p0.empty:
        merged = merged.merge(p0, on=["race_id", "horse_id"], how="left", validate="many_to_one")
    else:
        merged["p0_mu_pre"] = np.nan
    merged["elo_prior_boost"] = SCALE * (merged["mu_pre"] - merged["p0_mu_pre"])
    merged["record_type"] = merged["record_type"].fillna("final")
    merged = merged.sort_values(["race_id", "horse_id", "record_type", "race_date"], kind="stable")
    merged = merged.drop_duplicates(["race_id", "horse_id", "record_type"], keep="last").reset_index(drop=True)
    predict_dir = BASE_DIR / "p2b_forward"
    predictions = []
    for path in sorted(predict_dir.glob("predict_snapshots_*.csv")):
        predictions.append(read_snapshot(path, "predict"))
    if predictions:
        prediction = pd.concat(predictions, ignore_index=True)
        prediction = merge_master(prediction, master)
        if not p0.empty:
            prediction = prediction.merge(p0, on=["race_id", "horse_id"], how="left", validate="many_to_one")
        else:
            prediction["p0_mu_pre"] = np.nan
        prediction["elo_prior_boost"] = SCALE * (prediction["mu_pre"] - prediction["p0_mu_pre"]) if "p0_mu_pre" in prediction else np.nan
        prediction = prediction.sort_values(["race_id", "horse_id", "race_date"], kind="stable")
        prediction = prediction.drop_duplicates(["race_id", "horse_id", "record_type"], keep="last")
        merged = pd.concat([merged, prediction], ignore_index=True)
    auxiliary = load_auxiliary(master)
    return merged, auxiliary


def load_viewer_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    merged_path, auxiliary_path, fingerprint_path = cache_paths(config)
    fingerprint = source_fingerprint(config)
    if merged_path.exists() and auxiliary_path.exists() and fingerprint_path.exists():
        cached_fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if cached_fingerprint == fingerprint:
            return pd.read_parquet(merged_path), pd.read_parquet(auxiliary_path), fingerprint
    merged, auxiliary = build_cache(config)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(merged_path, index=False)
    auxiliary.to_parquet(auxiliary_path, index=False)
    fingerprint_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged, auxiliary, fingerprint
