from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

BASE = Path(__file__).resolve().parents[1]
P3 = BASE / "p3_grade"
for path in [BASE, P3]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forward_ops import copy_database, load_config
from p4_margin.margins import sequential_margins
from p3_grade.prepare_inputs import load_cached_inputs
from rating_core import file_sha256
from viewer_data import cleanup_access_lock_files

OUT = Path(__file__).resolve().parent
TRACK_LABELS = {1: "良", 2: "稍重", 3: "重", 4: "不良"}


def quantiles(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": len(clean), "mean": float(clean.mean()), "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)), "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)), "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)), "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def main() -> None:
    prepared, pedigree, _, input_manifest = load_cached_inputs()
    config = load_config(BASE / "forward_config.yaml")
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        db = copy_database(BASE / "stage2_input/BaoZ_2_snapshot.mdb", Path(temporary), config["database"].get("linked_tables", {}))
        if file_sha256(db) != input_manifest["source_db_sha256"]:
            raise ValueError("P3/P4の入力DBハッシュが一致しません")
        connection = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db};ReadOnly=1;"
        try:
            with pyodbc.connect(connection, readonly=True, autocommit=True) as con:
                raw = pd.read_sql(
                    """
                    SELECT s.[競走コード] AS race_id, s.[月日] AS race_date,
                           s.[主催者コード] AS organizer_code, s.[トラック種別コード] AS surface_code,
                           s.[場コード] AS venue_code, s.[場名] AS venue_name,
                           s.[距離] AS distance, s.[馬場状態コード] AS track_condition_code,
                           s.[一位入線タイム] AS first_time_raw,
                           r.[馬番] AS horse_number, r.[血統登録番号] AS horse_id,
                           r.[確定着順] AS official_finish, r.[異常区分コード] AS abnormal_code,
                           r.[タイム] AS time_raw, r.[着差] AS stored_margin_raw,
                           rm.[東西所属コード] AS affiliation_code
                    FROM ([レースT] AS s INNER JOIN [出走馬T] AS r ON s.[競走コード]=r.[競走コード])
                    INNER JOIN [出走馬マスタ] AS rm
                      ON r.[競走コード]=rm.[競走コード] AND r.[馬番]=rm.[馬番]
                    WHERE s.[月日] >= #2012-01-01# AND s.[月日] < #2025-01-01#
                      AND s.[データ区分]='7' AND s.[主催者コード]=1 AND s.[トラック種別コード] IN (0,1)
                    """,
                    con,
                )
                con.close()
        finally:
            gc.collect()
            cleanup_access_lock_files(temporary)
    raw["race_date"] = pd.to_datetime(raw["race_date"])
    raw["venue_code"] = pd.to_numeric(raw["venue_code"], errors="coerce")
    target = raw[
        raw["organizer_code"].eq(1) & raw["surface_code"].isin([0, 1])
        & raw["venue_code"].lt(101)
        & ~raw["venue_name"].fillna("").astype(str).str.contains("帯広", regex=False)
    ].copy()
    candidate = target[target["affiliation_code"].isin([1, 2]) & ~target["abnormal_code"].isin([1, 2, 3])].copy()
    for column in ["race_id", "horse_id", "horse_number", "official_finish", "abnormal_code", "distance", "track_condition_code"]:
        candidate[column] = pd.to_numeric(candidate[column], errors="raise").astype("int64")
    for column in ["time_raw", "stored_margin_raw", "first_time_raw"]:
        candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
        target[column] = pd.to_numeric(target[column], errors="coerce")
    candidate["time_seconds"] = candidate["time_raw"].where(candidate["time_raw"].gt(0)) / 10.0
    candidate["stored_margin_seconds"] = candidate["stored_margin_raw"].where(candidate["stored_margin_raw"].ge(0)) / 10.0
    target["time_seconds"] = target["time_raw"].where(target["time_raw"].gt(0)) / 10.0
    official_winners = target[target["official_finish"].eq(1)].groupby("race_id")["time_seconds"].min()
    candidate["official_winner_seconds"] = candidate["race_id"].map(official_winners)
    candidate["computed_official_margin_seconds"] = candidate["time_seconds"] - candidate["official_winner_seconds"]
    if len(raw) != 613963 or len(candidate) != 610676:
        raise AssertionError("走破タイム抽出件数が既存入力と一致しません")
    if set(candidate["race_id"]) != {race.race_id for race in prepared.races}:
        raise AssertionError("タイムデータと対象レースが一致しません")
    by_race = {race_id: group.sort_values("horse_number") for race_id, group in candidate.groupby("race_id", sort=False)}
    relative_records = []
    margin_records = []
    tied_spread_rows = 0
    for race in prepared.races:
        group = by_race[race.race_id].copy()
        horse_ids = prepared.horse_ids[race.horse_idx]
        if not np.array_equal(group["horse_id"].to_numpy(dtype=np.int64), horse_ids):
            raise AssertionError(f"馬ID・馬番の対応が不一致です: {race.race_id}")
        if not np.array_equal(group["abnormal_code"].to_numpy(dtype=np.int16), race.abnormal_code):
            raise AssertionError(f"異常区分が不一致です: {race.race_id}")
        positive = sorted(group.loc[group["official_finish"].gt(0), "official_finish"].unique())
        mapping = {official: index + 1 for index, official in enumerate(positive)}
        relative = group["official_finish"].map(mapping).fillna(0).astype(int).to_numpy()
        if not np.array_equal(relative, race.finish):
            raise AssertionError(f"相対着順が不一致です: {race.race_id}")
        times = group["time_seconds"].to_numpy(dtype=float)
        winner_times = times[race.finish == 1]
        winner_time = float(np.nanmin(winner_times)) if np.isfinite(winner_times).any() else np.nan
        cumulative = times - winner_time
        cumulative[race.finish == 0] = np.nan
        deltas, statuses = sequential_margins(race.finish, times, race.k_max)
        for rank in np.unique(race.finish[race.finish > 0]):
            rank_times = times[race.finish == rank]
            if len(rank_times) > 1 and np.isfinite(rank_times).all() and np.ptp(rank_times) > 0:
                tied_spread_rows += len(rank_times)
        group["relative_finish"] = race.finish
        group["relative_winner_seconds"] = winner_time
        group["cumulative_relative_seconds"] = cumulative
        relative_records.append(group[[
            "race_id", "race_date", "horse_id", "horse_number", "official_finish", "relative_finish",
            "abnormal_code", "distance", "track_condition_code", "time_raw", "time_seconds",
            "stored_margin_seconds", "computed_official_margin_seconds", "cumulative_relative_seconds",
        ]])
        margin_records.append({
            "race_id": int(race.race_id),
            "race_date": pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date),
            "distance": int(race.distance),
            "condition_code": int(race.condition_code),
            "track_condition_code": int(group["track_condition_code"].iloc[0]),
            "k_max": int(race.k_max),
            **{f"stage_{index}": float(deltas[index - 1]) for index in range(1, 6)},
            **{f"status_{index}": int(statuses[index - 1]) for index in range(1, 6)},
        })
    relative_times = pd.concat(relative_records, ignore_index=True)
    margins = pd.DataFrame(margin_records).sort_values("race_id").reset_index(drop=True)
    if len(relative_times) != 610676 or len(margins) != 43252:
        raise AssertionError("相対タイム出力件数が不正です")
    relative_times.to_csv(OUT / "relative_times.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    margins.to_csv(OUT / "race_margins.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d")

    ranked = relative_times[relative_times["relative_finish"].gt(0)].copy()
    missing_ranked = ranked[ranked["time_seconds"].isna()]
    comparison = ranked.dropna(subset=["stored_margin_seconds", "computed_official_margin_seconds"])
    stored_error = comparison["computed_official_margin_seconds"] - comparison["stored_margin_seconds"]
    speed = ranked["distance"] / ranked["time_seconds"] * 3.6
    abnormal_audit = relative_times.groupby("abnormal_code").agg(
        rows=("horse_id", "size"),
        missing_time=("time_seconds", lambda value: int(value.isna().sum())),
        present_time=("time_seconds", lambda value: int(value.notna().sum())),
    ).reset_index()
    abnormal_audit.to_csv(OUT / "time_abnormal_audit.csv", index=False)
    stage_rows = []
    for index in range(1, 6):
        status = margins[f"status_{index}"]
        delta = margins[f"stage_{index}"]
        stage_rows.append({
            "stage": index,
            "applicable": int(status.ne(0).sum()),
            "available": int(status.eq(1).sum()),
            "missing": int(status.eq(2).sum()),
            "negative_clipped": int(status.eq(3).sum()),
            "zero_seconds": int(status.eq(1).mul(delta.eq(0)).sum()),
            **{f"delta_{key}": value for key, value in quantiles(delta[status.eq(1)]).items()},
        })
    pd.DataFrame(stage_rows).to_csv(OUT / "margin_stage_summary.csv", index=False)

    margins["normalized_stage_1"] = margins["stage_1"] / (margins["distance"] / 1000.0)
    winner_times = ranked[ranked["relative_finish"].eq(1)].groupby("race_id")["time_seconds"].min()
    margins["winner_time_seconds"] = margins["race_id"].map(winner_times)
    distance_rows = []
    for distance, group in margins.groupby("distance"):
        available = group[group["status_1"].eq(1)]
        distance_rows.append({
            "distance": int(distance), "races": len(group), "available_margins": len(available),
            "winner_time_median": group["winner_time_seconds"].median(),
            "margin_mean": available["stage_1"].mean(), "margin_median": available["stage_1"].median(),
            "margin_p90": available["stage_1"].quantile(0.9),
            "normalized_median": available["normalized_stage_1"].median(),
            "normalized_p90": available["normalized_stage_1"].quantile(0.9),
        })
    pd.DataFrame(distance_rows).to_csv(OUT / "margin_scale_by_distance.csv", index=False)
    margins["track_condition"] = margins["track_condition_code"].map(TRACK_LABELS).fillna("UNKNOWN")
    margins["distance_band"] = pd.cut(
        margins["distance"], bins=[0, 1400, 1800, 2200, 2600, np.inf],
        labels=["1400m以下", "1500～1800m", "1900～2200m", "2300～2600m", "2700m以上"],
    )
    scale_rows = []
    for (band, track), group in margins.groupby(["distance_band", "track_condition"], observed=True):
        available = group[group["status_1"].eq(1)]
        scale_rows.append({
            "distance_band": str(band), "track_condition": track, "races": len(group),
            "available_margins": len(available), "margin_mean": available["stage_1"].mean(),
            "margin_median": available["stage_1"].median(), "margin_p90": available["stage_1"].quantile(0.9),
            "normalized_median": available["normalized_stage_1"].median(),
            "normalized_p90": available["normalized_stage_1"].quantile(0.9),
        })
    pd.DataFrame(scale_rows).to_csv(OUT / "margin_scale_by_condition.csv", index=False)
    cumulative_rows = []
    for rank, group in ranked[ranked["relative_finish"].le(5)].groupby("relative_finish"):
        cumulative_rows.append({"relative_finish": int(rank), **quantiles(group["cumulative_relative_seconds"])})
    pd.DataFrame(cumulative_rows).to_csv(OUT / "cumulative_margin_summary.csv", index=False)

    audit = {
        "time_column": "出走馬T.タイム",
        "margin_column": "出走馬T.着差",
        "unit_seconds_per_integer": 0.1,
        "raw_rows": len(raw), "target_rows": len(target), "retained_rows": len(candidate),
        "races": len(margins), "ranked_rows": len(ranked), "unranked_rows": len(relative_times) - len(ranked),
        "missing_time_all": int(relative_times["time_seconds"].isna().sum()),
        "missing_time_ranked": len(missing_ranked),
        "missing_time_ranked_rate": len(missing_ranked) / len(ranked),
        "stored_margin_comparison_rows": len(comparison),
        "stored_margin_mismatch_over_0_05s": int(stored_error.abs().gt(0.050001).sum()),
        "stored_margin_max_abs_error_seconds": float(stored_error.abs().max()),
        "negative_cumulative_rows": int(ranked["cumulative_relative_seconds"].lt(0).sum()),
        "tied_group_different_time_rows": tied_spread_rows,
        "speed_outliers_below_30_or_above_80_kmh": int(((speed < 30) | (speed > 80)).sum()),
        "negative_stage_margins_clipped": int(sum(margins[f"status_{index}"].eq(3).sum() for index in range(1, 6))),
        "missing_stage_margins_neutral": int(sum(margins[f"status_{index}"].eq(2).sum() for index in range(1, 6))),
        "winner_margin_raw": quantiles(margins.loc[margins["status_1"].eq(1), "stage_1"]),
        "winner_margin_distance_normalized": quantiles(margins.loc[margins["status_1"].eq(1), "normalized_stage_1"]),
        "track_condition_counts": margins["track_condition"].value_counts().to_dict(),
        "source_db_sha256": input_manifest["source_db_sha256"],
        "db_access": "temporary_copy_read_only",
    }
    (OUT / "time_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "stage_summary": stage_rows, "cumulative_summary": cumulative_rows}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
