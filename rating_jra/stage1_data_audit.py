from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

DB_PATH = Path(r"C:\BaoZ_AI\DB\BaoZ_2.mdb")
OUTPUT_PATH = Path(__file__).with_name("stage1_audit_result.json")
START_DATE = "2012-01-01"
END_DATE = "2025-01-01"
FIRST_APPEARANCE_AFTER = pd.Timestamp("2013-01-01")
YEARS = list(range(2012, 2025))
ABNORMAL_LABELS = {
    0: "正常",
    1: "出走取消",
    2: "発走除外",
    3: "競走除外",
    4: "競走中止",
    5: "失格",
    6: "落馬再騎乗",
    7: "降着",
}
AFFILIATION_LABELS = {1: "JRA_美浦", 2: "JRA_栗東", 3: "地方", 4: "海外"}
NON_START_ABNORMAL_CODES = {1, 2, 3}


def read_sql(con: pyodbc.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, con)


def missing_mask(series: pd.Series, zero_is_missing: bool = False) -> pd.Series:
    result = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        result = result | series.fillna("").astype(str).str.strip().eq("")
    if zero_is_missing:
        result = result | pd.to_numeric(series, errors="coerce").fillna(0).eq(0)
    return result


def missing_rate(series: pd.Series, zero_is_missing: bool = False) -> float:
    if len(series) == 0:
        return 0.0
    return float(missing_mask(series, zero_is_missing).mean())


def year_counts(frame: pd.DataFrame, value: str = "race_id") -> dict[str, int]:
    if frame.empty:
        return {}
    counts = frame.groupby(frame["race_date"].dt.year)[value].nunique()
    return {str(int(year)): int(count) for year, count in counts.items()}


def exclusion_category(row: pd.Series) -> str:
    year = int(row["year"])
    organizer = row["organizer_code"]
    venue_code = row["venue_code"]
    venue_name = "" if pd.isna(row["venue_name"]) else str(row["venue_name"]).strip()
    surface = row["surface_code"]
    if year >= 2025:
        return "LOCK_2025_PLUS"
    if organizer in (0, 3) or (pd.notna(venue_code) and venue_code >= 101):
        return "OVERSEAS"
    if "帯広" in venue_name:
        return "BANEI"
    if organizer == 2:
        return "LOCAL"
    if organizer == 1 and surface == 2:
        return "JRA_OBSTACLE"
    if organizer == 1 and surface in (0, 1):
        return "TARGET_JRA_FLAT"
    return "UNCLASSIFIED"


def exclusion_records(groups: pd.DataFrame, count_column: str) -> list[dict]:
    if groups.empty:
        return []
    categorized = groups.copy()
    categorized["category"] = categorized.apply(exclusion_category, axis=1)
    counts = categorized.groupby(["year", "category"])[count_column].sum()
    return [
        {"year": int(year), "category": category, "count": int(count)}
        for (year, category), count in sorted(counts.items())
    ]


def abnormal_counts(frame: pd.DataFrame) -> list[dict]:
    counts = frame["abnormal_code"].value_counts(dropna=False)
    records = []
    for code, label in ABNORMAL_LABELS.items():
        records.append({"abnormal_code": code, "label": label, "count": int(counts.get(code, 0))})
    unknown = counts[[index for index in counts.index if pd.notna(index) and index not in ABNORMAL_LABELS]]
    for code, count in unknown.items():
        records.append({"abnormal_code": int(code), "label": "不明", "count": int(count)})
    null_count = int(counts.get(np.nan, 0)) if np.nan in counts.index else int(frame["abnormal_code"].isna().sum())
    if null_count:
        records.append({"abnormal_code": None, "label": "欠損", "count": null_count})
    return records


def tied_diagnostics(candidate: pd.DataFrame) -> dict:
    ranked = candidate[pd.to_numeric(candidate["finish_position"], errors="coerce").fillna(0).gt(0)]
    group_sizes = ranked.groupby(["race_id", "finish_position"]).size()
    tied = group_sizes[group_sizes.gt(1)]
    if tied.empty:
        return {
            "tied_starts": 0,
            "tied_groups": 0,
            "races_with_ties": 0,
            "max_tie_size": 0,
            "max_permutation_combinations_per_race": 1,
            "races_over_120_permutations": 0,
        }
    tie_frame = tied.rename("tie_size").reset_index()
    permutations_by_race = tie_frame.groupby("race_id")["tie_size"].apply(
        lambda sizes: math.prod(math.factorial(int(size)) for size in sizes)
    )
    return {
        "tied_starts": int(tied.sum()),
        "tied_groups": int(len(tied)),
        "races_with_ties": int(tie_frame["race_id"].nunique()),
        "max_tie_size": int(tied.max()),
        "max_permutation_combinations_per_race": int(permutations_by_race.max()),
        "races_over_120_permutations": int(permutations_by_race.gt(120).sum()),
    }


def same_day_diagnostics(candidate: pd.DataFrame) -> dict:
    grouped = candidate.groupby(["horse_id", "race_date"]).agg(
        starts=("race_id", "nunique"), rows=("race_id", "size")
    )
    multiple = grouped[grouped["starts"].gt(1)].reset_index()
    if multiple.empty:
        return {
            "horse_date_groups": 0,
            "involved_starts": 0,
            "excess_starts": 0,
            "horses": 0,
            "dates": 0,
            "by_year": {},
            "examples": [],
        }
    multiple["year"] = multiple["race_date"].dt.year
    by_year = multiple.groupby("year").agg(
        horse_date_groups=("starts", "size"),
        involved_starts=("starts", "sum"),
        horses=("horse_id", "nunique"),
    )
    examples = multiple.sort_values(["race_date", "horse_id"]).head(20).copy()
    examples["race_date"] = examples["race_date"].dt.strftime("%Y-%m-%d")
    return {
        "horse_date_groups": int(len(multiple)),
        "involved_starts": int(multiple["starts"].sum()),
        "excess_starts": int((multiple["starts"] - 1).sum()),
        "horses": int(multiple["horse_id"].nunique()),
        "dates": int(multiple["race_date"].nunique()),
        "by_year": {
            str(int(year)): {key: int(value) for key, value in row.items()}
            for year, row in by_year.iterrows()
        },
        "examples": examples[["horse_id", "race_date", "starts", "rows"]].to_dict("records"),
    }


def date_relation(races: pd.DataFrame, column: str) -> dict:
    values = pd.to_datetime(races[column], errors="coerce")
    differences = (values.dt.normalize() - races["race_date"].dt.normalize()).dt.days
    valid = differences.dropna().astype(int)
    counts = Counter(int(value) for value in valid)
    delayed = races.loc[valid.index, ["race_id", "race_date"]].copy()
    delayed["candidate_date"] = values.loc[valid.index]
    delayed["days_after_race"] = valid
    delayed = delayed.sort_values("days_after_race", ascending=False).head(10)
    delayed["race_date"] = delayed["race_date"].dt.strftime("%Y-%m-%d")
    delayed["candidate_date"] = delayed["candidate_date"].dt.strftime("%Y-%m-%d")
    return {
        "column": column,
        "rows": int(len(races)),
        "missing_count": int(values.isna().sum()),
        "missing_rate": float(values.isna().mean()),
        "min_days_after_race": int(valid.min()) if len(valid) else None,
        "median_days_after_race": float(valid.median()) if len(valid) else None,
        "max_days_after_race": int(valid.max()) if len(valid) else None,
        "before_race_count": int(valid.lt(0).sum()),
        "same_day_count": int(valid.eq(0).sum()),
        "next_day_count": int(valid.eq(1).sum()),
        "over_1_day_count": int(valid.gt(1).sum()),
        "over_7_day_count": int(valid.gt(7).sum()),
        "over_30_day_count": int(valid.gt(30).sum()),
        "days_after_race_counts": {str(key): counts[key] for key in sorted(counts)},
        "largest_delays": delayed.to_dict("records"),
    }


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    connection_string = (
        "Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
        f"DBQ={DB_PATH};ReadOnly=1;"
    )
    with pyodbc.connect(connection_string, readonly=True, autocommit=True) as con:
        race_groups = read_sql(
            con,
            """
            SELECT Year([月日]) AS year, [主催者コード] AS organizer_code,
                   [トラック種別コード] AS surface_code, [場コード] AS venue_code,
                   [場名] AS venue_name, COUNT(*) AS races
            FROM [レースT]
            WHERE [月日] >= #2012-01-01#
            GROUP BY Year([月日]), [主催者コード], [トラック種別コード],
                     [場コード], [場名]
            """,
        )
        start_groups = read_sql(
            con,
            """
            SELECT Year(s.[月日]) AS year, s.[主催者コード] AS organizer_code,
                   s.[トラック種別コード] AS surface_code, s.[場コード] AS venue_code,
                   s.[場名] AS venue_name, COUNT(*) AS starts
            FROM [レースT] AS s INNER JOIN [出走馬T] AS r
              ON s.[競走コード] = r.[競走コード]
            WHERE s.[月日] >= #2012-01-01#
            GROUP BY Year(s.[月日]), s.[主催者コード], s.[トラック種別コード],
                     s.[場コード], s.[場名]
            """,
        )
        races = read_sql(
            con,
            """
            SELECT s.[競走コード] AS race_id, s.[月日] AS race_date,
                   s.[主催者コード] AS organizer_code,
                   s.[トラック種別コード] AS surface_code,
                   s.[トラックコード] AS track_code,
                   s.[場コード] AS venue_code, s.[場名] AS venue_name,
                   s.[競走番号] AS race_number, s.[距離] AS distance,
                   s.[競走条件コード] AS condition_code,
                   s.[競走条件名称] AS condition_name,
                   s.[登録頭数] AS n_registered,
                   s.[頭数] AS n_starters_recorded,
                   s.[初出走頭数] AS n_first_starters_recorded,
                   s.[混合] AS sex_mixed_flag,
                   s.[データ区分] AS race_data_division,
                   s.[データ作成年月日] AS race_data_created,
                   s.[発走時刻] AS scheduled_off_at,
                   s.[レコード作成fromtime] AS record_fromtime,
                   m.[競走名本題] AS race_name,
                   m.[競走記号コード] AS race_symbol_code,
                   m.[データ作成年月日] AS master_race_data_created,
                   m.[JRA競走コード] AS jra_race_code
            FROM [レースT] AS s INNER JOIN [レースマスタ] AS m
              ON s.[競走コード] = m.[競走コード]
            WHERE s.[主催者コード] = 1
              AND s.[トラック種別コード] IN (0, 1)
              AND s.[月日] >= #2012-01-01#
              AND s.[月日] < #2025-01-01#
            """,
        )
        runners = read_sql(
            con,
            """
            SELECT s.[競走コード] AS race_id, s.[月日] AS race_date,
                   s.[競走条件コード] AS condition_code,
                   r.[馬番] AS horse_number, r.[馬名] AS horse_name,
                   r.[血統登録番号] AS horse_id,
                   r.[所属] AS trainer_region_name,
                   r.[負担重量] AS weight_raw,
                   r.[入線順位] AS arrival_position,
                   r.[確定着順] AS finish_position,
                   r.[異常区分コード] AS abnormal_code,
                   r.[データ区分] AS runner_data_division,
                   r.[データ作成年月日] AS runner_data_created,
                   rm.[東西所属コード] AS affiliation_code,
                   rm.[同着区分] AS master_tie_code,
                   rm.[同着頭数] AS master_tie_count,
                   rm.[血統登録番号] AS master_horse_id,
                   rm.[負担重量] AS master_weight_raw,
                   rm.[入線順位] AS master_arrival_position,
                   rm.[確定着順] AS master_finish_position,
                   rm.[異常区分コード] AS master_abnormal_code,
                   rm.[データ作成年月日] AS master_runner_data_created
            FROM ([レースT] AS s INNER JOIN [出走馬T] AS r
              ON s.[競走コード] = r.[競走コード])
            INNER JOIN [出走馬マスタ] AS rm
              ON r.[競走コード] = rm.[競走コード]
             AND r.[馬番] = rm.[馬番]
            WHERE s.[主催者コード] = 1
              AND s.[トラック種別コード] IN (0, 1)
              AND s.[月日] >= #2012-01-01#
              AND s.[月日] < #2025-01-01#
            """,
        )
        pedigree = read_sql(
            con,
            """
            SELECT [血統登録番号] AS horse_id,
                   [父繁殖登録番号] AS sire_id,
                   [母父繁殖登録番号] AS bms_id
            FROM [産駒マスタ]
            """,
        )
        schema_specs = [
            ("race_id", "レースT", "競走コード", "レース識別子", False),
            ("organizer_code", "レースT", "主催者コード", "1=中央、2=地方。海外は場コード等も使用", False),
            ("race_date", "レースT", "月日", "開催日", False),
            ("result_available_at", None, None, "明示された結果利用可能日時列はなし", False),
            ("race_data_created", "レースT", "データ作成年月日", "結果利用可能日時の候補。意味の確定が必要", False),
            ("venue_code", "レースT", "場コード", "競馬場コード。101以降は海外", False),
            ("surface_code", "レースT", "トラック種別コード", "0=芝、1=ダート、2=障害", False),
            ("distance", "レースT", "距離", "距離（m）", False),
            ("condition_code", "レースT", "競走条件コード", "701=新馬、702=未出走、703=未勝利、999=オープン", False),
            ("n_registered", "レースT", "登録頭数", "出馬表登録時の頭数", False),
            ("n_starters_recorded", "レースT", "頭数", "登録頭数から取消・除外を除いた頭数", False),
            ("horse_id", "出走馬T", "血統登録番号", "馬ID", True),
            ("horse_number", "出走馬T", "馬番", "馬番", False),
            ("arrival_position", "出走馬T", "入線順位", "入線順位", False),
            ("finish_position", "出走馬T", "確定着順", "確定着順。順位なしは0", False),
            ("abnormal_code", "出走馬T", "異常区分コード", "出走状態コード", False),
            ("master_tie_code", "出走馬マスタ", "同着区分", "同着の補助列。実診断は確定着順の重複で確認", False),
            ("weight_kg", "出走馬T", "負担重量", "斤量。DB値は0.1kg単位", True),
            ("sire_id", "産駒マスタ", "父繁殖登録番号", "父の馬ID", True),
            ("bms_id", "産駒マスタ", "母父繁殖登録番号", "母父の馬ID", True),
            ("exchange_flag", None, None, "交流競走フラグは未検出。混合は牡牝混合なので不使用", False),
            ("affiliation_code", "出走馬マスタ", "東西所属コード", "当該出走時の所属。1/2=JRA、3=地方、4=海外", True),
            ("trainer_region_name", "出走馬T", "所属", "調教師所属地域名。所属体系の補助表示", False),
            ("sex_mixed_flag", "レースT", "混合", "牡馬と牝馬が混合して出走するか。交流フラグではない", False),
        ]
        schema_cursor = con.cursor()
        source_types = {}
        for _, table, column, _, _ in schema_specs:
            if table is None:
                continue
            try:
                rows = schema_cursor.columns(table=table, column=column).fetchall()
                source_types[(table, column)] = rows[0].type_name if rows else "不明"
            except (pyodbc.Error, UnicodeDecodeError):
                source_types[(table, column)] = "取得不可"

    for frame in (races, runners):
        frame["race_id"] = pd.to_numeric(frame["race_id"], errors="raise").astype("int64")
        frame["race_date"] = pd.to_datetime(frame["race_date"])
    for column in ["horse_id", "master_horse_id", "affiliation_code", "abnormal_code", "master_abnormal_code", "finish_position", "arrival_position"]:
        runners[column] = pd.to_numeric(runners[column], errors="coerce").astype("Int64")
    pedigree["horse_id"] = pd.to_numeric(pedigree["horse_id"], errors="coerce").astype("Int64")
    for column in ["sire_id", "bms_id"]:
        pedigree[column] = pd.to_numeric(pedigree[column], errors="coerce").astype("Int64")

    runners = runners.merge(
        races[[
            "race_id", "organizer_code", "venue_code", "surface_code", "distance",
            "n_registered", "n_starters_recorded", "race_data_created", "sex_mixed_flag",
        ]],
        on="race_id",
        how="left",
        validate="many_to_one",
    )
    runners["weight_kg"] = pd.to_numeric(runners["weight_raw"], errors="coerce") / 10.0
    runners.loc[runners["weight_kg"].le(0), "weight_kg"] = np.nan
    runners["horse_affiliation_circuit"] = runners["affiliation_code"].map(AFFILIATION_LABELS)
    runners["exchange_flag"] = pd.NA
    runners["result_available_at"] = pd.NaT
    runners = runners.merge(pedigree, on="horse_id", how="left", validate="many_to_one")

    assert races["race_id"].is_unique
    assert not runners.duplicated(["race_id", "horse_number"]).any()
    assert runners["race_date"].between(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE) - pd.Timedelta(days=1)).all()
    assert races["race_date"].between(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE) - pd.Timedelta(days=1)).all()
    assert runners["race_id"].isin(races["race_id"]).all()
    assert len(runners) == int(start_groups.loc[start_groups.apply(exclusion_category, axis=1).eq("TARGET_JRA_FLAT"), "starts"].sum())
    assert len(races) == int(race_groups.loc[race_groups.apply(exclusion_category, axis=1).eq("TARGET_JRA_FLAT"), "races"].sum())

    consistency = {}
    for left, right in [
        ("horse_id", "master_horse_id"),
        ("weight_raw", "master_weight_raw"),
        ("arrival_position", "master_arrival_position"),
        ("finish_position", "master_finish_position"),
        ("abnormal_code", "master_abnormal_code"),
    ]:
        both = runners[left].notna() & runners[right].notna()
        mismatch = both & runners[left].ne(runners[right])
        consistency[left] = {
            "compared_rows": int(both.sum()),
            "left_only_missing": int((runners[left].isna() & runners[right].notna()).sum()),
            "right_only_missing": int((runners[left].notna() & runners[right].isna()).sum()),
            "mismatches": int(mismatch.sum()),
        }

    retained = runners[runners["affiliation_code"].isin([1, 2])].copy()
    non_jra = runners[runners["affiliation_code"].isin([3, 4])].copy()
    unknown_affiliation = runners[~runners["affiliation_code"].isin([1, 2, 3, 4])].copy()
    candidate = retained[~retained["abnormal_code"].isin(NON_START_ABNORMAL_CODES)].copy()

    all_race_ids = races["race_id"]
    retained_sizes = retained.groupby("race_id").size().reindex(all_race_ids, fill_value=0)
    candidate_sizes = candidate.groupby("race_id").size().reindex(all_race_ids, fill_value=0)

    yearly = []
    for year in YEARS:
        race_subset = races[races["race_date"].dt.year.eq(year)]
        all_rows = runners[runners["race_date"].dt.year.eq(year)]
        retained_rows = retained[retained["race_date"].dt.year.eq(year)]
        candidate_rows = candidate[candidate["race_date"].dt.year.eq(year)]
        non_jra_rows = non_jra[non_jra["race_date"].dt.year.eq(year)]
        yearly.append({
            "year": year,
            "races": int(race_subset["race_id"].nunique()),
            "rows_before_non_jra_removal": int(len(all_rows)),
            "rows_after_non_jra_removal": int(len(retained_rows)),
            "actual_graph_starts": int(len(candidate_rows)),
            "non_jra_rows_removed": int(len(non_jra_rows)),
            "races_with_non_jra": int(non_jra_rows["race_id"].nunique()),
            "races_with_at_least_2_candidates": int(candidate_sizes[race_subset["race_id"]].ge(2).sum()),
        })

    non_jra_by_year = []
    for (year, circuit), group in non_jra.groupby([non_jra["race_date"].dt.year, "horse_affiliation_circuit"]):
        non_jra_by_year.append({
            "year": int(year),
            "circuit": circuit,
            "starts": int(len(group)),
            "horses": int(group["horse_id"].nunique()),
            "races": int(group["race_id"].nunique()),
        })
    non_jra_by_year.sort(key=lambda row: (row["year"], row["circuit"]))
    non_jra_totals = [
        {
            "circuit": circuit,
            "starts": int(len(group)),
            "horses": int(group["horse_id"].nunique()),
            "races": int(group["race_id"].nunique()),
        }
        for circuit, group in non_jra.groupby("horse_affiliation_circuit")
    ]
    non_jra_raw_labels = []
    for (year, circuit, raw_label), group in non_jra.groupby(
        [non_jra["race_date"].dt.year, "horse_affiliation_circuit", non_jra["trainer_region_name"].fillna("(NULL)")]
    ):
        non_jra_raw_labels.append({
            "year": int(year),
            "circuit": circuit,
            "trainer_region_name": raw_label,
            "starts": int(len(group)),
            "horses": int(group["horse_id"].nunique()),
        })

    same_day = same_day_diagnostics(candidate)

    history = candidate.sort_values(["horse_id", "race_date", "race_id", "horse_number"]).copy()
    history["n_graph_starts_pre"] = history.groupby("horse_id").cumcount()
    new_horse_starts = history[history["condition_code"].eq(701)]
    new_horse_violations = new_horse_starts[new_horse_starts["n_graph_starts_pre"].gt(0)]
    new_horse_by_year = []
    for year, group in new_horse_violations.groupby(new_horse_violations["race_date"].dt.year):
        new_horse_by_year.append({
            "year": int(year),
            "starts": int(len(group)),
            "horses": int(group["horse_id"].nunique()),
            "races": int(group["race_id"].nunique()),
        })

    first_appearance = history.drop_duplicates("horse_id", keep="first")
    other_route = first_appearance[
        first_appearance["race_date"].ge(FIRST_APPEARANCE_AFTER)
        & ~first_appearance["condition_code"].eq(701)
    ]
    other_route_by_year = {
        str(int(year)): int(count)
        for year, count in other_route.groupby(other_route["race_date"].dt.year).size().items()
    }

    ranked_candidate = candidate[pd.to_numeric(candidate["finish_position"], errors="coerce").fillna(0).gt(0)]
    ties = tied_diagnostics(candidate)
    tie_flag_counts = {
        str(int(code)) if pd.notna(code) else "NULL": int(count)
        for code, count in candidate["master_tie_code"].value_counts(dropna=False).items()
    }

    unexplained_rank = candidate[
        ~candidate["abnormal_code"].isin([4, 5])
        & pd.to_numeric(candidate["finish_position"], errors="coerce").fillna(0).eq(0)
    ]
    no_rank_explained = candidate[
        candidate["abnormal_code"].isin([4, 5])
        & pd.to_numeric(candidate["finish_position"], errors="coerce").fillna(0).eq(0)
    ]

    schema = []
    for logical_name, table, column, meaning, zero_missing in schema_specs:
        if logical_name in runners.columns:
            all_series = runners[logical_name]
            retained_series = retained[logical_name]
            pandas_type = str(all_series.dtype)
            all_missing = missing_rate(all_series, zero_missing)
            retained_missing = missing_rate(retained_series, zero_missing)
        else:
            pandas_type = "列なし"
            all_missing = 1.0
            retained_missing = 1.0
        schema.append({
            "logical_name": logical_name,
            "source_table": table,
            "source_column": column,
            "db_type": source_types.get((table, column)),
            "pandas_type": pandas_type,
            "meaning": meaning,
            "missing_rate_all_target": all_missing,
            "missing_rate_retained_jra": retained_missing,
        })

    result_dates = {
        "explicit_result_available_at_exists": False,
        "candidate": date_relation(races, "race_data_created"),
        "master_race_candidate": date_relation(races, "master_race_data_created"),
        "record_fromtime": {
            "rows": int(len(races)),
            "missing_count": int(races["record_fromtime"].isna().sum()),
            "missing_rate": missing_rate(races["record_fromtime"]),
        },
        "race_vs_master_created_mismatch_count": int(
            races["race_data_created"].ne(races["master_race_data_created"]).sum()
        ),
        "runner_vs_master_created_mismatch_count": int(
            runners["runner_data_created"].ne(runners["master_runner_data_created"]).sum()
        ),
        "race_data_division_counts": {
            str(key): int(value) for key, value in races["race_data_division"].value_counts(dropna=False).items()
        },
    }

    stop_reasons = [
        "交流競走フラグ列が存在しないため、交流競走数と交流フラグなしレースの非JRA出走を仕様どおり判定できない"
    ]
    if result_dates["candidate"]["over_30_day_count"]:
        stop_reasons.append(
            "明示されたresult_available_atがなく、データ作成年月日には30日超の開催日差があるため、結果利用可能時刻と後日修正の扱いを確定する必要がある"
        )
    if not unknown_affiliation.empty:
        stop_reasons.append(f"所属体系を判定できない出走が{len(unknown_affiliation)}件ある")
    if same_day["horse_date_groups"]:
        stop_reasons.append("同一馬の同日複数レース出走がある")
    if len(new_horse_violations):
        stop_reasons.append("新馬戦に発走前n_graph_starts>0の馬がいる")
    if len(unexplained_rank):
        stop_reasons.append("理由が不明な順位欠損がある")
    if ties["races_over_120_permutations"]:
        stop_reasons.append("同着順列組合せが120を超えるレースがある")
    sire_missing = missing_rate(retained["sire_id"], True)
    bms_missing = missing_rate(retained["bms_id"], True)
    if sire_missing >= 0.10 or bms_missing >= 0.10:
        stop_reasons.append("父または母父IDの欠損率が10%以上である")

    result = {
        "scope": {
            "database": str(DB_PATH),
            "analysis_period": [START_DATE, "2024-12-31"],
            "target": "レースT.主催者コード=1かつトラック種別コードIN(0,1)",
            "lock_handling": "2025年以降の行は抽出せず、DB内の理由別件数集計だけを実施",
            "new_horse_condition_code": 701,
            "unraced_condition_code": 702,
            "non_start_abnormal_codes": sorted(NON_START_ABNORMAL_CODES),
            "affiliation_source": "出走馬マスタ.東西所属コード",
            "affiliation_mapping": AFFILIATION_LABELS,
            "abnormal_mapping": ABNORMAL_LABELS,
            "basis": "行数は特記なしではJRA平地抽出直後、JRA所属限定後、取消・除外後を区別",
        },
        "schema": schema,
        "counts": {
            "races": int(races["race_id"].nunique()),
            "rows_before_non_jra_removal": int(len(runners)),
            "rows_after_non_jra_removal": int(len(retained)),
            "actual_graph_starts_after_non_jra_and_non_start_removal": int(len(candidate)),
            "horse_ids_before_non_jra_removal": int(runners["horse_id"].nunique()),
            "horse_ids_after_non_jra_removal": int(retained["horse_id"].nunique()),
            "new_horse_races": int(races["condition_code"].eq(701).sum()),
            "unraced_races": int(races["condition_code"].eq(702).sum()),
            "unclassified_source_groups": exclusion_records(race_groups[race_groups.apply(exclusion_category, axis=1).eq("UNCLASSIFIED")], "races"),
        },
        "yearly": yearly,
        "source_exclusions": {
            "category_rule": "2025年以降 > 海外 > ばんえい > 地方主催 > JRA障害 > JRA平地。相互排他的",
            "races": exclusion_records(race_groups, "races"),
            "rows": exclusion_records(start_groups, "starts"),
        },
        "non_jra": {
            "exchange_flag_exists": False,
            "exchange_race_count_by_year": None,
            "non_jra_on_races_without_exchange_flag": None,
            "limitation": "独立した交流フラグがない。混合は牡牝混合なので代理にしていない。非JRA出走ありレース数は交流競走数の代理ではない",
            "races_with_non_jra_by_year": year_counts(non_jra),
            "removed_by_year_and_circuit": non_jra_by_year,
            "removed_totals_by_circuit": non_jra_totals,
            "removed_by_year_and_raw_label": non_jra_raw_labels,
            "unknown_affiliation_starts": int(len(unknown_affiliation)),
            "unknown_affiliation_horses": int(unknown_affiliation["horse_id"].nunique()),
            "affiliation_missing_rate_all_target": missing_rate(runners["affiliation_code"], True),
        },
        "retained_counts": {
            "races_with_fewer_than_2_after_non_jra_only": int(retained_sizes.lt(2).sum()),
            "races_with_0_after_non_jra_only": int(retained_sizes.eq(0).sum()),
            "races_with_1_after_non_jra_only": int(retained_sizes.eq(1).sum()),
            "races_with_fewer_than_2_after_non_jra_and_non_start_removal": int(candidate_sizes.lt(2).sum()),
            "races_with_0_after_non_jra_and_non_start_removal": int(candidate_sizes.eq(0).sum()),
            "races_with_1_after_non_jra_and_non_start_removal": int(candidate_sizes.eq(1).sum()),
        },
        "key_missingness": {
            "basis": "JRA所属限定後。取消・除外も含む",
            "rows": int(len(retained)),
            "horse_id_missing_rate": missing_rate(retained["horse_id"], True),
            "sire_id_missing_count": int(missing_mask(retained["sire_id"], True).sum()),
            "sire_id_missing_rate": sire_missing,
            "bms_id_missing_count": int(missing_mask(retained["bms_id"], True).sum()),
            "bms_id_missing_rate": bms_missing,
            "weight_missing_count": int(missing_mask(retained["weight_kg"], True).sum()),
            "weight_missing_rate": missing_rate(retained["weight_kg"], True),
            "affiliation_missing_rate": missing_rate(retained["affiliation_code"], True),
            "unique_retained_horses": int(retained["horse_id"].nunique()),
            "unique_horses_with_sire_missing": int(retained.loc[missing_mask(retained["sire_id"], True), "horse_id"].nunique()),
            "unique_horses_with_bms_missing": int(retained.loc[missing_mask(retained["bms_id"], True), "horse_id"].nunique()),
            "candidate_weight_missing_count": int(missing_mask(candidate["weight_kg"], True).sum()),
            "candidate_races_with_weight_missing": int(candidate.loc[missing_mask(candidate["weight_kg"], True), "race_id"].nunique()),
        },
        "abnormal": {
            "all_target": abnormal_counts(runners),
            "retained_jra": abnormal_counts(retained),
            "unexplained_missing_rank_count": int(len(unexplained_rank)),
            "explained_no_rank_abnormal_4_or_5_count": int(len(no_rank_explained)),
            "positive_rank_starts": int(len(ranked_candidate)),
        },
        "ties": {
            **ties,
            "basis": "非JRA・取消・除外後の確定着順の重複。馬番順への置換はしていない",
            "master_tie_code_counts": tie_flag_counts,
        },
        "same_horse_same_day": same_day,
        "new_horse_history": {
            "new_horse_starts_checked": int(len(new_horse_starts)),
            "violating_starts": int(len(new_horse_violations)),
            "violating_horses": int(new_horse_violations["horse_id"].nunique()),
            "violating_races": int(new_horse_violations["race_id"].nunique()),
            "by_year": new_horse_by_year,
        },
        "other_route_suspects": {
            "definition": "グラフ内初登場日が2013-01-01以降で、初登場レースの競走条件コードが701以外の馬",
            "horse_count": int(other_route["horse_id"].nunique()),
            "by_first_appearance_year": other_route_by_year,
            "note": "地方・障害・海外からの移籍馬を含み得る。これらの除外結果は診断に使用していない",
        },
        "result_availability": result_dates,
        "source_consistency": consistency,
        "stop_reasons": stop_reasons,
        "stage": "1_ONLY",
        "modeling_performed": False,
    }
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "counts": result["counts"],
        "retained_counts": result["retained_counts"],
        "new_horse_history": result["new_horse_history"],
        "same_horse_same_day": {key: value for key, value in same_day.items() if key != "examples"},
        "other_route_suspects": result["other_route_suspects"],
        "stop_reasons": stop_reasons,
    }, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
