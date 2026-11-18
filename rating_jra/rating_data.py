from __future__ import annotations

import itertools
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

from rating_core import file_sha256

DB_PATH = Path(r"C:\BaoZ_AI\DB\BaoZ_2.mdb")
CACHE_DIR = Path(__file__).with_name("stage2_cache")
NON_START_ABNORMAL_CODES = {1, 2, 3}
REQUIRED_COLUMNS = {
    "race_id", "race_date", "organizer_code", "surface_code", "venue_code", "venue_name",
    "distance", "condition_code", "n_registered", "horse_id", "horse_number", "weight_kg",
    "finish_position", "abnormal_code", "affiliation_code",
}


@dataclass
class Race:
    race_index: int
    race_id: int
    date: int
    horse_idx: np.ndarray
    horse_number: np.ndarray
    weight_kg: np.ndarray
    finish: np.ndarray
    abnormal_code: np.ndarray
    orders: np.ndarray
    unranked: np.ndarray
    pre_nstarts: np.ndarray
    condition_code: int
    surface_code: int
    distance: int
    venue_code: int
    n_field_original: int
    n_field_before_non_jra: int
    n_removed_non_jra: int
    weight_missing_count: int
    result_version: str
    race_data_created: str

    @property
    def n(self) -> int:
        return len(self.horse_idx)

    @property
    def r(self) -> int:
        return int(np.count_nonzero(self.finish > 0))

    @property
    def k_max(self) -> int:
        return min(5, self.r, self.n - 1) if self.n >= 2 else 0

    @property
    def has_likelihood(self) -> bool:
        return self.n >= 2 and self.k_max > 0


@dataclass
class PreparedInput:
    races: list[Race]
    horse_ids: np.ndarray
    source_hash: str
    diagnostics: dict

    @property
    def total_starts(self) -> int:
        return sum(race.n for race in self.races)

    def evaluation_masks(self) -> dict[str, np.ndarray]:
        n = len(self.races)
        years = np.array([pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=int(race.date)) for race in self.races])
        years = np.array([value.year for value in years])
        eligible = np.array([race.has_likelihood for race in self.races], dtype=bool)
        not_debut = np.array([race.condition_code not in (701, 702) for race in self.races])
        few_first = np.array([np.count_nonzero(race.pre_nstarts == 0) < 4 for race in self.races])
        clean = np.array([race.n_removed_non_jra == 0 for race in self.races])
        tuning = (years >= 2017) & (years <= 2020)
        validation = (years >= 2021) & (years <= 2024)
        e = eligible & not_debut & few_first
        return {
            "years": years,
            "eligible": eligible,
            "E": e,
            "E_clean": e & clean,
            "F": eligible,
            "tuning": tuning,
            "validation": validation,
            "tuning_E": tuning & e,
            "tuning_E_clean": tuning & e & clean,
            "tuning_F": tuning & eligible,
            "validation_E": validation & e,
            "validation_E_clean": validation & e & clean,
            "validation_F": validation & eligible,
        }


def make_orders(finish: np.ndarray) -> np.ndarray:
    ranked_groups = []
    for rank in sorted(np.unique(finish[finish > 0])):
        ranked_groups.append(np.flatnonzero(finish == rank).tolist())
    if not ranked_groups:
        return np.zeros((1, 0), dtype=np.int16)
    permutation_groups = [list(itertools.permutations(group)) for group in ranked_groups]
    combination_count = int(np.prod([len(group) for group in permutation_groups]))
    if combination_count > 120:
        raise ValueError(f"同着順列組合せが120を超えています: {combination_count}")
    orders = []
    for combination in itertools.product(*permutation_groups):
        order = [horse for group in combination for horse in group]
        orders.append(order)
    return np.asarray(orders, dtype=np.int16)


def prepare_input(raw: pd.DataFrame, source_hash: str = "synthetic", allow_future: bool = False, allow_unranked: bool = False) -> PreparedInput:
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"必要列がありません: {sorted(missing)}")
    frame = raw.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"])
    locked = frame["race_date"] >= pd.Timestamp("2025-01-01")
    if locked.any() and not allow_future:
        raise ValueError(f"2025年以降のデータが{int(locked.sum())}行混入しています")
    if frame.duplicated(["race_id", "horse_number"]).any():
        raise ValueError("同一race_id・horse_numberの重複があります")
    frame["race_id"] = pd.to_numeric(frame["race_id"], errors="raise").astype("int64")
    frame["horse_id"] = pd.to_numeric(frame["horse_id"], errors="raise").astype("int64")
    target = frame[
        frame["organizer_code"].eq(1)
        & frame["surface_code"].isin([0, 1])
        & frame["venue_code"].lt(101)
        & ~frame["venue_name"].fillna("").astype(str).str.contains("帯広", regex=False)
    ].copy()
    if target.empty:
        raise ValueError("対象レースがありません")
    if ~target["affiliation_code"].isin([1, 2, 3, 4]).all():
        raise ValueError("所属体系を判定できない出走があります")
    candidate = target[target["affiliation_code"].isin([1, 2]) & ~target["abnormal_code"].isin(NON_START_ABNORMAL_CODES)].copy()
    if candidate.empty:
        raise ValueError("除外後の出走馬がありません")
    duplicate_day = candidate.groupby(["horse_id", "race_date"])["race_id"].nunique()
    if duplicate_day.gt(1).any():
        raise ValueError("同一馬の同日複数レース出走があります")
    horse_ids = np.sort(candidate["horse_id"].unique()).astype("int64")
    horse_map = {int(horse_id): index for index, horse_id in enumerate(horse_ids)}
    candidate["horse_idx"] = candidate["horse_id"].map(horse_map).astype("int32")
    candidate["date"] = (candidate["race_date"].astype("int64") // 86_400_000_000_000).astype("int32")
    candidate = candidate.sort_values(["date", "race_id", "horse_number"], kind="stable")

    races: list[Race] = []
    for race_index, (race_id, group) in enumerate(candidate.groupby("race_id", sort=False)):
        first = group.iloc[0]
        metadata_columns = ["race_date", "condition_code", "surface_code", "distance", "venue_code", "n_registered"]
        for column in metadata_columns:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"レース内で{column}が一致しません: {race_id}")
        official_finish = pd.to_numeric(group["finish_position"], errors="coerce").fillna(0).to_numpy(dtype=np.int16)
        abnormal = pd.to_numeric(group["abnormal_code"], errors="coerce").fillna(0).to_numpy(dtype=np.int16)
        unexplained = (official_finish == 0) & ~np.isin(abnormal, [4, 5])
        if unexplained.any() and not allow_unranked:
            raise ValueError(f"理由不明の順位欠損があります: {race_id}")
        if ((official_finish > 0) & np.isin(abnormal, [4, 5])).any():
            raise ValueError(f"順位なし異常区分と確定着順が矛盾しています: {race_id}")
        finish = np.zeros(len(group), dtype=np.int16)
        positive_ranks = sorted(np.unique(official_finish[official_finish > 0]))
        for relative_rank, official_rank in enumerate(positive_ranks, start=1):
            finish[official_finish == official_rank] = relative_rank
        orders = make_orders(finish)
        weights = pd.to_numeric(group["weight_kg"], errors="coerce").to_numpy(dtype=float)
        weights[weights <= 0] = np.nan
        missing_weight_count = int(np.isnan(weights).sum())
        if missing_weight_count:
            if np.isnan(weights).all():
                raise ValueError(f"全馬の斤量が欠損しています: {race_id}")
            weights[np.isnan(weights)] = np.nanmean(weights)
        raw_race = target[target["race_id"].eq(race_id)]
        n_removed = int(raw_race["affiliation_code"].isin([3, 4]).sum())
        result_version = str(first["result_version"]) if "result_version" in group.columns else "data_division=7"
        created = str(first["race_data_created"]) if "race_data_created" in group.columns else ""
        races.append(Race(
            race_index=race_index,
            race_id=int(race_id),
            date=int(first["date"]),
            horse_idx=group["horse_idx"].to_numpy(dtype=np.int32),
            horse_number=pd.to_numeric(group["horse_number"], errors="raise").to_numpy(dtype=np.int16),
            weight_kg=weights,
            finish=finish,
            abnormal_code=abnormal,
            orders=orders,
            unranked=np.flatnonzero(finish == 0).astype(np.int16),
            pre_nstarts=np.zeros(len(group), dtype=np.int32),
            condition_code=int(first["condition_code"]),
            surface_code=int(first["surface_code"]),
            distance=int(first["distance"]),
            venue_code=int(first["venue_code"]),
            n_field_original=int(first["n_registered"]),
            n_field_before_non_jra=int(len(raw_race)),
            n_removed_non_jra=n_removed,
            weight_missing_count=missing_weight_count,
            result_version=result_version,
            race_data_created=created,
        ))
    races.sort(key=lambda race: (race.date, race.race_id))
    for index, race in enumerate(races):
        race.race_index = index

    counts = np.zeros(len(horse_ids), dtype=np.int32)
    by_date: dict[int, list[Race]] = {}
    for race in races:
        by_date.setdefault(race.date, []).append(race)
    for date in sorted(by_date):
        day_races = by_date[date]
        for race in day_races:
            race.pre_nstarts = counts[race.horse_idx].copy()
            if race.condition_code == 701 and np.any(race.pre_nstarts > 0):
                raise ValueError(f"新馬戦に過去グラフ出走馬がいます: {race.race_id}")
        day_horses = np.unique(np.concatenate([race.horse_idx for race in day_races]))
        counts[day_horses] += 1

    diagnostics = {
        "raw_rows": int(len(frame)),
        "target_rows_before_exclusions": int(len(target)),
        "candidate_rows": int(len(candidate)),
        "races": len(races),
        "horses": int(len(horse_ids)),
        "non_jra_rows_removed": int(target["affiliation_code"].isin([3, 4]).sum()),
        "non_start_rows_removed": int((target["affiliation_code"].isin([1, 2]) & target["abnormal_code"].isin(NON_START_ABNORMAL_CODES)).sum()),
        "races_with_fewer_than_two": int(sum(race.n < 2 for race in races)),
        "races_without_likelihood": int(sum(not race.has_likelihood for race in races)),
        "weight_missing_starts": int(sum(race.weight_missing_count for race in races)),
        "races_with_weight_missing": int(sum(race.weight_missing_count > 0 for race in races)),
        "max_tie_permutations": int(max(len(race.orders) for race in races)),
    }
    return PreparedInput(races=races, horse_ids=horse_ids, source_hash=source_hash, diagnostics=diagnostics)


def load_raw_from_db(db_path: Path = DB_PATH, start_date: str = "2012-01-01", end_date: str | None = "2025-01-01", final_only: bool = False) -> pd.DataFrame:
    connection_string = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};ReadOnly=1;"
    end_clause = f"AND s.[月日] < #{end_date}#" if end_date is not None else ""
    final_clause = "AND s.[データ区分] = '7'" if final_only else ""
    sql = f"""
        SELECT s.[競走コード] AS race_id, s.[月日] AS race_date,
               s.[主催者コード] AS organizer_code,
               s.[トラック種別コード] AS surface_code,
               s.[場コード] AS venue_code, s.[場名] AS venue_name,
               s.[距離] AS distance, s.[競走条件コード] AS condition_code,
               s.[登録頭数] AS n_registered,
               s.[データ作成年月日] AS race_data_created,
               s.[データ区分] AS result_version,
               r.[馬番] AS horse_number, r.[血統登録番号] AS horse_id,
               r.[負担重量] AS weight_raw,
               r.[確定着順] AS finish_position,
               r.[異常区分コード] AS abnormal_code,
               rm.[東西所属コード] AS affiliation_code
        FROM ([レースT] AS s INNER JOIN [出走馬T] AS r
          ON s.[競走コード] = r.[競走コード])
        INNER JOIN [出走馬マスタ] AS rm
          ON r.[競走コード] = rm.[競走コード]
         AND r.[馬番] = rm.[馬番]
        WHERE s.[月日] >= #{start_date}#
          {end_clause}
          {final_clause}
          AND s.[主催者コード] = 1
          AND s.[トラック種別コード] IN (0, 1)
    """
    with pyodbc.connect(connection_string, readonly=True, autocommit=True) as con:
        frame = pd.read_sql(sql, con)
    frame["weight_kg"] = pd.to_numeric(frame["weight_raw"], errors="coerce") / 10.0
    frame.loc[frame["weight_kg"].le(0), "weight_kg"] = np.nan
    frame["result_version"] = "BaoZ_data_division_" + frame["result_version"].astype(str)
    return frame.drop(columns=["weight_raw"])


def load_or_prepare(db_path: Path = DB_PATH, use_cache: bool = True) -> PreparedInput:
    CACHE_DIR.mkdir(exist_ok=True)
    source_hash = file_sha256(db_path)
    cache_path = CACHE_DIR / f"prepared_input_{source_hash[:16]}.pkl"
    metadata_path = CACHE_DIR / f"prepared_input_{source_hash[:16]}_meta.json"
    if use_cache and cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_hash") == source_hash:
            with cache_path.open("rb") as handle:
                prepared = pickle.load(handle)
            if prepared.source_hash == source_hash:
                return prepared
    raw = load_raw_from_db(db_path)
    prepared = prepare_input(raw, source_hash=source_hash)
    with cache_path.open("wb") as handle:
        pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_path.write_text(json.dumps({
        "source_hash": source_hash,
        "database": str(db_path),
        "diagnostics": prepared.diagnostics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return prepared
