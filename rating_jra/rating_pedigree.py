from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

from rating_core import file_sha256, harville_inclusion_probabilities
from rating_initialization import FeatureBlock

CACHE_DIR = Path(__file__).with_name("stage3_cache")
STATUS_LABELS = {
    0: "NOT_USED",
    1: "OK",
    2: "NO_HISTORY",
    3: "ID_MISSING",
    4: "EXISTING_UNKNOWN",
}


def standardization_matches(actual: list[dict], expected: list[dict], rtol: float = 1.0e-9, atol: float = 1.0e-12) -> bool:
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected):
        if left.keys() != right.keys():
            return False
        for key in ["center", "scale", "raw_std"]:
            if not np.isclose(left[key], right[key], rtol=rtol, atol=atol):
                return False
        for key in ["feature", "side", "metric", "standardization_horses", "kappa"]:
            if left[key] != right[key]:
                return False
    return True


@dataclass
class PedigreeBase:
    candidates: pd.DataFrame
    history: pd.DataFrame
    cache_key: str


@dataclass
class StandardizedFeatures:
    kappa: int
    provider: dict[int, FeatureBlock]
    constants: list[dict]


def load_pedigree_ids(db_path: Path, fallback_to_horse_master: bool = False) -> pd.DataFrame:
    connection_string = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};ReadOnly=1;"
    with pyodbc.connect(connection_string, readonly=True, autocommit=True) as con:
        pedigree = pd.read_sql(
            """
            SELECT [血統登録番号] AS horse_id,
                   [父繁殖登録番号] AS sire_id,
                   [母父繁殖登録番号] AS bms_id
            FROM [産駒マスタ]
            """,
            con,
        )
        if fallback_to_horse_master:
            horse_master = pd.read_sql(
                """
                SELECT [血統登録番号] AS horse_id,
                       [繁殖登録番号1] AS master_sire_id,
                       [繁殖登録番号5] AS master_bms_id
                FROM [競走馬マスタ]
                """,
                con,
            )
            pedigree = pedigree.merge(horse_master, on="horse_id", how="outer", validate="many_to_one")
            for source, fallback in [("sire_id", "master_sire_id"), ("bms_id", "master_bms_id")]:
                pedigree[source] = pedigree[source].where(
                    pedigree[source].notna() & pedigree[source].ne(0),
                    pedigree[fallback],
                )
            pedigree = pedigree.drop(columns=["master_sire_id", "master_bms_id"])
        return pedigree


def build_or_load_base(
    snapshots_path: Path,
    db_path: Path,
    use_cache: bool = True,
    pedigree_loader=None,
) -> PedigreeBase:
    CACHE_DIR.mkdir(exist_ok=True)
    key = "v3_" + file_sha256(snapshots_path)[:16] + "_" + file_sha256(db_path)[:16]
    cache_path = CACHE_DIR / f"pedigree_base_{key}.pkl"
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    snapshots = pd.read_csv(snapshots_path, parse_dates=["race_date"])
    snapshots = snapshots.sort_values(["race_index"], kind="stable").reset_index(drop=True)
    snapshots["date"] = (snapshots["race_date"].astype("int64") // 86_400_000_000_000).astype(np.int32)
    eligible = snapshots["debut_type_code"].eq(1) & snapshots["n_graph_starts"].le(2)
    eligible_indices = np.flatnonzero(eligible.to_numpy())
    e_win = np.zeros(len(eligible_indices), dtype=float)
    e_top3 = np.zeros(len(eligible_indices), dtype=float)
    position = np.full(len(snapshots), -1, dtype=np.int32)
    position[eligible_indices] = np.arange(len(eligible_indices), dtype=np.int32)
    for _, group in snapshots.loc[eligible].groupby("race_index", sort=True):
        inclusion = harville_inclusion_probabilities(group["m_pre"].to_numpy(dtype=float), 3)
        target = position[group.index.to_numpy()]
        e_win[target] = inclusion[:, 0]
        e_top3[target] = inclusion.sum(axis=1)
    history = snapshots.loc[eligible, [
        "race_index", "race_date", "date", "horse_id", "finish_position", "n_graph_starts", "condition_code",
    ]].copy()
    history["e_win"] = e_win
    history["e_top3"] = e_top3
    history["actual_win"] = history["finish_position"].eq(1).astype(float)
    history["actual_top3"] = history["finish_position"].between(1, 3).astype(float)
    history["r_win"] = history["actual_win"] - history["e_win"]
    history["r_top3"] = history["actual_top3"] - history["e_top3"]
    pedigree = (pedigree_loader or load_pedigree_ids)(db_path)
    history = history.merge(pedigree, on="horse_id", how="left", validate="many_to_one")

    candidates = snapshots[[
        "race_index", "race_date", "date", "horse_id", "debut_type_code", "n_graph_starts", "condition_code",
    ]].copy()
    candidates = candidates.merge(pedigree, on="horse_id", how="left", validate="many_to_one")
    for side in ["sire", "bms"]:
        N, sum_win, sum_top3, status = aggregate_side(candidates, history, side)
        candidates[f"N_{side}"] = N
        candidates[f"{side}_sum_win"] = sum_win
        candidates[f"{side}_sum_top3"] = sum_top3
        candidates[f"{side}_status"] = status
    result = PedigreeBase(candidates=candidates, history=history, cache_key=key)
    if use_cache:
        with cache_path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def aggregate_side(candidates: pd.DataFrame, history: pd.DataFrame, side: str):
    id_column = f"{side}_id"
    n = len(candidates)
    N = np.zeros(n, dtype=np.int32)
    sum_win = np.zeros(n, dtype=float)
    sum_top3 = np.zeros(n, dtype=float)
    status = np.full(n, 3, dtype=np.int8)
    status[candidates["debut_type_code"].to_numpy() == 4] = 4
    eligible = candidates["debut_type_code"].isin([1, 2, 3]) & candidates[id_column].notna() & candidates[id_column].ne(0)
    status[eligible.to_numpy()] = 2
    eligible_history = history[history[id_column].notna() & history[id_column].ne(0)].copy()
    candidate_groups = candidates.loc[eligible].groupby(id_column).indices
    history_groups = eligible_history.groupby(id_column).indices
    eligible_positions = np.flatnonzero(eligible.to_numpy())
    candidate_dates = candidates["date"].to_numpy()
    history_dates = eligible_history["date"].to_numpy()
    history_win = eligible_history["r_win"].to_numpy(dtype=float)
    history_top3 = eligible_history["r_top3"].to_numpy(dtype=float)
    for pedigree_id, positions_in_subset in candidate_groups.items():
        if pedigree_id not in history_groups:
            continue
        positions = eligible_positions[positions_in_subset]
        hpos = history_groups[pedigree_id]
        order = np.argsort(history_dates[hpos], kind="stable")
        sorted_positions = hpos[order]
        dates = history_dates[sorted_positions]
        prefix_win = np.concatenate(([0.0], np.cumsum(history_win[sorted_positions])))
        prefix_top3 = np.concatenate(([0.0], np.cumsum(history_top3[sorted_positions])))
        cut = np.searchsorted(dates, candidate_dates[positions], side="left")
        N[positions] = cut
        sum_win[positions] = prefix_win[cut]
        sum_top3[positions] = prefix_top3[cut]
        status[positions] = np.where(cut > 0, 1, 2)
    return N, sum_win, sum_top3, status


def build_or_load_features(base: PedigreeBase, kappa: int, use_cache: bool = True) -> StandardizedFeatures:
    CACHE_DIR.mkdir(exist_ok=True)
    if kappa not in (20, 50, 100, 200):
        raise ValueError("kappa_s候補外です")
    cache_path = CACHE_DIR / f"features_{base.cache_key}_{kappa}.pkl"
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    frame = base.candidates.copy()
    tuning_debut = (
        frame["debut_type_code"].eq(1)
        & frame["n_graph_starts"].eq(0)
        & frame["condition_code"].eq(701)
        & frame["race_date"].between("2017-01-01", "2020-12-31")
    )
    X_raw = np.zeros((len(frame), 4), dtype=float)
    columns = ["sire", "sire", "bms", "bms"]
    metrics = ["win", "top3", "win", "top3"]
    for index, (side, metric) in enumerate(zip(columns, metrics)):
        N = frame[f"N_{side}"].to_numpy(dtype=float)
        sums = frame[f"{side}_sum_{metric}"].to_numpy(dtype=float)
        X_raw[:, index] = sums / (N + float(kappa))
    X = np.zeros_like(X_raw)
    constants = []
    for index, (side, metric) in enumerate(zip(columns, metrics)):
        status = frame[f"{side}_status"].to_numpy()
        valid = tuning_debut.to_numpy() & (status == 1)
        mean = float(X_raw[valid, index].mean()) if valid.any() else 0.0
        std = float(X_raw[valid, index].std()) if valid.any() else 0.0
        scale = std if std >= 1.0e-8 else 1.0
        ok = status == 1
        X[ok, index] = (X_raw[ok, index] - mean) / scale
        X[frame["debut_type_code"].to_numpy() == 4, index] = np.nan
        constants.append({
            "feature": f"X_{side}_{metric}",
            "side": side,
            "metric": metric,
            "center": mean,
            "scale": scale,
            "raw_std": std,
            "standardization_horses": int(valid.sum()),
            "kappa": kappa,
        })
    provider: dict[int, FeatureBlock] = {}
    for race_index, group in frame.groupby("race_index", sort=True):
        indices = group.index.to_numpy()
        provider[int(race_index)] = FeatureBlock(
            X=X[indices].copy(),
            N_sire=frame["N_sire"].to_numpy()[indices].astype(np.int32),
            N_bms=frame["N_bms"].to_numpy()[indices].astype(np.int32),
            sire_status=frame["sire_status"].to_numpy()[indices].astype(np.int8),
            bms_status=frame["bms_status"].to_numpy()[indices].astype(np.int8),
        )
    result = StandardizedFeatures(kappa=kappa, provider=provider, constants=constants)
    if use_cache:
        with cache_path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def pedigree_auxiliary(history: pd.DataFrame, side: str) -> pd.DataFrame:
    id_column = f"{side}_id"
    valid = history[history[id_column].notna() & history[id_column].ne(0)].copy()
    first = valid[valid["n_graph_starts"].eq(0)].groupby(id_column).agg(
        new_horse_starts=("horse_id", "size"),
        new_horse_wins=("actual_win", "sum"),
        new_horse_top3=("actual_top3", "sum"),
    )
    within3 = valid.groupby(id_column).agg(
        starts=("horse_id", "size"),
        offspring=("horse_id", "nunique"),
        wins_within3=("actual_win", "sum"),
    )
    result = first.join(within3, how="outer").fillna(0)
    result["new_horse_win_rate"] = result["new_horse_wins"] / result["new_horse_starts"]
    result["new_horse_top3_rate"] = result["new_horse_top3"] / result["new_horse_starts"]
    result["win_within3_rate"] = result["wins_within3"] / result["starts"]
    result["offspring_winning_within3"] = valid[valid["actual_win"].eq(1)].groupby(id_column)["horse_id"].nunique()
    result["offspring_winning_within3"] = result["offspring_winning_within3"].fillna(0)
    result["offspring_win_within3_rate"] = result["offspring_winning_within3"] / result["offspring"]
    return result.reset_index().rename(columns={id_column: f"{side}_id"})
