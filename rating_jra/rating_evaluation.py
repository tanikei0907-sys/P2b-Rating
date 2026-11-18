from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize_scalar

from rating_core import PRIZE_POINTS, harville_inclusion_probabilities, prize_ndcg_at_5
from rating_data import PreparedInput
from rating_engine import EngineResult

DEBUT_TYPE_LABELS = {
    0: "UNKNOWN",
    1: "新馬戦",
    2: "未勝利戦",
    3: "その他のクラス",
    4: "既存馬（デビュー時点不明）",
}
EXPERIENCE_LABELS = ["0", "1", "2", "3", "4～5", "6～10", "11以上"]


@njit(cache=True)
def winner_ce_from_flat(scores: np.ndarray, offsets: np.ndarray, winners: np.ndarray, log_tau: float) -> float:
    tau = np.exp(log_tau)
    total = 0.0
    for race_index in range(len(offsets) - 1):
        start = offsets[race_index]
        end = offsets[race_index + 1]
        maximum = -1.0e300
        for index in range(start, end):
            value = tau * scores[index]
            if value > maximum:
                maximum = value
        denominator = 0.0
        for index in range(start, end):
            denominator += np.exp(tau * scores[index] - maximum)
        log_denominator = maximum + np.log(denominator)
        winner_count = 0
        race_ce = 0.0
        for index in range(start, end):
            if winners[index]:
                race_ce += log_denominator - tau * scores[index]
                winner_count += 1
        if winner_count:
            total += race_ce / winner_count
    return total / (len(offsets) - 1)


def flat_race_inputs(snapshots: pd.DataFrame, race_indices: np.ndarray):
    selected = snapshots[snapshots["race_index"].isin(race_indices)].sort_values(["race_index", "horse_id"])
    counts = selected.groupby("race_index", sort=True).size().to_numpy(dtype=np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
    return (
        selected["m_pre"].to_numpy(dtype=float),
        offsets,
        selected["finish_position"].eq(1).to_numpy(),
    )


def fit_tau(snapshots: pd.DataFrame, tuning_f_mask: np.ndarray) -> dict:
    race_indices = np.flatnonzero(tuning_f_mask)
    scores, offsets, winners = flat_race_inputs(snapshots, race_indices)
    optimization = minimize_scalar(
        lambda log_tau: winner_ce_from_flat(scores, offsets, winners, float(log_tau)),
        bounds=(-8.0, 8.0),
        method="bounded",
        options={"xatol": 1.0e-9},
    )
    if not optimization.success:
        raise RuntimeError(f"tau最適化に失敗しました: {optimization.message}")
    return {
        "tau": float(np.exp(optimization.x)),
        "log_tau": float(optimization.x),
        "winner_ce": float(optimization.fun),
        "n_races": int(len(race_indices)),
        "n_starts": int(len(scores)),
        "optimization_success": bool(optimization.success),
        "race_set": "2017-2020の尤度あり全レース（F相当）",
    }


def add_win_probabilities(snapshots: pd.DataFrame, tau: float) -> pd.Series:
    scaled = tau * snapshots["m_pre"]
    offset = scaled.groupby(snapshots["race_index"]).transform("max")
    exp_scores = np.exp(scaled - offset)
    denominator = exp_scores.groupby(snapshots["race_index"]).transform("sum")
    return exp_scores / denominator


def race_metrics(prepared: PreparedInput, result: EngineResult) -> pd.DataFrame:
    masks = prepared.evaluation_masks()
    snapshots = result.snapshots
    grouped = snapshots.groupby("race_index")
    experience = grouped["n_graph_starts"].mean()
    zero_counts = grouped["n_graph_starts"].apply(lambda values: int((values == 0).sum()))
    rows = []
    for race in prepared.races:
        rows.append({
            "race_index": race.race_index,
            "race_id": race.race_id,
            "race_date": pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date),
            "year": masks["years"][race.race_index],
            "p0_nll": result.nll[race.race_index],
            "s0_nll": result.s0_nll[race.race_index],
            "delta_nll": result.nll[race.race_index] - result.s0_nll[race.race_index],
            "E": bool(masks["E"][race.race_index]),
            "E_clean": bool(masks["E_clean"][race.race_index]),
            "F": bool(masks["F"][race.race_index]),
            "tuning": bool(masks["tuning"][race.race_index]),
            "validation": bool(masks["validation"][race.race_index]),
            "condition_code": race.condition_code,
            "surface_code": race.surface_code,
            "distance": race.distance,
            "n_field_retained": race.n,
            "n_removed_nonJRA": race.n_removed_non_jra,
            "mean_n_graph_starts": float(experience.loc[race.race_index]),
            "n_zero_experience": int(zero_counts.loc[race.race_index]),
        })
    frame = pd.DataFrame(rows)
    frame["class"] = frame["condition_code"].map(class_label)
    frame["surface"] = frame["surface_code"].map({0: "芝", 1: "ダート"})
    frame["distance_band"] = pd.cut(
        frame["distance"], [0, 1200, 1600, 2000, 2400, np.inf],
        labels=["～1200m", "1201～1600m", "1601～2000m", "2001～2400m", "2401m～"],
    ).astype(str)
    frame["field_band"] = pd.cut(
        frame["n_field_retained"], [0, 10, 14, np.inf], labels=["～10", "11～14", "15以上"]
    ).astype(str)
    frame["experience_band"] = experience_band(frame["mean_n_graph_starts"])
    return frame


def class_label(code: int) -> str:
    if code == 701:
        return "新馬"
    if code == 702:
        return "未出走"
    if code == 703:
        return "未勝利"
    if code == 999:
        return "オープン"
    if 1 <= code <= 700:
        return f"賞金上限{code * 100}万"
    return f"その他({code})"


def experience_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values)
    return pd.cut(
        numeric,
        [-np.inf, 0, 1, 2, 3, 5, 10, np.inf],
        labels=EXPERIENCE_LABELS,
        include_lowest=True,
    ).astype(str)


def bootstrap_ci(delta: np.ndarray, seed: int, repetitions: int = 2000) -> tuple[float, float]:
    values = np.asarray(delta, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    batch_size = 32
    cursor = 0
    while cursor < repetitions:
        width = min(batch_size, repetitions - cursor)
        indices = rng.integers(0, n, size=(width, n))
        means[cursor:cursor + width] = values[indices].mean(axis=1)
        cursor += width
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def group_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256((str(base_seed) + "::" + "::".join(map(str, parts))).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def main_comparisons(metrics: pd.DataFrame, repetitions: int = 2000, base_seed: int = 20241117) -> pd.DataFrame:
    rows = []
    for period in ["tuning", "validation"]:
        period_frame = metrics[metrics[period]]
        for pool in ["E", "E_clean", "F"]:
            pool_frame = period_frame[period_frame[pool]]
            groups = [("POOLED", pool_frame)] + [
                (int(year), pool_frame[pool_frame["year"].eq(year)])
                for year in sorted(pool_frame["year"].unique())
            ]
            for year, group in groups:
                delta = group["delta_nll"].to_numpy(dtype=float)
                seed = group_seed(base_seed, period, pool, year)
                lower, upper = bootstrap_ci(delta, seed, repetitions)
                rows.append({
                    "period": period,
                    "pool": pool,
                    "year": year,
                    "n_races": len(group),
                    "p0_nll": group["p0_nll"].mean(),
                    "s0_nll": group["s0_nll"].mean(),
                    "delta_nll": group["delta_nll"].mean(),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": seed,
                })
    return pd.DataFrame(rows)


def layer_comparisons(metrics: pd.DataFrame, repetitions: int = 2000, base_seed: int = 20241117) -> pd.DataFrame:
    validation = metrics[metrics["validation"] & metrics["E"]]
    rows = []
    for layer_type, column in [
        ("発走前n_graph_starts_レース平均", "experience_band"),
        ("クラス", "class"),
        ("芝ダート", "surface"),
        ("距離帯", "distance_band"),
        ("頭数帯", "field_band"),
    ]:
        for level, group in validation.groupby(column, sort=False):
            seed = group_seed(base_seed, "layer", layer_type, level)
            lower, upper = bootstrap_ci(group["delta_nll"].to_numpy(dtype=float), seed, repetitions)
            rows.append({
                "layer_type": layer_type,
                "level": level,
                "n_races": len(group),
                "p0_nll": group["p0_nll"].mean(),
                "s0_nll": group["s0_nll"].mean(),
                "delta_nll": group["delta_nll"].mean(),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed,
            })
    return pd.DataFrame(rows)


def secondary_diagnostics(
    prepared: PreparedInput,
    result: EngineResult,
    tau: float,
    repetitions: int,
    base_seed: int,
) -> dict[str, pd.DataFrame | dict]:
    snapshots = result.snapshots.copy()
    snapshots["p_win"] = add_win_probabilities(snapshots, tau)
    snapshots["actual_win"] = snapshots["finish_position"].eq(1)
    snapshots["actual_top3"] = snapshots["finish_position"].between(1, 3)
    snapshots["actual_top5"] = snapshots["finish_position"].between(1, 5)
    validation_races = np.flatnonzero(prepared.evaluation_masks()["validation_E"])
    validation = snapshots[snapshots["race_index"].isin(validation_races)].copy()
    winner_ce_by_race = validation[validation["actual_win"]].groupby("race_index")["p_win"].apply(
        lambda probabilities: float((-np.log(probabilities)).mean())
    )
    top_rows = []
    ndcg_values = []
    for race_index, group in validation.groupby("race_index", sort=True):
        inclusion = harville_inclusion_probabilities(group["m_pre"].to_numpy(dtype=float), 5)
        top3 = np.clip(inclusion[:, :3].sum(axis=1), 0, 1)
        top5 = np.clip(inclusion.sum(axis=1), 0, 1)
        top_rows.append(pd.DataFrame({
            "race_index": race_index,
            "predicted_top3": top3,
            "predicted_top5": top5,
            "actual_top3": group["actual_top3"].to_numpy(),
            "actual_top5": group["actual_top5"].to_numpy(),
        }))
        ndcg_values.append(prize_ndcg_at_5(group["m_pre"].to_numpy(dtype=float), group["finish_position"].to_numpy()))
    top = pd.concat(top_rows, ignore_index=True)
    calibration = pd.DataFrame([
        {
            "metric": "勝ち馬",
            "n_starts": len(validation),
            "n_races": validation["race_index"].nunique(),
            "predicted_mean": validation["p_win"].mean(),
            "actual_mean": validation["actual_win"].mean(),
            "winner_ce": winner_ce_by_race.mean(),
        },
        {
            "metric": "3着内",
            "n_starts": len(top),
            "n_races": validation["race_index"].nunique(),
            "predicted_mean": top["predicted_top3"].mean(),
            "actual_mean": top["actual_top3"].mean(),
            "winner_ce": np.nan,
        },
        {
            "metric": "5着内",
            "n_starts": len(top),
            "n_races": validation["race_index"].nunique(),
            "predicted_mean": top["predicted_top5"].mean(),
            "actual_mean": top["actual_top5"].mean(),
            "winner_ce": np.nan,
        },
    ])
    validation["win_decile"] = pd.qcut(validation["p_win"], 10, labels=False, duplicates="drop") + 1
    deciles = validation.groupby("win_decile").agg(
        n_starts=("p_win", "size"),
        mean_predicted_win=("p_win", "mean"),
        actual_win_rate=("actual_win", "mean"),
        n_races=("race_index", "nunique"),
    ).reset_index()
    distribution = validation.assign(experience_band=experience_band(validation["n_graph_starts"])).groupby("experience_band", sort=False).agg(
        n_starts=("mu_pre", "size"),
        mu_mean=("mu_pre", "mean"),
        mu_std=("mu_pre", "std"),
        mu_q05=("mu_pre", lambda values: values.quantile(0.05)),
        mu_q25=("mu_pre", lambda values: values.quantile(0.25)),
        mu_median=("mu_pre", "median"),
        mu_q75=("mu_pre", lambda values: values.quantile(0.75)),
        mu_q95=("mu_pre", lambda values: values.quantile(0.95)),
        var_mean=("var_pre_persistent", "mean"),
        var_std=("var_pre_persistent", "std"),
        var_q05=("var_pre_persistent", lambda values: values.quantile(0.05)),
        var_median=("var_pre_persistent", "median"),
        var_q95=("var_pre_persistent", lambda values: values.quantile(0.95)),
    ).reset_index()
    rest = snapshots.copy()
    rest["actual_prize_points"] = rest["finish_position"].map(
        lambda rank: PRIZE_POINTS[int(rank) - 1] if 1 <= rank <= 6 else 0.0
    )
    rest = rest.sort_values(["horse_id", "race_date", "race_id"])
    rest["previous_prize_points"] = rest.groupby("horse_id")["actual_prize_points"].shift(1)
    rest = rest[rest["race_index"].isin(np.flatnonzero(prepared.evaluation_masks()["validation_E"]))].copy()
    rest["performance_change"] = rest["actual_prize_points"] - rest["previous_prize_points"]
    rest["rest_band"] = pd.cut(
        rest["days_since_start"],
        [-1, 0, 30, 90, 180, 365, np.inf],
        labels=["0日・初登場", "1～30日", "31～90日", "91～180日", "181～365日", "366日～"],
    ).astype(str)
    rest_summary = rest.groupby("rest_band", sort=False).agg(
        n_starts=("var_pre_eff", "size"),
        mean_var_pre_eff=("var_pre_eff", "mean"),
        mean_actual_prize_points=("actual_prize_points", "mean"),
        performance_change_count=("performance_change", "count"),
        mean_performance_change=("performance_change", "mean"),
        actual_top3_rate=("actual_top3", "mean"),
    ).reset_index()
    debut = snapshots[snapshots["debut_type_code"].eq(1) & snapshots["n_graph_starts"].eq(0) & snapshots["condition_code"].eq(701)][
        ["horse_id", "actual_win"]
    ].rename(columns={"actual_win": "debut_win"})
    subsequent = snapshots[snapshots["n_graph_starts"].between(1, 3)].merge(debut, on="horse_id", how="inner")
    per_horse = subsequent.groupby(["horse_id", "debut_win"]).agg(
        observations=("mu_post", "size"),
        mean_mu_post=("mu_post", "mean"),
    ).reset_index()
    new_horse = per_horse.groupby("debut_win").agg(
        horses=("horse_id", "nunique"),
        observations=("observations", "sum"),
        mean_subsequent_mu_post=("mean_mu_post", "mean"),
    ).reset_index()
    new_horse["debut_group"] = new_horse["debut_win"].map({True: "新馬戦勝ち馬", False: "新馬戦敗戦馬"})
    means = per_horse.groupby("debut_win")["mean_mu_post"].mean()
    difference = float(means.get(True, np.nan) - means.get(False, np.nan))
    order = pd.DataFrame({
        "metric": ["PrizeNDCG@5"],
        "n_races": [len(ndcg_values)],
        "mean": [float(np.nanmean(ndcg_values))],
        "definition": ["PRIZE_POINTSをgainとし、予測score順のNDCG@5。予測同点は平均gain"],
    })
    return {
        "calibration": calibration,
        "win_deciles": deciles,
        "mu_var_by_experience": distribution,
        "rest_summary": rest_summary,
        "new_horse_subsequent": new_horse,
        "new_horse_difference": {"winner_minus_loser_mean_mu_post": difference},
        "race_order_metric": order,
    }


def anchor_transition_summary(daily: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    frame = daily.copy()
    frame["asof_date"] = pd.to_datetime(frame["asof_date"])
    rows = []
    for crossing_date in frame.loc[frame["threshold_crossing"].ne(""), "asof_date"]:
        window = frame[frame["asof_date"].between(crossing_date - pd.Timedelta(days=window_days), crossing_date + pd.Timedelta(days=window_days))]
        for group_name, group in window.groupby("anchor_group"):
            rows.append({
                "crossing_date": crossing_date,
                "anchor_group": group_name,
                "n_days": len(group),
                "window_start": group["asof_date"].min(),
                "window_end": group["asof_date"].max(),
                "mu_bar_mean": group["mu_bar_D"].mean(),
                "mu_bar_median": group["mu_bar_D"].median(),
                "max_abs_daily_delta": group["mu_bar_delta_1d"].abs().max(),
                "active_latest_mu_std_mean": group["active_latest_mu_std"].mean(),
                "mu0_mean": group["mu0_mean"].mean(),
                "n_new_horses": int(group["n_new_horses"].sum()),
            })
    return pd.DataFrame(rows)


def anchor_svg(daily: pd.DataFrame, path: Path) -> None:
    width, height = 1200, 700
    margin = 70
    frame = daily.copy()
    dates = pd.to_datetime(frame["asof_date"])
    x_values = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    crossings = frame.index[frame["threshold_crossing"].ne("")].to_numpy()
    panels = [
        ("mu_bar_D", "mu_bar_D", 65, 300),
        ("n_active_365d", "n_active_365d", 370, 605),
    ]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for column, title, top, bottom in panels:
        y_values = frame[column].to_numpy(dtype=float)
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        if np.isclose(y_min, y_max):
            y_min -= 1
            y_max += 1
        points = []
        for x, y in zip(x_values, y_values):
            sx = margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)
            sy = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
            points.append(f"{sx:.2f},{sy:.2f}")
        parts.append(f'<text x="{margin}" y="{top - 15}" font-size="18">{title}</text>')
        parts.append(f'<line x1="{margin}" y1="{bottom}" x2="{width - margin}" y2="{bottom}" stroke="black"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1764ab" stroke-width="1.5"/>')
        parts.append(f'<text x="5" y="{top + 10}" font-size="12">{y_max:.4g}</text>')
        parts.append(f'<text x="5" y="{bottom}" font-size="12">{y_min:.4g}</text>')
        if column == "n_active_365d":
            y100 = bottom - (100 - y_min) / (y_max - y_min) * (bottom - top)
            parts.append(f'<line x1="{margin}" y1="{y100:.2f}" x2="{width - margin}" y2="{y100:.2f}" stroke="#c43c39" stroke-dasharray="5,4"/>')
            parts.append(f'<text x="{width - margin - 70}" y="{y100 - 5:.2f}" font-size="12" fill="#c43c39">N=100</text>')
        for index in crossings:
            x = x_values[index]
            sx = margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)
            parts.append(f'<line x1="{sx:.2f}" y1="{top}" x2="{sx:.2f}" y2="{bottom}" stroke="#e67e22" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{margin}" y="{height - 25}" font-size="14">{dates.min().date()} － {dates.max().date()}　オレンジ破線: N=100 crossing</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_model_outputs(prepared: PreparedInput, result: EngineResult, config_hash: str, output_dir: Path) -> None:
    snapshots = result.snapshots.copy()
    snapshots["prior_type"] = "P0"
    snapshots["debut_type"] = snapshots["debut_type_code"].map(DEBUT_TYPE_LABELS)
    snapshots["X_sire"] = np.nan
    snapshots["X_bms"] = np.nan
    snapshots["N_sire"] = 0
    snapshots["N_bms"] = 0
    snapshots["pedigree_status"] = "P0_NOT_USED"
    snapshots["config_sha256"] = config_hash
    leading = [
        "race_id", "race_date", "horse_id", "mu_pre", "var_pre_persistent", "var_pre_eff",
        "m_pre", "days_since_start", "n_graph_starts", "prior_type", "mu0", "debut_type",
        "X_sire", "X_bms", "N_sire", "N_bms", "pedigree_status", "n_field_original",
        "n_field_retained", "n_removed_nonJRA",
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
    state["config_sha256"] = config_hash
    state.to_csv(output_dir / "state_2024-12-31.csv", index=False, float_format="%.12g")
    daily = result.daily.copy()
    daily["asof_date"] = pd.to_datetime(daily["asof_date"], unit="D", origin="unix")
    daily.to_csv(output_dir / "mu_anchor_daily.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d")
    transitions = anchor_transition_summary(daily)
    transitions.to_csv(output_dir / "mu_anchor_transition_summary.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d")
    anchor_svg(daily, output_dir / "mu_anchor_timeseries.svg")
    events = pd.DataFrame({
        "update_sequence": np.arange(1, len(prepared.races) + 1),
        "event_id": [f"P0::{race.race_id}::{prepared.source_hash[:16]}" for race in prepared.races],
        "race_id": [race.race_id for race in prepared.races],
        "race_date": [pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date) for race in prepared.races],
        "result_version": [race.result_version for race in prepared.races],
        "input_db_sha256": prepared.source_hash,
        "config_sha256": config_hash,
    })
    if events["event_id"].duplicated().any() or events["race_id"].duplicated().any():
        raise AssertionError("更新イベントが重複しています")
    events.to_csv(output_dir / "result_events.csv", index=False, date_format="%Y-%m-%d")
