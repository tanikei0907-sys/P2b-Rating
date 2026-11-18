from __future__ import annotations

import numpy as np
import pandas as pd

from rating_evaluation import bootstrap_ci, group_seed

MODELS = ["P0", "P1", "P2a", "P2b"]


def combine_metrics(model_metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = model_metrics["P2b"][[
        "race_index", "race_id", "race_date", "year", "E", "E_clean", "F", "tuning", "validation",
        "condition_code", "surface_code", "distance", "n_field_retained", "n_removed_nonJRA",
        "mean_n_graph_starts", "n_zero_experience", "class", "surface", "distance_band", "field_band",
        "experience_band", "s0_nll",
    ]].copy()
    for model in MODELS:
        values = model_metrics[model][["race_id", "p0_nll"]].rename(columns={"p0_nll": f"{model}_nll"})
        base = base.merge(values, on="race_id", how="left", validate="one_to_one")
    return base


def pairwise_comparisons(combined: pd.DataFrame, repetitions: int = 2000, base_seed: int = 20241117) -> pd.DataFrame:
    rows = []
    for previous, model in zip(MODELS[:-1], MODELS[1:]):
        delta_column = f"{model}_minus_{previous}"
        combined[delta_column] = combined[f"{model}_nll"] - combined[f"{previous}_nll"]
        for period in ["tuning", "validation"]:
            period_frame = combined[combined[period]]
            for pool in ["E", "E_clean", "F"]:
                pool_frame = period_frame[period_frame[pool]]
                groups = [("POOLED", pool_frame)] + [
                    (int(year), pool_frame[pool_frame["year"].eq(year)])
                    for year in sorted(pool_frame["year"].unique())
                ]
                for year, group in groups:
                    seed = group_seed(base_seed, model, previous, period, pool, year)
                    lower, upper = bootstrap_ci(group[delta_column].to_numpy(dtype=float), seed, repetitions)
                    rows.append({
                        "comparison": f"{model}-{previous}",
                        "model": model,
                        "baseline": previous,
                        "period": period,
                        "pool": pool,
                        "year": year,
                        "n_races": len(group),
                        "model_nll": group[f"{model}_nll"].mean(),
                        "baseline_nll": group[f"{previous}_nll"].mean(),
                        "delta_nll": group[delta_column].mean(),
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_seed": seed,
                    })
    return pd.DataFrame(rows)


def debut_majority(snapshots: pd.DataFrame) -> pd.DataFrame:
    def majority(values: pd.Series) -> int:
        counts = values.value_counts()
        if len(counts) == 1 or counts.iloc[0] > counts.iloc[1]:
            return int(counts.index[0])
        return 0
    labels = {0: "MIXED", 1: "新馬戦", 2: "未勝利戦", 3: "その他のクラス", 4: "既存馬（デビュー時点不明）"}
    result = snapshots.groupby("race_index")["debut_type_code"].agg(majority).reset_index()
    result["debut_majority"] = result["debut_type_code"].map(labels)
    return result[["race_index", "debut_majority"]]


def pairwise_layers(combined: pd.DataFrame, repetitions: int = 2000, base_seed: int = 20241117) -> pd.DataFrame:
    validation = combined[combined["validation"] & combined["E"]].copy()
    rows = []
    for comparison, previous, model in [("P1-P0", "P0", "P1"), ("P2a-P1", "P1", "P2a"), ("P2b-P2a", "P2a", "P2b")]:
        delta_column = f"{model}_minus_{previous}"
        for layer_type, column in [
            ("発走前n_graph_starts_レース平均", "experience_band"),
            ("初出走形態_レース内多数", "debut_majority"),
            ("クラス", "class"),
            ("芝ダート", "surface"),
            ("距離帯", "distance_band"),
            ("頭数帯", "field_band"),
        ]:
            for level, group in validation.groupby(column, sort=False):
                seed = group_seed(base_seed, "stage3_layer", comparison, layer_type, level)
                lower, upper = bootstrap_ci(group[delta_column].to_numpy(dtype=float), seed, repetitions)
                rows.append({
                    "comparison": comparison,
                    "layer_type": layer_type,
                    "level": level,
                    "n_races": len(group),
                    "model_nll": group[f"{model}_nll"].mean(),
                    "baseline_nll": group[f"{previous}_nll"].mean(),
                    "delta_nll": group[delta_column].mean(),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": seed,
                })
    return pd.DataFrame(rows)


def stage3_judgment(combined: pd.DataFrame, repetitions: int = 2000, base_seed: int = 20241117) -> tuple[pd.DataFrame, dict]:
    frame = combined[combined["validation"] & combined["E"]].copy()
    frame["delta"] = frame["P2b_nll"] - frame["P2a_nll"]
    early = frame[frame["mean_n_graph_starts"].le(3.0)]
    experienced = frame[frame["mean_n_graph_starts"].gt(3.0)]
    seed = group_seed(base_seed, "stage3_judgment_early")
    lower, upper = bootstrap_ci(early["delta"].to_numpy(dtype=float), seed, repetitions)
    yearly = early.groupby("year")["delta"].mean()
    negative_years = int(yearly.lt(0).sum())
    early_delta = float(early["delta"].mean())
    experienced_delta = float(experienced["delta"].mean())
    criterion_1 = upper < 0
    criterion_2 = experienced_delta <= 0.0005
    criterion_3 = negative_years >= 3
    rows = [
        {"criterion": "経験<=3・検証EプールΔNLL 95%CI上限<0", "value": upper, "pass": criterion_1},
        {"criterion": "経験>=4・検証EプールΔNLL点推定<=+0.0005", "value": experienced_delta, "pass": criterion_2},
        {"criterion": "経験<=3・4年中3年以上でΔNLL<0", "value": negative_years, "pass": criterion_3},
        {"criterion": "3基準を全て満たすか（最終判断は依頼者）", "value": "MET" if criterion_1 and criterion_2 and criterion_3 else "NOT_MET", "pass": criterion_1 and criterion_2 and criterion_3},
    ]
    details = {
        "early_races": len(early),
        "early_delta_nll": early_delta,
        "early_ci95_lower": lower,
        "early_ci95_upper": upper,
        "early_negative_years": negative_years,
        "early_yearly_delta": {str(int(year)): float(value) for year, value in yearly.items()},
        "experienced_races": len(experienced),
        "experienced_delta_nll": experienced_delta,
        "criteria_met": bool(criterion_1 and criterion_2 and criterion_3),
    }
    return pd.DataFrame(rows), details


def pedigree_information(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {
        "全出走": snapshots,
        "初登場": snapshots[snapshots["n_graph_starts"].eq(0) & snapshots["debut_type_code"].isin([1, 2, 3])],
    }
    information_rows = []
    status_rows = []
    quantiles = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    for scope, frame in frames.items():
        for side in ["sire", "bms"]:
            values = frame[f"N_{side}"]
            information_rows.append({
                "scope": scope,
                "side": side,
                "rows": len(frame),
                "mean": values.mean(),
                "std": values.std(),
                **{f"q{int(q * 100):02d}": values.quantile(q) for q in quantiles},
            })
            status = frame[f"{side}_status_code"].map({0: "NOT_USED", 1: "OK", 2: "NO_HISTORY", 3: "ID_MISSING", 4: "EXISTING_UNKNOWN"}).value_counts(dropna=False)
            for label, count in status.items():
                status_rows.append({
                    "scope": scope,
                    "side": side,
                    "status": label,
                    "count": int(count),
                    "rate": float(count / len(frame)) if len(frame) else np.nan,
                })
    return pd.DataFrame(information_rows), pd.DataFrame(status_rows)
