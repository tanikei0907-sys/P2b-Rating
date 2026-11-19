from __future__ import annotations

import numpy as np
import pandas as pd


def sequential_margins(finish: np.ndarray, times: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.full(5, np.nan, dtype=float)
    statuses = np.zeros(5, dtype=np.int8)
    if k_max <= 0:
        return deltas, statuses
    ranks = sorted(np.unique(finish[finish > 0]))
    groups = []
    for rank in ranks:
        indices = np.flatnonzero(finish == rank)
        valid_times = times[indices][np.isfinite(times[indices])]
        group_time = float(valid_times.min()) if len(valid_times) else np.nan
        groups.append({"rank": int(rank), "size": len(indices), "time": group_time})
    stage = 0
    for group_index, group in enumerate(groups):
        for within_group in range(group["size"]):
            if stage >= k_max or stage >= 5:
                break
            if within_group + 1 < group["size"]:
                deltas[stage] = 0.0
                statuses[stage] = 1
            elif group_index + 1 < len(groups):
                next_time = groups[group_index + 1]["time"]
                if not np.isfinite(group["time"]) or not np.isfinite(next_time):
                    statuses[stage] = 2
                else:
                    difference = next_time - group["time"]
                    if difference < -1e-9:
                        deltas[stage] = 0.0
                        statuses[stage] = 3
                    else:
                        deltas[stage] = max(0.0, difference)
                        statuses[stage] = 1
            stage += 1
        if stage >= k_max or stage >= 5:
            break
    return deltas, statuses


def multiplier_arrays(margins: pd.DataFrame, prepared, config: dict) -> tuple[np.ndarray, np.ndarray]:
    indexed = margins.set_index("race_id").loc[[race.race_id for race in prepared.races]]
    deltas = indexed[[f"stage_{index}" for index in range(1, 6)]].to_numpy(dtype=float)
    if config["normalize"]:
        scale = indexed["distance"].to_numpy(dtype=float) / 1000.0
        if np.any(scale <= 0):
            raise ValueError("距離が正ではありません")
        deltas = deltas / scale[:, None]
    gamma = float(config["gamma"])
    cap = float(config["m_max"])
    if gamma < 0 or cap < 1:
        raise ValueError("マージン感度・上限が不正です")
    n = len(prepared.races)
    if config["method"] == "A":
        stages = np.minimum(cap, 1.0 + gamma * deltas)
        stages[~np.isfinite(deltas)] = 1.0
        return stages, np.ones(n, dtype=float)
    first = deltas[:, 0]
    second = deltas[:, 1]
    combined = first.copy()
    both = np.isfinite(first) & np.isfinite(second)
    combined[both] = 0.75 * first[both] + 0.25 * second[both]
    race_scales = np.clip(1.0 + gamma * combined, float(config.get("m_min", 0.5)), cap)
    race_scales[~np.isfinite(combined)] = 1.0
    return np.ones((n, 5), dtype=float), race_scales
