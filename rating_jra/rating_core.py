from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

PRIZE_POINTS = np.array([100, 40, 25, 15, 10, 0], dtype=float)
W_SHARE = PRIZE_POINTS[:5] / PRIZE_POINTS.sum()
PRIZE_GAPS = -np.diff(PRIZE_POINTS)
W_PTS = PRIZE_GAPS / PRIZE_GAPS.sum()
W_WIN = np.eye(5, dtype=float)[0]
W_WIN_HEAVY = np.array([80, 10, 5, 3, 2], dtype=float)
W_WIN_HEAVY = W_WIN_HEAVY / W_WIN_HEAVY.sum()
WEIGHTS = {
    "W_share": W_SHARE,
    "W_pts": W_PTS,
    "W_win": W_WIN,
    "W_win_heavy": W_WIN_HEAVY,
}
W_EVAL = W_SHARE.copy()
for _weight in WEIGHTS.values():
    assert _weight.shape == (5,)
    assert np.all(_weight >= 0)
    assert np.isclose(_weight.sum(), 1.0)


@dataclass(frozen=True)
class P0Config:
    var0: float
    q: float
    kappa: float
    beta_kg: float
    weight: str
    tau: float | None = None

    def to_dict(self) -> dict:
        return {
            "var0": self.var0,
            "q": self.q,
            "kappa": self.kappa,
            "beta_kg": self.beta_kg,
            "weight": self.weight,
            "tau": self.tau,
        }


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def effective_variance(var: np.ndarray, q: float, days: np.ndarray, var_cap: float) -> np.ndarray:
    return np.minimum(var + q * np.maximum(days, 0) / 30.0, var_cap)


def pl_single_order(
    scores: np.ndarray,
    order: np.ndarray,
    unranked: np.ndarray,
    k_max: int,
    weights: np.ndarray,
    need_info: bool = False,
):
    n = len(scores)
    gradient = np.zeros(n, dtype=float)
    diagonal = np.zeros(n, dtype=float)
    information = np.zeros((n, n), dtype=float) if need_info else None
    log_likelihood = 0.0
    for stage in range(k_max):
        candidates = np.concatenate((order[stage:], unranked))
        winner = order[stage]
        candidate_scores = scores[candidates]
        offset = float(np.max(candidate_scores))
        exp_scores = np.exp(candidate_scores - offset)
        denominator = float(exp_scores.sum())
        log_denominator = offset + np.log(denominator)
        probabilities = exp_scores / denominator
        weight = float(weights[stage])
        log_likelihood += weight * (scores[winner] - log_denominator)
        gradient[candidates] -= weight * probabilities
        gradient[winner] += weight
        diagonal[candidates] += weight * probabilities * (1.0 - probabilities)
        if need_info:
            block = -weight * np.outer(probabilities, probabilities)
            block[np.diag_indices(len(candidates))] += weight * probabilities
            information[np.ix_(candidates, candidates)] += block
    return log_likelihood, gradient, diagonal, information


def pl_race_single(
    scores: np.ndarray,
    orders: np.ndarray,
    unranked: np.ndarray,
    k_max: int,
    weights: np.ndarray,
    need_info: bool = False,
):
    n = len(scores)
    gradient = np.zeros(n, dtype=float)
    diagonal = np.zeros(n, dtype=float)
    information = np.zeros((n, n), dtype=float) if need_info else None
    log_likelihood = 0.0
    if k_max > 0 and len(scores) >= 2:
        for order in orders:
            ll, g, h, info = pl_single_order(scores, order, unranked, k_max, weights, need_info)
            log_likelihood += ll
            gradient += g
            diagonal += h
            if need_info:
                information += info
        divisor = float(len(orders))
        log_likelihood /= divisor
        gradient /= divisor
        diagonal /= divisor
        if need_info:
            information /= divisor
    return log_likelihood, gradient, diagonal, information


def diagonal_update(
    mu: np.ndarray,
    var: np.ndarray,
    weight_kg: np.ndarray,
    orders: np.ndarray,
    unranked: np.ndarray,
    finish: np.ndarray,
    config: P0Config,
    days: np.ndarray | int = 0,
    full_information: bool = False,
):
    days_array = np.broadcast_to(np.asarray(days, dtype=float), mu.shape)
    var_eff = effective_variance(var, config.q, days_array, config.var0)
    centered_weight = weight_kg - np.mean(weight_kg)
    scores = mu - config.beta_kg * centered_weight
    k_max = min(5, int(np.count_nonzero(finish > 0)), len(mu) - 1)
    _, gradient, diagonal, information = pl_race_single(
        scores, orders, unranked, k_max, WEIGHTS[config.weight], need_info=full_information
    )
    ranked = finish > 0
    mu_post = mu.copy()
    var_post = var.copy()
    if full_information:
        precision = np.diag(1.0 / var_eff) + config.kappa * information
        increment = config.kappa * np.linalg.solve(precision, gradient)
        covariance = np.linalg.inv(precision)
        mu_post[ranked] = mu[ranked] + increment[ranked]
        var_post[ranked] = np.diag(covariance)[ranked]
    elif k_max > 0 and len(mu) >= 2:
        precision = 1.0 / var_eff + config.kappa * diagonal
        mu_post[ranked] = mu[ranked] + config.kappa * gradient[ranked] / precision[ranked]
        var_post[ranked] = 1.0 / precision[ranked]
    return {
        "mu_post": mu_post,
        "var_post": var_post,
        "var_pre_eff": var_eff,
        "scores": scores,
        "gradient": gradient,
        "diagonal_information": diagonal,
        "information": information,
    }


@njit(cache=True)
def harville_inclusion_probabilities(scores: np.ndarray, max_places: int = 5) -> np.ndarray:
    n = len(scores)
    places = min(max_places, n)
    result = np.zeros((n, places), dtype=np.float64)
    if n == 0 or places == 0:
        return result
    scaled = np.exp(scores - np.max(scores))
    total = 0.0
    for value in scaled:
        total += value
    if places <= 3:
        if n == 1:
            result[0, 0] = 1.0
            return result
        for first in range(n):
            for second in range(n):
                if first == second:
                    continue
                pair_probability = scaled[first] / total * scaled[second] / (total - scaled[first])
                result[first, 0] += pair_probability
                result[second, 1] += pair_probability
                if places >= 3:
                    for third in range(n):
                        if third == first or third == second:
                            continue
                        probability = pair_probability * scaled[third] / (total - scaled[first] - scaled[second])
                        result[third, 2] += probability
        return result
    subset_count = 1 << n
    subset_sums = np.zeros(subset_count, dtype=np.float64)
    probabilities = np.zeros(subset_count, dtype=np.float64)
    probabilities[0] = 1.0
    current = np.zeros(subset_count, dtype=np.int64)
    following = np.zeros(subset_count, dtype=np.int64)
    current[0] = 0
    current_count = 1
    for stage in range(places):
        next_count = 0
        for index in range(current_count):
            subset = current[index]
            denominator = total - subset_sums[subset]
            base = probabilities[subset] / denominator
            for horse in range(n):
                bit = 1 << horse
                if subset & bit == 0:
                    probability = base * scaled[horse]
                    result[horse, stage] += probability
                    if stage + 1 < places:
                        new_subset = subset | bit
                        if probabilities[new_subset] == 0.0:
                            following[next_count] = new_subset
                            next_count += 1
                            subset_sums[new_subset] = subset_sums[subset] + scaled[horse]
                        probabilities[new_subset] += probability
        temporary = current
        current = following
        following = temporary
        current_count = next_count
    return result


def prize_ndcg_at_5(scores: np.ndarray, finish: np.ndarray) -> float:
    n = len(scores)
    if n == 0:
        return np.nan
    gains = np.zeros(n, dtype=float)
    for rank in np.unique(finish[finish > 0]):
        if 1 <= rank <= 5:
            gains[finish == rank] = PRIZE_POINTS[int(rank) - 1]
    predicted_groups = []
    for score in np.unique(scores)[::-1]:
        predicted_groups.append(np.flatnonzero(scores == score))
    discounts = 1.0 / np.log2(np.arange(n) + 2.0)
    dcg = 0.0
    cursor = 0
    for group in predicted_groups:
        width = len(group)
        dcg += float(gains[group].mean()) * float(discounts[cursor:cursor + width].sum())
        cursor += width
    ideal_gains = np.sort(gains)[::-1]
    ideal = float((ideal_gains * discounts).sum())
    return float(dcg / ideal) if ideal > 0 else np.nan
