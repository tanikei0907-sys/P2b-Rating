from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from rating_core import P0Config
from rating_data import PreparedInput
from rating_engine import run_engine
from rating_initialization import InitialRule


def early_race_mask(prepared: PreparedInput, base_mask: np.ndarray) -> np.ndarray:
    early = np.array([float(race.pre_nstarts.mean()) <= 3.0 for race in prepared.races], dtype=bool)
    return base_mask & early


def optimize_batch(
    prepared: PreparedInput,
    base_config: P0Config,
    evaluation_mask: np.ndarray,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    penalty_mask: np.ndarray,
    l2: float,
    rule_builder,
    feature_provider=None,
    model_name: str = "P1",
    maxiter: int = 40,
    epsilon: float = 1.0e-4,
) -> dict:
    x0 = np.asarray(x0, dtype=float)
    penalty_mask = np.asarray(penalty_mask, dtype=float)
    cache: dict[tuple[float, ...], float] = {}
    history = []
    counter = {"engine_runs": 0, "config_evaluations": 0}

    def evaluate(points: np.ndarray) -> np.ndarray:
        missing = [point for point in points if tuple(point) not in cache]
        if missing:
            rules = [rule_builder(point) for point in missing]
            rule = InitialRule(
                b=np.concatenate([item.b for item in rules], axis=0),
                var0_d=np.concatenate([item.var0_d for item in rules], axis=0),
                beta=np.concatenate([item.beta for item in rules], axis=0),
            )
            result = run_engine(
                prepared,
                [base_config] * len(missing),
                evaluation_mask=evaluation_mask,
                initial_rule=rule,
                feature_provider=feature_provider,
                model_name=model_name,
            )
            values = result.nll_sum / result.nll_count
            counter["engine_runs"] += 1
            counter["config_evaluations"] += len(missing)
            for point, value in zip(missing, values):
                cache[tuple(point)] = float(value)
        return np.array([cache[tuple(point)] for point in points], dtype=float)

    def objective_and_gradient(x: np.ndarray):
        dimension = len(x)
        points = [x]
        for index in range(dimension):
            plus = x.copy()
            minus = x.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            points.extend([plus, minus])
        values = evaluate(np.asarray(points))
        raw_nll = float(values[0])
        penalty = float(l2 * np.sum((x * penalty_mask) ** 2))
        gradient = (values[1::2] - values[2::2]) / (2.0 * epsilon)
        gradient = gradient + 2.0 * l2 * x * penalty_mask
        objective = raw_nll + penalty
        history.append({
            "x": x.copy(),
            "raw_nll": raw_nll,
            "penalty": penalty,
            "objective": objective,
            "gradient_max_abs": float(np.max(np.abs(gradient))),
        })
        return objective, gradient

    started = time.time()
    optimization = minimize(
        objective_and_gradient,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "maxls": 12, "ftol": 1.0e-11, "gtol": 1.0e-5},
    )
    best_x = np.asarray(optimization.x, dtype=float)
    raw_nll = float(evaluate(best_x[None, :])[0])
    penalty = float(l2 * np.sum((best_x * penalty_mask) ** 2))
    return {
        "success": bool(optimization.success),
        "message": str(optimization.message),
        "x": best_x,
        "raw_nll": raw_nll,
        "penalty": penalty,
        "objective": raw_nll + penalty,
        "nit": int(optimization.nit),
        "nfev": int(optimization.nfev),
        "engine_runs": counter["engine_runs"],
        "config_evaluations": counter["config_evaluations"],
        "elapsed_seconds": time.time() - started,
        "history": history,
        "n_races": int(evaluation_mask.sum()),
        "l2": l2,
        "epsilon": epsilon,
        "bounds": bounds,
        "x0": x0,
        "cache_nll": dict(cache),
    }
