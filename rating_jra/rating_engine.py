from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from rating_core import P0Config, WEIGHTS, W_EVAL
from rating_data import PreparedInput, Race
from rating_initialization import FeatureBlock, InitialRule


@njit(cache=True)
def pl_batch(
    scores: np.ndarray,
    orders: np.ndarray,
    unranked: np.ndarray,
    k_max: int,
    train_weights: np.ndarray,
    eval_weights: np.ndarray,
    need_eval: bool,
):
    config_count, n = scores.shape
    gradient = np.zeros((config_count, n), dtype=np.float64)
    diagonal = np.zeros((config_count, n), dtype=np.float64)
    eval_log_likelihood = np.zeros(config_count, dtype=np.float64)
    s0_log_likelihood = 0.0
    order_count = len(orders)
    for order_index in range(order_count):
        order = orders[order_index]
        for stage in range(k_max):
            winner = order[stage]
            candidate_count = len(order) - stage + len(unranked)
            maximum = -1.0e300
            for config_index in range(config_count):
                value = scores[config_index, winner]
                if value > maximum:
                    maximum = value
            maxima = np.full(config_count, -1.0e300, dtype=np.float64)
            for candidate_index in range(candidate_count):
                if candidate_index < len(order) - stage:
                    horse = order[stage + candidate_index]
                else:
                    horse = unranked[candidate_index - (len(order) - stage)]
                for config_index in range(config_count):
                    value = scores[config_index, horse]
                    if value > maxima[config_index]:
                        maxima[config_index] = value
            denominators = np.zeros(config_count, dtype=np.float64)
            for candidate_index in range(candidate_count):
                if candidate_index < len(order) - stage:
                    horse = order[stage + candidate_index]
                else:
                    horse = unranked[candidate_index - (len(order) - stage)]
                for config_index in range(config_count):
                    denominators[config_index] += np.exp(scores[config_index, horse] - maxima[config_index])
            if need_eval:
                for config_index in range(config_count):
                    log_probability = scores[config_index, winner] - (maxima[config_index] + np.log(denominators[config_index]))
                    eval_log_likelihood[config_index] += eval_weights[stage] * log_probability
                s0_log_likelihood += eval_weights[stage] * (-np.log(candidate_count))
            for candidate_index in range(candidate_count):
                if candidate_index < len(order) - stage:
                    horse = order[stage + candidate_index]
                else:
                    horse = unranked[candidate_index - (len(order) - stage)]
                for config_index in range(config_count):
                    probability = np.exp(scores[config_index, horse] - maxima[config_index]) / denominators[config_index]
                    weight = train_weights[config_index, stage]
                    gradient[config_index, horse] -= weight * probability
                    diagonal[config_index, horse] += weight * probability * (1.0 - probability)
            for config_index in range(config_count):
                gradient[config_index, winner] += train_weights[config_index, stage]
    divisor = float(order_count)
    for config_index in range(config_count):
        for horse in range(n):
            gradient[config_index, horse] /= divisor
            diagonal[config_index, horse] /= divisor
        eval_log_likelihood[config_index] /= divisor
    s0_log_likelihood /= divisor
    return gradient, diagonal, eval_log_likelihood, s0_log_likelihood


@dataclass
class EngineResult:
    nll_sum: np.ndarray
    nll_count: int
    nll: np.ndarray | None
    s0_nll: np.ndarray | None
    snapshots: pd.DataFrame | None
    daily: pd.DataFrame | None
    state: pd.DataFrame
    skip_mu_var_updates: int
    initial_zero_anchor_days: int
    fallback_anchor_days: int


SNAPSHOT_DTYPES = {
    "race_id": "int64", "race_date": "int32", "race_index": "int32", "horse_id": "int64",
    "horse_number": "int16", "finish_position": "int16", "abnormal_code": "int16",
    "weight_kg": "float64", "mu_pre": "float64", "var_pre_persistent": "float64",
    "var_pre_eff": "float64", "m_pre": "float64", "days_since_start": "int32",
    "n_graph_starts": "int32", "mu0": "float64", "debut_type_code": "int8",
    "n_field_original": "int16", "n_field_retained": "int16", "n_removed_nonJRA": "int16",
    "condition_code": "int16", "surface_code": "int8", "distance": "int16",
    "skip_mu_var_update": "bool", "mu_post": "float64", "var_post": "float64",
    "mu_bar_D": "float64", "X_sire_win": "float64", "X_sire_top3": "float64",
    "X_bms_win": "float64", "X_bms_top3": "float64", "N_sire": "int32", "N_bms": "int32",
    "sire_status_code": "int8", "bms_status_code": "int8",
}


def debut_code(condition_code: int, date: int) -> int:
    first_year_end = (pd.Timestamp("2013-01-01") - pd.Timestamp("1970-01-01")).days
    if condition_code == 701:
        return 1
    if date < first_year_end:
        return 4
    if condition_code == 703:
        return 2
    return 3


class BatchP0Engine:
    def __init__(
        self,
        prepared: PreparedInput,
        configs: list[P0Config],
        evaluation_mask: np.ndarray | None = None,
        record: bool = False,
        initial_rule: InitialRule | None = None,
        feature_provider: dict[int, FeatureBlock] | None = None,
        dry_run: bool = False,
    ):
        if not configs:
            raise ValueError("設定がありません")
        if record and len(configs) != 1:
            raise ValueError("スナップショット保存は単一設定でのみ実行できます")
        self.prepared = prepared
        self.configs = configs
        self.config_count = len(configs)
        self.horse_count = len(prepared.horse_ids)
        self.evaluation_mask = evaluation_mask
        self.record = record
        self.dry_run = dry_run
        self.initial_rule = initial_rule
        if initial_rule is not None:
            initial_rule.validate(self.config_count)
        self.feature_provider = feature_provider or {}
        self.train_weights = np.asarray([WEIGHTS[config.weight] for config in configs], dtype=float)
        self.var0 = np.asarray([config.var0 for config in configs], dtype=float)
        self.q = np.asarray([config.q for config in configs], dtype=float)
        self.kappa = np.asarray([config.kappa for config in configs], dtype=float)
        self.beta = np.asarray([config.beta_kg for config in configs], dtype=float)
        self.mu = np.zeros((self.config_count, self.horse_count), dtype=float)
        self.var = np.repeat(self.var0[:, None], self.horse_count, axis=1)
        self.var_cap = np.repeat(self.var0[:, None], self.horse_count, axis=1)
        self.seen = np.zeros(self.horse_count, dtype=bool)
        self.last_start = np.full(self.horse_count, -1, dtype=np.int32)
        self.n_starts = np.zeros(self.horse_count, dtype=np.int32)
        self.active = np.zeros(self.horse_count, dtype=bool)
        self.expiry_heap: list[tuple[int, int, int]] = []
        self.debut_types = np.zeros(self.horse_count, dtype=np.int8)
        self.mu0_state = np.zeros((1, self.horse_count), dtype=float) if record else None
        self.snapshots = {
            name: np.zeros(prepared.total_starts if record else 0, dtype=dtype)
            for name, dtype in SNAPSHOT_DTYPES.items()
        }
        self.row_cursor = 0
        self.nll_sum = np.zeros(self.config_count, dtype=float)
        self.nll_count = 0
        nll_size = max((race.race_index for race in prepared.races), default=-1) + 1
        self.nll = np.full(nll_size, np.nan, dtype=float) if record else None
        self.s0_nll = np.full(nll_size, np.nan, dtype=float) if record else None
        self.daily_records: list[dict] = []
        self.skip_mu_var_updates = 0
        self.initial_zero_anchor_days = 0
        self.fallback_anchor_days = 0

    def run(self) -> EngineResult:
        by_date: dict[int, list[Race]] = {}
        for race in self.prepared.races:
            by_date.setdefault(race.date, []).append(race)
        previous_anchor_group = None
        previous_mu_bar = None
        for date in sorted(by_date):
            day_races = by_date[date]
            expired = False
            while self.expiry_heap and self.expiry_heap[0][0] < date:
                expiry, horse, start_date = heapq.heappop(self.expiry_heap)
                if self.last_start[horse] == start_date and self.last_start[horse] < date - 365:
                    self.active[horse] = False
                    expired = True
            known = self.last_start >= 0
            active_count = int(self.active.sum())
            if active_count:
                population = self.active
                mu_bar = self.mu[:, population].mean(axis=1)
            elif known.any():
                population = known
                mu_bar = self.mu[:, population].mean(axis=1)
                self.fallback_anchor_days += 1
            else:
                population = known
                mu_bar = np.zeros(self.config_count, dtype=float)
                self.initial_zero_anchor_days += 1
            if self.record:
                if population.any():
                    latest = self.mu[0, population]
                    active_std = float(latest.std())
                    active_median = float(np.median(latest))
                else:
                    active_std = np.nan
                    active_median = np.nan
                anchor_group = "AT_OR_ABOVE_N_MIN" if active_count >= 100 else "BELOW_N_MIN"
                if previous_anchor_group is None or previous_anchor_group == anchor_group:
                    crossing = ""
                elif anchor_group == "AT_OR_ABOVE_N_MIN":
                    crossing = "UP"
                else:
                    crossing = "DOWN"
                self.daily_records.append({
                    "asof_date": date,
                    "n_active_365d": active_count,
                    "mu_bar_D": float(mu_bar[0]),
                    "active_latest_mu_std": active_std,
                    "active_latest_mu_median": active_median,
                    "mu_bar_delta_1d": float(mu_bar[0] - previous_mu_bar) if previous_mu_bar is not None else np.nan,
                    "n_new_horses": 0,
                    "mu0_mean": np.nan,
                    "anchor_group": anchor_group,
                    "threshold_crossing": crossing,
                })
                previous_anchor_group = anchor_group
                previous_mu_bar = float(mu_bar[0])

            day_horses = np.unique(np.concatenate([race.horse_idx for race in day_races]))
            new_horses = day_horses[~self.seen[day_horses]]
            if len(new_horses):
                if self.initial_rule is None:
                    self.mu[:, new_horses] = mu_bar[:, None]
                    self.var[:, new_horses] = self.var0[:, None]
                    if self.record:
                        self.mu0_state[0, new_horses] = mu_bar[0]
                    for race in day_races:
                        local_new = new_horses[np.isin(new_horses, race.horse_idx)]
                        if len(local_new):
                            self.debut_types[local_new] = debut_code(race.condition_code, date)
                else:
                    self._initialize_rule_new_horses(new_horses, day_races, mu_bar, date)
                if self.record:
                    self.daily_records[-1]["n_new_horses"] = int(len(new_horses))
                    self.daily_records[-1]["mu0_mean"] = float(self.mu[0, new_horses].mean())
                self.seen[new_horses] = True

            for race in day_races:
                horses = race.horse_idx
                n = race.n
                last = self.last_start[horses]
                days = np.where(last >= 0, np.maximum(0, date - last), 0).astype(np.int32)
                if not np.array_equal(self.n_starts[horses], race.pre_nstarts):
                    raise AssertionError(f"出走数の事前状態が不一致です: {race.race_id}")
                mu_pre = self.mu[:, horses].copy()
                var_pre = self.var[:, horses].copy()
                var_eff = np.minimum(
                    var_pre + self.q[:, None] * days[None, :] / 30.0,
                    self.var_cap[:, horses],
                )
                centered_weight = race.weight_kg - race.weight_kg.mean()
                scores = mu_pre - self.beta[:, None] * centered_weight[None, :]
                need_eval = self.record or (self.evaluation_mask is not None and self.evaluation_mask[race.race_index])
                if race.has_likelihood:
                    gradient, diagonal, eval_ll, s0_ll = pl_batch(
                        scores,
                        race.orders,
                        race.unranked,
                        race.k_max,
                        self.train_weights,
                        W_EVAL,
                        bool(need_eval),
                    )
                else:
                    gradient = np.zeros_like(scores)
                    diagonal = np.zeros_like(scores)
                    eval_ll = np.zeros(self.config_count, dtype=float)
                    s0_ll = 0.0
                if need_eval and race.has_likelihood:
                    if self.record:
                        self.nll[race.race_index] = -eval_ll[0]
                        self.s0_nll[race.race_index] = -s0_ll
                    else:
                        self.nll_sum -= eval_ll
                    self.nll_count += 1 if not self.record else 0
                ranked = race.finish > 0
                mu_post = mu_pre.copy()
                var_post = var_pre.copy()
                if race.has_likelihood and ranked.any():
                    precision = 1.0 / var_eff[:, ranked] + self.kappa[:, None] * diagonal[:, ranked]
                    mu_post[:, ranked] = mu_pre[:, ranked] + self.kappa[:, None] * gradient[:, ranked] / precision
                    var_post[:, ranked] = 1.0 / precision
                if not self.dry_run:
                    self.mu[:, horses] = mu_post
                    self.var[:, horses] = var_post
                    self.skip_mu_var_updates += int(np.count_nonzero(~ranked))
                if self.record:
                    self._record_race(race, days, mu_pre, var_pre, var_eff, scores, mu_post, var_post, mu_bar[0])

            if not self.dry_run:
                self.last_start[day_horses] = date
                self.n_starts[day_horses] += 1
                self.active[day_horses] = True
                for horse in day_horses:
                    heapq.heappush(self.expiry_heap, (date + 365, int(horse), date))

        state = pd.DataFrame({
            "horse_id": self.prepared.horse_ids,
            "mu": self.mu[0],
            "var": self.var[0],
            "last_start_date": self.last_start,
            "n_graph_starts": self.n_starts,
            "var_cap": self.var_cap[0],
            "prior_type": getattr(self, "model_name", "P0"),
            "debut_type_code": self.debut_types,
        })
        snapshots = self._snapshots_frame() if self.record else None
        daily = pd.DataFrame(self.daily_records) if self.record else None
        return EngineResult(
            nll_sum=self.nll_sum.copy(),
            nll_count=self.nll_count,
            nll=self.nll,
            s0_nll=self.s0_nll,
            snapshots=snapshots,
            daily=daily,
            state=state,
            skip_mu_var_updates=self.skip_mu_var_updates,
            initial_zero_anchor_days=self.initial_zero_anchor_days,
            fallback_anchor_days=self.fallback_anchor_days,
        )

    def _initialize_rule_new_horses(self, new_horses, day_races, mu_bar, date):
        for race in day_races:
            local_new = new_horses[np.isin(new_horses, race.horse_idx)]
            if len(local_new):
                self.debut_types[local_new] = debut_code(race.condition_code, date)
        initial_mu = np.repeat(mu_bar[:, None], len(new_horses), axis=1)
        for code in (1, 2, 3):
            mask = self.debut_types[new_horses] == code
            if mask.any():
                targets = new_horses[mask]
                initial_mu[:, mask] += self.initial_rule.b[:, code - 1, None]
                self.var[:, targets] = self.initial_rule.var0_d[:, code - 1, None]
                self.var_cap[:, targets] = self.initial_rule.var0_d[:, code - 1, None]
        if np.any(self.initial_rule.beta != 0):
            positions = {int(horse): index for index, horse in enumerate(new_horses)}
            for race in day_races:
                mask = np.isin(race.horse_idx, new_horses)
                globals_in_race = race.horse_idx[mask]
                eligible = self.debut_types[globals_in_race] != 4
                if not eligible.any():
                    continue
                local_indices = np.flatnonzero(mask)[eligible]
                globals_in_race = globals_in_race[eligible]
                block = self.feature_provider[race.race_index]
                block.validate(race.n)
                X = block.X[local_indices]
                if not np.isfinite(X).all():
                    raise ValueError("初期化に使う血統特徴量が非有限です")
                targets = np.array([positions[int(horse)] for horse in globals_in_race], dtype=np.int32)
                initial_mu[:, targets] += self.initial_rule.beta @ X.T
        self.mu[:, new_horses] = initial_mu
        if self.record:
            self.mu0_state[0, new_horses] = initial_mu[0]

    def _record_race(self, race, days, mu_pre, var_pre, var_eff, scores, mu_post, var_post, mu_bar):
        start = self.row_cursor
        end = start + race.n
        horses = race.horse_idx
        values = {
            "race_id": race.race_id,
            "race_date": race.date,
            "race_index": race.race_index,
            "horse_id": self.prepared.horse_ids[horses],
            "horse_number": race.horse_number,
            "finish_position": race.finish,
            "abnormal_code": race.abnormal_code,
            "weight_kg": race.weight_kg,
            "mu_pre": mu_pre[0],
            "var_pre_persistent": var_pre[0],
            "var_pre_eff": var_eff[0],
            "m_pre": scores[0],
            "days_since_start": days,
            "n_graph_starts": race.pre_nstarts,
            "mu0": self.mu0_state[0, horses],
            "debut_type_code": self.debut_types[horses],
            "n_field_original": race.n_field_original,
            "n_field_retained": race.n,
            "n_removed_nonJRA": race.n_removed_non_jra,
            "condition_code": race.condition_code,
            "surface_code": race.surface_code,
            "distance": race.distance,
            "skip_mu_var_update": race.finish == 0,
            "mu_post": mu_post[0],
            "var_post": var_post[0],
            "mu_bar_D": mu_bar,
        }
        block = self.feature_provider.get(race.race_index, FeatureBlock.empty(race.n))
        block.validate(race.n)
        values.update({
            "X_sire_win": block.X[:, 0],
            "X_sire_top3": block.X[:, 1],
            "X_bms_win": block.X[:, 2],
            "X_bms_top3": block.X[:, 3],
            "N_sire": block.N_sire,
            "N_bms": block.N_bms,
            "sire_status_code": block.sire_status,
            "bms_status_code": block.bms_status,
        })
        for name, value in values.items():
            self.snapshots[name][start:end] = value
        self.row_cursor = end

    def _snapshots_frame(self) -> pd.DataFrame:
        if self.row_cursor != self.prepared.total_starts:
            raise AssertionError("スナップショット行数が不一致です")
        frame = pd.DataFrame({name: values.copy() for name, values in self.snapshots.items()})
        frame["race_date"] = pd.to_datetime(frame["race_date"], unit="D", origin="unix")
        return frame


def run_engine(
    prepared: PreparedInput,
    configs: list[P0Config],
    evaluation_mask: np.ndarray | None = None,
    record: bool = False,
    initial_rule: InitialRule | None = None,
    feature_provider: dict[int, FeatureBlock] | None = None,
    model_name: str = "P0",
) -> EngineResult:
    engine = BatchP0Engine(
        prepared,
        configs,
        evaluation_mask=evaluation_mask,
        record=record,
        initial_rule=initial_rule,
        feature_provider=feature_provider,
    )
    engine.model_name = model_name
    return engine.run()
