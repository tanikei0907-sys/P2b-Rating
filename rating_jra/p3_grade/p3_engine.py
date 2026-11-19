from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from rating_core import P0Config, WEIGHTS, W_EVAL, harville_inclusion_probabilities
from rating_data import PreparedInput
from rating_engine import BatchP0Engine, debut_code, pl_batch
from rating_initialization import FeatureBlock, InitialRule


class P3Engine(BatchP0Engine):
    def __init__(
        self,
        prepared: PreparedInput,
        config: P0Config,
        initial_rule: InitialRule,
        pedigree: pd.DataFrame,
        grade_weights: np.ndarray,
        record: bool,
        standardization: list[dict],
    ):
        super().__init__(prepared, [config], record=record, initial_rule=initial_rule)
        if len(grade_weights) != len(prepared.races):
            raise ValueError("レース格重みの件数が不正です")
        if not np.isfinite(grade_weights).all() or np.any(grade_weights <= 0):
            raise ValueError("レース格重みは正の有限値である必要があります")
        self.grade_weights = np.asarray(grade_weights, dtype=float)
        self.model_name = "P3"
        self.mu0_state = np.zeros((1, self.horse_count), dtype=float)
        pedigree_indexed = pedigree.drop_duplicates("horse_id").set_index("horse_id")
        self.sire_ids = pedigree_indexed["sire_id"].reindex(prepared.horse_ids).to_numpy(dtype=float)
        self.bms_ids = pedigree_indexed["bms_id"].reindex(prepared.horse_ids).to_numpy(dtype=float)
        self.statistics = {
            "sire": [{}, {}, {}],
            "bms": [{}, {}, {}],
        }
        self.history_records: list[dict] = []
        self.history = pd.DataFrame()
        self.train_weights = np.asarray([WEIGHTS[config.weight]], dtype=float)
        if len(standardization) != 4:
            raise ValueError("血統標準化定数は4件である必要があります")
        self.standardization = standardization

    def _feature_block(self, horses: np.ndarray) -> FeatureBlock:
        n = len(horses)
        X = np.zeros((n, 4), dtype=float)
        N_values = np.zeros((n, 2), dtype=np.int32)
        statuses = np.full((n, 2), 3, dtype=np.int8)
        for local, horse in enumerate(horses):
            if self.debut_types[horse] == 4:
                X[local, :] = np.nan
                statuses[local, :] = 4
                continue
            for side_index, (side, ids) in enumerate([("sire", self.sire_ids), ("bms", self.bms_ids)]):
                pedigree_id = ids[horse]
                if not np.isfinite(pedigree_id) or pedigree_id == 0:
                    continue
                key = int(pedigree_id)
                count, sum_win, sum_top3 = self.statistics[side]
                N = count.get(key, 0)
                N_values[local, side_index] = N
                statuses[local, side_index] = 1 if N > 0 else 2
                if N == 0:
                    continue
                raw = [sum_win.get(key, 0.0) / (N + 20.0), sum_top3.get(key, 0.0) / (N + 20.0)]
                for metric_index, value in enumerate(raw):
                    feature_index = side_index * 2 + metric_index
                    constants = self.standardization[feature_index]
                    X[local, feature_index] = (value - constants["center"]) / constants["scale"]
        return FeatureBlock(X=X, N_sire=N_values[:, 0], N_bms=N_values[:, 1], sire_status=statuses[:, 0], bms_status=statuses[:, 1])

    def _initialize_new_horses(self, new_horses: np.ndarray, day_races, mu_bar: float, date: int) -> None:
        for race in day_races:
            local_new = new_horses[np.isin(new_horses, race.horse_idx)]
            if len(local_new):
                self.debut_types[local_new] = debut_code(race.condition_code, date)
        provider_races = day_races if self.record else [
            race for race in day_races
            if np.any(np.isin(race.horse_idx, new_horses) & (self.debut_types[race.horse_idx] != 4))
        ]
        self.feature_provider = {race.race_index: self._feature_block(race.horse_idx) for race in provider_races}
        initial_mu = np.full(len(new_horses), mu_bar, dtype=float)
        positions = {int(horse): index for index, horse in enumerate(new_horses)}
        for code in (1, 2, 3):
            targets = new_horses[self.debut_types[new_horses] == code]
            if len(targets):
                indices = np.array([positions[int(horse)] for horse in targets], dtype=np.int32)
                initial_mu[indices] += self.initial_rule.b[0, code - 1]
                self.var[0, targets] = self.initial_rule.var0_d[0, code - 1]
                self.var_cap[0, targets] = self.initial_rule.var0_d[0, code - 1]
        for race in day_races:
            local_indices = np.flatnonzero(
                np.isin(race.horse_idx, new_horses)
                & (self.debut_types[race.horse_idx] != 4)
            )
            if not len(local_indices):
                continue
            eligible = race.horse_idx[local_indices]
            X = self.feature_provider[race.race_index].X[local_indices]
            if not np.isfinite(X).all():
                raise ValueError("新規馬の血統特徴量が非有限です")
            indices = np.array([positions[int(horse)] for horse in eligible], dtype=np.int32)
            initial_mu[indices] += self.initial_rule.beta[0] @ X.T
        self.mu[0, new_horses] = initial_mu
        self.mu0_state[0, new_horses] = initial_mu
        self.seen[new_horses] = True
        if self.record:
            self.daily_records[-1]["n_new_horses"] = int(len(new_horses))
            self.daily_records[-1]["mu0_mean"] = float(initial_mu.mean())

    def _residuals(self, race, scores: np.ndarray) -> list[dict]:
        horses = race.horse_idx
        eligible = (self.debut_types[horses] == 1) & (race.pre_nstarts <= 2)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return []
        probabilities = harville_inclusion_probabilities(scores[indices], 3)
        records = []
        for row, local in enumerate(indices):
            horse = int(horses[local])
            actual_win = float(race.finish[local] == 1)
            actual_top3 = float(1 <= race.finish[local] <= 3)
            e_win = float(probabilities[row, 0])
            e_top3 = float(probabilities[row].sum())
            records.append({
                "race_index": int(race.race_index),
                "race_date": pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date),
                "date": int(race.date),
                "horse_id": int(self.prepared.horse_ids[horse]),
                "finish_position": int(race.finish[local]),
                "n_graph_starts": int(race.pre_nstarts[local]),
                "condition_code": int(race.condition_code),
                "e_win": e_win,
                "e_top3": e_top3,
                "actual_win": actual_win,
                "actual_top3": actual_top3,
                "r_win": actual_win - e_win,
                "r_top3": actual_top3 - e_top3,
                "sire_id": self.sire_ids[horse],
                "bms_id": self.bms_ids[horse],
            })
        return records

    def _apply_residuals(self, records: list[dict]) -> None:
        for record in records:
            for side in ["sire", "bms"]:
                value = record[f"{side}_id"]
                if not np.isfinite(value) or value == 0:
                    continue
                key = int(value)
                count, sum_win, sum_top3 = self.statistics[side]
                count[key] = count.get(key, 0) + 1
                sum_win[key] = sum_win.get(key, 0.0) + record["r_win"]
                sum_top3[key] = sum_top3.get(key, 0.0) + record["r_top3"]

    def run(self):
        by_date: dict[int, list] = {}
        for race in self.prepared.races:
            by_date.setdefault(race.date, []).append(race)
        previous_anchor_group = None
        previous_mu_bar = None
        for date in sorted(by_date):
            day_races = by_date[date]
            while self.expiry_heap and self.expiry_heap[0][0] < date:
                _, horse, start_date = heapq.heappop(self.expiry_heap)
                if self.last_start[horse] == start_date and self.last_start[horse] < date - 365:
                    self.active[horse] = False
            known = self.last_start >= 0
            active_count = int(self.active.sum())
            if active_count:
                population = self.active
                mu_bar = float(self.mu[0, population].mean())
            elif known.any():
                population = known
                mu_bar = float(self.mu[0, population].mean())
                self.fallback_anchor_days += 1
            else:
                population = known
                mu_bar = 0.0
                self.initial_zero_anchor_days += 1
            if self.record:
                latest = self.mu[0, population]
                anchor_group = "AT_OR_ABOVE_N_MIN" if active_count >= 100 else "BELOW_N_MIN"
                crossing = "" if previous_anchor_group is None or previous_anchor_group == anchor_group else "UP" if anchor_group == "AT_OR_ABOVE_N_MIN" else "DOWN"
                self.daily_records.append({
                    "asof_date": date,
                    "n_active_365d": active_count,
                    "mu_bar_D": mu_bar,
                    "active_latest_mu_std": float(latest.std()) if population.any() else np.nan,
                    "active_latest_mu_median": float(np.median(latest)) if population.any() else np.nan,
                    "mu_bar_delta_1d": mu_bar - previous_mu_bar if previous_mu_bar is not None else np.nan,
                    "n_new_horses": 0,
                    "mu0_mean": np.nan,
                    "anchor_group": anchor_group,
                    "threshold_crossing": crossing,
                })
                previous_anchor_group = anchor_group
                previous_mu_bar = mu_bar
            day_horses = np.unique(np.concatenate([race.horse_idx for race in day_races]))
            new_horses = day_horses[~self.seen[day_horses]]
            if len(new_horses):
                self._initialize_new_horses(new_horses, day_races, mu_bar, date)
            else:
                self.feature_provider = {race.race_index: self._feature_block(race.horse_idx) for race in day_races} if self.record else {}
            pending_history = []
            for race in day_races:
                horses = race.horse_idx
                last = self.last_start[horses]
                days = np.where(last >= 0, np.maximum(0, date - last), 0).astype(np.int32)
                if not np.array_equal(self.n_starts[horses], race.pre_nstarts):
                    raise AssertionError(f"出走数の事前状態が不一致です: {race.race_id}")
                mu_pre = self.mu[:, horses].copy()
                var_pre = self.var[:, horses].copy()
                var_eff = np.minimum(var_pre + self.q[:, None] * days[None, :] / 30.0, self.var_cap[:, horses])
                centered_weight = race.weight_kg - race.weight_kg.mean()
                scores = mu_pre - self.beta[:, None] * centered_weight[None, :]
                if race.has_likelihood:
                    gradient, diagonal, eval_ll, s0_ll = pl_batch(
                        scores, race.orders, race.unranked, race.k_max,
                        self.train_weights, W_EVAL, True,
                    )
                else:
                    gradient = np.zeros_like(scores)
                    diagonal = np.zeros_like(scores)
                    eval_ll = np.zeros(1, dtype=float)
                    s0_ll = 0.0
                if race.has_likelihood:
                    if self.record:
                        self.nll[race.race_index] = -eval_ll[0]
                        self.s0_nll[race.race_index] = -s0_ll
                    else:
                        self.nll_sum[0] -= eval_ll[0]
                        self.nll_count += 1
                grade_weight = float(self.grade_weights[race.race_index])
                gradient *= grade_weight
                diagonal *= grade_weight
                ranked = race.finish > 0
                mu_post = mu_pre.copy()
                var_post = var_pre.copy()
                if race.has_likelihood and ranked.any():
                    precision = 1.0 / var_eff[:, ranked] + self.kappa[:, None] * diagonal[:, ranked]
                    mu_post[:, ranked] = mu_pre[:, ranked] + self.kappa[:, None] * gradient[:, ranked] / precision
                    var_post[:, ranked] = 1.0 / precision
                self.mu[:, horses] = mu_post
                self.var[:, horses] = var_post
                self.skip_mu_var_updates += int(np.count_nonzero(~ranked))
                if self.record:
                    self._record_race(race, days, mu_pre, var_pre, var_eff, scores, mu_post, var_post, mu_bar)
                race_history = self._residuals(race, scores[0])
                pending_history.extend(race_history)
                if self.record:
                    self.history_records.extend(race_history)
            self._apply_residuals(pending_history)
            self.last_start[day_horses] = date
            self.n_starts[day_horses] += 1
            self.active[day_horses] = True
            for horse in day_horses:
                heapq.heappush(self.expiry_heap, (date + 365, int(horse), date))
        from rating_engine import EngineResult
        state = pd.DataFrame({
            "horse_id": self.prepared.horse_ids,
            "mu": self.mu[0],
            "var": self.var[0],
            "last_start_date": self.last_start,
            "n_graph_starts": self.n_starts,
            "var_cap": self.var_cap[0],
            "prior_type": "P3",
            "debut_type_code": self.debut_types,
        })
        snapshots = self._snapshots_frame() if self.record else None
        daily = pd.DataFrame(self.daily_records) if self.record else None
        self.history = pd.DataFrame(self.history_records)
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
