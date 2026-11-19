from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
P3 = BASE / "p3_grade"
for path in [BASE, P3]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from p3_grade.p3_engine import P3Engine
from rating_core import W_EVAL
from rating_engine import EngineResult, pl_batch


class P4Engine(P3Engine):
    def __init__(
        self,
        prepared,
        config,
        initial_rule,
        pedigree,
        standardization,
        method: str,
        stage_multipliers: np.ndarray,
        race_scales: np.ndarray,
        record: bool,
    ):
        if method not in ["A", "B"]:
            raise ValueError("P4方式が不正です")
        if len(race_scales) != len(prepared.races) or stage_multipliers.shape != (len(prepared.races), 5):
            raise ValueError("マージン倍率の形状が不正です")
        if not np.isfinite(race_scales).all() or np.any(race_scales <= 0) or not np.isfinite(stage_multipliers).all() or np.any(stage_multipliers <= 0):
            raise ValueError("マージン倍率は正の有限値である必要があります")
        self.method = method
        self.stage_multipliers = np.asarray(stage_multipliers, dtype=float)
        self.race_scales = np.asarray(race_scales, dtype=float)
        super().__init__(
            prepared, config, initial_rule, pedigree,
            race_scales if method == "B" else np.ones(len(prepared.races)),
            record, standardization,
        )
        self.model_name = "P4"

    def run(self):
        if self.method == "B":
            result = super().run()
            result.state["prior_type"] = "P4"
            return result
        return self._run_margin_stages()

    def _run_margin_stages(self):
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
                train_weights = self.train_weights * self.stage_multipliers[race.race_index][None, :]
                if race.has_likelihood:
                    gradient, diagonal, eval_ll, s0_ll = pl_batch(
                        scores, race.orders, race.unranked, race.k_max,
                        train_weights, W_EVAL, True,
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
        state = pd.DataFrame({
            "horse_id": self.prepared.horse_ids,
            "mu": self.mu[0],
            "var": self.var[0],
            "last_start_date": self.last_start,
            "n_graph_starts": self.n_starts,
            "var_cap": self.var_cap[0],
            "prior_type": "P4",
            "debut_type_code": self.debut_types,
        })
        snapshots = self._snapshots_frame() if self.record else None
        daily = pd.DataFrame(self.daily_records) if self.record else None
        self.history = pd.DataFrame(self.history_records)
        return EngineResult(
            nll_sum=self.nll_sum.copy(), nll_count=self.nll_count, nll=self.nll, s0_nll=self.s0_nll,
            snapshots=snapshots, daily=daily, state=state,
            skip_mu_var_updates=self.skip_mu_var_updates,
            initial_zero_anchor_days=self.initial_zero_anchor_days,
            fallback_anchor_days=self.fallback_anchor_days,
        )
