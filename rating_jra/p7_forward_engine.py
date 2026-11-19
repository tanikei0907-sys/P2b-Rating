from __future__ import annotations

import json

import numpy as np
import pandas as pd

from p7_propagation.p7_engine import P7Engine


class P7ProductionEngine(P7Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_propagation = np.zeros(self.horse_count, dtype=float)
        self.horse_index = {int(horse_id): index for index, horse_id in enumerate(self.prepared.horse_ids)}
        for name in ["delta_turf_pre", "delta_dirt_pre", "propagation_correction_pre"]:
            self.snapshots[name] = np.zeros(self.prepared.total_starts, dtype=float)

    def _process_race(self, race, date):
        horses = race.horse_idx
        delta_turf = self.surface_delta[0, horses].copy()
        delta_dirt = self.surface_delta[1, horses].copy()
        pending = self.pending_propagation[horses].copy()
        start = self.row_cursor
        result = super()._process_race(race, date)
        end = self.row_cursor
        self.snapshots["delta_turf_pre"][start:end] = delta_turf
        self.snapshots["delta_dirt_pre"][start:end] = delta_dirt
        self.snapshots["propagation_correction_pre"][start:end] = pending
        self.pending_propagation[horses] = 0.0
        return result

    def _propagate(self, date, day_horses, day_deltas, day_variances, day_races):
        start = len(self.propagation_events)
        super()._propagate(date, day_horses, day_deltas, day_variances, day_races)
        for event in self.propagation_events[start:]:
            target = self.horse_index[event["target_horse_id"]]
            self.pending_propagation[target] += float(event["applied_delta_mu"])


def interaction_history_frame(engine: P7ProductionEngine, config_hash: str, source_hash: str) -> pd.DataFrame:
    rows = []
    for source, history in engine.interaction_history.items():
        for past in history:
            opponents = {
                str(int(engine.prepared.horse_ids[target])): float(weight)
                for target, weight in past["opponents"].items()
            }
            rows.append({
                "source_horse_id": int(engine.prepared.horse_ids[source]),
                "race_id": int(past["race_id"]),
                "race_date": pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=int(past["date"])),
                "opponents_json": json.dumps(opponents, sort_keys=True, separators=(",", ":")),
                "config_sha256": config_hash,
                "source_db_sha256": source_hash,
            })
    return pd.DataFrame(rows, columns=[
        "source_horse_id", "race_id", "race_date", "opponents_json", "config_sha256", "source_db_sha256",
    ]).sort_values(["source_horse_id", "race_date", "race_id"]).reset_index(drop=True)
