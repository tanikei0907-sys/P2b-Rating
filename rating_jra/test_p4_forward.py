from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from p4_forward_ops import EXPECTED_P4_HASH, OUTPUT_FILES, P4ForwardError, load_config, predict_snapshot, update
from test_forward_ops import dummy_rows, loader_for

BASE = Path(__file__).resolve().parent


def timed_dummy_rows() -> pd.DataFrame:
    raw = dummy_rows()
    raw["time_raw"] = np.where(raw["finish_position"].gt(0), 1000 + raw["finish_position"] * 10, 0)
    raw["stored_margin_raw"] = 0
    raw["track_condition_code"] = 1
    return raw


def test_config(root: Path) -> dict:
    payload = json.loads((BASE / "p4_margin/forward_config.yaml").read_text(encoding="utf-8"))
    payload["database"]["path"] = str(BASE / "stage2_input/BaoZ_2_snapshot.mdb")
    payload["database"]["linked_tables"] = {"産駒マスタ": r"C:\BaoZ_AI\DB\MasterDB\BaoZ-RA.mdb"}
    payload["output"] = {
        "directory": str(root / "p4_forward"), "backup_directory": str(root / "backups"),
        "temp_directory": str(root / "work"),
    }
    path = root / "forward_config.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_config(path)


class P4ForwardTest(unittest.TestCase):
    @unittest.skipUnless((BASE / "stage2_input/BaoZ_2_snapshot.mdb").is_file(), "実DBスナップショットが必要です")
    def test_update_idempotency_margin_and_predict_dry_run(self):
        raw = timed_dummy_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            out = Path(config["output"]["directory"])
            loader = loader_for(raw)
            first = update(config, "2025-01-04", raw_loader=loader)
            self.assertEqual(first["status"], "UPDATED")
            self.assertEqual(first["new_races"], 2)
            snapshots = pd.read_csv(out / OUTPUT_FILES["snapshots"])
            self.assertEqual(len(snapshots), 6)
            self.assertTrue(snapshots["config_sha256"].eq(EXPECTED_P4_HASH).all())
            margins = pd.read_csv(out / "margins_2025plus.csv.gz")
            multipliers = pd.read_csv(out / "margin_multipliers_2025plus.csv.gz")
            self.assertTrue(margins["stage_1"].eq(1.0).all())
            self.assertTrue(multipliers["stage_multiplier_1"].eq(2.0).all())
            first_hashes = {name: (out / name).read_bytes() for name in OUTPUT_FILES.values()}
            second = update(config, "2025-01-04", raw_loader=loader)
            self.assertEqual(second["status"], "NO_NEW_RACES")
            self.assertEqual(first_hashes, {name: (out / name).read_bytes() for name in OUTPUT_FILES.values()})
            third = update(config, "2025-01-05", raw_loader=loader)
            self.assertEqual(third["new_races"], 1)
            snapshots = pd.read_csv(out / OUTPUT_FILES["snapshots"])
            self.assertEqual(len(snapshots), 10)
            self.assertEqual(snapshots["race_index"].tolist(), [0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
            state = pd.read_csv(out / OUTPUT_FILES["state"])
            self.assertEqual(state["last_start_date"].max(), "2025-01-05")
            operational_hashes = {name: (out / name).read_bytes() for name in OUTPUT_FILES.values()}
            prediction = predict_snapshot(config, "2025-01-06", raw_loader=loader)
            self.assertEqual(prediction["status"], "PREDICT_SNAPSHOT_WRITTEN")
            self.assertEqual(operational_hashes, {name: (out / name).read_bytes() for name in OUTPUT_FILES.values()})
            predicted = pd.read_csv(prediction["output"])
            self.assertEqual(len(predicted), 4)
            self.assertFalse({"mu_post", "var_post", "skip_mu_var_update"} & set(predicted.columns))
            expected = 1500 + 400 / np.log(10) * (predicted["mu_pre"] - predicted["mu_bar_D"])
            np.testing.assert_allclose(predicted["elo_pre"], expected, atol=1e-12, rtol=0)
            for row in predicted.itertuples():
                current = state[state["horse_id"].eq(row.horse_id)].iloc[0]
                self.assertEqual(row.n_graph_starts, current["n_graph_starts"])

    def test_hash_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = json.loads((BASE / "p4_margin/forward_config.yaml").read_text(encoding="utf-8"))
            payload["model"]["config_sha256"] = "0" * 64
            payload["output"] = {"directory": str(root / "out"), "backup_directory": str(root / "backup"), "temp_directory": str(root / "work")}
            path = root / "forward_config.yaml"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(P4ForwardError):
                load_config(path)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
