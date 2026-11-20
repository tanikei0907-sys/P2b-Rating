from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from p7_forward_ops import EXPECTED_P7_HASH, OUTPUT_FILES, P7ForwardError, load_config, predict_snapshot, update
from test_forward_ops import dummy_rows, loader_for

BASE = Path(__file__).resolve().parent


def timed_dummy_rows() -> pd.DataFrame:
    raw = dummy_rows()
    raw["time_raw"] = np.where(raw["finish_position"].gt(0), 1000 + raw["finish_position"] * 10, 0)
    raw["stored_margin_raw"] = 0
    raw["track_condition_code"] = 1
    return raw


def test_config(root: Path) -> dict:
    payload = json.loads((BASE / "p7_forward/forward_config.yaml").read_text(encoding="utf-8"))
    payload["database"]["path"] = str(BASE / "stage2_input/BaoZ_2_snapshot.mdb")
    payload["database"]["linked_tables"] = {"産駒マスタ": r"C:\BaoZ_AI\DB\MasterDB\BaoZ-RA.mdb"}
    payload["model"]["p0_state_path"] = str(BASE / "stage2_output/state_2024-12-31.csv")
    payload["output"] = {
        "directory": str(root / "p7_forward"), "backup_directory": str(root / "backups"),
        "temp_directory": str(root / "work"),
    }
    path = root / "forward_config.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_config(path)


class P7ForwardTest(unittest.TestCase):
    @unittest.skipUnless((BASE / "stage2_input/BaoZ_2_snapshot.mdb").is_file(), "実DBスナップショットが必要です")
    def test_update_idempotency_interactions_and_predict_dry_run(self):
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
            self.assertTrue(snapshots["config_sha256"].eq(EXPECTED_P7_HASH).all())
            self.assertTrue(np.isfinite(snapshots[["mu_base_pre", "surface_offset_pre", "delta_turf_pre", "delta_dirt_pre", "age_drift", "propagation_correction_pre"]].to_numpy(dtype=float)).all())
            np.testing.assert_allclose(snapshots["elo_pre"], snapshots["elo_base_pre"] + snapshots["elo_surface_adjustment"], atol=1e-10, rtol=0)
            lemon = snapshots[snapshots["horse_id"].eq(2018110015)].iloc[0]
            frozen = pd.read_csv(BASE / "p7_propagation/state_2024-12-31.csv").set_index("horse_id")
            self.assertAlmostEqual(lemon["delta_dirt_pre"], frozen.loc[2018110015, "delta_dirt"])
            self.assertEqual(lemon["n_graph_starts"], frozen.loc[2018110015, "n_graph_starts"])
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
            interactions = pd.read_csv(out / OUTPUT_FILES["interactions"])
            latest = interactions[interactions["source_horse_id"].eq(2018110015)].iloc[-1]
            self.assertEqual(latest["race_id"], 900002)
            self.assertTrue(np.isfinite(state["propagation_pending"]).all())
            operational_hashes = {name: (out / name).read_bytes() for name in OUTPUT_FILES.values()}
            prediction = predict_snapshot(config, "2025-01-06", raw_loader=loader)
            self.assertEqual(prediction["status"], "PREDICT_SNAPSHOT_WRITTEN")
            self.assertEqual(operational_hashes, {name: (out / name).read_bytes() for name in OUTPUT_FILES.values()})
            predicted = pd.read_csv(prediction["output"])
            self.assertEqual(len(predicted), 4)
            self.assertFalse({"mu_post", "var_post", "skip_mu_var_update", "mu_base_post", "surface_offset_post"} & set(predicted.columns))
            expected = 1500 + 400 / np.log(10) * (predicted["mu_pre"] - predicted["mu_bar_D"])
            np.testing.assert_allclose(predicted["elo_pre"], expected, atol=1e-12, rtol=0)
            np.testing.assert_allclose(predicted["elo_pre"], predicted["elo_base_pre"] + predicted["elo_surface_adjustment"], atol=1e-12, rtol=0)
            for row in predicted.itertuples():
                current = state[state["horse_id"].eq(row.horse_id)].iloc[0]
                self.assertEqual(row.n_graph_starts, current["n_graph_starts"])
            p0_predicted = pd.read_csv(prediction["p0_output"])
            self.assertEqual(len(p0_predicted), 4)
            self.assertEqual(set(p0_predicted["race_id"]), set(predicted["race_id"]))
            self.assertEqual(prediction["p0_confirmed_races_applied_in_memory"], 3)
            self.assertFalse(prediction["p0_state_files_modified"])
            self.assertTrue(p0_predicted["config_sha256"].eq(prediction["p0_config_sha256"]).all())
            p0_expected = 1500 + 400 / np.log(10) * (p0_predicted["mu_pre"] - p0_predicted["mu_bar_D"])
            np.testing.assert_allclose(p0_predicted["elo_pre"], p0_expected, atol=1e-12, rtol=0)
            self.assertFalse({"mu_post", "var_post"} & set(p0_predicted.columns))

    def test_hash_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = json.loads((BASE / "p7_forward/forward_config.yaml").read_text(encoding="utf-8"))
            payload["model"]["config_sha256"] = "0" * 64
            payload["output"] = {"directory": str(root / "out"), "backup_directory": str(root / "backup"), "temp_directory": str(root / "work")}
            path = root / "forward_config.yaml"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(P7ForwardError):
                load_config(path)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
