from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from forward_ops import (
    EXPECTED_P2B_HASH,
    OUTPUT_FILES,
    ForwardError,
    load_config,
    predict_snapshot,
    restore_backup,
    resolve_path,
    update,
)
from rating_pedigree import build_or_load_base, load_pedigree_ids
from test_rating import race_rows

BASE = Path(__file__).resolve().parent


def dummy_rows() -> pd.DataFrame:
    rows = []
    definitions = [
        (900000, "2025-01-03", [1990000001, 1990000004, 1990000002, 1990000003], [1, 2, 0, 0], [0, 0, 0, 1], [1, 2, 3, 1], 701, "BaoZ_data_division_7"),
        (900001, "2025-01-04", [2019105219, 2018110015, 2019105283, 1990000001, 1990000002, 1990000003], [1, 2, 3, 4, 5, 0], [0, 0, 0, 0, 0, 1], [1, 2, 1, 2, 3, 1], 703, "BaoZ_data_division_7"),
        (900002, "2025-01-05", [2019105219, 2018110015, 2019105283, 1990000001, 1990000002, 1990000003], [2, 1, 3, 4, 5, 0], [0, 0, 0, 0, 0, 1], [1, 2, 1, 2, 3, 1], 703, "BaoZ_data_division_7"),
        (900003, "2025-01-06", [2019105219, 2018110015, 2019105283, 1990000001, 1990000002, 1990000003], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 2, 1, 2, 3, 1], 703, "BaoZ_data_division_1"),
    ]
    for race_id, date, horses, finishes, abnormal, affiliations, condition, version in definitions:
        for index, horse_id in enumerate(horses):
            rows.append({
                "race_id": race_id,
                "race_date": pd.Timestamp(date),
                "organizer_code": 1,
                "surface_code": 0,
                "venue_code": 5,
                "venue_name": "東京",
                "distance": 1600,
                "condition_code": condition,
                "n_registered": len(horses),
                "horse_id": horse_id,
                "horse_number": index + 1,
                "weight_kg": 54.0 + index,
                "finish_position": finishes[index],
                "abnormal_code": abnormal[index],
                "affiliation_code": affiliations[index],
                "result_version": version,
                "race_data_created": date,
            })
    return pd.DataFrame(rows)


def loader_for(raw: pd.DataFrame):
    def loader(db_path, start_date, end_date, final_only):
        frame = raw.copy()
        if final_only:
            frame = frame[frame["result_version"].eq("BaoZ_data_division_7")]
        frame = frame[frame["race_date"].ge(pd.Timestamp(start_date))]
        if end_date is not None:
            frame = frame[frame["race_date"].lt(pd.Timestamp(end_date))]
        return frame
    return loader


def expected_p2b_mu0(row, config):
    p1 = config["_model"]["base_p1"]["p1"]
    beta = config["_model"]["p2"]["beta"]
    b = {1: p1["b"]["新馬戦"], 2: p1["b"]["未勝利戦"], 3: p1["b"]["その他のクラス"]}[int(row["debut_type_code"])]
    return (
        row["mu_bar_D"] + b
        + beta["beta_sw"] * (row["X_sire_win"] if pd.notna(row["X_sire_win"]) else 0.0)
        + beta["beta_st"] * (row["X_sire_top3"] if pd.notna(row["X_sire_top3"]) else 0.0)
        + beta["beta_bw"] * (row["X_bms_win"] if pd.notna(row["X_bms_win"]) else 0.0)
        + beta["beta_bt"] * (row["X_bms_top3"] if pd.notna(row["X_bms_top3"]) else 0.0)
    )


class ForwardOpsTest(unittest.TestCase):
    @unittest.skipUnless((BASE / "stage2_input/BaoZ_2_snapshot.mdb").is_file(), "公開リポジトリでは実DBを同梱しません")
    def test_update_idempotency_predict_dry_run_elo_and_rollback(self):
        raw = dummy_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "forward_config.yaml"
            config_payload = json.loads((BASE / "forward_config.yaml").read_text(encoding="utf-8"))
            config_payload["database"]["path"] = str(BASE / "stage2_input" / "BaoZ_2_snapshot.mdb")
            config_payload["database"]["linked_tables"] = {
                "産駒マスタ": r"C:\BaoZ_AI\DB\MasterDB\BaoZ-RA.mdb"
            }
            config_payload["output"] = {
                "directory": str(root / "p2b_forward"),
                "backup_directory": str(root / "backups"),
                "temp_directory": str(root / "work"),
            }
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")
            config = load_config(config_path)
            out = Path(config["output"]["directory"])
            loader = loader_for(raw)
            base = build_or_load_base(BASE / "stage2_output/snapshots.csv.gz", BASE / "stage2_input/BaoZ_2_snapshot.mdb", use_cache=False)
            pedigree_row = base.history[["sire_id", "bms_id"]].dropna().iloc[0]
            dummy_pedigree = pd.DataFrame({"horse_id": [1990000001], "sire_id": [pedigree_row["sire_id"]], "bms_id": [pedigree_row["bms_id"]]})

            def pedigree_with_dummy(db_path, **kwargs):
                original = load_pedigree_ids(db_path)
                return pd.concat([original, dummy_pedigree], ignore_index=True)

            with patch("rating_pedigree.load_pedigree_ids", side_effect=pedigree_with_dummy):
                first = update(config, "2025-01-04", raw_loader=loader)
                self.assertEqual(first["status"], "UPDATED")
                self.assertEqual(first["new_races"], 2)
                snapshots = pd.read_csv(out / OUTPUT_FILES["snapshots"])
                self.assertEqual(len(snapshots), 6)
                self.assertEqual(snapshots["n_removed_nonJRA"].iloc[0], 1)
                self.assertEqual(snapshots["n_field_retained"].iloc[0], 2)
                self.assertEqual(snapshots["n_field_retained"].max(), 4)
                new_horse = snapshots[snapshots["horse_id"].eq(1990000001)].iloc[0]
                self.assertAlmostEqual(new_horse["mu0"], expected_p2b_mu0(new_horse, config), places=12)
                missing_pedigree = snapshots[snapshots["horse_id"].eq(1990000004)].iloc[0]
                self.assertAlmostEqual(missing_pedigree["mu0"], missing_pedigree["mu_bar_D"] + config["_model"]["base_p1"]["p1"]["b"]["新馬戦"], places=12)
                self.assertEqual(new_horse["var_pre_eff"], 4.0)
                events = pd.read_csv(out / OUTPUT_FILES["events"])
                self.assertEqual(len(events), 2)
                self.assertTrue(events["race_id"].is_unique)
                first_hashes = {name: (out / name).stat().st_size for name in OUTPUT_FILES.values()}

                second = update(config, "2025-01-04", raw_loader=loader)
                self.assertEqual(second["status"], "NO_NEW_RACES")
                second_hashes = {name: (out / name).stat().st_size for name in OUTPUT_FILES.values()}
                self.assertEqual(first_hashes, second_hashes)

                third = update(config, "2025-01-05", raw_loader=loader)
                self.assertEqual(third["status"], "UPDATED")
                self.assertEqual(third["new_races"], 1)
                events = pd.read_csv(out / OUTPUT_FILES["events"])
                self.assertEqual(len(events), 3)
                self.assertTrue(events["race_id"].is_unique)
                snapshots = pd.read_csv(out / OUTPUT_FILES["snapshots"])
                self.assertEqual(len(snapshots), 10)
                state = pd.read_csv(out / OUTPUT_FILES["state"])
                self.assertEqual(state["last_start_date"].max(), "2025-01-05")
                residual = pd.read_csv(out / OUTPUT_FILES["residual"])
                self.assertGreaterEqual(len(residual), 2)
                anchor = pd.read_csv(out / OUTPUT_FILES["anchor"])
                self.assertEqual(len(anchor), 3)
                operational_hashes = {name: (out / name).stat().st_size for name in OUTPUT_FILES.values()}

                prediction = predict_snapshot(config, "2025-01-06", raw_loader=loader)
                self.assertEqual(prediction["status"], "PREDICT_SNAPSHOT_WRITTEN")
                self.assertFalse(prediction["state_files_modified"])
                after_prediction = {name: (out / name).stat().st_size for name in OUTPUT_FILES.values()}
                self.assertEqual(operational_hashes, after_prediction)
                predicted = pd.read_csv(prediction["output"])
                self.assertEqual(len(predicted), 4)
                self.assertFalse({"mu_post", "var_post", "skip_mu_var_update"} & set(predicted.columns))
                expected_elo = 1500 + (400 / np.log(10)) * (predicted["mu_pre"] - predicted["mu_bar_D"])
                np.testing.assert_allclose(predicted["elo_pre"], expected_elo, atol=1e-12, rtol=0)
                for row in predicted.itertuples():
                    current = state[state["horse_id"].eq(row.horse_id)].iloc[0]
                    self.assertEqual(row.n_graph_starts, current["n_graph_starts"])

                backup_dir = Path(third["backup"])
                self.assertTrue(backup_dir.exists())
                restore_backup(out, backup_dir)
                restored_events = pd.read_csv(out / OUTPUT_FILES["events"])
                self.assertEqual(len(restored_events), 2)

    @unittest.skipUnless((BASE / "stage2_input/BaoZ_2_snapshot.mdb").is_file(), "公開リポジトリでは実DBを同梱しません")
    def test_predict_new_horse_prior(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "forward_config.yaml"
            payload = json.loads((BASE / "forward_config.yaml").read_text(encoding="utf-8"))
            payload["database"]["path"] = str(BASE / "stage2_input" / "BaoZ_2_snapshot.mdb")
            payload["database"]["linked_tables"] = {"産駒マスタ": r"C:\BaoZ_AI\DB\MasterDB\BaoZ-RA.mdb"}
            payload["output"] = {"directory": str(root / "out"), "backup_directory": str(root / "backup"), "temp_directory": str(root / "work")}
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(config_path)
            raw = pd.DataFrame(race_rows(900010, "2025-01-06", [1990000005, 1990000006], [1, 2], [54, 55], condition_code=701))
            raw["race_date"] = pd.to_datetime(raw["race_date"])
            base = build_or_load_base(BASE / "stage2_output/snapshots.csv.gz", BASE / "stage2_input/BaoZ_2_snapshot.mdb", use_cache=False)
            pedigree_row = base.history[["sire_id", "bms_id"]].dropna().iloc[0]
            dummy = pd.DataFrame({
                "horse_id": [1990000005, 1990000006],
                "sire_id": [pedigree_row["sire_id"], pedigree_row["sire_id"]],
                "bms_id": [pedigree_row["bms_id"], pedigree_row["bms_id"]],
            })
            with patch("rating_pedigree.load_pedigree_ids", side_effect=lambda db_path, **kwargs: pd.concat([load_pedigree_ids(db_path), dummy], ignore_index=True)):
                result = predict_snapshot(config, "2025-01-06", raw_loader=loader_for(raw))
            predicted = pd.read_csv(result["output"])
            for horse_id in [1990000005, 1990000006]:
                row = predicted[predicted["horse_id"].eq(horse_id)].iloc[0]
                self.assertAlmostEqual(row["mu0"], expected_p2b_mu0(row, config), places=12)
                self.assertEqual(row["var_pre_eff"], 4.0)

    def test_project_relative_config_path(self):
        self.assertEqual(resolve_path("rating_jra/forward_config.yaml"), BASE / "forward_config.yaml")
        self.assertEqual(resolve_path("forward_config.yaml"), BASE / "forward_config.yaml")

    def test_hash_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "forward_config.yaml"
            payload = json.loads((BASE / "forward_config.yaml").read_text(encoding="utf-8"))
            payload["model"]["config_sha256"] = "0" * 64
            payload["output"] = {"directory": str(root / "out"), "backup_directory": str(root / "backup"), "temp_directory": str(root / "work")}
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ForwardError):
                load_config(config_path)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
