from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from viewer_data import SCALE, merge_master, read_snapshot

BASE = Path(__file__).resolve().parent


def minimal_master(data_divisions: list[str]) -> dict[str, pd.DataFrame]:
    runners = pd.DataFrame([
        {
            "race_id": 900001, "race_date": pd.Timestamp("2026-11-22"), "venue_name": "東京",
            "race_number": 11, "distance": 1600, "surface_code": 0, "official_field_size": 2,
            "horse_number": horse, "horse_id": 2023000000 + horse, "horse_name": f"馬{horse}",
            "official_finish": 0 if division != "7" else horse, "win_payout": 0,
            "place_payout": 0, "abnormal_code": 0, "data_division": division,
            "race_name": "テスト特別", "condition_name": "2歳1勝クラス", "grade_code": "",
            "race_age_code": 11,
        }
        for division in data_divisions for horse in [1, 2]
    ])
    horses = pd.DataFrame({"horse_id": [2023000001, 2023000002], "horse_name": ["馬1", "馬2"]})
    breeding = pd.DataFrame({"pedigree_id": pd.Series(dtype="int64"), "pedigree_name": pd.Series(dtype="str")})
    pedigree = pd.DataFrame({"horse_id": [2023000001, 2023000002], "sire_id": [None, None], "bms_id": [None, None]})
    return {"runners": runners, "horses": horses, "breeding": breeding, "pedigree": pedigree}


def viewer_rows() -> pd.DataFrame:
    rows = []
    for mode in ["final", "predict"]:
        for horse in [1, 2]:
            rows.append({
                "race_id": 900001, "race_date": pd.Timestamp("2026-11-22"), "record_type": mode,
                "horse_id": 2023000000 + horse, "horse_name": f"馬{horse}", "venue_name": "東京",
                "race_number": 11, "race_age_code": 11, "grade_code": "", "condition_code": 5,
                "race_name": "テスト特別", "condition_name": "2歳1勝クラス", "distance": 1600,
                "surface_code": 0, "official_finish": horse if mode == "final" else 0,
                "elo_pre": 1500 + horse, "elo_sd": 100.0, "elo_prior_boost": 10.0,
                "elo_post": 1501 + horse if mode == "final" else np.nan,
                "elo_delta": float(horse) if mode == "final" else np.nan,
                "mu_pre": 0.1 * horse, "var_pre_eff": 0.4, "n_graph_starts": 3,
                "days_since_start": 28,
            })
    return pd.DataFrame(rows)


class ViewerDataTest(unittest.TestCase):
    def test_predict_snapshot_derives_elo_and_preserves_missing_post(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predict_snapshots_20261122.csv"
            pd.DataFrame([
                {"race_id": 900001, "race_date": "2026-11-22", "horse_id": 2023000001,
                 "mu_pre": 0.1, "var_pre_eff": 0.25, "mu_bar_D": 0.05}
            ]).to_csv(path, index=False)
            frame = read_snapshot(path, "predict")
        row = frame.iloc[0]
        self.assertAlmostEqual(row["elo_pre"], 1500 + SCALE * 0.05)
        self.assertAlmostEqual(row["elo_sd"], SCALE * 0.5)
        self.assertTrue(pd.isna(row["elo_post"]))
        self.assertTrue(pd.isna(row["elo_delta"]))
        self.assertEqual(row["record_type"], "predict")

    def test_master_prefers_final_and_retains_unconfirmed(self):
        snapshot = viewer_rows()[["race_id", "horse_id", "race_date", "record_type"]]
        confirmed = merge_master(snapshot, minimal_master(["1", "7"]))
        self.assertEqual(confirmed["official_finish"].tolist(), [1, 2, 1, 2])
        provisional = merge_master(snapshot, minimal_master(["1"]))
        self.assertTrue(provisional["official_finish"].eq(0).all())
        self.assertTrue(provisional["race_name"].eq("テスト特別").all())

    def test_four_tabs_accept_predict_rows_without_post_values(self):
        auxiliary = pd.DataFrame([{
            "side": "sire", "sire_id": 1, "bms_id": None, "pedigree_name": "父テスト",
            "offspring": 10, "starts": 30, "new_horse_win_rate": 0.1,
            "new_horse_top3_rate": 0.3, "offspring_win_within3_rate": 0.4,
            "residual_win_mean": 0.01, "residual_top3_mean": 0.02, "residual_starts": 30,
        }])
        with patch("viewer_data.load_viewer_config", return_value={}), patch("viewer_data.load_viewer_data", return_value=(viewer_rows(), auxiliary, {})):
            app = AppTest.from_file(str(BASE / "rating_viewer.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            mode = next(widget for widget in app.selectbox if widget.label == "表示区分")
            mode.set_value("predict").run()
            self.assertFalse(app.exception)
            self.assertEqual(mode.value, "predict")

    def test_auxiliary_uses_2017plus_and_includes_forward_residuals(self):
        from viewer_data import load_auxiliary
        history = pd.DataFrame([
            {"race_date": "2016-12-01", "horse_id": 10, "n_graph_starts": 0, "sire_id": 1, "bms_id": 2, "actual_win": 1.0, "actual_top3": 1.0, "r_win": 0.9, "r_top3": 0.8},
            {"race_date": "2025-01-01", "horse_id": 11, "n_graph_starts": 0, "sire_id": 1, "bms_id": 2, "actual_win": 0.0, "actual_top3": 0.0, "r_win": -0.1, "r_top3": -0.2},
            {"race_date": "2025-02-01", "horse_id": 11, "n_graph_starts": 1, "sire_id": 1, "bms_id": 2, "actual_win": 1.0, "actual_top3": 1.0, "r_win": 0.8, "r_top3": 0.7},
            {"race_date": "2025-03-01", "horse_id": 11, "n_graph_starts": 2, "sire_id": 1, "bms_id": 2, "actual_win": 0.0, "actual_top3": 0.0, "r_win": -0.2, "r_top3": -0.3},
        ])
        with patch("viewer_data.pd.read_csv", side_effect=lambda path, **kwargs: history.iloc[:1].copy() if "stage3_output" in str(path) else history.iloc[1:].copy()):
            auxiliary = load_auxiliary(minimal_master(["7"]))
        row = auxiliary[auxiliary["side"].eq("sire") & auxiliary["sire_id"].eq(1)].iloc[0]
        self.assertEqual(row["offspring"], 1)
        self.assertEqual(row["starts"], 3)
        self.assertEqual(row["new_horse_starts"], 1)
        self.assertEqual(row["offspring_winning_within3"], 1)
        self.assertAlmostEqual(row["residual_win_mean"], 0.5 / 3.0)
        self.assertTrue(auxiliary["sire_id"].dropna().isin([1]).all())

    def test_packaged_cache_is_used_without_database(self):
        import json
        from viewer_data import cache_paths, load_viewer_data, source_fingerprint
        with tempfile.TemporaryDirectory() as temporary:
            config = {"paths": {"cache_dir": temporary}}
            merged_path, auxiliary_path, fingerprint_path = cache_paths(config)
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            expected = viewer_rows()
            expected.to_parquet(merged_path, index=False)
            pd.DataFrame({"side": ["sire"]}).to_parquet(auxiliary_path, index=False)
            fingerprint = source_fingerprint(config)
            fingerprint_path.write_text(json.dumps(fingerprint), encoding="utf-8")
            with patch("viewer_data.load_master", side_effect=AssertionError("DBは不要です")):
                actual, _, _ = load_viewer_data(config)
            pd.testing.assert_frame_equal(actual, expected)
            self.assertTrue(all("mtime_ns" not in item and "sha256" in item for item in fingerprint.values() if isinstance(item, dict)))
            self.assertNotIn(str(BASE).replace("\\", "/"), json.dumps(fingerprint))

    def test_cached_viewer_does_not_import_database_modules(self):
        import subprocess
        import sys
        code = "import sys; import viewer_data; assert 'pyodbc' not in sys.modules; assert 'forward_ops' not in sys.modules; assert 'rating_pedigree' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=BASE, check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
