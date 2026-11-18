from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import W_EVAL, file_sha256, pl_race_single
from rating_data import load_or_prepare, make_orders
from rating_pedigree import build_or_load_base, build_or_load_features, standardization_matches

OUT = Path(__file__).with_name("stage3_output")


def main() -> None:
    config_hashes = {}
    for model in ["P1", "P2a", "P2b"]:
        path = OUT / f"config_{model.lower()}.yaml"
        digest = file_sha256(path)
        assert digest == path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
        config_hashes[model] = digest
    p0_hash = file_sha256("stage2_output/config_p0.yaml")
    assert p0_hash == "a06b15ef4e9affe345807b500a6b46fa769a6e1939ad0f309805d4a639a303c8"
    p1 = json.loads((OUT / "config_p1.yaml").read_text(encoding="utf-8"))
    p2a = json.loads((OUT / "config_p2a.yaml").read_text(encoding="utf-8"))
    p2b = json.loads((OUT / "config_p2b.yaml").read_text(encoding="utf-8"))
    assert p1["base_p0_config_sha256"] == p0_hash
    assert p2a["base_p1_config_sha256"] == config_hashes["P1"]
    assert p2b["base_p1_config_sha256"] == config_hashes["P1"]
    assert p1["input_db_sha256"] == p2a["input_db_sha256"] == p2b["input_db_sha256"]
    prepared = load_or_prepare(Path("stage2_input/BaoZ_2_snapshot.mdb"))
    base = build_or_load_base(Path("stage2_output/snapshots.csv.gz"), Path("stage2_input/BaoZ_2_snapshot.mdb"))
    assert len(base.history) == 144377
    assert base.history["n_graph_starts"].max() <= 2
    assert base.history["n_graph_starts"].min() == 0
    for row in base.candidates.iloc[::55000].itertuples():
        for side in ["sire", "bms"]:
            pedigree_id = getattr(row, f"{side}_id")
            if pd.isna(pedigree_id) or pedigree_id == 0 or row.debut_type_code == 4:
                continue
            history = base.history[(base.history[f"{side}_id"] == pedigree_id) & (base.history["date"] < row.date)]
            assert int(getattr(row, f"N_{side}")) == len(history)
            assert np.isclose(getattr(row, f"{side}_sum_win"), history["r_win"].sum())
            assert np.isclose(getattr(row, f"{side}_sum_top3"), history["r_top3"].sum())
    max_nll_error = {}
    for model in ["P1", "P2a", "P2b"]:
        model_dir = OUT / model.lower()
        snapshots = pd.read_csv(model_dir / "snapshots.csv.gz", parse_dates=["race_date"])
        assert len(snapshots) == 610676
        assert snapshots["race_id"].nunique() == 43252
        assert snapshots["race_date"].max() < pd.Timestamp("2025-01-01")
        assert snapshots["prior_type"].eq(model).all()
        assert snapshots["config_sha256"].eq(config_hashes[model]).all()
        assert snapshots["mu0"].notna().all()
        required = {"X_sire", "X_bms", "N_sire", "N_bms", "pedigree_status", "X_sire_win", "X_sire_top3", "X_bms_win", "X_bms_top3"}
        assert required <= set(snapshots.columns)
        state = pd.read_csv(model_dir / "state_2024-12-31.csv")
        assert state["config_sha256"].eq(config_hashes[model]).all()
        assert state["prior_type"].eq(model).all()
        assert state["mu0"].notna().all()
        metrics = pd.read_csv(model_dir / "race_metrics.csv.gz").set_index("race_id")
        errors = []
        for race_id in metrics.index[::2200]:
            group = snapshots[snapshots["race_id"].eq(race_id)]
            finish = group["finish_position"].to_numpy(dtype=np.int16)
            orders = make_orders(finish)
            unranked = np.flatnonzero(finish == 0).astype(np.int16)
            k_max = min(5, int(np.count_nonzero(finish > 0)), len(group) - 1)
            value = -pl_race_single(group["m_pre"].to_numpy(dtype=float), orders, unranked, k_max, W_EVAL)[0]
            errors.append(abs(value - metrics.loc[race_id, "p0_nll"]))
        max_nll_error[model] = float(max(errors))
        assert max(errors) < 1.0e-9
        events = pd.read_csv(model_dir / "result_events.csv")
        assert len(events) == 43252 and events["event_id"].is_unique and events["race_id"].is_unique
    for model, config in [("P2a", p2a), ("P2b", p2b)]:
        features = build_or_load_features(base, int(config["p2"]["kappa_s"]))
        assert standardization_matches(features.constants, config["p2"]["standardization"])
    for debut, code in [("新馬戦", 1), ("未勝利戦", 2), ("その他のクラス", 3)]:
        state = pd.read_csv(OUT / "p1" / "state_2024-12-31.csv")
        actual = state.loc[state["debut_type_code"].eq(code), "var_cap"]
        assert np.allclose(actual, p1["p1"]["var0_d"][debut])
    comparisons = pd.read_csv(OUT / "pairwise_comparisons.csv")
    pooled = comparisons[(comparisons["comparison"] == "P2b-P2a") & (comparisons["period"] == "validation") & (comparisons["pool"] == "E")]
    assert len(pooled) == 5
    judgment = json.loads((OUT / "stage3_summary.json").read_text(encoding="utf-8"))["judgment"]
    assert len(judgment["early_yearly_delta"]) == 4
    report = (OUT / "stage3_report.md").read_text(encoding="utf-8")
    assert "最終判断は依頼者" in report
    verification = {
        "status": "PASS",
        "config_sha256": config_hashes,
        "p0_config_sha256": p0_hash,
        "input_db_sha256": prepared.source_hash,
        "models": ["P0", "P1", "P2a", "P2b"],
        "snapshot_rows_per_model": 610676,
        "pedigree_history_rows": len(base.history),
        "max_independent_nll_error": max_nll_error,
        "judgment": judgment,
        "future_scripts_executed": False,
    }
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
