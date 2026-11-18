from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forward_ops import EXPECTED_P2B_HASH, OUTPUT_FILES, load_config, update
from rating_core import file_sha256


def verify_forward_outputs(check_idempotency: bool = False) -> dict:
    out = Path("p2b_forward")
    manifest_path = out / "latest_run_manifest.json"
    if not manifest_path.exists():
        raise AssertionError("latest_run_manifest.jsonがありません")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "UPDATED"
    assert manifest["config_sha256"] == EXPECTED_P2B_HASH
    assert (out / f"run_manifest_{manifest['run_id']}.json").exists()
    events = pd.read_csv(out / OUTPUT_FILES["events"])
    snapshots = pd.read_csv(out / OUTPUT_FILES["snapshots"])
    posts = pd.read_csv(out / OUTPUT_FILES["posts"])
    assert events["race_id"].is_unique and events["event_id"].is_unique
    assert len(events) == manifest["new_races"]
    assert not snapshots.duplicated(["race_id", "horse_id"]).any()
    assert not posts.duplicated(["race_id", "horse_id"]).any()
    assert len(snapshots) == len(posts)
    assert set(events["race_id"]) == set(snapshots["race_id"])
    assert snapshots["config_sha256"].eq(EXPECTED_P2B_HASH).all()
    assert posts["config_sha256"].eq(EXPECTED_P2B_HASH).all()
    source_hashes = set(events["input_db_sha256"].unique())
    assert len(source_hashes) == 1
    assert snapshots["source_db_sha256"].eq(next(iter(source_hashes))).all()
    assert np.allclose(
        snapshots["elo_pre"],
        1500 + (400 / np.log(10)) * (snapshots["mu_pre"] - snapshots["mu_bar_D"]),
        atol=1e-12,
        rtol=0,
    )
    state = pd.read_csv(out / OUTPUT_FILES["state"])
    p0_state = pd.read_csv(out / OUTPUT_FILES["p0_state"])
    assert state["config_sha256"].eq(EXPECTED_P2B_HASH).all()
    assert p0_state["config_sha256"].eq("a06b15ef4e9affe345807b500a6b46fa769a6e1939ad0f309805d4a639a303c8").all()
    assert state["last_start_date"].max() == manifest["last_race_date"]
    anchor = pd.read_csv(out / OUTPUT_FILES["anchor"])
    assert not anchor["asof_date"].duplicated().any()
    assert anchor["asof_date"].max() == manifest["last_race_date"]
    residual = pd.read_csv(out / OUTPUT_FILES["residual"])
    assert residual["n_graph_starts"].max() <= 2
    assert not residual.duplicated(["race_index", "horse_id"]).any()
    p0_snapshots = pd.read_csv(out / OUTPUT_FILES["p0_snapshots"])
    assert not p0_snapshots.duplicated(["race_id", "horse_id"]).any()
    for name, expected_hash in manifest["output_hashes"].items():
        assert file_sha256(out / name) == expected_hash, name
    backup = Path(manifest["backup"])
    assert backup.exists()
    report = {
        "status": "PASS",
        "run_id": manifest["run_id"],
        "config_sha256": EXPECTED_P2B_HASH,
        "input_db_sha256": next(iter(source_hashes)),
        "new_races": manifest["new_races"],
        "first_race_date": manifest["first_race_date"],
        "last_race_date": manifest["last_race_date"],
        "snapshot_rows": len(snapshots),
        "post_update_rows": len(posts),
        "p0_snapshot_rows": len(p0_snapshots),
        "residual_history_rows": len(residual),
        "anchor_days": len(anchor),
        "state_horses": len(state),
        "backup": str(backup),
        "output_hashes": manifest["output_hashes"],
    }
    if check_idempotency:
        before = {name: file_sha256(out / name) for name in OUTPUT_FILES.values()}
        config = load_config("forward_config.yaml")
        second = update(config, None)
        assert second["status"] == "NO_NEW_RACES"
        after = {name: file_sha256(out / name) for name in OUTPUT_FILES.values()}
        assert before == after
        report["idempotency"] = {"status": "NO_NEW_RACES", "files_unchanged": True}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-idempotency", action="store_true")
    args = parser.parse_args()
    report = verify_forward_outputs(args.check_idempotency)
    Path("p2b_forward/forward_completion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
