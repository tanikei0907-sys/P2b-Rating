from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import P0Config, file_sha256
from rating_data import load_or_prepare
from rating_engine import run_engine
from rating_evaluation import race_metrics, secondary_diagnostics
from rating_initialization import InitialRule
from rating_pedigree import build_or_load_base, build_or_load_features, pedigree_auxiliary
from stage2_search import library_versions
from stage3_io import save_stage3_model
from stage3_metrics import (
    combine_metrics,
    debut_majority,
    pairwise_comparisons,
    pairwise_layers,
    pedigree_information,
    stage3_judgment,
)
from stage3_p1 import default_json

OUT = Path(__file__).with_name("stage3_output")


def config_with_hash(path: Path) -> tuple[dict, str]:
    digest = file_sha256(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
    if digest != expected:
        raise ValueError(f"configハッシュが一致しません: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "FROZEN_BEFORE_STAGE3_VALIDATION" and payload.get("model") != "P0":
        raise ValueError(f"凍結前のconfigです: {path}")
    return payload, digest


def rule_for_p1(config: dict) -> InitialRule:
    b = np.array([config["p1"]["b"]["新馬戦"], config["p1"]["b"]["未勝利戦"], config["p1"]["b"]["その他のクラス"]])
    var = np.array([config["p1"]["var0_d"]["新馬戦"], config["p1"]["var0_d"]["未勝利戦"], config["p1"]["var0_d"]["その他のクラス"]])
    return InitialRule(b=b[None, :], var0_d=var[None, :], beta=np.zeros((1, 4)))


def rule_for_p2(config: dict) -> InitialRule:
    rule = rule_for_p1(config["base_p1"])
    beta = config["p2"]["beta"]
    rule.beta = np.array([[beta["beta_sw"], beta["beta_st"], beta["beta_bw"], beta["beta_bt"]]])
    return rule


def main() -> None:
    prepared = load_or_prepare(Path("stage2_input/BaoZ_2_snapshot.mdb"))
    p0_config, p0_hash = config_with_hash(Path("stage2_output/config_p0.yaml"))
    p1_config, p1_hash = config_with_hash(OUT / "config_p1.yaml")
    p2a_config, p2a_hash = config_with_hash(OUT / "config_p2a.yaml")
    p2b_config, p2b_hash = config_with_hash(OUT / "config_p2b.yaml")
    if p1_config["base_p0_config_sha256"] != p0_hash:
        raise ValueError("P1の基準P0ハッシュが一致しません")
    if p2a_config["base_p1_config_sha256"] != p1_hash or p2b_config["base_p1_config_sha256"] != p1_hash:
        raise ValueError("P2の基準P1ハッシュが一致しません")
    p0 = P0Config(**p0_config["p0"])
    base = build_or_load_base(Path("stage2_output/snapshots.csv.gz"), Path("stage2_input/BaoZ_2_snapshot.mdb"))
    model_metrics = {"P0": pd.read_csv("stage2_output/race_metrics.csv.gz")}
    model_summaries = {}
    debut_majority_frame = None
    for model, config, config_hash in [("P1", p1_config, p1_hash), ("P2a", p2a_config, p2a_hash), ("P2b", p2b_config, p2b_hash)]:
        model_dir = OUT / model.lower()
        rule = rule_for_p1(config) if model == "P1" else rule_for_p2(config)
        provider = None
        if model != "P1":
            features = build_or_load_features(base, int(config["p2"]["kappa_s"]))
            provider = features.provider
        result = run_engine(
            prepared,
            [p0],
            record=True,
            initial_rule=rule,
            feature_provider=provider,
            model_name=model,
        )
        save_stage3_model(prepared, result, model, config_hash, model_dir)
        model_metrics[model] = race_metrics(prepared, result)
        model_metrics[model].to_csv(model_dir / "race_metrics.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
        secondary = secondary_diagnostics(prepared, result, p0.tau, 2000, 20241117)
        for name, value in secondary.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(model_dir / f"{name}.csv", index=False, float_format="%.12g")
        daily = result.daily
        model_summaries[model] = {
            "config_sha256": config_hash,
            "skip_mu_var_updates": result.skip_mu_var_updates,
            "anchor_max_abs_daily_delta": float(daily["mu_bar_delta_1d"].abs().max()),
            "anchor_crossings": int(daily["threshold_crossing"].ne("").sum()),
            "new_horse_difference": secondary["new_horse_difference"],
        }
        if model == "P2b":
            information, status = pedigree_information(result.snapshots)
            information.to_csv(model_dir / "pedigree_information.csv", index=False, float_format="%.12g")
            status.to_csv(model_dir / "pedigree_status.csv", index=False, float_format="%.12g")
            debut_majority_frame = debut_majority(result.snapshots)
        del result, secondary, provider
        gc.collect()

    combined = combine_metrics(model_metrics)
    combined = combined.merge(debut_majority_frame, on="race_index", how="left", validate="one_to_one")
    combined.to_csv(OUT / "combined_race_metrics.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    comparisons = pairwise_comparisons(combined)
    comparisons.to_csv(OUT / "pairwise_comparisons.csv", index=False, float_format="%.12g")
    layers = pairwise_layers(combined)
    layers.to_csv(OUT / "pairwise_layers.csv", index=False, float_format="%.12g")
    judgment, judgment_details = stage3_judgment(combined)
    judgment.to_csv(OUT / "judgment.csv", index=False)
    base.history.to_csv(OUT / "pedigree_residual_history.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    pedigree_auxiliary(base.history, "sire").to_csv(OUT / "sire_auxiliary.csv", index=False, float_format="%.12g")
    pedigree_auxiliary(base.history, "bms").to_csv(OUT / "bms_auxiliary.csv", index=False, float_format="%.12g")
    summary = {
        "stage": "3_P1_P2a_P2b",
        "input_db_sha256": prepared.source_hash,
        "p0_config_sha256": p0_hash,
        "model_summaries": model_summaries,
        "judgment": judgment_details,
        "final_judgment_assigned_to": "依頼者",
        "library_versions": library_versions(),
        "input_diagnostics": prepared.diagnostics,
        "pedigree_history_rows": len(base.history),
        "pedigree_sire_missing_rows": int(base.history["sire_id"].isna().sum()),
        "pedigree_bms_missing_rows": int(base.history["bms_id"].isna().sum()),
        "future_data_used": False,
        "validation_settings_changed_after_results": False,
    }
    (OUT / "stage3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=default_json), encoding="utf-8")
    (OUT / "input_manifest.json").write_text(json.dumps({
        "input_db_sha256": prepared.source_hash,
        "p0_config_sha256": p0_hash,
        "p1_config_sha256": p1_hash,
        "p2a_config_sha256": p2a_hash,
        "p2b_config_sha256": p2b_hash,
        "library_versions": library_versions(),
        "result_policy": p0_config["result_policy"],
        "result_event_count": len(prepared.races),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=default_json))


if __name__ == "__main__":
    main()
