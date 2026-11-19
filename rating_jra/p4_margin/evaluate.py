from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
P3 = BASE / "p3_grade"
for path in [BASE, P3]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from p4_margin.common import run_configuration
from p3_grade.evaluate import cluster_bootstrap
from p3_grade.grades import CATEGORIES
from p4_margin.margins import multiplier_arrays
from p3_grade.common import frozen_model
from p3_grade.prepare_inputs import load_cached_inputs
from rating_core import file_sha256
from rating_evaluation import race_metrics

OUT = Path(__file__).resolve().parent


def comparison_row(frame: pd.DataFrame, protocol: dict) -> dict:
    delta = frame["P4_nll"] - frame["P2b_nll"]
    stats = cluster_bootstrap(frame, delta, protocol["bootstrap"]["repetitions"], protocol["bootstrap"]["seed"])
    return {
        "n_races": len(frame), "starts": int(frame["n_field_retained"].sum()),
        "P2b_nll": float(frame["P2b_nll"].mean()), "P4_nll": float(frame["P4_nll"].mean()),
        "delta_P4_minus_P2b": float(delta.mean()), "ci95_lower": stats["ci95_lower"],
        "ci95_upper": stats["ci95_upper"], "p_one_sided_improvement": stats["p_one_sided_improvement"],
        "bootstrap_repetitions": protocol["bootstrap"]["repetitions"], "bootstrap_seed": protocol["bootstrap"]["seed"],
    }


def main() -> None:
    protocol = json.loads((OUT / "evaluation_protocol.json").read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_P4_VALIDATION":
        raise ValueError("評価手順が凍結されていません")
    config_path = OUT / "selected_config.json"
    config_hash = file_sha256(config_path)
    if config_hash != config_path.with_suffix(".json.sha256").read_text(encoding="ascii").strip():
        raise ValueError("P4選択configのハッシュが不一致です")
    selected = json.loads(config_path.read_text(encoding="utf-8"))
    if selected["status"] != "FROZEN_BEFORE_P4_VALIDATION":
        raise ValueError("P4が選択時点で凍結されていません")
    prepared, pedigree, races, _ = load_cached_inputs()
    margins = pd.read_csv(OUT / "race_margins.csv")
    model = frozen_model()
    config = {
        "method": selected["method"], "normalize": selected["normalize"], "gamma": selected["gamma"],
        "m_max": selected["m_max"], "grid_stage": "frozen",
    }
    if selected["method"] == "B":
        config["m_min"] = selected["m_min"]
    stages, scales = multiplier_arrays(margins, prepared, config)
    result, engine = run_configuration(prepared, pedigree, margins, model, config, record=True)
    from forward_ops import enrich_snapshots, state_frame
    snapshots = enrich_snapshots(result, engine, "P4", config_hash)
    margin_index = margins.set_index("race_id")
    snapshots["winner_margin_seconds"] = snapshots["race_id"].map(margin_index["stage_1"])
    snapshots["winner_margin_normalized"] = snapshots["winner_margin_seconds"] / (snapshots["distance"] / 1000.0)
    snapshots["race_margin_scale"] = snapshots["race_id"].map(pd.Series(scales, index=[race.race_id for race in prepared.races]))
    snapshots.to_csv(OUT / "snapshots.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    snapshots[["race_id", "race_date", "horse_id", "mu_post", "var_post", "config_sha256"]].to_csv(
        OUT / "post_updates.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d"
    )
    state_frame(result, engine, "P4", config_hash).to_csv(OUT / "state_2024-12-31.csv", index=False, float_format="%.12g")
    daily = result.daily.copy()
    daily["asof_date"] = pd.to_datetime(daily["asof_date"], unit="D", origin="unix")
    daily["config_sha256"] = config_hash
    daily.to_csv(OUT / "mu_anchor_daily.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d")
    history = engine.history.copy()
    history["config_sha256"] = config_hash
    history.to_csv(OUT / "pedigree_residual_history.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    pd.DataFrame([{
        "race_id": race.race_id,
        "race_date": pd.Timestamp(1970, 1, 1) + pd.Timedelta(days=race.date),
        "result_version": race.result_version, "config_sha256": config_hash,
    } for race in prepared.races]).to_csv(OUT / "result_events.csv", index=False, date_format="%Y-%m-%d")
    multiplier_frame = pd.DataFrame({
        "race_id": [race.race_id for race in prepared.races],
        **{f"stage_multiplier_{index}": stages[:, index - 1] for index in range(1, 6)},
        "race_scale": scales,
    })
    multiplier_frame.to_csv(OUT / "margin_multipliers.csv.gz", index=False, compression="gzip", float_format="%.17g")
    selected_metrics = race_metrics(prepared, result)
    selected_metrics.to_csv(OUT / "race_metrics.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    tuning_nll = selected_metrics[selected_metrics["year"].le(2020) & selected_metrics["F"]]["p0_nll"].mean()
    if not np.isclose(tuning_nll, selected["tuning_nll"], atol=1e-12, rtol=0):
        raise AssertionError("P4の学習期間NLLが探索時と一致しません")
    model_summary = {
        "config_sha256": config_hash, "races": len(prepared.races), "starts": len(snapshots),
        "residual_history_rows": len(history), "skip_mu_var_updates": result.skip_mu_var_updates,
        "anchor_fallback_days": result.fallback_anchor_days, "initial_zero_anchor_days": result.initial_zero_anchor_days,
        "tuning_nll_reproduced": float(tuning_nll),
    }
    (OUT / "model_summary.json").write_text(json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    del snapshots, history, daily, result, engine
    gc.collect()

    baseline = pd.read_csv(BASE / "stage3_output/p2b/race_metrics.csv.gz")
    baseline_columns = [
        "race_id", "race_date", "year", "E", "E_clean", "F", "validation", "condition_code",
        "n_field_retained", "mean_n_graph_starts", "p0_nll",
    ]
    combined = baseline[baseline_columns].rename(columns={"p0_nll": "P2b_nll"}).merge(
        selected_metrics[["race_id", "p0_nll"]].rename(columns={"p0_nll": "P4_nll"}), on="race_id", how="left", validate="one_to_one"
    ).merge(races[["race_id", "category"]], on="race_id", how="left", validate="one_to_one")
    if combined[["P2b_nll", "P4_nll"]].isna().any().any():
        raise ValueError("NLL比較に欠損があります")
    validation = combined[combined["validation"]].copy()
    validation.to_csv(OUT / "validation_race_comparison.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    pools = {
        "F": validation[validation["F"]], "E": validation[validation["E"]],
        "E_clean": validation[validation["E_clean"]],
        "E_experience_le3": validation[validation["E"] & validation["mean_n_graph_starts"].le(3.0)],
        "E_experience_ge4": validation[validation["E"] & validation["mean_n_graph_starts"].gt(3.0)],
    }
    overall_rows = [{"pool": name, "year": "POOLED", **comparison_row(frame, protocol)} for name, frame in pools.items()]
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUT / "validation_overall.csv", index=False, float_format="%.12g")
    yearly_rows = []
    for pool in ["F", "E"]:
        for year, frame in pools[pool].groupby("year"):
            yearly_rows.append({"pool": pool, "year": int(year), **comparison_row(frame, protocol)})
    pd.DataFrame(yearly_rows).to_csv(OUT / "validation_by_year.csv", index=False, float_format="%.12g")
    grade_rows = []
    for category in CATEGORIES:
        frame = pools["F"][pools["F"]["category"].eq(category)]
        grade_rows.append({"category": category, **comparison_row(frame, protocol)})
    pd.DataFrame(grade_rows).to_csv(OUT / "validation_by_grade.csv", index=False, float_format="%.12g")

    primary = overall[overall["pool"].eq("F")].iloc[0]
    adopted = bool(primary["delta_P4_minus_P2b"] < 0 and primary["ci95_upper"] < 0)
    judgment = {
        "status": "P4_CANDIDATE_CONFIRMED" if adopted else "P2B_RETAINED",
        "adopted_model": "P4" if adopted else "P2b",
        "reason": "検証Fの標準PL-NLL改善が開催日クラスター95%CIで有意" if adopted else "検証Fの標準PL-NLL改善が有意ではありません",
        "primary": primary.to_dict(),
        "p2b_config_sha256": file_sha256(BASE / "stage3_output/config_p2b.yaml"),
        "p4_config_sha256": config_hash,
        "evaluation_protocol_sha256": file_sha256(OUT / "evaluation_protocol.json"),
        "validation_settings_changed_after_results": False,
    }
    (OUT / "judgment.json").write_text(json.dumps(judgment, ensure_ascii=False, indent=2), encoding="utf-8")
    output_names = [
        "selected_config.json", "margin_multipliers.csv.gz", "snapshots.csv.gz", "post_updates.csv.gz",
        "state_2024-12-31.csv", "mu_anchor_daily.csv", "pedigree_residual_history.csv.gz",
        "race_metrics.csv.gz", "validation_overall.csv", "validation_by_year.csv", "validation_by_grade.csv", "judgment.json",
    ]
    final = {
        "adopted_model": judgment["adopted_model"],
        "adopted_config_sha256": config_hash if adopted else judgment["p2b_config_sha256"],
        "p4_research_output_hashes": {name: file_sha256(OUT / name) for name in output_names},
    }
    (OUT / "final_model.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model_summary": model_summary, "judgment": judgment, "overall": overall_rows}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
