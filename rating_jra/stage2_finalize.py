from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import P0Config, file_sha256
from rating_data import load_or_prepare
from rating_engine import run_engine
from rating_evaluation import (
    fit_tau,
    layer_comparisons,
    main_comparisons,
    race_metrics,
    save_model_outputs,
    secondary_diagnostics,
)
from stage2_search import library_versions, write_yaml_json

OUTPUT_DIR = Path(__file__).with_name("stage2_output")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> None:
    pre_path = OUTPUT_DIR / "config_p0_pre_tau.yaml"
    expected_hash = pre_path.with_suffix(pre_path.suffix + ".sha256").read_text(encoding="ascii").strip()
    if file_sha256(pre_path) != expected_hash:
        raise ValueError("探索時configのハッシュが一致しません")
    config = json.loads(pre_path.read_text(encoding="utf-8"))
    prepared = load_or_prepare(Path(__file__).with_name("stage2_input") / "BaoZ_2_snapshot.mdb")
    if prepared.source_hash != config["input_db_sha256"]:
        raise ValueError("探索時と入力DBハッシュが変化しています")
    selected = P0Config(**config["p0"])
    started = time.time()
    result = run_engine(prepared, [selected], record=True)
    engine_seconds = time.time() - started
    masks = prepared.evaluation_masks()
    tau_info = fit_tau(result.snapshots, masks["tuning_F"])
    config["status"] = "FROZEN_BEFORE_VALIDATION_EVALUATION"
    config["p0"]["tau"] = tau_info["tau"]
    config["tau_selection"] = tau_info
    config["model_implementation"] = "対角近似オンライン更新（厳密な多変数Newton更新ではない）"
    config_hash = write_yaml_json(config, OUTPUT_DIR / "config_p0.yaml")

    save_model_outputs(prepared, result, config_hash, OUTPUT_DIR)
    metrics = race_metrics(prepared, result)
    metrics.to_csv(OUTPUT_DIR / "race_metrics.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    repetitions = int(config["bootstrap"]["repetitions"])
    seed = int(config["bootstrap"]["seed"])
    comparisons = main_comparisons(metrics, repetitions, seed)
    comparisons.to_csv(OUTPUT_DIR / "main_comparisons.csv", index=False, float_format="%.12g")
    layers = layer_comparisons(metrics, repetitions, seed)
    layers.to_csv(OUTPUT_DIR / "layer_comparisons.csv", index=False, float_format="%.12g")
    secondary = secondary_diagnostics(prepared, result, tau_info["tau"], repetitions, seed)
    for name, value in secondary.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, float_format="%.12g")

    pooled = comparisons[
        comparisons["period"].eq("validation")
        & comparisons["pool"].eq("E")
        & comparisons["year"].eq("POOLED")
    ].iloc[0]
    validation_years = comparisons[
        comparisons["period"].eq("validation")
        & comparisons["pool"].eq("E")
        & ~comparisons["year"].eq("POOLED")
    ]
    negative_years = int(validation_years["delta_nll"].lt(0).sum())
    criterion_1 = bool(pooled["ci95_upper"] < 0)
    criterion_2 = negative_years >= 3
    judgment = pd.DataFrame([
        {"criterion": "検証EプールのΔNLL 95%CI上限 < 0", "value": pooled["ci95_upper"], "pass": criterion_1},
        {"criterion": "検証4年のうち3年以上でΔNLL < 0", "value": negative_years, "pass": criterion_2},
        {"criterion": "段階2判定", "value": "PASS" if criterion_1 and criterion_2 else "FAIL", "pass": criterion_1 and criterion_2},
    ])
    judgment.to_csv(OUTPUT_DIR / "judgment.csv", index=False)
    daily = result.daily.copy()
    daily["asof_date"] = pd.to_datetime(daily["asof_date"], unit="D", origin="unix")
    anchor_review = {
        "n_days": len(daily),
        "crossing_count": int(daily["threshold_crossing"].ne("").sum()),
        "max_abs_mu_bar_delta_1d": float(daily["mu_bar_delta_1d"].abs().max()),
        "p99_abs_mu_bar_delta_1d": float(daily["mu_bar_delta_1d"].abs().quantile(0.99)),
        "initial_zero_anchor_days": result.initial_zero_anchor_days,
        "fallback_anchor_days": result.fallback_anchor_days,
        "largest_daily_changes": daily.assign(abs_delta=daily["mu_bar_delta_1d"].abs()).nlargest(10, "abs_delta")[
            ["asof_date", "n_active_365d", "mu_bar_D", "mu_bar_delta_1d", "n_new_horses", "mu0_mean", "threshold_crossing"]
        ].to_dict("records"),
    }
    summary = {
        "stage": "2_P0_ONLY",
        "selected_config": config["p0"],
        "config_sha256": config_hash,
        "input_db_sha256": prepared.source_hash,
        "selection_nll_tuning_E": config["selection"]["eval_pl_nll"],
        "tau_selection": tau_info,
        "engine_seconds": engine_seconds,
        "input_diagnostics": prepared.diagnostics,
        "skip_mu_var_updates": result.skip_mu_var_updates,
        "validation_judgment": {
            "criterion_1_ci_upper_below_zero": criterion_1,
            "criterion_2_at_least_three_negative_years": criterion_2,
            "negative_years": negative_years,
            "overall_pass": criterion_1 and criterion_2,
            "validation_E_pooled": pooled.to_dict(),
        },
        "anchor_review": anchor_review,
        "new_horse_difference": secondary["new_horse_difference"],
        "limitations": [
            "対角近似オンライン更新であり、厳密な多変数Newton更新ではない",
            "K<5では重みを再正規化せず、少頭数等では1レースの証拠量が小さくなる",
            "非JRA除外後の条件付き相対順位であり、実際のレース全体の勝ち馬とは限らない",
            "result_available_atは存在せず、データ作成年月日は利用可能時刻の代替にしていない",
            "交流競走フラグがないため交流の識別・層別は行わない",
            "新馬戦以外の初登場には地方・障害・海外からの移籍馬が含まれ得る",
        ],
    }
    (OUTPUT_DIR / "stage2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    (OUTPUT_DIR / "input_manifest.json").write_text(json.dumps({
        "input_db_sha256": prepared.source_hash,
        "config_sha256": config_hash,
        "diagnostics": prepared.diagnostics,
        "result_event_count": len(prepared.races),
        "library_versions": library_versions(),
        "result_policy": config["result_policy"],
    }, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    shutil.copyfile("stage2_test_diagnostics.json", OUTPUT_DIR / "test_diagnostics.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
