from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import P0Config, file_sha256
from rating_data import load_or_prepare
from rating_initialization import InitialRule
from rating_optimize import early_race_mask, optimize_batch
from rating_pedigree import build_or_load_base, build_or_load_features
from stage2_search import write_yaml_json
from stage3_p1 import default_json

OUT = Path(__file__).with_name("stage3_output")
L2_BETA = 0.01
KAPPAS = [20, 50, 100, 200]


def serializable_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in {"history", "cache_nll"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["P2a", "P2b"], required=True)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    model = args.model
    p1_path = OUT / "config_p1.yaml"
    p1_hash = file_sha256(p1_path)
    if p1_hash != p1_path.with_suffix(p1_path.suffix + ".sha256").read_text(encoding="ascii").strip():
        raise ValueError("P1 configハッシュが一致しません")
    p1_config = json.loads(p1_path.read_text(encoding="utf-8"))
    p0 = P0Config(**p1_config["base_p0"]["p0"])
    prepared = load_or_prepare(Path("stage2_input/BaoZ_2_snapshot.mdb"))
    if prepared.source_hash != p1_config["input_db_sha256"]:
        raise ValueError("入力DBがP1と一致しません")
    base = build_or_load_base(Path("stage2_output/snapshots.csv.gz"), Path("stage2_input/BaoZ_2_snapshot.mdb"))
    masks = prepared.evaluation_masks()
    selection_mask = early_race_mask(prepared, masks["tuning_E"])
    b_values = np.array([
        p1_config["p1"]["b"]["新馬戦"],
        p1_config["p1"]["b"]["未勝利戦"],
        p1_config["p1"]["b"]["その他のクラス"],
    ])
    var_values = np.array([
        p1_config["p1"]["var0_d"]["新馬戦"],
        p1_config["p1"]["var0_d"]["未勝利戦"],
        p1_config["p1"]["var0_d"]["その他のクラス"],
    ])
    aggregate = args.aggregate
    if aggregate and model == "P2a":
        dimension = 1
    elif aggregate:
        dimension = 2
    else:
        dimension = 2 if model == "P2a" else 4

    def beta_from_x(x):
        if model == "P2a" and aggregate:
            return np.array([x[0], x[0], 0.0, 0.0])
        if model == "P2a":
            return np.array([x[0], x[1], 0.0, 0.0])
        if aggregate:
            return np.array([x[0], x[0], x[1], x[1]])
        return np.asarray(x, dtype=float)

    results = []
    for kappa in KAPPAS:
        features = build_or_load_features(base, kappa)

        def builder(x):
            return InitialRule(
                b=b_values[None, :].copy(),
                var0_d=var_values[None, :].copy(),
                beta=beta_from_x(x)[None, :],
            )

        result_path = OUT / f"{model.lower()}_kappa_{kappa}_{'aggregate' if aggregate else 'full'}_result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["x"] = np.asarray(result["x"], dtype=float)
            result["x0"] = np.asarray(result["x0"], dtype=float)
            print(f"{model} κ={kappa}: cached", flush=True)
        else:
            print(f"{model} κ={kappa}: start", flush=True)
            result = optimize_batch(
                prepared=prepared,
                base_config=p0,
                evaluation_mask=selection_mask,
                x0=np.zeros(dimension),
                bounds=[(-1.0, 1.0)] * dimension,
                penalty_mask=np.ones(dimension),
                l2=L2_BETA,
                rule_builder=builder,
                feature_provider=features.provider,
                model_name=model,
                maxiter=45,
            )
            result["kappa"] = kappa
            result["standardization"] = features.constants
            result["aggregate"] = aggregate
            result_path.write_text(
                json.dumps({key: value for key, value in result.items() if key != "cache_nll"}, ensure_ascii=False, default=default_json),
                encoding="utf-8",
            )
            print(f"{model} κ={kappa}: done objective={result['objective']}", flush=True)
        results.append(result)
        if not result["success"]:
            raise RuntimeError(f"{model} κ={kappa}の最適化が失敗しました: {result['message']}。集約版を別途実行してください")
    best = min(results, key=lambda item: item["objective"])
    x = best["x"]
    beta = beta_from_x(x)
    payload = {
        "model": model,
        "status": "FROZEN_BEFORE_STAGE3_VALIDATION",
        "base_p1": p1_config,
        "base_p1_config_sha256": p1_hash,
        "input_db_sha256": prepared.source_hash,
        "p2": {
            "kappa_s": int(best["kappa"]),
            "beta": {
                "beta_sw": float(beta[0]),
                "beta_st": float(beta[1]),
                "beta_bw": float(beta[2]),
                "beta_bt": float(beta[3]),
            },
            "aggregate_win_top3_coefficients": aggregate,
            "standardization": best["standardization"],
        },
        "selection": {
            "metric": "調整期間E・レース内平均n_graph_starts<=3のEvalPLNLL + L2",
            "period": "2017-01-01/2020-12-31",
            "n_races": best["n_races"],
            "l2_beta": L2_BETA,
            "raw_nll": best["raw_nll"],
            "penalty": best["penalty"],
            "objective": best["objective"],
            "success": best["success"],
            "message": best["message"],
            "kappa_candidates": KAPPAS,
            "aggregate_reason": "最適化不安定時の集約を明示的に実行" if aggregate else "なし",
        },
    }
    config_hash = write_yaml_json(payload, OUT / f"config_{model.lower()}.yaml")
    serializable = {
        "model": model,
        "config_sha256": config_hash,
        "selected_kappa": best["kappa"],
        "selected_beta": payload["p2"]["beta"],
        "aggregate": aggregate,
        "results": [serializable_result(result) for result in results],
    }
    (OUT / f"{model.lower()}_optimization.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2, default=default_json), encoding="utf-8"
    )
    pd.DataFrame([{
        "kappa": result["kappa"],
        "raw_nll": result["raw_nll"],
        "penalty": result["penalty"],
        "objective": result["objective"],
        "success": result["success"],
        "message": result["message"],
        "nit": result["nit"],
        "nfev": result["nfev"],
        "elapsed_seconds": result["elapsed_seconds"],
        "aggregate": aggregate,
        **{f"x_{index}": float(value) for index, value in enumerate(result["x"])},
    } for result in results]).to_csv(OUT / f"{model.lower()}_kappa_summary.csv", index=False)
    pd.DataFrame([
        {"iteration": index, **{f"x_{i}": float(row["x"][i]) for i in range(dimension)},
         "raw_nll": row["raw_nll"], "penalty": row["penalty"], "objective": row["objective"],
         "gradient_max_abs": row["gradient_max_abs"]}
        for index, row in enumerate(best["history"])
    ]).to_csv(OUT / f"{model.lower()}_optimization_history.csv", index=False)
    print(json.dumps({
        "model": model,
        "config_sha256": config_hash,
        "selected_kappa": best["kappa"],
        "beta": payload["p2"]["beta"],
        "raw_nll": best["raw_nll"],
        "objective": best["objective"],
        "aggregate": aggregate,
        "kappa_summaries": [{"kappa": result["kappa"], "objective": result["objective"], "raw_nll": result["raw_nll"]} for result in results],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
