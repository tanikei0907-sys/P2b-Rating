from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rating_core import P0Config, file_sha256
from rating_data import load_or_prepare
from rating_initialization import InitialRule
from rating_optimize import early_race_mask, optimize_batch
from stage2_search import write_yaml_json

OUT = Path(__file__).with_name("stage3_output")
L2_B = 0.01


def default_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    p0_path = Path("stage2_output/config_p0.yaml")
    p0_hash = file_sha256(p0_path)
    if p0_hash != p0_path.with_suffix(p0_path.suffix + ".sha256").read_text(encoding="ascii").strip():
        raise ValueError("P0 configハッシュが一致しません")
    p0_config = json.loads(p0_path.read_text(encoding="utf-8"))
    p0 = P0Config(**p0_config["p0"])
    prepared = load_or_prepare(Path("stage2_input/BaoZ_2_snapshot.mdb"))
    if prepared.source_hash != p0_config["input_db_sha256"]:
        raise ValueError("入力DBがP0と一致しません")
    masks = prepared.evaluation_masks()
    selection_mask = early_race_mask(prepared, masks["tuning_E"])

    def builder(x):
        return InitialRule(
            b=np.asarray(x[:3], dtype=float)[None, :],
            var0_d=np.exp(np.asarray(x[3:6], dtype=float))[None, :],
            beta=np.zeros((1, 4)),
        )

    result = optimize_batch(
        prepared=prepared,
        base_config=p0,
        evaluation_mask=selection_mask,
        x0=np.array([0.0, 0.0, 0.0, np.log(2.0), np.log(2.0), np.log(2.0)]),
        bounds=[(-1.0, 1.0)] * 3 + [(np.log(0.25), np.log(4.0))] * 3,
        penalty_mask=np.array([1, 1, 1, 0, 0, 0]),
        l2=L2_B,
        rule_builder=builder,
        model_name="P1",
        maxiter=45,
    )
    if not result["success"] or not np.isfinite(result["objective"]):
        raise RuntimeError(f"P1最適化が失敗しました: {result['message']}")
    x = result["x"]
    payload = {
        "model": "P1",
        "status": "FROZEN_BEFORE_STAGE3_VALIDATION",
        "base_p0": p0_config,
        "base_p0_config_sha256": p0_hash,
        "input_db_sha256": prepared.source_hash,
        "p1": {
            "b": {
                "新馬戦": float(x[0]),
                "未勝利戦": float(x[1]),
                "その他のクラス": float(x[2]),
            },
            "var0_d": {
                "新馬戦": float(np.exp(x[3])),
                "未勝利戦": float(np.exp(x[4])),
                "その他のクラス": float(np.exp(x[5])),
            },
        },
        "selection": {
            "metric": "調整期間E・レース内平均n_graph_starts<=3のEvalPLNLL + L2",
            "period": "2017-01-01/2020-12-31",
            "n_races": result["n_races"],
            "l2_b": L2_B,
            "raw_nll": result["raw_nll"],
            "penalty": result["penalty"],
            "objective": result["objective"],
            "bounds": result["bounds"],
            "x0": result["x0"].tolist(),
            "success": result["success"],
            "message": result["message"],
        },
    }
    config_hash = write_yaml_json(payload, OUT / "config_p1.yaml")
    serializable = {key: value for key, value in result.items() if key not in {"history", "cache_nll"}}
    serializable["config_sha256"] = config_hash
    (OUT / "p1_optimization.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=default_json), encoding="utf-8")
    pd.DataFrame([
        {"iteration": index, **{f"x_{i}": float(row["x"][i]) for i in range(6)},
         "raw_nll": row["raw_nll"], "penalty": row["penalty"], "objective": row["objective"],
         "gradient_max_abs": row["gradient_max_abs"]}
        for index, row in enumerate(result["history"])
    ]).to_csv(OUT / "p1_optimization_history.csv", index=False)
    print(json.dumps({
        "model": "P1",
        "config_sha256": config_hash,
        "b": payload["p1"]["b"],
        "var0_d": payload["p1"]["var0_d"],
        "raw_nll": result["raw_nll"],
        "objective": result["objective"],
        "n_races": result["n_races"],
        "success": result["success"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
