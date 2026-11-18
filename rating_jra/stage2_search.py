from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import pandas as pd

from rating_core import P0Config, canonical_hash, file_sha256
from rating_data import load_or_prepare
from rating_engine import run_engine

OUTPUT_DIR = Path(__file__).with_name("stage2_output")


def all_configs() -> list[P0Config]:
    return [
        P0Config(var0, q, kappa, beta, weight)
        for var0, q, kappa, beta, weight in itertools.product(
            [0.5, 1.0, 2.0],
            [0.001, 0.004, 0.01],
            [0.5, 0.7, 1.0],
            [0.0, 0.03],
            ["W_share", "W_pts", "W_win", "W_win_heavy"],
        )
    ]


def write_yaml_json(value: dict, path: Path) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    prepared = load_or_prepare(Path(__file__).with_name("stage2_input") / "BaoZ_2_snapshot.mdb")
    masks = prepared.evaluation_masks()
    configs = all_configs()
    assert len(configs) == 216
    started = time.time()
    result = run_engine(prepared, configs, evaluation_mask=masks["tuning_E"])
    elapsed = time.time() - started
    nll = result.nll_sum / result.nll_count
    rows = []
    for config_id, (config, value) in enumerate(zip(configs, nll)):
        rows.append({
            "config_id": config_id,
            **config.to_dict(),
            "eval_pl_nll_tuning_E": value,
            "n_races": result.nll_count,
        })
    search = pd.DataFrame(rows).sort_values("eval_pl_nll_tuning_E").reset_index(drop=True)
    search.insert(0, "rank", np_rank(search))
    search.to_csv(OUTPUT_DIR / "hyperparameter_search.csv", index=False)
    best_row = search.iloc[0]
    best = P0Config(
        var0=float(best_row["var0"]),
        q=float(best_row["q"]),
        kappa=float(best_row["kappa"]),
        beta_kg=float(best_row["beta_kg"]),
        weight=str(best_row["weight"]),
    )
    sensitivity = []
    for parameter, values in {
        "var0": [0.5, 1.0, 2.0],
        "q": [0.001, 0.004, 0.01],
        "kappa": [0.5, 0.7, 1.0],
        "beta_kg": [0.0, 0.03],
        "weight": ["W_share", "W_pts", "W_win", "W_win_heavy"],
    }.items():
        for value in values:
            criteria = True
            for other in ["var0", "q", "kappa", "beta_kg", "weight"]:
                expected = value if other == parameter else getattr(best, other)
                criteria = criteria & search[other].eq(expected)
            matched = search.loc[criteria]
            if len(matched) != 1:
                raise AssertionError((parameter, value, len(matched)))
            row = matched.iloc[0]
            sensitivity.append({
                "parameter": parameter,
                "value": value,
                "config_id": int(row["config_id"]),
                "eval_pl_nll_tuning_E": float(row["eval_pl_nll_tuning_E"]),
                "delta_from_selected": float(row["eval_pl_nll_tuning_E"] - best_row["eval_pl_nll_tuning_E"]),
            })
    pd.DataFrame(sensitivity).to_csv(OUTPUT_DIR / "hyperparameter_sensitivity.csv", index=False)
    config_payload = {
        "model": "P0",
        "status": "SELECTED_BEFORE_TAU",
        "selection": {
            "metric": "EvalPLNLL",
            "period": "2017-01-01/2020-12-31",
            "race_set": "E",
            "strategy": "FULL_GRID_216",
            "config_id": int(best_row["config_id"]),
            "eval_pl_nll": float(best_row["eval_pl_nll_tuning_E"]),
        },
        "p0": best.to_dict(),
        "periods": {
            "warmup": "2012-01-01/2016-12-31",
            "tuning": "2017-01-01/2020-12-31",
            "validation": "2021-01-01/2024-12-31",
            "locked": "2025-01-01以降",
        },
        "input_db_sha256": prepared.source_hash,
        "prize_points": [100, 40, 25, 15, 10, 0],
        "evaluation_weight": "W_share",
        "weight_renormalized_by_K": False,
        "bootstrap": {"repetitions": 2000, "seed": 20241117},
        "anchor_diagnostic": {"N_min": 100, "window_days": 30},
        "result_policy": "開催日順に抽出時点の確定結果を1回だけ処理。データ作成年月日は使用しない。",
    }
    config_payload["selection_payload_hash"] = canonical_hash(config_payload)
    config_hash = write_yaml_json(config_payload, OUTPUT_DIR / "config_p0_pre_tau.yaml")
    metadata = {
        "source_hash": prepared.source_hash,
        "search_strategy": "FULL_GRID_216",
        "configurations_executed": len(configs),
        "tuning_E_races": result.nll_count,
        "full_period_online_update": "2012-01-01/2024-12-31",
        "validation_metrics_used_for_selection": False,
        "elapsed_seconds": elapsed,
        "selected_config": best.to_dict(),
        "selected_pre_tau_config_sha256": config_hash,
        "input_diagnostics": prepared.diagnostics,
        "library_versions": library_versions(),
    }
    (OUTPUT_DIR / "search_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "elapsed_seconds": elapsed,
        "configurations": len(configs),
        "tuning_E_races": result.nll_count,
        "selected": best.to_dict(),
        "selected_nll": float(best_row["eval_pl_nll_tuning_E"]),
        "top5": search.head(5).to_dict("records"),
        "config_sha256": config_hash,
    }, ensure_ascii=False, indent=2))


def np_rank(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(range(1, len(frame) + 1), index=frame.index, dtype="int64")


def library_versions() -> dict:
    import numba
    import numpy
    import pyodbc
    import scipy
    import sys
    return {
        "python": sys.version,
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "pyodbc": pyodbc.version,
    }


if __name__ == "__main__":
    main()
