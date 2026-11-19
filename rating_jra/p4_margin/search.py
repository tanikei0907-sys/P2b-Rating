from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
P3 = BASE / "p3_grade"
for path in [BASE, P3]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from p4_margin.common import run_configuration
from p3_grade.prepare_inputs import load_cached_inputs
from p3_grade.common import frozen_model, subset_until
from rating_core import canonical_hash, file_sha256

OUT = Path(__file__).resolve().parent
SEARCH_LOG = OUT / "tuning_search.jsonl"
GAMMAS = [0, 0.25, 0.5, 1.0, 2.0, 4.0]
CAPS = [1.25, 1.5, 2.0, 3.0]


def key_for(config: dict) -> str:
    return canonical_hash({key: value for key, value in config.items() if key != "grid_stage"})


def coarse_grid() -> list[dict]:
    configs = []
    for method in ["A", "B"]:
        for normalize in [False, True]:
            for gamma in GAMMAS:
                for cap in CAPS:
                    config = {"method": method, "normalize": normalize, "gamma": gamma, "m_max": cap, "grid_stage": "coarse"}
                    if method == "B":
                        config["m_min"] = 0.5
                    configs.append(config)
    return configs


def load_results() -> list[dict]:
    if not SEARCH_LOG.exists():
        return []
    return [json.loads(line) for line in SEARCH_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def config_row(config: dict) -> dict:
    return {
        "config_key": key_for(config), "method": config["method"], "normalize": config["normalize"],
        "gamma": config["gamma"], "m_max": config["m_max"], "m_min": config.get("m_min"),
        "grid_stage": config["grid_stage"],
        "parameters_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
    }


def best_of(results: list[dict], method: str | None = None, normalize: bool | None = None) -> dict:
    subset = [
        row for row in results if row["status"] == "PASS"
        and (method is None or row["method"] == method)
        and (normalize is None or row["normalize"] == normalize)
    ]
    if not subset:
        raise ValueError("探索結果がありません")
    return min(subset, key=lambda row: (row["nll"], row["config_key"]))


def refine(config: dict) -> list[dict]:
    def neighbors(value: float, grid: list[float], count: int, upper: float) -> list[float]:
        index = grid.index(value)
        differences = []
        if index > 0:
            differences.append(value - grid[index - 1])
        if index + 1 < len(grid):
            differences.append(grid[index + 1] - value)
        step = min(differences) if differences else 0.25
        return np.linspace(max(0, value - step), min(upper, value + step), count).tolist()

    configs = []
    for gamma in neighbors(float(config["gamma"]), GAMMAS, 5, 8.0):
        for cap in neighbors(float(config["m_max"]), CAPS, 3, 4.0):
            item = {
                "method": config["method"], "normalize": config["normalize"],
                "gamma": float(gamma), "m_max": float(cap), "grid_stage": "fine",
            }
            if config["method"] == "B":
                item["m_min"] = 0.5
            configs.append(item)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    prepared_full, pedigree, _, _ = load_cached_inputs()
    prepared = subset_until(prepared_full, "2020-12-31")
    tuning_ids = {race.race_id for race in prepared.races}
    margins = pd.read_csv(OUT / "race_margins.csv")
    margins = margins[margins["race_id"].isin(tuning_ids)].copy()
    if margins["race_id"].nunique() != len(prepared.races):
        raise ValueError("探索期間のマージンが不足しています")
    model = frozen_model()
    if file_sha256(BASE / "stage3_output/config_p2b.yaml") != "3577cbc7a6a2c3e58bfdeda6666b9aa9d7369866f4c6fa49902dac6b8779dbe9":
        raise ValueError("P2b configハッシュが不一致です")
    results = load_results()

    def run_one(config: dict):
        key = key_for(config)
        existing = next((row for row in reversed(results) if row["config_key"] == key), None)
        if existing is not None:
            return existing
        control = next((row for row in results if row["gamma"] == 0 and row["status"] == "PASS"), None)
        if config["gamma"] == 0 and control is not None:
            row = {
                **config_row(config), "status": "PASS", "nll": control["nll"], "nll_sum": control["nll_sum"],
                "n_races": control["n_races"], "elapsed_seconds": 0.0, "reused_unit_control": True,
            }
            with SEARCH_LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)
            return row
        started = time.perf_counter()
        result, _ = run_configuration(prepared, pedigree, margins, model, config, record=False)
        elapsed = time.perf_counter() - started
        if result.nll_count != 29936 or not np.isfinite(result.nll_sum[0]):
            raise ValueError(f"探索NLLが不正です: {config}")
        row = {
            **config_row(config), "status": "PASS", "nll": float(result.nll_sum[0] / result.nll_count),
            "nll_sum": float(result.nll_sum[0]), "n_races": int(result.nll_count),
            "elapsed_seconds": elapsed, "reused_unit_control": False,
        }
        with SEARCH_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        results.append(row)
        print(json.dumps({"candidate": len(results), "parameters": config, "nll": row["nll"], "elapsed_seconds": elapsed}, ensure_ascii=False), flush=True)
        return row

    control_config = {"method": "A", "normalize": False, "gamma": 0.0, "m_max": 1.25, "grid_stage": "coarse"}
    control = run_one(control_config)
    p3_control_path = P3 / "selected_config.json"
    if p3_control_path.exists():
        expected_control = json.loads(p3_control_path.read_text(encoding="utf-8"))["control_nll"]
        if not np.isclose(control["nll"], expected_control, atol=1e-12, rtol=0):
            raise AssertionError("P4の倍率1対照がP3対照と一致しません")
    if args.benchmark:
        print(json.dumps({"benchmark": control, "tuning_races": len(prepared.races)}, ensure_ascii=False, indent=2))
        return

    for config in coarse_grid():
        run_one(config)
    for method in ["A", "B"]:
        for normalize in [False, True]:
            best = json.loads(best_of(results, method, normalize)["parameters_json"])
            for config in refine(best):
                run_one(config)
    selected = best_of(results)
    selected_config = json.loads(selected["parameters_json"])
    baseline = pd.read_csv(BASE / "stage3_output/p2b/race_metrics.csv.gz")
    baseline_period = baseline[baseline["year"].between(2012, 2020) & baseline["F"]]
    payload = {
        "model": "P4", "status": "FROZEN_BEFORE_P4_VALIDATION",
        "method": selected["method"], "normalize": selected["normalize"], "gamma": selected["gamma"],
        "m_max": selected["m_max"], "m_min": selected["m_min"],
        "tuning_period": "2012-01-01/2020-12-31",
        "tuning_objective": "全尤度ありレースの標準PLレースNLL平均（F相当）",
        "tuning_nll": selected["nll"], "tuning_races": selected["n_races"], "control_nll": control["nll"],
        "p2b_tuning_nll_same_period_F": float(baseline_period["p0_nll"].mean()),
        "candidate_count": len({row["config_key"] for row in results}),
        "best_by_method": {method: best_of(results, method) for method in ["A", "B"]},
        "best_by_group": {
            f"{method}_{'normalized' if normalize else 'raw'}": best_of(results, method, normalize)
            for method in ["A", "B"] for normalize in [False, True]
        },
        "margin_formulas": {
            "A": "stage_weight = W_share[k] * min(m_max, 1 + gamma*f(delta_k)); missing/not_applicable=1",
            "B": "g=0.75*f(delta_1)+0.25*f(delta_2) if both available, otherwise f(delta_1); race_scale=clip(1+gamma*g,m_min,m_max)",
            "f": "delta/(distance/1000) if normalize else delta (seconds)",
        },
        "p2b_config_sha256": file_sha256(BASE / "stage3_output/config_p2b.yaml"),
        "input_db_sha256": prepared.source_hash,
        "time_audit_sha256": file_sha256(OUT / "time_audit.json"),
        "prior_policy": "P1・P2b係数と標準化定数は固定。初期3走残差だけをP4のas-of軌道から逐次生成",
        "evaluation_nll": "標準W_share PL-NLL。マージン倍率は更新のみ",
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_period_untouched_during_selection": True,
    }
    path = OUT / "selected_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config_hash = file_sha256(path)
    path.with_suffix(".json.sha256").write_text(config_hash, encoding="ascii")
    pd.DataFrame(results).sort_values(["nll", "config_key"]).to_csv(OUT / "tuning_search.csv", index=False)
    (OUT / "search_summary.json").write_text(json.dumps({"best_by_method": payload["best_by_method"], "best_by_group": payload["best_by_group"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "FROZEN_BEFORE_P4_VALIDATION", "selected": payload, "config_sha256": config_hash}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
