from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from rating_core import file_sha256
from rating_evaluation import layer_comparisons, main_comparisons, race_metrics, secondary_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("stage3_output/config_p2b.yaml"))
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--prepared-pickle", type=Path, required=True)
    parser.add_argument("--result-pickle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if file_sha256(args.config) != args.config_hash:
        raise ValueError("configハッシュが一致しません")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config["status"].startswith("FROZEN"):
        raise ValueError("凍結configが必要です")
    if config["model"] == "P0":
        p0_config = config
    elif config["model"] == "P1":
        p0_config = config["base_p0"]
    else:
        p0_config = config["base_p1"]["base_p0"]
    if p0_config["p0"].get("tau") is None:
        raise ValueError("P0温度を含む凍結configが必要です")
    with args.prepared_pickle.open("rb") as handle:
        prepared = pickle.load(handle)
    with args.result_pickle.open("rb") as handle:
        result = pickle.load(handle)
    if result.snapshots is None or result.nll is None or result.s0_nll is None:
        raise ValueError("評価に必要なsnapshot・NLLがありません")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = race_metrics(prepared, result)
    metrics.to_csv(args.output_dir / "race_metrics.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    repetitions = int(p0_config["bootstrap"]["repetitions"])
    seed = int(p0_config["bootstrap"]["seed"])
    main_comparisons(metrics, repetitions, seed).to_csv(args.output_dir / "main_comparisons.csv", index=False, float_format="%.12g")
    layer_comparisons(metrics, repetitions, seed).to_csv(args.output_dir / "layer_comparisons.csv", index=False, float_format="%.12g")
    secondary = secondary_diagnostics(prepared, result, float(p0_config["p0"]["tau"]), repetitions, seed)
    for name, value in secondary.items():
        if hasattr(value, "to_csv"):
            value.to_csv(args.output_dir / f"{name}.csv", index=False, float_format="%.12g")
    (args.output_dir / "eval_manifest.json").write_text(json.dumps({
        "config_sha256": args.config_hash,
        "input_db_sha256": prepared.source_hash,
        "races": len(prepared.races),
        "snapshots": len(result.snapshots),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
