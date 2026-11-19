from __future__ import annotations

import argparse
import json
import sys

from p7_forward_ops import EXPECTED_P7_HASH, load_config, predict_snapshot, update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["update", "predict_snapshot"])
    parser.add_argument("--config", default="p7_forward/forward_config.yaml")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--config-hash", default=None)
    args = parser.parse_args()
    try:
        if args.config_hash is not None and args.config_hash != EXPECTED_P7_HASH:
            raise ValueError("指定されたP7 configハッシュが一致しません")
        config = load_config(args.config)
        result = update(config, args.end_date) if args.mode == "update" else predict_snapshot(config, args.date)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "ERROR", "mode": args.mode, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
