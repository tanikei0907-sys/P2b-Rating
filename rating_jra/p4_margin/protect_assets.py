from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from rating_core import file_sha256

OUT = Path(__file__).resolve().parent
MANIFEST = OUT / "protected_assets_before.json"


def protected_files() -> list[Path]:
    files = []
    for relative in ["stage2_output", "stage3_output/p2b", "p2b_forward"]:
        files.extend(path for path in (BASE / relative).rglob("*") if path.is_file())
    files.append(BASE / "stage3_output/config_p2b.yaml")
    return sorted(set(files))


def snapshot() -> dict:
    return {
        path.relative_to(BASE).as_posix(): {"size": path.stat().st_size, "sha256": file_sha256(path)}
        for path in protected_files()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = snapshot()
    if args.check:
        before = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
        added = sorted(set(current) - set(before))
        removed = sorted(set(before) - set(current))
        changed = sorted(key for key in set(before) & set(current) if before[key] != current[key])
        report = {"status": "PASS" if not added and not removed and not changed else "FAIL", "protected_files": len(before), "added": added, "removed": removed, "changed": changed}
        (OUT / "protected_assets_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(1)
    else:
        payload = {"status": "FROZEN", "protected_files": len(current), "files": current}
        MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "FROZEN", "protected_files": len(current), "manifest": str(MANIFEST)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
