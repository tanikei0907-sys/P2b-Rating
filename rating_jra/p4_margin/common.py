from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
P3 = BASE / "p3_grade"
for path in [BASE, P3]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from p4_margin.margins import multiplier_arrays
from p4_margin.p4_engine import P4Engine


def run_configuration(prepared, pedigree, margins, model, config, record):
    p0, rule, payload = model
    stages, scales = multiplier_arrays(margins, prepared, config)
    engine = P4Engine(
        prepared, p0, rule, pedigree, payload["p2"]["standardization"],
        config["method"], stages, scales, record,
    )
    return engine.run(), engine
