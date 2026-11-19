from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
TRACK_LABELS = {1: "良", 2: "稍重", 3: "重", 4: "不良"}


def main() -> None:
    relative = pd.read_csv(
        OUT / "relative_times.csv.gz",
        usecols=["race_id", "race_date", "horse_id", "official_finish", "relative_finish", "time_seconds", "stored_margin_seconds", "cumulative_relative_seconds"],
    )
    margins = pd.read_csv(OUT / "race_margins.csv")
    ranked = relative[relative["relative_finish"].gt(0)]
    negative = ranked[ranked["cumulative_relative_seconds"].lt(0)]
    negative.to_csv(OUT / "negative_cumulative_audit.csv", index=False)
    anomalies = []
    for index in range(1, 6):
        frame = margins[margins[f"status_{index}"].eq(3)][["race_id", "race_date", "distance", "track_condition_code"]].copy()
        frame["stage"] = index
        anomalies.append(frame)
    pd.concat(anomalies, ignore_index=True).to_csv(OUT / "negative_stage_audit.csv", index=False)
    margins["track_condition"] = margins["track_condition_code"].map(TRACK_LABELS).fillna("UNKNOWN")
    margins["normalized_stage_1"] = margins["stage_1"] / (margins["distance"] / 1000.0)
    rows = []
    for track, group in margins.groupby("track_condition"):
        available = group[group["status_1"].eq(1)]
        rows.append({
            "track_condition": track, "races": len(group), "available_margins": len(available),
            "margin_mean": available["stage_1"].mean(), "margin_median": available["stage_1"].median(),
            "margin_p90": available["stage_1"].quantile(0.9),
            "normalized_median": available["normalized_stage_1"].median(),
            "normalized_p90": available["normalized_stage_1"].quantile(0.9),
        })
    pd.DataFrame(rows).to_csv(OUT / "margin_scale_by_track.csv", index=False)
    path = OUT / "time_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit.update({
        "stored_margin_missing_ranked": int(ranked["stored_margin_seconds"].isna().sum()),
        "stored_margin_missing_ranked_rate": float(ranked["stored_margin_seconds"].isna().mean()),
        "missing_time_unranked": int(relative.loc[relative["relative_finish"].eq(0), "time_seconds"].isna().sum()),
        "present_time_unranked": int(relative.loc[relative["relative_finish"].eq(0), "time_seconds"].notna().sum()),
        "negative_cumulative_audit_file": "negative_cumulative_audit.csv",
        "negative_stage_audit_file": "negative_stage_audit.csv",
    })
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stored_margin_missing_ranked": audit["stored_margin_missing_ranked"], "track_summary": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
