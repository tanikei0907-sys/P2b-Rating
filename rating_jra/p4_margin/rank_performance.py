from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from forward_ops import copy_database, load_config
from rating_core import file_sha256
from stage2_report import f, markdown
from viewer_data import cleanup_access_lock_files

OUT = Path(__file__).resolve().parent / "rank_performance"


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = frame.groupby(group_columns, sort=True).agg(
        races=("race_id", "nunique"), starts=("horse_id", "size"),
        wins=("official_win", "sum"), win_payout_total_yen=("win_payout", "sum"),
        top3=("official_top3", "sum"), place_wins=("official_place_win", "sum"),
        place_payout_total_yen=("place_payout", "sum"),
    ).reset_index()
    result["win_rate_pct"] = 100 * result["wins"] / result["starts"]
    result["win_recovery_pct"] = result["win_payout_total_yen"] / result["starts"]
    result["top3_rate_pct"] = 100 * result["top3"] / result["starts"]
    result["place_hit_rate_pct"] = 100 * result["place_wins"] / result["starts"]
    result["place_recovery_pct"] = result["place_payout_total_yen"] / result["starts"]
    return result


def main() -> None:
    OUT.mkdir(exist_ok=True)
    config_hash = file_sha256(BASE / "p4_margin/selected_config.json")
    if config_hash != "436f08ffb840a60ef33354fc28fa9434d0193a0460c73ba8c8f29d8038a26ba1":
        raise ValueError("P4 configハッシュが不一致です")
    metrics = pd.read_csv(BASE / "p4_margin/race_metrics.csv.gz", usecols=["race_id", "validation", "F"])
    valid_races = set(metrics.loc[metrics["validation"] & metrics["F"], "race_id"])
    snapshots = pd.read_csv(
        BASE / "p4_margin/snapshots.csv.gz",
        usecols=["race_id", "race_date", "horse_id", "horse_number", "mu_pre", "m_pre", "elo_pre", "config_sha256"],
        parse_dates=["race_date"],
    )
    snapshots = snapshots[snapshots["race_id"].isin(valid_races)].copy()
    if not snapshots["config_sha256"].eq(config_hash).all():
        raise ValueError("P4スナップショットのconfigハッシュが不一致です")
    if not np.allclose(snapshots["mu_pre"], snapshots["m_pre"]):
        raise AssertionError("P4ではmu_pre=m_preという前提が崩れています")
    snapshots["rating_rank"] = snapshots.groupby("race_id")["mu_pre"].rank(method="min", ascending=False).astype("int16")
    config = load_config(BASE / "forward_config.yaml")
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        db = copy_database(BASE / "stage2_input/BaoZ_2_snapshot.mdb", Path(temporary), config["database"].get("linked_tables", {}))
        connection = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db};ReadOnly=1;"
        try:
            with pyodbc.connect(connection, readonly=True, autocommit=True) as con:
                official = pd.read_sql(
                    """
                    SELECT s.[競走コード] AS race_id, r.[血統登録番号] AS horse_id,
                           s.[頭数] AS official_field_size,
                           r.[確定着順] AS official_finish,
                           r.[単勝配当] AS win_payout, r.[複勝配当] AS place_payout
                    FROM [レースT] AS s INNER JOIN [出走馬T] AS r ON s.[競走コード]=r.[競走コード]
                    WHERE s.[主催者コード]=1 AND s.[トラック種別コード] IN (0,1)
                      AND s.[月日] >= #2021-01-01# AND s.[月日] < #2025-01-01#
                      AND s.[データ区分]='7' AND r.[データ区分]='7'
                    """,
                    con,
                )
                con.close()
        finally:
            gc.collect()
            cleanup_access_lock_files(temporary)
    official["race_id"] = official["race_id"].astype("int64")
    official["horse_id"] = official["horse_id"].astype("int64")
    official = official[official["race_id"].isin(valid_races)]
    if official.duplicated(["race_id", "horse_id"]).any():
        raise AssertionError("公式結果のrace_id・horse_idが重複しています")
    detail = snapshots.merge(official, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    if detail[["official_finish", "win_payout", "place_payout"]].isna().all(axis=1).any():
        raise AssertionError("公式結果を結合できていない出走があります")
    detail["official_finish"] = pd.to_numeric(detail["official_finish"], errors="coerce").fillna(0)
    detail["official_win"] = detail["official_finish"].eq(1)
    detail["official_top3"] = detail["official_finish"].between(1, 3)
    detail["official_place_win"] = detail["official_finish"].ge(1) & detail["official_finish"].le(np.where(detail["official_field_size"].le(7), 2, 3))
    if (detail["official_win"] & detail["win_payout"].isna()).any() or (detail["official_place_win"] & detail["place_payout"].isna()).any():
        raise AssertionError("勝ち馬券に対応する配当欠損があります")
    detail["win_payout"] = pd.to_numeric(detail["win_payout"], errors="coerce").fillna(0)
    detail["place_payout"] = pd.to_numeric(detail["place_payout"], errors="coerce").fillna(0)
    if (detail.loc[detail["official_win"], "win_payout"] <= 0).any():
        raise AssertionError("1着なのに単勝配当が0以下です")
    if (detail.loc[detail["official_place_win"], "place_payout"] <= 0).any():
        raise AssertionError("複勝対象着順なのに複勝配当が0以下です")
    detail["year"] = detail["race_date"].dt.year
    detail = detail[[
        "race_id", "race_date", "year", "horse_id", "horse_number", "mu_pre", "elo_pre", "rating_rank",
        "official_field_size", "official_finish", "official_win", "official_top3", "official_place_win",
        "win_payout", "place_payout",
    ]]
    detail.to_csv(OUT / "p4_rating_rank_performance_detail.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")
    pooled = summarize(detail, ["rating_rank"])
    pooled.to_csv(OUT / "p4_rating_rank_performance.csv", index=False, float_format="%.12g")
    yearly = summarize(detail, ["year", "rating_rank"])
    yearly.to_csv(OUT / "p4_rating_rank_performance_by_year.csv", index=False, float_format="%.12g")
    overall = summarize(detail.assign(rating_rank="全体"), ["rating_rank"])
    tie_mask = detail.groupby("race_id")["mu_pre"].transform(lambda values: values.duplicated(keep=False))
    metadata = {
        "model": "P4", "config_sha256": config_hash,
        "rating": "発走前mu_preのレース内降順。P4ではm_preと同じ",
        "tie_rule": "method=minの競技順位。馬番・馬IDで同点を解消しない",
        "period": "2021-01-01/2024-12-31", "race_set": "F（尤度を計算できる全レース）",
        "outcome": "公式確定着順。複勝的中は7頭以下2着、8頭以上3着",
        "payout": "出走馬T.単勝配当・複勝配当。各出走を100円購入した払戻総額/投資総額",
        "starts": len(detail), "races": detail["race_id"].nunique(),
        "races_with_rating_ties": int(detail.loc[tie_mask, "race_id"].nunique()),
        "rating_tied_starts": int(tie_mask.sum()),
        "small_field_third_without_place_payout": int((detail["official_field_size"].le(7) & detail["official_finish"].eq(3) & detail["place_payout"].le(0)).sum()),
        "overall": overall.iloc[0].to_dict(),
        "db_access": "temporary_copy_read_only",
    }
    (OUT / "p4_rating_rank_performance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_detail_path = BASE / "stage3_output/p2b_rating_rank_performance_detail.csv.gz"
    if baseline_detail_path.exists():
        baseline = pd.read_csv(baseline_detail_path)
        comparison = detail.merge(
            baseline[["race_id", "horse_id", "official_finish", "official_field_size", "win_payout", "place_payout"]],
            on=["race_id", "horse_id"], how="left", suffixes=("", "_p2b"), validate="one_to_one",
        )
        for column in ["official_finish", "official_field_size", "win_payout", "place_payout"]:
            if not np.allclose(comparison[column], comparison[f"{column}_p2b"], atol=0, rtol=0):
                raise AssertionError(f"P2b分析と公式結果が不一致です: {column}")

    rows = []
    for row in pooled.itertuples():
        rows.append([
            int(row.rating_rank), f(row.races), f(row.starts), f(row.wins), f(row.win_rate_pct, 2) + "%",
            f(row.win_recovery_pct, 2) + "%", f(row.top3_rate_pct, 2) + "%",
            f(row.place_wins), f(row.place_hit_rate_pct, 2) + "%", f(row.place_recovery_pct, 2) + "%",
        ])
    total = overall.iloc[0]
    rows.append(["全体", f(total["races"]), f(total["starts"]), f(total["wins"]), f(total["win_rate_pct"], 2) + "%", f(total["win_recovery_pct"], 2) + "%", f(total["top3_rate_pct"], 2) + "%", f(total["place_wins"]), f(total["place_hit_rate_pct"], 2) + "%", f(total["place_recovery_pct"], 2) + "%"])
    lines = [
        "# P4 レーティング順位別成績（2021～2024年）", "",
        "発走前`mu_pre`をレース内で降順にした、F（尤度を計算できる全レース）の成績です。同点は`method=min`で同一順位とし、馬番・馬IDでは同点を解消していません。", "",
        "単勝率は公式1着、複勝率は7頭以下のレースで2着以内、8頭以上で3着以内の的中率です。3着内率は小頭数の複勝対象外も含む参考指標です。回収率は各出走を100円ずつ購入した場合の払戻総額÷投資総額で、全出走を分母としています。", "",
        markdown(["順位", "レース数", "件数", "1着数", "単勝率", "単勝回収率", "3着内率", "複勝的中数", "複勝率", "複勝回収率"], rows), "",
        f"対象は{len(detail):,}出走・{detail['race_id'].nunique():,}レースです。同点出走は{int(tie_mask.sum()):,}件、同点レースは{detail.loc[tie_mask, 'race_id'].nunique():,}件です。7頭以下の公式3着で複勝対象外は{metadata['small_field_third_without_place_payout']:,}件です。配当は事後評価にのみ使用し、レーティング学習・順位付けには使用していません。",
    ]
    (OUT / "P4_rating_rank_performance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
