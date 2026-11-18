from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

from rating_data import load_or_prepare
from stage2_report import f, markdown

OUT = Path(__file__).with_name("stage3_output")
DB = Path(__file__).with_name("stage2_input") / "BaoZ_2_snapshot.mdb"


def load_official_results(race_ids: list[int]) -> pd.DataFrame:
    connection_string = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={DB};ReadOnly=1;"
    sql = """
        SELECT s.[競走コード] AS race_id, r.[血統登録番号] AS horse_id,
               s.[頭数] AS official_field_size,
               r.[確定着順] AS official_finish,
               r.[単勝配当] AS win_payout,
               r.[複勝配当] AS place_payout
        FROM [レースT] AS s INNER JOIN [出走馬T] AS r
          ON s.[競走コード] = r.[競走コード]
        WHERE s.[主催者コード] = 1
          AND s.[トラック種別コード] IN (0, 1)
          AND s.[月日] >= #2021-01-01#
          AND s.[月日] < #2025-01-01#
    """
    with pyodbc.connect(connection_string, readonly=True, autocommit=True) as con:
        results = pd.read_sql(sql, con)
    results["race_id"] = results["race_id"].astype("int64")
    results["horse_id"] = results["horse_id"].astype("int64")
    results = results[results["race_id"].isin(race_ids)]
    return results


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = frame.groupby(group_columns, sort=True).agg(
        races=("race_id", "nunique"),
        starts=("horse_id", "size"),
        wins=("official_win", "sum"),
        win_payout_total_yen=("win_payout", "sum"),
        top3=("official_top3", "sum"),
        place_wins=("official_place_win", "sum"),
        place_payout_total_yen=("place_payout", "sum"),
    ).reset_index()
    result["win_rate_pct"] = 100 * result["wins"] / result["starts"]
    result["win_recovery_pct"] = result["win_payout_total_yen"] / result["starts"]
    result["top3_rate_pct"] = 100 * result["top3"] / result["starts"]
    result["place_hit_rate_pct"] = 100 * result["place_wins"] / result["starts"]
    result["place_recovery_pct"] = result["place_payout_total_yen"] / result["starts"]
    return result


def main() -> None:
    prepared = load_or_prepare(DB)
    validation_f = prepared.evaluation_masks()["validation_F"]
    valid_race_indices = set(np.flatnonzero(validation_f).tolist())
    snapshots = pd.read_csv(
        OUT / "p2b" / "snapshots.csv.gz",
        usecols=["race_id", "race_index", "race_date", "horse_id", "mu_pre", "m_pre"],
        parse_dates=["race_date"],
    )
    snapshots = snapshots[snapshots["race_index"].isin(valid_race_indices)].copy()
    if not np.allclose(snapshots["mu_pre"], snapshots["m_pre"]):
        raise AssertionError("最終configではmu_pre=m_preという前提が崩れています")
    snapshots["rating_rank"] = snapshots.groupby("race_id")["mu_pre"].rank(method="min", ascending=False).astype("int16")
    race_ids = sorted(snapshots["race_id"].unique().tolist())
    official = load_official_results(race_ids)
    if official.duplicated(["race_id", "horse_id"]).any():
        raise AssertionError("公式結果のrace_id・horse_idが重複しています")
    detail = snapshots.merge(official, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    if detail[["official_finish", "win_payout", "place_payout"]].isna().all(axis=1).any():
        raise AssertionError("公式結果を結合できていない出走があります")
    detail["official_finish"] = pd.to_numeric(detail["official_finish"], errors="coerce").fillna(0)
    detail["official_win"] = detail["official_finish"].eq(1)
    detail["official_top3"] = detail["official_finish"].between(1, 3)
    detail["official_place_win"] = detail["official_finish"].ge(1) & detail["official_finish"].le(np.where(detail["official_field_size"].le(7), 2, 3))
    missing_win = detail["official_win"] & detail["win_payout"].isna()
    missing_place = detail["official_place_win"] & detail["place_payout"].isna()
    if missing_win.any() or missing_place.any():
        raise AssertionError("勝ち馬券に対応する配当欠損があります")
    detail["win_payout"] = pd.to_numeric(detail["win_payout"], errors="coerce").fillna(0)
    detail["place_payout"] = pd.to_numeric(detail["place_payout"], errors="coerce").fillna(0)
    if (detail.loc[detail["official_win"], "win_payout"] <= 0).any():
        raise AssertionError("1着なのに単勝配当が0以下です")
    if (detail.loc[detail["official_place_win"], "place_payout"] <= 0).any():
        raise AssertionError("複勝対象着順なのに複勝配当が0以下です")
    detail["year"] = detail["race_date"].dt.year
    detail = detail[["race_id", "race_date", "year", "horse_id", "mu_pre", "rating_rank", "official_field_size", "official_finish", "official_win", "official_top3", "official_place_win", "win_payout", "place_payout"]]
    detail.to_csv(OUT / "p2b_rating_rank_performance_detail.csv.gz", index=False, compression="gzip", float_format="%.12g", date_format="%Y-%m-%d")

    pooled = summarize(detail, ["rating_rank"])
    pooled.to_csv(OUT / "p2b_rating_rank_performance.csv", index=False, float_format="%.12g")
    yearly = summarize(detail, ["year", "rating_rank"])
    yearly.to_csv(OUT / "p2b_rating_rank_performance_by_year.csv", index=False, float_format="%.12g")
    overall = summarize(detail.assign(rating_rank="全体"), ["rating_rank"])
    tie_mask = detail.groupby("race_id")["mu_pre"].transform(lambda values: values.duplicated(keep=False))
    metadata = {
        "model": "P2b",
        "rating": "発走前mu_preのレース内降順。最終configではm_preと同じ",
        "tie_rule": "method=minの競技順位。馬番・馬IDで同点を解消しない",
        "period": "2021-01-01/2024-12-31",
        "race_set": "F（尤度を計算できる全レース）",
        "outcome": "公式確定着順による1着・3着内。複勝的中は7頭以下2着、8頭以上3着",
        "payout": "出走馬T.単勝配当・複勝配当。各出走を100円購入した場合の払戻総額/投資総額",
        "starts": int(len(detail)),
        "races": int(detail["race_id"].nunique()),
        "races_with_rating_ties": int(detail.loc[tie_mask, "race_id"].nunique()),
        "rating_tied_starts": int(tie_mask.sum()),
        "small_field_third_without_place_payout": int((detail["official_field_size"].le(7) & detail["official_finish"].eq(3) & detail["place_payout"].le(0)).sum()),
        "overall": overall.iloc[0].to_dict(),
    }
    (OUT / "p2b_rating_rank_performance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# P2b レーティング順位別成績（2021～2024年）", ""]
    lines.append("発走前`mu_pre`をレース内で降順に順位付けした、F（尤度を計算できる全レース）の成績です。同点は`method=min`で同一順位とし、馬番・馬IDでは順序を付けていません。")
    lines.append("")
    lines.append("勝率は公式1着、3着内率は公式3着内の割合です。複勝的中率は、7頭以下のレースでは2着以内、8頭以上では3着以内とした馬券の的中率です。回収率は、各出走を100円ずつ購入した場合の払戻総額を投資総額で割った値で、全て出走行を分母としています。")
    lines.append("")
    rows = []
    for row in pooled.itertuples():
        rows.append([
            int(row.rating_rank), f(row.races), f(row.starts), f(row.wins), f(row.win_rate_pct, 2) + "%",
            f(row.win_recovery_pct, 2) + "%", f(row.top3), f(row.top3_rate_pct, 2) + "%", f(row.place_hit_rate_pct, 2) + "%", f(row.place_recovery_pct, 2) + "%",
        ])
    total = overall.iloc[0]
    rows.append(["全体", f(total["races"]), f(total["starts"]), f(total["wins"]), f(total["win_rate_pct"], 2) + "%", f(total["win_recovery_pct"], 2) + "%", f(total["top3"]), f(total["top3_rate_pct"], 2) + "%", f(total["place_hit_rate_pct"], 2) + "%", f(total["place_recovery_pct"], 2) + "%"])
    lines.append(markdown(["順位", "レース数", "件数", "1着数", "単勝率", "単勝回収率", "3着内数", "3着内率", "複勝的中率", "複勝回収率"], rows))
    lines.append("")
    lines.append(f"対象は{len(detail):,}出走・{detail['race_id'].nunique():,}レースです。レーティング同点がある出走は{int(tie_mask.sum()):,}件、同点があるレースは{detail.loc[tie_mask, 'race_id'].nunique():,}件です。7頭以下のレースで公式3着となり複勝対象外の出走は{metadata['small_field_third_without_place_payout']:,}件あります。回収率は払戻金を使用する事後評価で、レーティング学習や順位付けに配当は使っていません。")
    (OUT / "p2b_rating_rank_performance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
