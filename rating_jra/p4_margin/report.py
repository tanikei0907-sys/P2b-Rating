from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from rating_core import file_sha256

OUT = Path(__file__).resolve().parent


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ])


def nll(value) -> str:
    return f"{float(value):.9f}"


def integer(value) -> str:
    return f"{int(value):,}"


def number(value, digits=3) -> str:
    return f"{float(value):.{digits}f}"


def config_label(row: dict) -> str:
    config = json.loads(row["parameters_json"])
    return f"γ={config['gamma']:.8g}, cap={config['m_max']:.8g}"


def main() -> None:
    selected = json.loads((OUT / "selected_config.json").read_text(encoding="utf-8"))
    judgment = json.loads((OUT / "judgment.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "evaluation_protocol.json").read_text(encoding="utf-8"))
    verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "time_audit.json").read_text(encoding="utf-8"))
    model_summary = json.loads((OUT / "model_summary.json").read_text(encoding="utf-8"))
    stages = pd.read_csv(OUT / "margin_stage_summary.csv")
    cumulative = pd.read_csv(OUT / "cumulative_margin_summary.csv")
    distances = pd.read_csv(OUT / "margin_scale_by_distance.csv")
    tracks = pd.read_csv(OUT / "margin_scale_by_track.csv")
    overall = pd.read_csv(OUT / "validation_overall.csv")
    yearly = pd.read_csv(OUT / "validation_by_year.csv")
    by_grade = pd.read_csv(OUT / "validation_by_grade.csv")
    abnormal = pd.read_csv(OUT / "time_abnormal_audit.csv")
    lines = []
    add = lines.append
    add("# P4：走破タイム差を用いたオンライン更新重みの検証報告")
    add("")
    if judgment["status"] == "P4_CANDIDATE_CONFIRMED":
        add(f"**結論：P4を新採用候補として確定します。検証Fの標準PL-NLL差は{nll(judgment['primary']['delta_P4_minus_P2b'])}、95%CI上限は{nll(judgment['primary']['ci95_upper'])}で、有意な改善を確認しました。**")
    elif judgment['primary']['delta_P4_minus_P2b'] > 0 and judgment['primary']['ci95_lower'] > 0:
        add(f"**結論：P2bを引き続き最終採用モデルとして維持します。検証Fの標準PL-NLL差は{nll(judgment['primary']['delta_P4_minus_P2b'])}、95%CIは[{nll(judgment['primary']['ci95_lower'])}, {nll(judgment['primary']['ci95_upper'])}]で、P4は有意に悪化しました。**")
    else:
        add(f"**結論：P2bを引き続き最終採用モデルとして維持します。検証Fの標準PL-NLL差は{nll(judgment['primary']['delta_P4_minus_P2b'])}、95%CIは[{nll(judgment['primary']['ci95_lower'])}, {nll(judgment['primary']['ci95_upper'])}]で、有意な改善は確認できませんでした。**")
    add("")

    add("## 1. 資産保護とモデル設計")
    add("")
    add(f"- P2b config SHA-256：`{judgment['p2b_config_sha256']}`")
    add(f"- P4選択config SHA-256：`{judgment['p4_config_sha256']}`")
    add("- P0・P2b・P2b継続更新出力と凍結configは変更していません。DBは一時コピー経由の読み取り専用接続です。")
    add("- 探索は2012～2020年、検証は2021～2024年です。探索時には検証期間のレースを入力へ含めていません。")
    add("- P1デビュー形態、P2b血統係数・κ=20・標準化定数、P0のκ・q=0.01・分散、W_shareを維持しています。")
    add("- 初期3走残差はP4の発走前スコアから計算し、開催日終了後に血統集計へ反映しています。")
    add("- **評価NLLは常に標準W_share PL-NLLです。マージン倍率は勾配・対角情報量の更新だけに使用しています。**")
    add("")
    add("方式Aは段階kの`W_share[k]`へ`min(m_max, 1+γf(Δt_k))`を乗じます。方式Bは`g=0.75f(Δt_1)+0.25f(Δt_2)`（Δt_2がない場合はf(Δt_1)）を用い、レース倍率を`clip(1+γg,m_min,m_max)`とします。距離正規化時は`f(Δt)=Δt/(距離/1000)`、それ以外は秒です。")
    add("")

    add("## 2. 走破タイム・着差データ監査")
    add("")
    add("`出走馬T.タイム`と`出走馬T.着差`は整数の0.1秒単位です。全43,252レース・613,963行から、既存と同じ除外後610,676出走を再構成しました。ranked出走608,710行の走破タイム欠損は**0件**です。")
    add("")
    add(table(["項目", "件数・結果"], [
        ["ranked出走 / タイム欠損", f"{integer(audit['ranked_rows'])} / {integer(audit['missing_time_ranked'])}"],
        ["順位なし出走 / タイム欠損", f"{integer(audit['unranked_rows'])} / {integer(audit['missing_time_unranked'])}"],
        ["DB着差が欠損のranked出走", f"{integer(audit['stored_margin_missing_ranked'])}（{audit['stored_margin_missing_ranked_rate']:.2%}）"],
        ["保存着差との照合対象 / 0.05秒超の不一致", f"{integer(audit['stored_margin_comparison_rows'])} / {integer(audit['stored_margin_mismatch_over_0_05s'])}"],
        ["保存着差との最大絶対誤差", f"{audit['stored_margin_max_abs_error_seconds']:.3g}秒"],
        ["速度30～80km/h外", integer(audit['speed_outliers_below_30_or_above_80_kmh'])],
        ["同着内でタイムが異なる行", integer(audit['tied_group_different_time_rows'])],
        ["負の累積相対着差", integer(audit['negative_cumulative_rows'])],
        ["負の段階着差（0へclip）", integer(audit['negative_stage_margins_clipped'])],
        ["モデル着差の欠損（倍率1）", integer(audit['missing_stage_margins_neutral'])],
    ]))
    add("")
    add("異常区分別の内訳は次のとおりです。コード4はタイム欠損、コード5と7はタイムが記録されています。")
    add("")
    add(table(["異常コード", "行数", "タイム欠損", "タイムあり"], [[int(row.abnormal_code), integer(row.rows), integer(row.missing_time), integer(row.present_time)] for row in abnormal.itertuples()]))
    add("")
    add("負の累積差は2012-10-28の1行（相対10着、-0.2秒）でした。負の段階差6件は絶対値を取らず0へclipしています。詳細は`negative_cumulative_audit.csv`と`negative_stage_audit.csv`に保存しています。")
    add("")

    add("### 段階別・累積着差の分布")
    add("")
    add(table(["段階", "適用可能", "利用可能", "欠損", "負値clip", "0秒", "中央値", "P90", "P95", "最大"], [[int(row.stage), integer(row.applicable), integer(row.available), integer(row.missing), integer(row.negative_clipped), integer(row.zero_seconds), number(row.delta_median, 2), number(row.delta_p90, 2), number(row.delta_p95, 2), number(row.delta_max, 2)] for row in stages.itertuples()]))
    add("")
    add(table(["相対着順", "馬数", "累積差中央値", "P90", "P95", "最大"], [[int(row.relative_finish), integer(row.count), number(row.median, 2), number(row.p90, 2), number(row.p95, 2), number(row.max, 2)] for row in cumulative.itertuples()]))
    add("")

    add("### 距離・馬場状態によるスケール")
    add("")
    selected_distances = [1000, 1200, 1600, 1800, 2000, 2400, 3000, 3600]
    add(table(["距離", "レース数", "勝ちタイム中央値", "1-2着差中央値", "P90", "正規化中央値", "正規化P90"], [[integer(row.distance), integer(row.races), number(row.winner_time_median, 1), number(row.margin_median, 2), number(row.margin_p90, 2), number(row.normalized_median, 3), number(row.normalized_p90, 3)] for row in distances[distances["distance"].isin(selected_distances)].itertuples()]))
    add("")
    add(table(["馬場", "レース数", "生着差平均", "中央値", "P90", "正規化中央値", "正規化P90"], [[row.track_condition, integer(row.races), number(row.margin_mean, 3), number(row.margin_median, 2), number(row.margin_p90, 2), number(row.normalized_median, 3), number(row.normalized_p90, 3)] for row in tracks.itertuples()]))
    add("")
    add("生着差のP90は良0.5秒、稍重・重0.6秒、不良0.7秒でした。距離正規化は長距離の着差を小さくし、例えば1000mと2000mでは中央値の尺度が異なります。どちらの尺度がNLLに有効かは探索で選定しています。全距離・距離帯別の結果もCSVに保存しています。")
    add("")

    add("## 3. 探索結果と採用パラメータ")
    add("")
    group_rows = []
    for key, row in selected["best_by_group"].items():
        group_rows.append([key, config_label(row), nll(row["nll"]), nll(row["nll"] - selected["control_nll"])])
    add(table(["方式・尺度", "最良γ・cap", "探索NLL", "倍率1対照との差"], group_rows))
    add("")
    add(f"全{selected['candidate_count']}候補から、**方式{selected['method']}、距離正規化={selected['normalize']}、γ={selected['gamma']:.8g}、m_max={selected['m_max']:.8g}、m_min={selected['m_min']}**を凍結しました。探索NLLは{nll(selected['tuning_nll'])}、倍率1対照は{nll(selected['control_nll'])}、同期間の保存済みP2bは{nll(selected['p2b_tuning_nll_same_period_F'])}です。")
    add("")

    add("## 4. 2021～2024年の標準PL-NLL")
    add("")
    add("ΔはP4-P2bで負の値が改善です。開催日単位20,000回のペア・クラスターブートストラップで95%CIを求めました。")
    add("")
    add(table(["プール", "レース数", "P2b", "P4", "ΔNLL", "95%CI", "片側p"], [[row.pool, integer(row.n_races), nll(row.P2b_nll), nll(row.P4_nll), nll(row.delta_P4_minus_P2b), f"[{nll(row.ci95_lower)}, {nll(row.ci95_upper)}]", f"{row.p_one_sided_improvement:.5g}"] for row in overall.itertuples()]))
    add("")
    add("### 年別")
    add("")
    add(table(["プール", "年", "レース数", "P2b", "P4", "ΔNLL", "95%CI"], [[row.pool, int(row.year), integer(row.n_races), nll(row.P2b_nll), nll(row.P4_nll), nll(row.delta_P4_minus_P2b), f"[{nll(row.ci95_lower)}, {nll(row.ci95_upper)}]"] for row in yearly.itertuples()]))
    add("")
    add("### 10クラス別（F）")
    add("")
    add(table(["クラス", "レース数", "出走数", "P2b", "P4", "ΔNLL", "95%CI"], [[row.category, integer(row.n_races), integer(row.starts), nll(row.P2b_nll), nll(row.P4_nll), nll(row.delta_P4_minus_P2b), f"[{nll(row.ci95_lower)}, {nll(row.ci95_upper)}]"] for row in by_grade.itertuples()]))
    add("")
    maiden = by_grade[by_grade["category"].eq("未勝利")].iloc[0]
    new_race = by_grade[by_grade["category"].eq("新馬")].iloc[0]
    one_win = by_grade[by_grade["category"].eq("1勝クラス")].iloc[0]
    add(f"改善は全クラス一様ではありません。未勝利は{nll(maiden['delta_P4_minus_P2b'])}と大きく改善した一方、新馬は{nll(new_race['delta_P4_minus_P2b'])}と悪化し、1勝クラスも{nll(one_win['delta_P4_minus_P2b'])}でした。全体での有意改善と層別の逆転を区別し、運用時もクラス別の監視が必要です。")
    add("")

    add("## 5. 最終判定と独立検証")
    add("")
    add(f"判定：**{judgment['status']}**。{judgment['reason']}。")
    add("")
    add(f"評価プロトコルSHA-256：`{file_sha256(OUT / 'evaluation_protocol.json')}`。設定・出力は`p4_margin/`に限定し、P2b運用へは反映していません。")
    add("")
    add(table(["独立検査", "結果"], [[name, "PASS" if item["pass"] else "FAIL"] for name, item in verification["checks"].items()]))
    add("")
    add(f"更新再計算の最大誤差はmu={verification['metrics']['max_update_mu_error']:.3g}、var={verification['metrics']['max_update_var_error']:.3g}、標準PL-NLL={verification['metrics']['max_nll_error']:.3g}です。P4専用テスト{verification['metrics']['tests_passed']}件と、保護資産の不変性も確認しています。")
    add("")

    add("## 6. 出力ファイル")
    add("")
    add("- `time_audit.json` / `relative_times.csv.gz` / `race_margins.csv`：タイム監査と相対着差")
    add("- `margin_scale_by_distance.csv` / `margin_scale_by_condition.csv`：距離・馬場別の尺度")
    add("- `tuning_search.csv` / `selected_config.json`：探索と凍結設定")
    add("- `margin_multipliers.csv.gz` / `snapshots.csv.gz` / `state_2024-12-31.csv`：P4更新結果")
    add("- `validation_overall.csv` / `validation_by_year.csv` / `validation_by_grade.csv`：検証NLL")
    add("- `verification.json` / `protected_assets_verification.json` / `final_model.json`：監査と最終判定")
    add("")
    add("NLLは着順予測の対数損失であり、回収率の改善を直接意味しません。P4を候補化する場合も、この判定は指定された標準PL-NLLのホールドアウト比較に基づきます。")
    (OUT / "P4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "REPORT_WRITTEN", "path": str(OUT / "P4_report.md"), "judgment": judgment["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
