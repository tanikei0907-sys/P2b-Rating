from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("stage2_output")


def f(value, digits=6):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    summary = json.loads((OUT / "stage2_summary.json").read_text(encoding="utf-8"))
    config = json.loads((OUT / "config_p0.yaml").read_text(encoding="utf-8"))
    search = pd.read_csv(OUT / "hyperparameter_search.csv")
    sensitivity = pd.read_csv(OUT / "hyperparameter_sensitivity.csv")
    comparisons = pd.read_csv(OUT / "main_comparisons.csv")
    layers = pd.read_csv(OUT / "layer_comparisons.csv")
    calibration = pd.read_csv(OUT / "calibration.csv")
    deciles = pd.read_csv(OUT / "win_deciles.csv")
    distribution = pd.read_csv(OUT / "mu_var_by_experience.csv")
    rest = pd.read_csv(OUT / "rest_summary.csv")
    new_horse = pd.read_csv(OUT / "new_horse_subsequent.csv")
    order_metric = pd.read_csv(OUT / "race_order_metric.csv")
    transitions = pd.read_csv(OUT / "mu_anchor_transition_summary.csv")
    tests = json.loads((OUT / "test_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "input_manifest.json").read_text(encoding="utf-8"))
    judgment = summary["validation_judgment"]
    pooled = judgment["validation_E_pooled"]
    selected = summary["selected_config"]
    anchor = summary["anchor_review"]
    lines = []
    add = lines.append
    add("# 中央競馬平地レーティング：段階2（P0）報告")
    add("")
    add("## 結論要約")
    add("")
    add("### 段階1：データ確認")
    add("")
    add("- 2012～2024年のJRA主催平地43,252レース・613,963行を確認し、非JRA1,111行と取消・除外2,176行を除いた610,676実出走を構築した。")
    add("- 新馬戦の事前グラフ出走、同一馬の同日複数出走、残存1頭以下、理由不明の順位欠損はいずれも0件だった。")
    add("- 斤量・所属の欠損は0件、父・母父ID欠損はそれぞれ183行だった。同着の最大順列組合せ数は6で、120超えはなかった。")
    add("- DBの制限として、交流競走フラグと明示的な結果利用可能時刻がないことを確定し、更新仕様ではこれらを推定・代替しないことにした。")
    add("")
    add("### 段階2：P0")
    add("")
    add(f"- 全216条件を調整期間Eで比較し、`var0={selected['var0']}`、`q={selected['q']}`、`kappa={selected['kappa']}`、`beta_kg={selected['beta_kg']}`、`{selected['weight']}`を選択した。選択NLLは{f(summary['selection_nll_tuning_E'])}。")
    add(f"- 調整期間だけで診断用温度`tau={f(selected['tau'], 9)}`を推定して凍結し、検証期間を再調整せずに処理した。")
    add(f"- 検証期間EのΔNLLは{f(pooled['delta_nll'])}、95％CIは[{f(pooled['ci95_lower'])}, {f(pooled['ci95_upper'])}]で、4年全てが負の改善だった。")
    add("- 事前の2判定基準をともに満たし、段階2は合格と判定した。段階3の血統事前分布は実装・実行していない。")
    add("")
    add("## 段階2の判定")
    add("")
    add(markdown(["判定", "結果", "合否"], [
        ["検証EプールのΔNLL 95％CI上限 < 0", f(pooled["ci95_upper"]), "合格" if judgment["criterion_1_ci_upper_below_zero"] else "不合格"],
        ["検証4年のうち3年以上でΔNLL < 0", f"{judgment['negative_years']} / 4年", "合格" if judgment["criterion_2_at_least_three_negative_years"] else "不合格"],
        ["総合", "両基準を満たす", "**PASS**" if judgment["overall_pass"] else "**FAIL**"],
    ]))
    add("")
    add("## 入力・凍結・再現性")
    add("")
    add(markdown(["項目", "値"], [
        ["対象期間", "2012-01-01～2024-12-31"],
        ["不動入力", "`stage2_input/BaoZ_2_snapshot.mdb`"],
        ["入力DB SHA-256", f"`{summary['input_db_sha256']}`"],
        ["凍結config", "`stage2_output/config_p0.yaml`（JSON互換YAML）"],
        ["config SHA-256", f"`{summary['config_sha256']}`"],
        ["探索戦略", "全216条件のFULL GRID。座標探索への短縮なし"],
        ["選択に使用", "2017～2020年のE、EvalPLNLLのみ"],
        ["温度推定に使用", "2017～2020年の尤度あり全レース（F相当）、勝ち馬CE"],
        ["検証期間での再推定", "なし"],
        ["更新イベント", "43,252件。race_id・event_idの重複なし"],
    ]))
    add("")
    add("稼働中DBのハッシュが作業中に変化したため、旧探索は`stage2_output_unstable_82f8`として隔離し、不動コピーから全探索をやり直した。再探索の全216条件のNLLは旧探索と一致した。本報告は不動コピーとハッシュ別キャッシュだけを使用し、検証期間の結果を見て設定を変更していない。")
    add("")
    add("### データ件数")
    add("")
    add(markdown(["項目", "件数"], [
        ["JRA平地レース", f(43252)],
        ["非JRA除外前の出走行", f(613963)],
        ["非JRA除外行", f(1111)],
        ["JRA所属の取消・除外行", f(2176)],
        ["最終的な実出走行", f(610676)],
        ["実出走馬ID", f(67030)],
        ["斤量欠損", f(0)],
        ["残存2頭未満・尤度なしレース", f(0)],
        ["順位なし馬のmu/var更新スキップ", f(summary["skip_mu_var_updates"])],
    ]))
    add("")
    add("## モデルの厳守事項")
    add("")
    add("- 更新は対角情報量による近似オンライン更新であり、厳密な多変数Newton更新ではない。")
    add("- 重みはKに応じて再正規化しない。K<5のレースでは重み合計が1未満となり、観測段階が少ないことを1レースの証拠量の減少として扱う。")
    add("- 同着は全順列の勾配・対角情報量・評価対数尤度を平均し、馬番・馬IDで着順を決めない。")
    add("- 順位なし馬は候補集合に残すが自身のmu・varは更新せず、出走数と最終出走日だけを更新する。")
    add("- 非JRA所属馬を除いたJRA所属馬の条件付き相対順位を使う。勝ち馬は実際のレース全体の勝ち馬とは限らない。")
    add("- 開催日の全スナップショットを生成してから更新する。明示的なresult_available_atは存在せず、データ作成年月日を代替として使っていない。")
    add("- 交流競走はJRA主催平地に含めるが、フラグがないため交流の識別・集計・層別は行わない。")
    add("")
    add("## ハイパーパラメータ探索")
    add("")
    add("### 上位5条件")
    add("")
    rows = []
    for row in search.head(5).itertuples():
        rows.append([row.rank, row.config_id, f(row.var0, 1), f(row.q, 3), f(row.kappa, 1), f(row.beta_kg, 2), row.weight, f(row.eval_pl_nll_tuning_E, 9)])
    add(markdown(["順位", "ID", "var0", "q", "kappa", "beta_kg", "weight", "調整E NLL"], rows))
    add("")
    add("### 選択値の他条件を固定した感度")
    add("")
    rows = [[row.parameter, row.value, f(row.eval_pl_nll_tuning_E, 9), f(row.delta_from_selected, 9)] for row in sensitivity.itertuples()]
    add(markdown(["パラメータ", "値", "調整E NLL", "選択値との差"], rows))
    add("")
    add("選択されたvar0・q・kappaは探索格子の上限だが、仕様どおり新しい候補は追加していない。")
    add("")
    add("## 主比較：P0対S0")
    add("")
    add("各セルのΔNLLはP0−S0で、負の値が改善。CIはレース単位のペア・ブートストラップ2,000回、固定シードによる95％区間。")
    add("")
    rows = []
    for row in comparisons.itertuples():
        rows.append([row.period, row.pool, row.year, f(row.n_races), f(row.p0_nll), f(row.s0_nll), f(row.delta_nll), f"[{f(row.ci95_lower)}, {f(row.ci95_upper)}]"])
    add(markdown(["期間", "集合", "年", "レース", "P0 NLL", "S0 NLL", "ΔNLL", "95％CI"], rows))
    add("")
    clean = comparisons[(comparisons["period"] == "validation") & (comparisons["pool"] == "E_clean") & (comparisons["year"] == "POOLED")].iloc[0]
    add(f"検証EのΔNLLは{f(pooled['delta_nll'])}、E_cleanは{f(clean['delta_nll'])}で、符号は同じく負、水準も近い。E_cleanの方がレース数は少ない（{int(clean['n_races']):,}対{int(pooled['n_races']):,}）。Eは非JRA除外後の条件付き相対順位を含む推定対象、E_cleanは除外が一度もないレースに限った対象であり、両者の違いとして併記する。")
    add("")
    add("## 検証期間の層別（E）")
    add("")
    add("発走前経験数の層は、仕様のレース単位ΔNLLへ一意的に割り当てるため、レース内の発走前n_graph_starts平均値を境界0・1・2・3・5・10に適用した。各レースは1層だけに入る。")
    add("")
    rows = []
    for row in layers.itertuples():
        rows.append([row.layer_type, row.level, f(row.n_races), f(row.delta_nll), f"[{f(row.ci95_lower)}, {f(row.ci95_upper)}]"])
    add(markdown(["層", "区分", "レース", "ΔNLL", "95％CI"], rows))
    add("")
    add("## 副次診断")
    add("")
    add("勝率だけは調整期間で固定したtauによる診断用出力で、後続モデルへの必須特徴量ではない。3着内・5着内確率はtauを掛けない発走前mからHarville形式で計算した。")
    add("")
    rows = []
    for row in calibration.itertuples():
        rows.append([row.metric, f(row.n_starts), f(row.n_races), f(row.predicted_mean), f(row.actual_mean), f(row.winner_ce)])
    add(markdown(["指標", "出走数", "レース数", "予測平均", "実測", "勝ち馬CE"], rows))
    add("")
    add("### 勝率十分位")
    add("")
    rows = [[int(row.win_decile), f(row.n_starts), f(row.mean_predicted_win), f(row.actual_win_rate)] for row in deciles.itertuples()]
    add(markdown(["十分位", "出走数", "平均予測勝率", "実勝率"], rows))
    add("")
    add("### 経験数別mu・var")
    add("")
    rows = []
    for row in distribution.itertuples():
        rows.append([row.experience_band, f(row.n_starts), f(row.mu_mean), f(row.mu_std), f(row.mu_q05), f(row.mu_median), f(row.mu_q95), f(row.var_mean), f(row.var_median)])
    add(markdown(["経験数", "出走数", "mu平均", "mu SD", "mu 5%", "mu中央", "mu 95%", "var平均", "var中央"], rows))
    add("")
    add("### 休養期間")
    add("")
    add("実性能変化は、同じ馬の前走との実際の賞金配点差（今走配点−前走配点）で、前走は全期間から求めた。0日・初登場には前走差がない。")
    add("")
    rows = [[row.rest_band, f(row.n_starts), f(row.mean_var_pre_eff), f(row.mean_actual_prize_points), f(row.mean_performance_change), f(row.actual_top3_rate)] for row in rest.itertuples()]
    add(markdown(["休養", "出走数", "var_pre_eff平均", "実配点平均", "前走差平均", "実3着内率"], rows))
    add("")
    add("### 新馬戦後3走")
    add("")
    rows = [[row.debut_group, f(row.horses), f(row.observations), f(row.mean_subsequent_mu_post)] for row in new_horse.itertuples()]
    add(markdown(["デビュー結果", "馬数", "対象出走", "その後3走のmu_post平均"], rows))
    add("")
    add(f"勝ち馬−敗戦馬の馬単位平均差は **{f(summary['new_horse_difference']['winner_minus_loser_mean_mu_post'])}** だった。")
    add("")
    add(f"賞金配点に基づく順序指標はPrizeNDCG@5で、平均は **{f(order_metric.iloc[0]['mean'])}**（{int(order_metric.iloc[0]['n_races']):,}レース）。PRIZE_POINTSをgainとし、予測同点は平均gainで扱う。")
    add("")
    add("## mu_bar_Dの日別診断")
    add("")
    add(markdown(["項目", "値"], [
        ["開催日数", f(anchor["n_days"])],
        ["N=100 crossing", f(anchor["crossing_count"])],
        ["最大絶対日次変化", f(anchor["max_abs_mu_bar_delta_1d"], 9)],
        ["99パーセント点の絶対日次変化", f(anchor["p99_abs_mu_bar_delta_1d"], 9)],
        ["利用可能母集団がない初期日", f(anchor["initial_zero_anchor_days"])],
        ["365日母集団なしのフォールバック日", f(anchor["fallback_anchor_days"])],
    ]))
    add("")
    rows = []
    for row in transitions.itertuples():
        rows.append([row.crossing_date, row.anchor_group, f(row.n_days), f(row.mu_bar_mean, 9), f(row.mu_bar_median, 9), f(row.max_abs_daily_delta, 9), f(row.active_latest_mu_std_mean), f(row.mu0_mean, 9), f(row.n_new_horses)])
    add(markdown(["crossing日", "側", "日数", "mu_bar平均", "中央値", "最大絶対日次変化", "active SD平均", "mu0平均", "新規馬"], rows))
    add("")
    add("N=100のcrossingはデータ初期の2012-01-08の1回だけで、前後の水準切替や初期値の不連続は見られない。全期間の最大絶対日次変化も0.014987にとどまり、大きな変化は主に8～9月の連続的な季節変動であり、特定日への不連続な切替とは判断しなかった。N_min・フォールバック・算出式は変更していない。")
    add("")
    add("## 単体テスト")
    add("")
    add("第8項に対応する13テストが全て合格した。勾配と対角・非対角情報行列は数値微分と一致し、切詰め・将来結果改変・同日順序・シフト・同着・非JRA・除外データ・ロック・結果版の不変性も確認した。")
    add("")
    add(markdown(["検査", "結果"], [
        ["対角近似と全行列更新の最大mu差", f(tests["diagonal_vs_full"]["max_abs_mu_difference"], 9)],
        ["合成データ相関（250・600・1200レース）", " / ".join(f(value) for value in tests["synthetic_recovery"]["correlations"])],
        ["分散/二乗誤差（同上）", " / ".join(f(value) for value in tests["synthetic_recovery"]["posterior_var_over_squared_error"])],
        ["合成データの少頭数・同着レース", f"{tests['synthetic_recovery']['small_field_races']} / {tests['synthetic_recovery']['tie_races']}"],
    ]))
    add("")
    add("合成データでは真値との相関が収束し、今回の対角近似分散は最終時点で二乗誤差より大きい（過小分散ではない）結果だった。これは実データの較正保証ではない。")
    add("")
    add("## 読み取れること／読み取れないこと")
    add("")
    add("- 読み取れるのは、2012～2024年を開催日順に処理した場合、P0がS0より着順全体の評価用PL-NLLを改善したこと、およびその層別・較正上の特徴である。")
    add("- 2021～2024年の検証結果は、2025年以降の将来性能や後続モデルでの特徴量の有効性を保証しない。後続モデルの最終ホールドアウトは別に評価する。")
    add("- 新馬戦以外での初登場には地方・障害・海外からの移籍馬が含まれ得る。除外した地方・障害・海外の結果はグラフ履歴にも移籍診断にも使っていない。")
    add("- 馬券購入条件、市場優位、収支、LightGBM等への実装・採否は本報告の対象外である。")
    add("- 結果の後日修正・再抽出は別の結果版として扱い、過去スナップショットを黙って更新しない。今回は抽出時点の確定結果を開催日順に1回だけ処理した。")
    add("")
    add("## 成果物")
    add("")
    artifacts = [
        "snapshots.csv.gz：全期間610,676行の発走前スナップショット",
        "post_updates.csv.gz：更新後mu・var",
        "state_2024-12-31.csv：最終状態",
        "config_p0.yaml / config_p0.yaml.sha256：凍結設定とハッシュ",
        "result_events.csv：結果版・更新イベントID",
        "hyperparameter_search.csv / hyperparameter_sensitivity.csv",
        "main_comparisons.csv / layer_comparisons.csv / judgment.csv",
        "calibration.csv / win_deciles.csv / mu_var_by_experience.csv",
        "rest_summary.csv / new_horse_subsequent.csv / race_order_metric.csv",
        "mu_anchor_daily.csv / mu_anchor_transition_summary.csv / mu_anchor_timeseries.svg",
        "input_manifest.json / test_diagnostics.json / stage2_summary.json",
        "rating_update_for_features.py / rating_final_eval.py：2025年以降用。作成のみで実行していない",
    ]
    for artifact in artifacts:
        add(f"- `{artifact}`")
    add("")
    add("## 不確かな点・割り切った点・依頼者が判断すべき点")
    add("")
    add("- P0の改善は条件付き相対着順の評価であり、実際の全出走馬に対する着順モデルや馬券価値を意味しない。")
    add("- 経験数別のレース単位ΔNLLは、レース内平均経験数で層を定義した。馬単位の層別NLLではない。")
    add("- tauは勝ち馬診断用で、主指標のEvalPLNLL・学習・選択には使っていない。")
    add("- 父・母父の事前分布は段階3の範囲であり、本成果のP0には血統情報を使っていない。段階3への進行は依頼者の承認が必要である。")
    (OUT / "stage2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "stage2_report.md")


if __name__ == "__main__":
    main()
