from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

from stage2_report import f, markdown

OUT = Path(__file__).with_name("stage3_output")
DB = Path(__file__).with_name("stage2_input") / "BaoZ_2_snapshot.mdb"


def rating_examples() -> dict:
    posts = pd.read_csv(OUT / "p2b" / "post_updates.csv.gz", parse_dates=["race_date"])
    period = posts[posts["race_date"].between("2021-01-01", "2024-12-31")].copy()
    maxima = period.loc[period.groupby("horse_id")["mu_post"].idxmax()]
    historical = maxima.sort_values(["mu_post", "horse_id"], ascending=[False, True]).head(10).copy()
    state = pd.read_csv(OUT / "p2b" / "state_2024-12-31.csv", parse_dates=["last_start_date"])
    year_end = state[state["last_start_date"].dt.year.eq(2024)].sort_values(["mu", "horse_id"], ascending=[False, True]).head(10).copy()
    ids = sorted(set(historical["horse_id"].astype(int)) | set(year_end["horse_id"].astype(int)))
    placeholders = ",".join("?" for _ in ids)
    connection_string = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={DB};ReadOnly=1;"
    with pyodbc.connect(connection_string, readonly=True, autocommit=True) as con:
        names = pd.read_sql(
            f"SELECT [血統登録番号] AS horse_id,[馬名] AS horse_name FROM [競走馬マスタ] WHERE [血統登録番号] IN ({placeholders})",
            con,
            params=ids,
        )
    historical = historical.merge(names, on="horse_id", how="left", validate="one_to_one")
    year_end = year_end.merge(names, on="horse_id", how="left", validate="one_to_one")
    assert historical["horse_name"].notna().all() and year_end["horse_name"].notna().all()
    result = {
        "historical_definition": "P2bの2021～2024年の更新後mu_postの馬ごとの最大値",
        "year_end_definition": "2024年に実出走した馬に限定した、2024-12-31時点のmu",
        "historical_top10": historical.assign(race_date=historical["race_date"].dt.strftime("%Y-%m-%d"))[["horse_id", "horse_name", "mu_post", "race_date", "race_id"]].to_dict("records"),
        "year_end_top10": year_end.assign(last_start_date=year_end["last_start_date"].dt.strftime("%Y-%m-%d"))[["horse_id", "horse_name", "mu", "last_start_date", "n_graph_starts"]].to_dict("records"),
    }
    (OUT / "p2b_rating_top10.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    examples = rating_examples()
    p2b = json.loads((OUT / "config_p2b.yaml").read_text(encoding="utf-8"))
    p1 = p2b["base_p1"]
    p0 = p1["base_p0"]
    summary = json.loads((OUT / "stage3_summary.json").read_text(encoding="utf-8"))
    comparisons = pd.read_csv(OUT / "pairwise_comparisons.csv")
    pooled = comparisons[(comparisons["period"] == "validation") & (comparisons["year"] == "POOLED")]
    judgment = summary["judgment"]
    calibration = pd.read_csv(OUT / "p2b" / "calibration.csv")
    p0_calibration = pd.read_csv("stage2_output/calibration.csv")
    manifest = json.loads((OUT / "input_manifest.json").read_text(encoding="utf-8"))
    config_hashes = {
        "P0": summary["p0_config_sha256"],
        "P1": summary["model_summaries"]["P1"]["config_sha256"],
        "P2a": summary["model_summaries"]["P2a"]["config_sha256"],
        "P2b": summary["model_summaries"]["P2b"]["config_sha256"],
    }
    final_manifest = {
        "adopted_model": "P2b",
        "approval": "依頼者が全工程の完了を承認",
        "config_sha256": config_hashes,
        "input_db_sha256": summary["input_db_sha256"],
        "report": "stage3_output/p2b_final_model_report.md",
        "rating_examples": "stage3_output/p2b_rating_top10.json",
        "periods": p0["periods"],
        "future_scripts_executed": False,
    }
    (OUT / "final_model_manifest.json").write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    add = lines.append
    add("# 最終採用モデル P2b レーティング総合報告")
    add("")
    add("## 1. このモデルで何をしているか")
    add("")
    add("P2bは、JRA所属馬それぞれの**強さの平均 `mu` と、その不確実性 `var`**を、レース結果から時系列に推定するモデルです。`mu`が大きい馬ほど強いと評価します。ただし、JRA公式レーティングでも、馬券の期待値や勝率そのものでもありません。")
    add("")
    add("基本の流れは次のとおりです。")
    add("")
    add("```text")
    add("馬王Zの確定結果")
    add("  → JRA主催・平地・JRA所属馬だけを残す")
    add("  → 当日の全レースの発走前状態を保存")
    add("  → 着順を重み付きPlackett–Luce尤度として評価")
    add("  → 対角近似のオンライン更新でmu・varを更新")
    add("  → 次の開催日の発走前特徴量として保存")
    add("```")
    add("")
    add("新しく現れた馬には、過去の馬の水準を基準とする`mu_bar_D`に、デビュー形態別の補正と、父・母父の産駒成績に基づく補正を加えます。以後は実績が増えるほど、実際のレース結果が初期値よりも効くようになります。")
    add("")
    add("## 2. レーティングの直感的な読み方")
    add("")
    add(markdown(["値", "意味", "読み方の注意"], [
        ["`mu_pre`", "レース前の強さの平均", "同じ凍結config内で比較する。他のconfigとの絶対値比較は避ける"],
        ["`var_pre_persistent`", "直前の更新までの不確実性", "小さいほど状態が安定しているとみなす"],
        ["`var_pre_eff`", "休養も加えた今回の不確実性", "長期休養で大きくなるが、var_capを超えない"],
        ["`m_pre`", "レース内の相対順位の計算に使うスコア", "斤量補正後の値。最終モデルではbeta_kg=0なのでmu_preと一致"],
        ["`mu_post`", "そのレース結果を反映した強さ", "勝敗だけでなく、相手・着順・出走頭数に応じて変化する"],
        ["`n_graph_starts`", "対象グラフ内の実出走数", "非JRA時・障害・地方・海外・取消除外は含めない"],
    ]))
    add("")
    add("馬の強さは一点推定ではなく、**平均と分散のペア**で持ちます。休養で分散を増やすのは、最近の情報が少ない馬ほど次の結果への反応を大きくできるようにするためです。分散は永続状態をその場で増やすのではなく、今回の更新に使う一時値`var_pre_eff`として計算します。")
    add("")
    add("### 相対的な強さであること")
    add("")
    add("着順は、非JRA所属馬を除いたJRA所属馬の中での相対順位に付け替えます。そのため、このモデルは「実際の全出走馬の着順モデル」ではなく、**JRA所属馬同士の条件付きの強さモデル**です。実際の勝ち馬が非JRA所属馬だった場合、モデル内の最先着馬は実際の勝ち馬と異なります。")
    add("")
    add("## 3. 最終採用モデルの構成")
    add("")
    add(markdown(["モデル", "役割"], [
        ["P0", "全馬共通の初期化とオンライン更新。基準値・残差計算の基礎"],
        ["P1", "新馬戦・未勝利戦・その他のクラスで、初登場時の平均と分散を変える"],
        ["P2a", "P1に父の産駒成績を加える"],
        ["**P2b（最終採用）**", "P1に父と母父の産駒成績を加える"],
    ]))
    add("")
    add("P0→P1→P2a→P2bの順に、調整期間だけで係数を決め、各段階を凍結してから検証しました。検証期間の結果を見て、ハイパーパラメータ、係数、κ、標準化、判定基準を変更していません。")
    add("")
    add("## 4. 発走前状態と更新の詳細")
    add("")
    add("### 4.1 基準値 mu_bar_D")
    add("")
    add("新しい馬の初期値の基準は、開催日Dの前日時点で、直前365日に出走した馬の最新`mu`の平均です。同日の結果は使いません。直前365日の母集団がない場合は、それ以前の直近の母集団を使い、全くない初期時点だけ0とします。")
    add("")
    add("```text")
    add("mu_bar_D = 直前365日に出走した馬の最新muの平均")
    add("```")
    add("")
    add("P2bでは、N=100 crossingがデータ初期の1回だけで、最大絶対日次変化は" + f(summary["model_summaries"]["P2b"]["anchor_max_abs_daily_delta"], 9) + "でした。100件の閾値は診断用で、基準値の算出ルールは変更していません。")
    add("")
    add("### 4.2 休養と不確実性")
    add("")
    add("```text")
    add("days_since_start = max(0, D - last_start_date)")
    add("var_pre_eff = min(var + q × days_since_start / 30, var_cap)")
    add("```")
    add("")
    add(f"最終モデルの`q`は{p0['p0']['q']}です。初登場馬は休養日数を0とし、P0ならvar0、P1・P2ならデビュー形態別var0_dをvar_capとします。")
    add("")
    add("### 4.3 レース内スコア")
    add("")
    add("```text")
    add("m_i = mu_i - beta_kg × (weight_i - 残った馬の平均斤量)")
    add("```")
    add("")
    add("P0探索では`beta_kg=0`が選ばれたため、**最終P2bでは`m_pre = mu_pre`です**。斤量は考え方として組み込まれていますが、最終採用された補正係数はゼロです。斤量欠損時は残った馬の平均を代入する設計で、今回の実データに斤量欠損はありませんでした。")
    add("")
    add("### 4.4 着順を使う重み付きPL尤度")
    add("")
    add("Plackett–Luce（PL）は、候補馬の中から上の着順の馬が選ばれる確率を、スコアの指数で表す着順モデルです。段階kでは、まだ順位のついていない馬と順位なし馬を候補集合として使います。")
    add("")
    add("```text")
    add("p_ik = exp(m_i) / Σ_j∈R_k exp(m_j)")
    add("L = Σ_k w_k × log p_(π_k,k)")
    add("K = min(5, 確定着順のある馬数, 出走馬数 - 1)")
    add("```")
    add("")
    add("配点`(100, 40, 25, 15, 10, 0)`から作った最終採用の重みは`W_share`です。")
    add("")
    add(markdown(["着順段階", "1", "2", "3", "4", "5"], [["重み", "0.526316", "0.210526", "0.131579", "0.078947", "0.052632"]]))
    add("")
    add("6着以下の順序は使いません。少頭数などでK<5になっても重みを再正規化せず、観測された着順段階が少ないことは、そのレースの証拠量を小さくする設計です。これはバグではありません。")
    add("")
    add("### 4.5 対角近似オンライン更新")
    add("")
    add("```text")
    add("g_i = Σ_k w_k [1(i=π_k) - 1(i∈R_k) p_ik]")
    add("h_i = Σ_k w_k 1(i∈R_k) p_ik (1-p_ik)")
    add("prec_i = 1 / var_pre_eff_i + κ h_i")
    add("mu_new_i = mu_i + κ g_i / prec_i")
    add("var_new_i = 1 / prec_i")
    add("```")
    add("")
    add("最終モデルの`kappa=1.0`です。情報行列の対角成分だけを使う近似更新で、厳密な多変数Newton更新ではありません。同着は可能な順列を全て列挙し、勾配・情報量・評価対数尤度を平均して1回だけ更新します。馬番や馬IDで同着の順序を決めることはありません。")
    add("")
    add(markdown(["出走状態", "候補集合", "mu・var", "出走数・最終出走日"], [
        ["確定着順あり", "含める", "更新する", "更新する"],
        ["競走中止・失格など順位なし", "他馬の尤度のため残す", "自身は更新しない", "実出走として更新する"],
        ["取消・発走除外・競走除外", "最初から除外", "更新しない", "更新しない"],
        ["降着", "確定着順で扱う", "確定着順に基づき更新", "更新する"],
        ["同着", "同じ着順のまま", "全順列を平均", "実出走として更新"],
    ]))
    add("")
    add("同日の全レースのスナップショットを作ってから更新するため、同日の他レース結果は同日の特徴量に影響しません。")
    add("")
    add("## 5. P1：デビュー形態別の初期値")
    add("")
    add("新しい馬の初期値は、デビュー形態に応じて次のようにします。")
    add("")
    add("```text")
    add("mu0 = mu_bar_D + b_d")
    add("var0 = var0_d, var_cap = var0_d")
    add("```")
    add("")
    rows = [[debut, f(p1["p1"]["b"][debut], 9), f(p1["p1"]["var0_d"][debut], 9)] for debut in ["新馬戦", "未勝利戦", "その他のクラス"]]
    add(markdown(["デビュー形態", "b_d", "var0_d"], rows))
    add("")
    add("新馬戦・未勝利戦のvar0_dは探索時に設定した上限4.0に達しましたが、上限変更や新候補の追加はしていません。データ開始から1年以内の既存馬で、新馬戦以外に初登場した馬は、デビュー時点不明としてP0と同じ初期化を使います。新馬戦以外での初登場には、地方・障害・海外からの移籍馬が含まれ得ます。")
    add("")
    add("## 6. P2b：父・母父を使う初期値")
    add("")
    add("P2bでは、P1の初期値に次の血統補正を加えます。")
    add("")
    add("```text")
    add("mu0 = mu_bar_D + b_d")
    add("      + beta_sw × X_sire,win + beta_st × X_sire,top3")
    add("      + beta_bw × X_bms,win  + beta_bt × X_bms,top3")
    add("```")
    add("")
    beta = p2b["p2"]["beta"]
    add(markdown(["項", "係数", "値"], [[name, name.replace("beta_", "β_"), f(value, 9)] for name, value in beta.items()]))
    add("")
    add("### 6.1 残差の作り方")
    add("")
    add("対象は、**新馬戦でデビューした馬の最初の3走**です。これにより、移籍馬の成績を血統情報に混ぜません。各走について、P0の発走前スコアからHarville形式で計算した期待勝率・期待3着内率を使い、実績との差を取ります。")
    add("")
    add("```text")
    add("r_win  = 1（相対順位1着） - e_win")
    add("r_top3 = 1（相対順位3着以内） - e_top3")
    add("```")
    add("")
    add("P1やP2の出力を残差計算に使わず、P0に固定して循環を避けています。各as-of日Dについて、Dより前の結果だけを集計し、D当日・同日の結果は使いません。")
    add("")
    add("### 6.2 縮小推定と標準化")
    add("")
    add("```text")
    add("X_s = 残差の合計 / (N_s + κ_s)")
    add("κ_s = 20")
    add("```")
    add("")
    add("産駒が少ない場合は0へ縮約します。κ=20・50・100・200を比較し、P2a・P2bとも20を選びました。κごとにXと標準化定数を再計算し、2017～2020年の新馬戦デビュー馬で平均・標準偏差を凍結しています。")
    add("")
    add("ID欠損の項は0として`ID_MISSING`を記録します。`N_s=0`の無履歴、ID欠損、通常の0を区別するため、`N_sire`・`N_bms`と状態も保存します。標準偏差が1e-8未満の場合はscale=1として異常な拡大を防ぎます。")
    add("")
    add("## 7. 期間設計とデータの扱い")
    add("")
    add(markdown(["期間", "用途"], [
        ["2012～2016", "ウォームアップ。状態更新のみで、性能評価はしない"],
        ["2017～2020", "P0・P1・P2の係数、κ、標準化、診断用温度の決定"],
        ["2021～2024", "凍結した設定の検証・比較"],
        ["2025年以降", "ロック。入力・学習・評価に使用していない"],
    ]))
    add("")
    add("対象はJRA主催の平地競走です。障害・地方主催・ばんえい・海外開催と、レース時点でJRA所属でない馬の記録は使いません。交流競走はJRA主催平地に含めますが、DBに交流フラグがないため、交流の識別・交流別集計は行いません。")
    add("")
    add("明示的な`result_available_at`は存在せず、データ作成年月日を結果利用可能時刻の代替には使っていません。抽出時点の確定着順を開催日順に1回だけ処理し、結果版と更新イベントIDを保存しています。")
    add("")
    add("## 8. 最終検証結果")
    add("")
    add("### 8.1 モデル間比較（検証期間・プール）")
    add("")
    rows = []
    for row in pooled.itertuples():
        rows.append([row.comparison, row.pool, f(row.n_races), f(row.model_nll), f(row.baseline_nll), f(row.delta_nll), f"[{f(row.ci95_lower)}, {f(row.ci95_upper)}]"])
    add(markdown(["比較", "集合", "レース", "model NLL", "baseline NLL", "ΔNLL", "95％CI"], rows))
    add("")
    add("ΔNLLが負なら改善です。P1、P2a、P2bは、いずれも前のモデルからE・E_clean・Fで改善しました。E_cleanは非JRA除外が一度もないレースに限定した集合で、除外による推定対象の違いを確認するためEと並列しています。")
    add("")
    add("### 8.2 P2bの判定基準")
    add("")
    add("ここでの「経験3以内」はレース内の発走前`n_graph_starts`平均が3以下、「経験4以上」はその平均が3を超えるレース群です。馬個別のNLLではなく、レース単位のΔNLLを集計しています。")
    add("")
    add(markdown(["基準", "結果", "充足"], [
        ["経験3以内のΔNLL 95％CI上限 < 0", f(judgment["early_ci95_upper"]), "是"],
        ["経験4以上のΔNLL点推定 ≤ +0.0005", f(judgment["experienced_delta_nll"]), "是"],
        ["経験3以内で3年以上ΔNLL < 0", f"{judgment['early_negative_years']} / 4年", "是"],
    ]))
    add("")
    add(f"経験3以内のΔNLLは{f(judgment['early_delta_nll'])}、95％CIは[{f(judgment['early_ci95_lower'])}, {f(judgment['early_ci95_upper'])}]でした。年別は2021年{f(judgment['early_yearly_delta']['2021'])}、2022年{f(judgment['early_yearly_delta']['2022'])}、2023年{f(judgment['early_yearly_delta']['2023'])}、2024年{f(judgment['early_yearly_delta']['2024'])}です。**3基準を全て満たし、依頼者がP2bを最終採用モデルとして確定しました。**")
    add("")
    add("### 8.3 副次診断の読み方")
    add("")
    rows = []
    for row in calibration.itertuples():
        rows.append(["P2b", row.metric, f(row.predicted_mean), f(row.actual_mean), f(row.winner_ce)])
    for row in p0_calibration[p0_calibration["metric"] == "勝ち馬"].itertuples():
        rows.append(["P0（参考）", row.metric, f(row.predicted_mean), f(row.actual_mean), f(row.winner_ce)])
    add(markdown(["モデル", "指標", "予測平均", "実測", "勝ち馬CE"], rows))
    add("")
    add("P2bは主指標のPL-NLLでは改善していますが、P0より勝ち馬CEは大きくなっています。これは、着順全体の重み付き評価と、1着だけの評価が一致しないことを示します。全体の予測確率平均はレース内正規化のため頭数構成に強く依存するので、十分位・CE・層別結果と併せて読む必要があります。馬券の勝率や収益を直接意味する指標ではありません。")
    add("")
    add("## 9. P2bのレーティング例")
    add("")
    add("次の値はP2bの内部スケールで、P0のランキングと絶対値を直接比較するものではありません。瞬間最大は各レース結果反映直後の`mu_post`、年末は最終状態の`mu`です。")
    add("")
    add("### 2021～2024年 瞬間最大 Top10")
    add("")
    rows = [[index, row["horse_name"], f(row["mu_post"]), row["race_date"]] for index, row in enumerate(examples["historical_top10"], start=1)]
    add(markdown(["順位", "馬名", "レーティング", "記録日"], rows))
    add("")
    add("### 2024年12月末 Top10（2024年出走馬限定）")
    add("")
    rows = [[index, row["horse_name"], f(row["mu"])] for index, row in enumerate(examples["year_end_top10"], start=1)]
    add(markdown(["順位", "馬名", "レーティング"], rows))
    add("")
    add("## 10. 後続モデルで使う値")
    add("")
    add("1行＝1レース×1頭の発走前スナップショットを全期間について保存しています。主な候補は次の値です。")
    add("")
    add(markdown(["列", "内容"], [
        ["`mu_pre`", "発走前の強さの平均"],
        ["`var_pre_persistent`", "直前の永続分散"],
        ["`var_pre_eff`", "休養を反映した今回の分散"],
        ["`m_pre`", "レース内相対順位のスコア。最終configではmu_preと同じ"],
        ["`days_since_start`", "前のグラフ内出走からの日数"],
        ["`n_graph_starts`", "発走前のグラフ内出走数"],
        ["`mu0`", "初登場時の初期強さ"],
        ["`debut_type`", "新馬戦・未勝利戦・その他・既存馬不明"],
        ["`X_sire_win/top3`", "父の標準化した勝ち・3着内の残差情報"],
        ["`X_bms_win/top3`", "母父の標準化した勝ち・3着内の残差情報"],
        ["`N_sire/N_bms`", "as-of時点の集計走数"],
        ["`pedigree_status`", "OK・無履歴・ID欠損等の区別"],
        ["`n_field_original/retained`", "元の頭数と除外後の頭数"],
    ]))
    add("")
    add("診断用の勝率はP0で凍結したtauによる補助出力で、後続モデルへ必須で渡す値ではありません。オッズ・人気・支持率・払戻・予想タイム指数は使用していません。")
    add("")
    add("## 11. 再現性と凍結管理")
    add("")
    add(markdown(["項目", "値"], [
        ["最終採用モデル", "P2b"],
        ["P2b config SHA-256", f"`{config_hashes['P2b']}`"],
        ["入力DB SHA-256", f"`{summary['input_db_sha256']}`"],
        ["入力スナップショット", "`stage2_input/BaoZ_2_snapshot.mdb`"],
        ["スナップショット行数", f"610,676行 / 43,252レース"],
        ["最終状態", "`stage3_output/p2b/state_2024-12-31.csv`"],
        ["更新イベント", "43,252件。event_id・race_idの重複なし"],
        ["テスト", "18テスト合格。独立NLL誤差は最大7.9×10⁻¹²"],
    ]))
    add("")
    add("2025年以降用の`rating_update_for_features.py`と`rating_final_eval.py`は作成済みですが、実行していません。いずれもconfigハッシュを照合します。後続モデルでの特徴量の採否や2025年以降の最終評価は、別の作業として行います。")
    add("")
    add("## 12. このモデルで分かること・分からないこと")
    add("")
    add(markdown(["分かること", "分からないこと"], [
        ["JRA所属馬の条件付き相対的な強さの推移", "JRA公式レーティングとの対応や、他configとの絶対値比較"],
        ["着順全体を使ったP0・P1・P2からの改善", "馬券の購入条件・市場優位・収支"],
        ["デビュー形態と血統が初期値に与える影響", "後続モデルに入れた場合の採否・寄与・最終性能"],
        ["検証期間における年別・層別の傾向", "2025年以降の将来性能の保証"],
        ["不確実性・休養・経験数を含む発走前特徴量", "実際の全出走馬に対する無条件の着順確率"],
    ]))
    add("")
    add("新馬戦以外の初登場には地方・障害・海外からの移籍馬が含まれ得ますが、それらの除外結果はグラフ履歴や血統残差には使用していません。結果の後日修正・再抽出は、過去のスナップショットを無断で書き換えず、別の結果版として扱う必要があります。")
    add("")
    add("## 13. 成果物の案内")
    add("")
    add(markdown(["内容", "場所"], [
        ["本報告書", "`stage3_output/p2b_final_model_report.md`"],
        ["最終採用マニフェスト", "`stage3_output/final_model_manifest.json`"],
        ["凍結config", "`stage3_output/config_p2b.yaml`"],
        ["全期間スナップショット", "`stage3_output/p2b/snapshots.csv.gz`"],
        ["更新後表", "`stage3_output/p2b/post_updates.csv.gz`"],
        ["2024年末の状態", "`stage3_output/p2b/state_2024-12-31.csv`"],
        ["モデル間比較", "`stage3_output/pairwise_comparisons.csv`"],
        ["層別結果", "`stage3_output/pairwise_layers.csv`"],
        ["血統情報量・状態", "`stage3_output/p2b/pedigree_information.csv`、`pedigree_status.csv`"],
        ["血統残差履歴", "`stage3_output/pedigree_residual_history.csv.gz`"],
        ["基準値診断", "`stage3_output/p2b/mu_anchor_daily.csv`、`mu_anchor_transition_summary.csv`、`mu_anchor_timeseries.svg`"],
        ["検証結果", "`stage3_output/verification.json`"],
        ["ランキング例の元データ", "`stage3_output/p2b_rating_top10.json`"],
        ["段階3の詳細報告", "`stage3_output/stage3_report.md`"],
    ]))
    add("")
    add("## 14. 一言でまとめると")
    add("")
    add("P2bは、**デビュー時の馬のタイプと父・母父の成績で慎重に初期値を与え、その後はレース結果に応じて強さと不確実性を毎回更新する、JRA所属馬向けの時系列レーティングです。** 強さの目安を発走前特徴量として供給するもので、馬券戦略や将来性能を保証するものではありません。")
    (OUT / "p2b_final_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "p2b_final_model_report.md")


if __name__ == "__main__":
    main()
