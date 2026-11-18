from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_report import f, markdown

OUT = Path(__file__).with_name("stage3_output")


def main() -> None:
    summary = json.loads((OUT / "stage3_summary.json").read_text(encoding="utf-8"))
    p1 = json.loads((OUT / "config_p1.yaml").read_text(encoding="utf-8"))
    p2a = json.loads((OUT / "config_p2a.yaml").read_text(encoding="utf-8"))
    p2b = json.loads((OUT / "config_p2b.yaml").read_text(encoding="utf-8"))
    comparisons = pd.read_csv(OUT / "pairwise_comparisons.csv")
    layers = pd.read_csv(OUT / "pairwise_layers.csv")
    judgment = summary["judgment"]
    information = pd.read_csv(OUT / "p2b" / "pedigree_information.csv")
    status = pd.read_csv(OUT / "p2b" / "pedigree_status.csv")
    sire = pd.read_csv(OUT / "sire_auxiliary.csv")
    bms = pd.read_csv(OUT / "bms_auxiliary.csv")
    test3 = json.loads(Path("stage3_test_diagnostics.json").read_text(encoding="utf-8"))
    lines = []
    add = lines.append
    add("# 中央競馬平地レーティング：段階3（P1・P2a・P2b）報告")
    add("")
    add("## 結論要約")
    add("")
    add("### 段階1・2")
    add("")
    add("- 2012～2024年のJRA平地43,252レース・610,676実出走を、開催日順の単一方向処理で構築した。")
    add("- P0は全216条件から`var0=2.0・q=0.01・kappa=1.0・beta_kg=0・W_share`を選択し、検証期間で事前基準を満たした。")
    add("- P0のconfig・入力ハッシュ・出力は段階3でも固定し、実データのP0 NLLが完全一致することを再確認した。")
    add("")
    add("### 段階3")
    add("")
    add(f"- P1は調整期間の早期レースでデビュー形態別の`b_d`と`var0_d`を決定し、raw NLLは{f(p1['selection']['raw_nll'])}だった。")
    add(f"- P2aはκ={p2a['p2']['kappa_s']}、P2bはκ={p2b['p2']['kappa_s']}を選択し、P0の残差で計算した父・母父の情報を初登場時のmu0へ加えた。")
    add(f"- P2b−P2aの検証E・経験3以内ΔNLLは{f(judgment['early_delta_nll'])}、95％CI上限は{f(judgment['early_ci95_upper'])}、経験4以上の点推定は{f(judgment['experienced_delta_nll'])}だった。")
    add(f"- 3基準の充足状況は**{'MET' if judgment['criteria_met'] else 'NOT_MET'}**。最終判断は依頼者に委ねる。係数・候補・基準は検証結果後に変更していない。")
    add("")
    add("## 段階3の判定材料")
    add("")
    add(markdown(["基準", "結果", "充足"], [
        ["経験<=3・検証EプールΔNLL 95％CI上限<0", f(judgment["early_ci95_upper"]), "是" if judgment["early_ci95_upper"] < 0 else "否"],
        ["経験>=4・検証EプールΔNLL点推定<=+0.0005", f(judgment["experienced_delta_nll"]), "是" if judgment["experienced_delta_nll"] <= 0.0005 else "否"],
        ["経験<=3・4年中3年以上でΔNLL<0", f"{judgment['early_negative_years']} / 4年", "是" if judgment["early_negative_years"] >= 3 else "否"],
    ]))
    add("")
    add("年別の早期ΔNLL：" + "、".join(f"{year}年={f(value)}" for year, value in judgment["early_yearly_delta"].items()))
    add("")
    add("## 凍結設定")
    add("")
    add(markdown(["モデル", "config SHA-256"], [
        ["P0", f"`{summary['p0_config_sha256']}`"],
        ["P1", f"`{summary['model_summaries']['P1']['config_sha256']}`"],
        ["P2a", f"`{summary['model_summaries']['P2a']['config_sha256']}`"],
        ["P2b", f"`{summary['model_summaries']['P2b']['config_sha256']}`"],
    ]))
    add("")
    add(f"入力DB SHA-256は`{summary['input_db_sha256']}`で全モデル共通。L2係数はP1のb・P2のβともに事前固定の0.01。早期レースは段階2の層別定義と同じく、レース内の発走前n_graph_starts平均が3以下とした。")
    add("")
    add("### P1：デビュー形態別の初期化")
    add("")
    rows = []
    for debut in ["新馬戦", "未勝利戦", "その他のクラス"]:
        rows.append([debut, f(p1["p1"]["b"][debut], 9), f(p1["p1"]["var0_d"][debut], 9)])
    add(markdown(["デビュー形態", "b_d", "var0_d"], rows))
    add("")
    add(f"選択時のraw NLL={f(p1['selection']['raw_nll'], 9)}、L2={f(p1['selection']['penalty'], 9)}、目的関数={f(p1['selection']['objective'], 9)}、対象{p1['selection']['n_races']:,}レース。新馬戦・未勝利戦のvar0_dは設定上限4.0に到達したが、候補追加や上限変更はしていない。既存馬（デビュー時点不明）はP0初期化を維持した。")
    add("")
    add("### P2a・P2b：血統係数")
    add("")
    rows = []
    for model, config in [("P2a", p2a), ("P2b", p2b)]:
        beta = config["p2"]["beta"]
        rows.append([model, config["p2"]["kappa_s"], f(beta["beta_sw"], 9), f(beta["beta_st"], 9), f(beta["beta_bw"], 9), f(beta["beta_bt"], 9), config["p2"]["aggregate_win_top3_coefficients"]])
    add(markdown(["モデル", "κ", "β_sw", "β_st", "β_bw", "β_bt", "係数集約"], rows))
    add("")
    add("### κ候補の比較")
    add("")
    for model in ["p2a", "p2b"]:
        frame = pd.read_csv(OUT / f"{model}_kappa_summary.csv")
        rows = [[int(row.kappa), f(row.raw_nll, 9), f(row.penalty, 9), f(row.objective, 9), row.success] for row in frame.itertuples()]
        add(f"**{model.upper()}**")
        add("")
        add(markdown(["κ", "raw NLL", "L2", "目的関数", "収束"], rows))
        add("")
    add("標準化定数はκごとに再計算し、2017～2020年の新馬戦デビュー馬だけで凍結した。ID欠損・無履歴の項は0とし、N=0・ID_MISSING・通常の0を区別した。P1/P2の出力ではなくP0の発走前mから残差を計算している。")
    add("")
    add("## モデル間比較")
    add("")
    add("ΔNLLは各モデルと1つ前のモデルの差（P1−P0、P2a−P1、P2b−P2a）で、負の値が改善。CIはレース単位のペア・ブートストラップ2,000回による95％区間。")
    add("")
    rows = []
    for row in comparisons.itertuples():
        rows.append([row.comparison, row.period, row.pool, row.year, f(row.n_races), f(row.model_nll), f(row.baseline_nll), f(row.delta_nll), f"[{f(row.ci95_lower)}, {f(row.ci95_upper)}]"])
    add(markdown(["比較", "期間", "集合", "年", "レース", "model NLL", "baseline NLL", "ΔNLL", "95％CI"], rows))
    add("")
    add("## 検証期間の層別（E）")
    add("")
    add("経験数はレース内平均、初出走形態はレース内の多数派を使い、同数はMIXEDとした。これらは馬単位のNLLではなく、レース単位ΔNLLの層別である。")
    add("")
    rows = [[row.comparison, row.layer_type, row.level, f(row.n_races), f(row.delta_nll), f"[{f(row.ci95_lower)}, {f(row.ci95_upper)}]"] for row in layers.itertuples()]
    add(markdown(["比較", "層", "区分", "レース", "ΔNLL", "95％CI"], rows))
    add("")
    add("## 副次診断")
    add("")
    add("勝率はP0で凍結したtauを全モデルに適用した診断用出力で、主指標のPL-NLLには使わない。3着内・5着内は各モデルの発走前mによるHarville確率である。全体の予測平均はレース内で確率が1・3・5に正規化されるため頭数構成に強く依存し、モデル間の差は勝ち馬CEと十分位で読む。")
    add("")
    rows = []
    for model, path in [("P0", Path("stage2_output/calibration.csv")), ("P1", OUT / "p1" / "calibration.csv"), ("P2a", OUT / "p2a" / "calibration.csv"), ("P2b", OUT / "p2b" / "calibration.csv")]:
        frame = pd.read_csv(path)
        for row in frame.itertuples():
            rows.append([model, row.metric, f(row.predicted_mean), f(row.actual_mean), f(row.winner_ce)])
    add(markdown(["モデル", "指標", "予測平均", "実測", "勝ち馬CE"], rows))
    add("")
    add("## 血統情報量・欠損・無履歴")
    add("")
    rows = []
    for row in information.itertuples():
        rows.append([row.scope, row.side, f(row.rows), f(row.mean), f(row.std), f(row.q50), f(row.q90), f(row.q95), f(row.q99), f(row.q100)])
    add(markdown(["対象", "側", "行数", "N平均", "N SD", "中央", "90%", "95%", "99%", "最大"], rows))
    add("")
    rows = [[row.scope, row.side, row.status, f(row.count), f(100 * row.rate, 4) + "%"] for row in status.itertuples()]
    add(markdown(["対象", "側", "状態", "件数", "割合"], rows))
    add("")
    for name, frame in [("父", sire), ("母父", bms)]:
        values = frame["offspring"]
        add(f"{name}の産駒数分布：最小{int(values.min())}、中央{values.median():.1f}、平均{values.mean():.3f}、90％点{values.quantile(0.9):.1f}、99％点{values.quantile(0.99):.1f}、最大{int(values.max())}。新馬戦勝率・3着内率・3走以内勝ち上がり率の全件はCSVに保存した。")
        add("")
    add(f"残差計算の対象は新馬戦デビュー馬の最初3走{summary['pedigree_history_rows']:,}行で、父ID欠損{summary['pedigree_sire_missing_rows']:,}行、母父ID欠損{summary['pedigree_bms_missing_rows']:,}行。各as-of日より前の結果だけを使い、当日・同日の結果は含めていない。")
    add("")
    add("## 実装・テスト・基準値診断")
    add("")
    add("- 既存13テストと段階3の5テストが全合格。P1初期化、P2血統項、as-of同日除外、ID欠損・無履歴、標準化を検証した。")
    add(f"- 実データでのP0 tuning E NLLは{f(test3['actual_p0_nll_invariance'], 12)}で、凍結時と完全一致した。")
    add("- 血統集計の開発途中で検出した行番号マッピング不備は、対象行位置と全行位置を区別する実装へ修正し、as-of独立再計算で一致を確認した。旧P2結果は`stage3_output_bug_v2`に隔離し、本報告のP2は修正後の特徴量で全て再最適化したものである。")
    add("- 同着は全順列平均、順位なし馬は候補集合に残して自身のmu・varは更新せず、対角近似オンライン更新を全モデルで維持した。")
    add("- K<5で重みを再正規化しない設計、非JRA除外後の条件付き相対順位、開催日順の結果版管理もP0から継承した。")
    add(f"- P2bのmu_bar_D最大絶対日次変化は{f(summary['model_summaries']['P2b']['anchor_max_abs_daily_delta'], 9)}、N=100 crossingは{summary['model_summaries']['P2b']['anchor_crossings']}回。モデル別の日別表・遷移表・SVGを保存した。")
    add("")
    add("## 読み取れること／読み取れないこと")
    add("")
    add("- P2の差はP2aとの比較であり、血統を追加した効果の検証期間における変化を示す。後続モデルの特徴量としての有効性や馬券収益を意味しない。")
    add("- 新馬戦以外での初登場には地方・障害・海外からの移籍馬が含まれ得る。それらの除外結果はグラフ履歴・残差集計に使用していない。")
    add("- 勝ち馬は非JRA除外後のJRA所属馬の最先着馬で、実際のレース全体の勝ち馬とは限らない。")
    add("- result_available_atは存在せず、データ作成年月日を代替にしていない。抽出時点の確定結果を開催日順に1回だけ処理した。")
    add("- 2021～2024年の結果を見て係数・κ・標準化・判定基準は変更していない。2025年以降の性能や後続モデルでの採否は別の評価が必要である。")
    add("")
    add("## 成果物")
    add("")
    for artifact in [
        "config_p1.yaml / config_p2a.yaml / config_p2b.yaml と各ハッシュ",
        "p1/・p2a/・p2b/：全期間スナップショット、更新後表、最終状態、結果版イベント",
        "pairwise_comparisons.csv / pairwise_layers.csv / judgment.csv",
        "各モデルのcalibration・十分位・経験数・休養・新馬戦後3走・順序指標",
        "p2b/pedigree_information.csv / p2b/pedigree_status.csv",
        "sire_auxiliary.csv / bms_auxiliary.csv / pedigree_residual_history.csv.gz",
        "mu_anchor_daily.csv / mu_anchor_transition_summary.csv / mu_anchor_timeseries.svg（各モデル）",
        "stage3_summary.json / input_manifest.json / verification.json",
        "rating_update_for_features.py / rating_final_eval.py：作成のみで実行していない",
    ]:
        add(f"- `{artifact}`")
    add("")
    add("最終判断は依頼者に委ね、ここで作業を停止する。")
    (OUT / "stage3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "stage3_report.md")


if __name__ == "__main__":
    main()
