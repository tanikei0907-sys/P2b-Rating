# P7 2025年以降 運用ガイド

最終採用モデルはP7です。P1デビュー形態、P2b血統事前分布、P4生タイム差倍率、P5芝・ダート分離、P6年齢ドリフトを継承し、方式Aの二段階対戦伝播`L=1, η=0.075, τ=30日, clip=±0.25`を開催日終了後に適用します。評価用NLLは標準PLのままです。

```text
3246e20ccee1102638213235d6b9e6f373079de6165fda16ddc3ae76d4b9e174
```

起動時にP7・P6・P4・P2b configの実ファイルハッシュを照合します。事前分布・標準化定数・更新パラメータは再学習しません。

## 1. ファイル構成

| ファイル | 用途 |
|---|---|
| `p7_update_for_features.py` | `update`・`predict_snapshot`のCLI入口 |
| `p7_forward_ops.py` | 状態継承、冪等更新、予測、バックアップ |
| `p7_forward_engine.py` | P7本番用の両馬場・伝播量の記録 |
| `p7_forward/forward_config.yaml` | DB、P7凍結資産、出力先、開始日 |
| `p7_forward/` | 2025年以降の出力とバックアップ |
| `../run_rating_update.bat` / `.ps1` | 確定結果の更新 |
| `../run_predict_snapshot.bat` / `.ps1` | 発走前スナップショット生成 |

`stage2_output/`、`stage3_output/p2b/`、`p2b_forward/`、`p4_margin/`、`p4_forward/`、`p5_surface/`、`p6_age/`、`p7_propagation/`の既存資産は読み取り専用です。P7の更新対象は常に`p7_forward/`です。

## 2. 実行前の準備

1. プロジェクト直下でPythonと`requirements.txt`の依存関係を用意します。
2. `p7_forward/forward_config.yaml`の`database.path`に最新の馬王Z DBを設定します。
3. `database.linked_tables`に出走馬・レース・馬・血統マスタのリンク先を設定します。
4. DBとリンク先を一時コピーし、一時DB内のリンクだけを付け替えて読み取り専用で接続します。元DBの内容・リンク設定は変更しません。
5. 出力先の相対パスは`rating_jra`基準です。

## 3. 確定結果の更新

2024-12-31のP7最終状態・残差履歴・直近1走対戦履歴から、DB内の最新確定日までを開催日順に処理します。

```bat
run_rating_update.bat
```

PowerShellでは`.\run_rating_update.ps1`を実行します。特定日までに限定する場合は`--end-date YYYY-MM-DD`または`-EndDate`を使用します。直接CLIは次のとおりです。

```bash
python p7_update_for_features.py update --config p7_forward/forward_config.yaml
```

処理後は`p7_forward/latest_run_manifest.json`の`status`、`new_races`、出力ハッシュを確認します。新しい確定レースがない場合は`NO_NEW_RACES`となり、状態・履歴・バックアップ・manifestを更新しません。

## 4. 発走前予測

```bat
run_predict_snapshot.bat --date 2026-11-22
```

PowerShellでは`-Date 2026-11-22`を使用します。日付を省略すると、当日以降の最も近い開催日を選びます。出力は`p7_forward/predict_snapshots_<YYYYMMDD>.csv`です。

同じ発走前状態を使うP0予測も`p7_forward/p0_predict_snapshots_<YYYYMMDD>.csv`に生成します。P0状態の確定日から当日までの確定結果をメモリ上で追いつき計算し、当日分はP0もドライランです。P0状態ファイルは保存・更新しません。

このモードは発走前情報だけを生成し、`state_latest.csv`・対戦履歴・更新イベント・過去スナップショット・残差履歴を変更しません。更新後を示す列は出力しません。最新確定日以前への将来状態の適用は拒否します。

## 5. 出力

| ファイル | 内容 |
|---|---|
| `snapshots_2025plus.csv.gz` | 発走前の総合・基礎・両馬場・年齢・伝播量とElo |
| `post_updates_2025plus.csv.gz` | 更新後`mu_post`・`var_post` |
| `state_latest.csv` | 基礎体力、両馬場オフセット、対戦伝播保留量を含む最新全馬状態 |
| `pedigree_residual_history_2025plus.csv.gz` | P7軌道から生成した新馬デビュー馬の最初3走残差 |
| `mu_anchor_daily_2025plus.csv` | 日別`mu_bar_D` |
| `result_events_2025plus.csv` | race_id・結果版・処理日時・DBハッシュ |
| `propagation_events_2025plus.csv.gz` | 伝播元・伝播先・過去レース・適用量 |
| `interaction_history_latest.csv.gz` | 次回更新へ引き継ぐ直近1走対戦履歴 |
| `margins_2025plus.csv.gz` | 段階別タイム差と利用状態 |
| `margin_multipliers_2025plus.csv.gz` | P4の段階別更新倍率 |
| `latest_run_manifest.json` | 最新実行の期間・件数・ハッシュ・バックアップ先 |

Elo風表示は`1500 + (400 / ln(10)) × (mu_pre - mu_bar_D)`です。基礎体力Eloと馬場適性補正Eloも出力します。

## 6. 冪等性・バックアップ・復元

更新前に既存出力を`p7_forward/backups/<run_id>/`へコピーし、一時領域で全出力を完成させてから置き換えます。失敗時は直前の出力を復元します。同じrace_idは更新イベントで二重処理を防ぎ、結果版が変わった場合や処理済み日への遅延追加がある場合は停止します。

ロールバックは、対象run_idのバックアップにある10種の出力と`latest_run_manifest.json`を`p7_forward/`へ戻します。バックアップにないファイルは、その実行後に新規作成されたものです。凍結資産へは復元ファイルを上書きしません。

## 7. テストと監査

```bash
python -m unittest -v test_p7_forward
python p7_forward_verify.py --check-idempotency
```

ダミーデータで初回更新・差分更新・二重更新防止・対戦履歴・予測時の状態不変・Elo分解を確認します。実出力の監査では、全レース更新、連続性10頭、血統as-of、残差、伝播イベント、最終状態、出力ハッシュ、保護資産を検査します。

## 8. GitHub同期

4本のラッパーはローカル処理成功後に`github_sync.py`を実行します。P7出力・予測・必要な凍結config・本番コード・P7ビューワーキャッシュをcommitしてpushします。DB・バックアップ・作業ファイル・ログ・研究用の全期間伝播ログは同期対象外です。

pushに失敗してもローカル出力は保持されます。回復後に`python rating_jra/github_sync.py`を再実行してください。リモートが進んでいる場合は強制pushせず、変更を確認して統合します。
