# P4 2025年以降 運用ガイド

最終採用モデルはP4です。方式Aの段階別マージン変調PLを使用し、生タイム差、γ=2、m_max=2で更新します。評価用NLLは標準PLのままです。

```text
436f08ffb840a60ef33354fc28fa9434d0193a0460c73ba8c8f29d8038a26ba1
```

起動時に`p4_margin/selected_config.json`の実ファイルハッシュを照合します。P1の`b_d`・`var0_d`、P2bの血統係数・κ=20・標準化定数、P0の更新設定は再学習しません。

## 1. ファイル構成

| ファイル | 用途 |
|---|---|
| `p4_update_for_features.py` | `update`・`predict_snapshot`のCLI入口 |
| `p4_forward_ops.py` | P4の状態継承、着差重み、冪等更新、予測、バックアップ |
| `p4_margin/forward_config.yaml` | DB、P4凍結資産、出力先、開始日 |
| `p4_forward/` | 2025年以降の出力とバックアップ |
| `../run_rating_update.bat` / `.ps1` | 確定結果の更新 |
| `../run_predict_snapshot.bat` / `.ps1` | 発走前スナップショット生成 |

`stage2_output/`、`stage3_output/p2b/`、`p2b_forward/`のP0・P2b資産は読み取り専用です。P4の更新対象は常に`p4_forward/`です。

## 2. 実行前の準備

1. プロジェクト直下でPython 3.14と`requirements.txt`の依存関係を用意します。
2. `p4_margin/forward_config.yaml`の`database.path`に最新の馬王Z DBを設定します。
3. `database.linked_tables`に出走馬・レース・馬・血統マスタのリンク先を設定します。
4. DBとリンク先を一時コピーし、一時DB内のリンクだけを付け替えて読み取り専用で接続します。元DBの内容・リンク設定は変更しません。
5. 出力先の相対パスは`rating_jra`基準です。

## 3. 確定結果の更新

2024-12-31のP4最終状態と残差履歴から、DB内の最新確定日までを開催日順に処理します。

```bat
run_rating_update.bat
```

PowerShellでは次を実行します。

```powershell
.\run_rating_update.ps1
```

特定日までに限定する場合は`--end-date YYYY-MM-DD`または`-EndDate`を使用します。直接CLIを使う場合は次のとおりです。

```bash
python p4_update_for_features.py update --config p4_margin/forward_config.yaml
```

処理後は`p4_forward/latest_run_manifest.json`の`status`、`new_races`、出力ハッシュを確認します。新しい確定レースがない場合は`NO_NEW_RACES`となり、状態・履歴・バックアップを更新しません。

## 4. 発走前予測

```bat
run_predict_snapshot.bat --date 2026-11-22
```

PowerShellでは`-Date 2026-11-22`を使用します。日付を省略すると、当日以降の最も近い開催日を選びます。出力は`p4_forward/predict_snapshots_<YYYYMMDD>.csv`です。

このモードは発走前情報だけを生成し、`state_latest.csv`・更新イベント・過去スナップショット・残差履歴を変更しません。`mu_post`・`var_post`・`skip_mu_var_update`は出力しません。最新確定日以前への将来状態の適用は拒否します。

## 5. 出力

| ファイル | 内容 |
|---|---|
| `snapshots_2025plus.csv.gz` | 発走前状態・Elo・着差の追記 |
| `post_updates_2025plus.csv.gz` | 更新後`mu_post`・`var_post` |
| `state_latest.csv` | P4の最新全馬状態 |
| `pedigree_residual_history_2025plus.csv.gz` | P4軌道から生成した新馬デビュー馬の最初3走残差 |
| `mu_anchor_daily_2025plus.csv` | 日別`mu_bar_D` |
| `result_events_2025plus.csv` | race_id・結果版・処理日時・DBハッシュ |
| `margins_2025plus.csv.gz` | 段階別タイム差と利用状態 |
| `margin_multipliers_2025plus.csv.gz` | P4の段階別更新倍率 |
| `latest_run_manifest.json` | 最新実行の期間・件数・ハッシュ・バックアップ先 |

Elo風表示は`1500 + (400 / ln(10)) × (mu_pre - mu_bar_D)`です。PL更新に使う`mu_pre`・`m_pre`自体は変更しません。

## 6. 冪等性・バックアップ・復元

更新前に既存出力を`p4_forward/backups/<run_id>/`へコピーし、一時領域で全出力を完成させてから置き換えます。失敗時は直前の出力を復元します。同じrace_idは更新イベントで二重処理を防ぎ、結果版が変わった場合や処理済み日への遅延追加がある場合は停止します。

ロールバックは、対象run_idのバックアップにある8種のCSV出力を`p4_forward/`へ戻し、対応する実行マニフェストを確認します。バックアップにないファイルは、その実行後に新規作成されたものです。P0・P2bの凍結資産へは復元ファイルを上書きしません。

## 7. テストと監査

```bash
python -m unittest -v test_p4_forward
python p4_forward_verify.py --check-idempotency
```

ダミーデータで、初回更新・差分更新・二重更新防止・着差倍率・予測時の状態不変・Elo一致を確認します。実出力の監査では、全レースの更新、血統as-of特徴量、残差、日別基準値、最終状態、出力ハッシュを検査します。

## 8. GitHub同期

4本のラッパーはローカル処理成功後に`github_sync.py`を実行します。P4出力・予測・P4ビューワーキャッシュだけをcommitしてpushします。DB・バックアップ・作業ファイル・ログは同期対象外です。

pushに失敗してもローカル出力は保持されます。回復後に`python rating_jra/github_sync.py`を再実行してください。リモートが進んでいる場合は強制pushせず、変更を確認して統合します。
