# P2b 2025年以降 運用ガイド

最終採用モデルはP2bです。凍結configのSHA-256は次の値で固定されています。

```text
3577cbc7a6a2c3e58bfdeda6666b9aa9d7369866f4c6fa49902dac6b8779dbe9
```

起動時に`stage3_output/config_p2b.yaml`の実ファイルハッシュを照合します。不一致なら、DBコピーや出力作成より先にエラー終了します。P1の`b_d`・`var0_d`、P2bの`κ=20`・`β`・血統標準化定数は再学習せず適用します。

## 1. ファイル構成

| ファイル | 用途 |
|---|---|
| `rating_update_for_features.py` | `update`・`predict_snapshot`のCLI入口 |
| `forward_ops.py` | 状態継承、冪等更新、予測、バックアップの本体 |
| `forward_config.yaml` | DB、凍結資産、出力先、開始日の設定 |
| `p2b_forward/` | 2025年以降の出力とバックアップ。初回実行時に作成 |
| `../run_rating_update.bat` / `.ps1` | 確定結果の更新ラッパー |
| `../run_predict_snapshot.bat` / `.ps1` | 発走前スナップショット生成ラッパー |

`stage2_output/`と`stage3_output/p2b/`にある2012～2024年の凍結資産は読み取り専用です。更新対象は常に`p2b_forward/`内のファイルです。

## 2. 実行前の準備

1. プロジェクト直下`C:\BaoZ_AI\keiba_ai_project_v3`を開きます。
2. Python 3.14と、開発時と同じ`numpy・pandas・scipy・numba・pyodbc`を利用できることを確認します。
3. 馬王Zの最新DBパスを`forward_config.yaml`の`database.path`に設定します。既定値は`C:\BaoZ_AI\DB\BaoZ_2.mdb`です。
4. 出力先を変える場合は`output.directory`を編集します。相対パスは`rating_jra`基準です。
5. DBは実行ごとに一時ディレクトリへコピーし、そのコピーへ読み取り専用で接続します。コピー中に元DBが変化した場合は中止するため、まず再実行してください。
6. `出走馬T`は`BaoZ_2.ex.mdb`、`出走馬マスタ`・`レースマスタ`・`産駒マスタ`は`BaoZ-RA.mdb`のリンクテーブルです。これらも一時コピーし、一時MDB内の`TableDef.Connect`だけをコピー先へ付け替えます。元DBのリンク設定・内容は変更しません。パスが異なる場合は`database.linked_tables`を編集してください。

## 3. 初回の追いつき更新

2024-12-31の最終状態から、DB内の最新確定日までを一括処理します。

### BAT

```bat
run_rating_update.bat
```

### PowerShell

```powershell
.\run_rating_update.ps1
```

特定日までに限定する場合は、以下のようにします。

```bat
run_rating_update.bat --end-date 2025-12-31
```

```powershell
.\run_rating_update.ps1 -EndDate 2025-12-31
```

### 実装変更後の全期間再構築

血統補完などの実装変更で、過去の初期値も再計算する場合は、次を実行します。

```bash
python rating_update_for_features.py rebuild --config forward_config.yaml
```

`p2b_forward/`を`p2b_forward_legacy/<実行ID>/`へ移してから、2024-12-31の凍結状態から全期間を再構築します。旧出力は削除されず、退避先に残ります。再構築後は`latest_run_manifest.json`と`forward_completion_report.json`を確認してください。

### 処理後の確認

- `p2b_forward/latest_run_manifest.json`の`status`が`UPDATED`
- `result_events_2025plus.csv`のrace_idが重複していない
- `state_latest.csv`の最終出走日が更新されている
- `p2b_forward/update_last.log`

新しい確定レースがない場合は`NO_NEW_RACES`となり、状態・履歴・バックアップは更新しません。

## 4. 毎週の運用

### 確定結果が出た後

1. 馬王Zを更新します。
2. `run_rating_update.bat`または`run_rating_update.ps1`を実行します。
3. すでに処理済みのrace_idは更新イベントで除外されます。処理済みレースの結果版が変わっている場合は、無断で再計算せず停止します。

### 予想を出す前

当日の出馬表から、発走前特徴量だけを生成します。

```bat
run_predict_snapshot.bat --date 2025-01-05
```

```powershell
.\run_predict_snapshot.ps1 -Date 2025-01-05
```

日付を省略すると、DB内の当日以降の最も近い開催日を選びます。出力は`p2b_forward/predict_snapshots_<YYYYMMDD>.csv`です。

このモードは完全なドライランで、`state_latest.csv`・更新イベント・過去スナップショット・残差履歴を変更しません。出力は発走前情報だけで、`mu_post`・`var_post`・`skip_mu_var_update`は含めません。読み込む状態は、最新の確定更新が完了した`state_latest.csv`です。まだ更新がない場合は、凍結された2024-12-31の状態を直接読みます。

## 5. 出力

| ファイル | 内容 |
|---|---|
| `snapshots_2025plus.csv.gz` | 発走前スナップショットの追記。`elo_pre`を含む |
| `post_updates_2025plus.csv.gz` | 更新後`mu_post`・`var_post` |
| `state_latest.csv` | P2bの最新全馬状態 |
| `pedigree_residual_history_2025plus.csv.gz` | 新馬戦デビュー馬の最初3走の残差履歴 |
| `mu_anchor_daily_2025plus.csv` | 日別の`mu_bar_D`と基準値診断 |
| `result_events_2025plus.csv` | race_id・結果版・更新日時・入力DBハッシュ |
| `state_p0_residual_latest.csv` | P2bの血統残差を計算するためのP0最新状態 |
| `p0_residual_snapshots_2025plus.csv.gz` | 残差計算を継続するためのP0発走前スナップショット |
| `run_manifest_<run_id>.json` | 実行日時、処理レース数、出力ハッシュ、バックアップ先 |

`elo_pre`は次の表示用換算値で、PL更新に使う`mu_pre`や`m_pre`は変更しません。

```text
elo_pre = 1500 + (400 / ln(10)) × (mu_pre - mu_bar_D)
```

## 6. 冪等性とバックアップ

更新は以下の順で行います。

1. 既存の更新イベントを確認し、未処理race_idだけを選びます。
2. 直前の`state_latest.csv`と関連出力を`p2b_forward/backups/<run_id>/`へコピーします。
3. 一時領域で全出力を完成させてから置き換えます。
4. 出力中に失敗した場合は、バックアップから復元してエラー終了します。

したがって、同じ日付範囲を再実行しても、同じrace_idは二重更新されません。結果版が異なる場合や既存ファイルのキー重複・configハッシュ不一致がある場合は停止します。

## 7. ロールバック

1. 更新・予測ジョブを停止します。
2. `latest_run_manifest.json`の`backup`、または`p2b_forward/backups/`の実行日時から、復元したい直前状態の`<run_id>`を選びます。
3. そのフォルダにある次のファイルを`p2b_forward/`へコピーして戻します。
   - `snapshots_2025plus.csv.gz`
   - `post_updates_2025plus.csv.gz`
   - `state_latest.csv`
   - `pedigree_residual_history_2025plus.csv.gz`
   - `mu_anchor_daily_2025plus.csv`
   - `result_events_2025plus.csv`
   - `state_p0_residual_latest.csv`
   - `p0_residual_snapshots_2025plus.csv.gz`
   - `latest_run_manifest.json`
4. バックアップに存在しないファイルは、ロールバック対象時点より後に作られたものなので削除します。
5. 更新を再実行し、イベント数・最終状態・ハッシュを確認します。

2012～2024年の凍結資産をバックアップから上書きする必要はありません。常に`p2b_forward/`内で復元してください。

## 8. 主なエラーと対処

| 症状 | 対処 |
|---|---|
| `config_p2b.yamlのハッシュ不一致` | 凍結configが変更されています。正しいconfigを復元し、ハッシュを確認します。 |
| DBコピー中に元DBが変更された | 馬王Zの更新を停止して再実行します。 |
| 処理済みレースの結果版が変わった | 無断で上書きせず、変更内容を確認します。必要なら別の結果版として再構築を検討します。 |
| 予測対象のレースがない | `predict_date`と出馬表のデータ区分を確認します。`update`は確定区分7だけを処理します。 |
| 出力中に失敗した | 直前バックアップから復元されます。`update_last.log`を確認し、原因を解消して再実行します。 |
| `NO_NEW_RACES` | 新しい確定レースがありません。状態と履歴は変更されていません。 |

## 9. テスト

`rating_jra`を作業ディレクトリとして、以下を実行します。

```bash
python -m unittest -v test_forward_ops
```

ダミーの2025年データを使い、次を確認します。

- 2024-12-31の状態から数日分を開催日順に更新できる
- 同じ更新を再実行してもrace_idが二重登録されず、出力が変わらない
- `predict_snapshot`で状態・履歴が変化しない
- `elo_pre`が`mu_pre`と`mu_bar_D`から厳密に一致する

テスト用の一時ファイルは完了後に削除します。本番DBの2025年以降の結果はテストに使用しません。

既存モデル・運用・ビューワーを一括検証する場合は`python -m unittest discover -p 'test_*.py' -v`を実行します。ビューワーの予測表示・未確定出馬表の結合・2017年以降の血統集計も確認します。

本番の2025年以降出力を読み取り専用で全件監査する場合は、次を実行します。DBは一時コピーして接続し、凍結資産は変更しません。

```bash
python forward_comprehensive_audit.py
python forward_verify.py --check-idempotency
```

結果は`forward_comprehensive_audit.json`と`p2b_forward/forward_completion_report.json`に保存します。新たな確定レースがDBへ追加された場合は`unprocessed_final_races`に件数を記録し、過去出力の不整合とは区別します。

## 10. GitHubへの自動同期

プロジェクト直下の`run_rating_update.bat` / `.ps1`と`run_predict_snapshot.bat` / `.ps1`は、ローカル処理成功後に`github_sync.py`を実行します。同期の前にビューワーキャッシュを再生成し、レーティング出力・予測CSV・キャッシュだけをcommitして`origin`へpushします。DB・バックアップ・作業ファイル・ログは同期対象外です。

`NO_NEW_RACES`でも未pushの変更があればpushします。pushに失敗してもローカル出力は保持されます。認証・通信を回復した後に`python rating_jra/github_sync.py`を再実行してください。リモートが進んでいる場合は強制pushせず、変更を確認して統合します。テスト用に`--no-push`と`--no-cache-refresh`を指定できます。

GitHub Actionsはユニットテストを実行します。馬王Z DBはローカルPCにあるため、GitHub側のスケジュール実行ではDBを更新しません。
