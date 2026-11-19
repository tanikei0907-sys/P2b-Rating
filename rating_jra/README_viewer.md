# P7 JRAレーティング ビューワー

`rating_viewer.py`は、最終採用モデルP7の出力をブラウザで閲覧・分析するアプリです。Python + Streamlit + Plotlyを使用します。

## 起動

プロジェクト直下で次を実行します。

```bat
python -m pip install -r requirements.txt
run_viewer.bat
```

ブラウザで`http://127.0.0.1:8501`を開きます。終了するには起動ウィンドウで`Ctrl+C`を押します。初回または入力更新後はキャッシュ再生成に約1分かかることがあります。

## 設定とハッシュ

`viewer_config.yaml`でDB・リンク先・出力・キャッシュのディレクトリを指定します。起動時に`p7_propagation/selected_config.json`のSHA-256を照合します。

```text
3246e20ccee1102638213235d6b9e6f373079de6165fda16ddc3ae76d4b9e174
```

不一致なら起動しません。P1・P2bの事前分布、P4の着差倍率、P5の馬場別状態、P6の年齢傾斜、P7の対戦伝播パラメータは変更しません。

## キャッシュ

`cache/viewer_merged_2017plus.parquet`、`viewer_auxiliary.parquet`、`viewer_source_fingerprint.json`を使用します。入力CSV・アプリ・設定の内容ハッシュとサイズが変わったときだけ再生成します。パスやmtimeが違っても内容が同じなら再利用するため、GitHubからのクローンでも同梱キャッシュを直接利用できます。

キャッシュはP7データだけで再生成され、P4・P2bキャッシュを結合・混用しません。DBは一時コピーして読み取り専用で接続し、一時ファイルとAccessの`.ldb`は削除します。

## タブ

### レース別

開催日・競馬場・レースを選び、レーティング順位、総合Elo、基礎体力Elo、今回の芝・ダート適性補正Elo、`±elo_sd`、P0比、着順、`elo_post`・`elo_delta`を表示します。横棒グラフとエラーバーも表示します。

レース選択欄には年齢・クラス・グレードを表示し、名称が空のレースに「名称未設定」は表示しません。`p7_forward/predict_snapshots_*.csv`があるレースは「発走前（未確定）」を選択できます。未確定出馬表は確定結果がなくてもDBから補完し、確定版があれば確定版を優先します。予測行の`elo_post`・`elo_delta`は空欄です。

### 馬別推移

馬名の部分一致または馬IDから1頭を選び、2017年以降のElo推移、`±1 SD`、全戦績を表示します。戦績には基礎体力Eloと馬場適性補正Eloも含みます。開催日・レース名・着順・休養日数・Elo変動を確認できます。累積出走数はウォームアップ期間を含むスナップショットの値です。

### ランキング

指定日から直近365日以内に出走した馬の最新Eloで現役TOP100を表示します。芝・ダートと最小出走数を指定できます。指定日以降の結果は使いません。

### 血統評価

種牡馬・母父別の産駒数、新馬戦勝率・3着内率、3走以内勝ち上がり率、初期3走残差を検索・並べ替えできます。2017年以降のP7残差だけを使用し、2012～2016年は集計から除外します。2016年以前に新馬戦を終えた馬の新馬戦勝率は算出不可として空欄にします。

## Elo風表示

```text
elo_pre = 1500 + SCALE × (mu_pre - mu_bar_D)
elo_base_pre = 1500 + SCALE × (mu_base_pre - mu_bar_D)
elo_surface_adjustment = SCALE × surface_offset_pre
elo_pre = elo_base_pre + elo_surface_adjustment
elo_post = 1500 + SCALE × (mu_post - mu_bar_D)
elo_delta = elo_post - elo_pre
elo_sd = SCALE × sqrt(var_pre_eff)
elo_prior_boost = SCALE × (p7_mu_pre - p0_mu_pre)
SCALE = 400 / ln(10)
```

Eloは表示用換算で、PL更新には使いません。`elo_prior_boost`はP7とP0の総合差であり、血統だけの純粋な補正ではありません。P0との比較には既存の読み取り専用P0スナップショットを使用します。

## 期間と入力

表示は2017-01-01以降に限定します。

- 2017～2024年：`p7_propagation/snapshots.csv.gz`、`post_updates.csv.gz`、`mu_anchor_daily.csv`
- 2025年以降：`p7_forward/snapshots_2025plus.csv.gz`、`post_updates_2025plus.csv.gz`、`mu_anchor_daily_2025plus.csv`
- 当日予想：`p7_forward/predict_snapshots_*.csv`

確定・予測は`(race_id, horse_id, record_type)`で重複排除して区別します。

## Streamlit Cloud

リポジトリ`tanikei0907-sys/P2b-Rating`、ブランチ`main`、Main file path`rating_jra/rating_viewer.py`、Python 3.13を指定します。`requirements.txt`と`packages.txt`を使用し、Secrets・馬王Z DBは不要です。更新・予測ラッパーがP7データとキャッシュをpushするため、Cloudでも最新の公開データを閲覧できます。

## トラブル対応

- P7 configハッシュ不一致：正しい`selected_config.json`を復元してください。
- DBロック：馬王Zを終了して再実行してください。一時MDBとリンク先を使用します。
- キャッシュが古い：入力の内容が変れば自動再生成されます。`cache/`内のParquetを削除しても再生成できます。
- 馬名が見つからない：馬IDでも検索してください。馬名マスタで補完しています。
- 起動確認：`viewer_server.log`とHTTP `200 OK`を確認してください。
