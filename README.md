# P2b-Rating

JRA平地競走を対象とした、P4レーティングの算出・オンライン更新・予測・閲覧用プロジェクトです。リポジトリ名は初期モデルの履歴を保持しています。

最終採用モデルは、P2bをベースに着差を更新重みへ組み込んだP4です。方式Aの段階別マージン変調PLを採用し、生タイム差、γ=2、上限2を使用します。評価NLLは標準PLのままです。

```text
P4 config SHA-256: 436f08ffb840a60ef33354fc28fa9434d0193a0460c73ba8c8f29d8038a26ba1
```

## ビューワーを開く

WindowsとPython 3.14を用意し、リポジトリ直下で実行します。

```bat
python -m pip install -r requirements.txt
run_viewer.bat
```

ブラウザで`http://127.0.0.1:8501`を開きます。2017年以降の馬名・レース情報を結合したParquetキャッシュを同梱しているため、**馬王ZのDBがない環境でも閲覧できます**。

キャッシュは内容ハッシュで照合します。P4のCSV・設定・コードが変わったときだけ再生成され、その場合は馬王Z DBが必要です。

## Streamlit Cloud

GitHubリポジトリ`tanikei0907-sys/P2b-Rating`の`main`ブランチを指定し、Main file pathを`rating_jra/rating_viewer.py`としてデプロイします。CloudのPython versionは3.13を選択します。`requirements.txt`と`packages.txt`を使用し、Streamlit Secretsや馬王Z DBは不要です。

`main`へのpush後にCloud側へ変更を反映します。更新・予測ラッパーはP4データとキャッシュもpushするため、Cloudでも最新の公開データを閲覧できます。馬王Zを使った更新そのものはローカルPCで実行します。

## データと公開範囲

- P4の凍結設定、2012～2024年のP4スナップショット・状態・血統残差
- 2025年以降のP4継続更新出力、発走前予測、ビューワーキャッシュ
- モデル実装、更新・予測・同期スクリプト、テスト、運用文書
- P0・P2bの比較用資産は既存の読み取り専用データとして保持

馬王ZのDB、参照PDF、DBスナップショット、開発途中の出力、バックアップ、作業ファイル、ログは公開しません。ビューワーキャッシュには馬名・レース名・着順等が含まれます。データの公開・利用条件は利用者が確認してください。

## 確定結果の更新と新規予測

更新・予測を行うPCには馬王ZとMicrosoft Access ODBCドライバーが必要です。`rating_jra/p4_margin/forward_config.yaml`と`rating_jra/viewer_config.yaml`に自身のDBパスとリンク先を設定してください。DBは一時コピーして読み取り専用で接続します。

```bat
run_rating_update.bat
run_predict_snapshot.bat --date 2026-11-22
```

PowerShellでも実行できます。

```powershell
.\run_rating_update.ps1
.\run_predict_snapshot.ps1 -Date 2026-11-22
```

各ラッパーはローカル処理成功後、P4のビューワーキャッシュを更新し、対象データだけをcommitして`origin`の現在ブランチへpushします。予測では状態・結果履歴を変更せず、予測CSVとキャッシュだけを公開します。`NO_NEW_RACES`でも未pushのデータがあればpushします。

pushに失敗してもローカル出力は保持されます。認証・通信を回復して次を再実行してください。

```bat
python rating_jra/github_sync.py
```

リモート側が進んでいる場合は、強制pushせず変更を確認して統合します。GitHub Actionsはテストを実行しますが、馬王Z DBへは接続しません。

## テスト

```bat
cd rating_jra
python -m unittest discover -p "test_*.py" -v
```

実DBを必要とする検証は、DBがない環境ではskipされます。同期処理のテストは一時的なローカルGitリポジトリを使用し、GitHubを変更しません。

詳細は`rating_jra/README_forward_ops.md`、`rating_jra/README_viewer.md`、`rating_jra/p4_margin/P4_report.md`を参照してください。`rating_jra/rating_spec.md`は初期P2bモデルの仕様を記録した資料です。
