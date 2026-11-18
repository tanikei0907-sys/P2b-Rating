# P2b-Rating

JRA平地競走を対象とした、凍結済みP2bレーティングの算出・オンライン更新・予測・閲覧用プロジェクトです。発走前スナップショット、Elo風レーティング、キャリア推移、ランキング、血統評価をStreamlit + Plotlyで表示します。

最終採用モデルのconfig SHA-256は次で固定されています。

```text
3577cbc7a6a2c3e58bfdeda6666b9aa9d7369866f4c6fa49902dac6b8779dbe9
```

## ビューワーを開く

WindowsとPython 3.14を用意し、リポジトリ直下で次を実行します。

```bat
python -m pip install -r requirements.txt
run_viewer.bat
```

ブラウザで `http://127.0.0.1:8501` を開きます。2017年以降の馬名・レース情報を結合したParquetキャッシュも同梱しているため、**馬王ZのDBがない環境でも、同梱時点のデータをそのまま閲覧できます**。

キャッシュは内容ハッシュで照合します。CSV・設定・コードが変わったときだけ再生成され、その場合は馬王Z DBが必要です。

## データと公開範囲

- 凍結済みP0/P2bの設定と、運用に必要な2012年以降のレーティング出力
- 2017～2024年のP2bスナップショット、2025年以降の継続更新出力
- 発走前予測CSV、ビューワー用の2017年以降キャッシュ
- モデル実装、更新・予測・同期スクリプト、テスト、運用文書

馬王ZのDB（`.mdb`）、参照PDF、DBスナップショット、開発途中の出力、バックアップ、作業ファイル、ログは公開しません。ビューワーキャッシュには公開用の馬名・レース名・着順等が含まれます。データの公開・利用条件は利用者が確認してください。

## 確定結果の更新と新規予測

更新・予測を行うPCに馬王ZとMicrosoft Access ODBCドライバーが必要です。`rating_jra/forward_config.yaml`と`rating_jra/viewer_config.yaml`に、自身のDBパスとリンク先を設定してください。DBは一時コピーして読み取り専用で接続します。

```bat
run_rating_update.bat
run_predict_snapshot.bat --date 2026-11-22
```

PowerShellからは次も利用できます。

```powershell
.\run_rating_update.ps1
.\run_predict_snapshot.ps1 -Date 2026-11-22
```

各ラッパーはローカル処理が成功した後、ビューワーキャッシュを更新し、対象データだけをcommitして`origin`の現在ブランチへpushします。予測モードではレーティング状態・結果履歴を変更せず、予測CSVとキャッシュだけを公開します。`NO_NEW_RACES`でも、未pushのデータがあればpushします。

初回のみ、GitHubへのpush権限を設定します。

```bat
git remote add origin https://github.com/tanikei0907-sys/P2b-Rating.git
git branch -M main
```

認証にはGitHub CLIの`gh auth login`または、利用者のGit認証を使用できます。pushに失敗してもローカル出力は保持されます。認証・通信を回復して次を再実行してください。

```bat
python rating_jra/github_sync.py
```

リモート側が進んでいる場合は、強制pushせず変更を確認して統合します。GitHub Actionsはテストを実行しますが、馬王Z DBへは接続できないため、**データ更新は馬王ZのあるローカルPCでラッパーを実行したときに行われます**。

## テスト

```bat
cd rating_jra
python -m unittest discover -p "test_*.py" -v
```

実DBを必要とする検証は、DBがない環境ではskipされます。同期処理のテストは一時的なローカルGitリポジトリを使用し、GitHubを変更しません。

詳細は`rating_jra/rating_spec.md`、`rating_jra/README_forward_ops.md`、`rating_jra/README_viewer.md`を参照してください。
