from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNC_PATHS = [
    "rating_jra/p4_forward/snapshots_2025plus.csv.gz",
    "rating_jra/p4_forward/post_updates_2025plus.csv.gz",
    "rating_jra/p4_forward/state_latest.csv",
    "rating_jra/p4_forward/pedigree_residual_history_2025plus.csv.gz",
    "rating_jra/p4_forward/mu_anchor_daily_2025plus.csv",
    "rating_jra/p4_forward/result_events_2025plus.csv",
    "rating_jra/p4_forward/margins_2025plus.csv.gz",
    "rating_jra/p4_forward/margin_multipliers_2025plus.csv.gz",
    "rating_jra/p4_forward/latest_run_manifest.json",
    "rating_jra/cache/viewer_merged_2017plus.parquet",
    "rating_jra/cache/viewer_auxiliary.parquet",
    "rating_jra/cache/viewer_source_fingerprint.json",
    ":(glob)rating_jra/p4_forward/predict_snapshots_*.csv",
]


def run_git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Gitコマンドが失敗しました")
    return result


def sync_github(root: Path = PROJECT_ROOT, push: bool = True, refresh_cache: bool = True) -> dict:
    root = root.resolve()
    if run_git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        raise RuntimeError("Gitリポジトリではありません")
    lock_path = root / ".git/p4-rating-sync.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("別のGitHub同期が実行中です。完了後に再実行してください") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
            lock.write(f"pid={os.getpid()}\n")
        if refresh_cache:
            from viewer_data import load_viewer_config, load_viewer_data
            load_viewer_data(load_viewer_config(root / "rating_jra/viewer_config.yaml"))
        paths = [path for path in SYNC_PATHS if not path.startswith(":(glob)") and (root / path).exists()]
        if list(root.glob("rating_jra/p4_forward/predict_snapshots_*.csv")):
            paths.append(":(glob)rating_jra/p4_forward/predict_snapshots_*.csv")
        if not paths:
            raise RuntimeError("同期対象の出力ファイルがありません")
        run_git(root, "add", "--", *paths)
        changed = run_git(root, "diff", "--cached", "--quiet", "--", *paths, check=False).returncode != 0
        commit = None
        if changed:
            message = "Update P4 rating data: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            run_git(root, "commit", "-m", message, "--", *paths)
            commit = run_git(root, "rev-parse", "HEAD").stdout.strip()
        if push:
            remotes = run_git(root, "remote").stdout.splitlines()
            if "origin" not in remotes:
                raise RuntimeError("Git remote 'origin'が設定されていません")
            branch = run_git(root, "branch", "--show-current").stdout.strip()
            if not branch:
                raise RuntimeError("ブランチがdetached HEAD状態です")
            run_git(root, "push", "origin", branch)
        return {
            "status": "PUBLISHED" if push and changed else "NO_DATA_CHANGES" if not changed else "COMMITTED_LOCALLY",
            "commit": commit,
            "pushed": push,
            "cache_refreshed": refresh_cache,
        }
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P4の出力とビューワーキャッシュをGitHubへ公開します")
    parser.add_argument("--no-push", action="store_true", help="ローカルcommitまで実行します")
    parser.add_argument("--no-cache-refresh", action="store_true", help="テスト用。キャッシュを再生成しません")
    args = parser.parse_args()
    result = sync_github(push=not args.no_push, refresh_cache=not args.no_cache_refresh)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
