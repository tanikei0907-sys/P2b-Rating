from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from github_sync import run_git, sync_github


class GithubSyncTest(unittest.TestCase):
    def make_repository(self, root: Path) -> None:
        subprocess.run(["git", "init", "-b", "main", str(root)], check=True, capture_output=True)
        run_git(root, "config", "user.name", "P2b Test")
        run_git(root, "config", "user.email", "p2b-test@example.invalid")
        output = root / "rating_jra/p2b_forward/snapshots_2025plus.csv.gz"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"test-snapshot")
        cache = root / "rating_jra/cache/viewer_source_fingerprint.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{}", encoding="utf-8")

    def test_data_commit_is_idempotent_and_excludes_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            self.make_repository(root)
            first = sync_github(root, push=False, refresh_cache=False)
            self.assertEqual(first["status"], "COMMITTED_LOCALLY")
            second = sync_github(root, push=False, refresh_cache=False)
            self.assertEqual(second["status"], "NO_DATA_CHANGES")
            self.assertEqual(run_git(root, "rev-list", "--count", "HEAD").stdout.strip(), "1")
            database = root / "rating_jra/stage2_input/BaoZ_2_snapshot.mdb"
            database.parent.mkdir()
            database.write_bytes(b"not-for-publication")
            prediction = root / "rating_jra/p2b_forward/predict_snapshots_20261122.csv"
            prediction.write_text("race_id,horse_id\n1,2\n", encoding="utf-8")
            third = sync_github(root, push=False, refresh_cache=False)
            self.assertEqual(third["status"], "COMMITTED_LOCALLY")
            published = run_git(root, "ls-tree", "-r", "--name-only", "HEAD").stdout
            self.assertIn("predict_snapshots_20261122.csv", published)
            self.assertNotIn(".mdb", published)

    def test_push_also_publishes_an_existing_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            remote = Path(temporary) / "remote.git"
            self.make_repository(root)
            sync_github(root, push=False, refresh_cache=False)
            subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
            run_git(root, "remote", "add", "origin", str(remote))
            result = sync_github(root, push=True, refresh_cache=False)
            self.assertEqual(result["status"], "NO_DATA_CHANGES")
            self.assertTrue(result["pushed"])
            remote_head = run_git(remote, "rev-parse", "main").stdout.strip()
            self.assertEqual(remote_head, run_git(root, "rev-parse", "HEAD").stdout.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
