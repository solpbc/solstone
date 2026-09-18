# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Family 9: Dry-run snapshot invariance and scratch cleanup."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_test_release_server
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"
        self.prefix.mkdir(parents=True, exist_ok=True)
        # Put a sentinel file in prefix to ensure directory exists
        (self.prefix / "sentinel.txt").write_text("initial state\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot_dir(self, directory: Path) -> dict[str, str]:
        files = {}
        for p in directory.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(directory))
                files[rel] = p.read_text(errors="replace")
        return files

    def test_dry_run_snapshot_invariance(self):
        with ephemeral_keypair("test dry run") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin)
            try:
                installer = self.work_dir / "install.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                )

                before_snapshot = self.snapshot_dir(self.prefix)

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--dry-run", "--components", "all", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Dry run failed: {proc.stderr}\n{proc.stdout}")

                after_snapshot = self.snapshot_dir(self.prefix)
                self.assertEqual(before_snapshot, after_snapshot, "Target prefix modified during --dry-run")

                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["lock_state"], "unlocked")
                self.assertEqual(res["snapshot_consistency"], "unlocked")
                self.assertEqual(res["receipt_paths"], [])
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
