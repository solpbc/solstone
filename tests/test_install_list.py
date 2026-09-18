# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Family 12: Catalogue listing (--list) mode."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_test_release_server
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallList(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"
        self.prefix.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_mode_does_not_download_archives_or_mutate(self):
        with ephemeral_keypair("test list") as (sec, pub, pin):
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

                # 1. Test Human format
                proc_h = subprocess.run(
                    [str(installer), "--list", "--prefix", str(self.prefix)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_h.returncode, 0, f"Failed: {proc_h.stderr}")
                self.assertIn("Solstone Platform Catalogue", proc_h.stdout)
                self.assertIn("journal", proc_h.stdout)
                self.assertIn("desktop", proc_h.stdout)
                self.assertIn("tmux", proc_h.stdout)

                # 2. Test JSON format
                proc_j = subprocess.run(
                    [str(installer), "--list", "--json", "--prefix", str(self.prefix)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_j.returncode, 0, f"Failed: {proc_j.stderr}")
                res = json.loads(proc_j.stdout.strip())
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["root_code"], "list")

                # Verify target prefix remained completely empty
                prefix_files = list(self.prefix.rglob("*"))
                self.assertEqual(len(prefix_files), 0, f"Prefix was modified: {prefix_files}")
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
