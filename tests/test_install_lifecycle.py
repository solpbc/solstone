# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Family 13: Lifecycle, collision detection, receipt management, and uninstallation."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_test_release_server
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"
        self.prefix.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_journal_and_cli_exclusivity_refusal(self):
        with ephemeral_keypair("test exclusivity") as (sec, pub, pin):
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

                # Passing both journal and cli in --components must refuse with source-ambiguous
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "journal,cli", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "refusal")
                self.assertEqual(res["root_code"], "source-ambiguous")
            finally:
                server.stop()

    def test_receipt_creation(self):
        with ephemeral_keypair("test receipt") as (sec, pub, pin):
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

                env = os.environ.copy()
                data_home = self.work_dir / "xdg_data"
                env["XDG_DATA_HOME"] = str(data_home)

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "journal,desktop,tmux", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")

                receipt_path = data_home / "solstone" / "install.conf"
                self.assertTrue(receipt_path.is_file(), f"Receipt file not created: {receipt_path}")
                content = receipt_path.read_text(encoding="utf-8")
                self.assertIn("[component:journal]", content)
                self.assertIn("[component:desktop]", content)
                self.assertIn("[component:tmux]", content)
                self.assertIn("phase=complete", content)
            finally:
                server.stop()

    def test_uninstall_desktop(self):
        with ephemeral_keypair("test uninstall") as (sec, pub, pin):
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

                env = os.environ.copy()
                config_home = self.work_dir / "xdg_config"
                data_home = self.work_dir / "xdg_data"
                env["XDG_CONFIG_HOME"] = str(config_home)
                env["XDG_DATA_HOME"] = str(data_home)

                # 1. Install journal and desktop
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "journal,desktop", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc.returncode, 0, f"Install failed: {proc.stderr}\n{proc.stdout}")

                # Check autostart desktop entry was created by desktop handler
                autostart = config_home / "autostart" / "solstone-desktop.desktop"
                self.assertTrue(autostart.is_file())

                # Check journal binary exists
                journal_bin = self.prefix / "bin" / "journal"
                self.assertTrue(journal_bin.is_file())

                # 2. Uninstall desktop
                proc2 = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc2.returncode, 0, f"Uninstall failed: {proc2.stderr}\n{proc2.stdout}")
                res2 = json.loads(proc2.stdout.strip())
                self.assertEqual(res2["status"], "success")
                self.assertEqual(res2["root_code"], "uninstalled")

                # Assert handler-managed autostart file is gone
                self.assertFalse(autostart.exists(), "Desktop autostart file should have been removed")

                # Assert lock file still present (never unlinked)
                lock_file = self.prefix / ".solstone-platform.lock"
                self.assertTrue(lock_file.exists(), "Lock file inode was deleted")

                # Assert journal payload still present
                self.assertTrue(journal_bin.is_file(), "Journal payload was deleted by desktop uninstall")
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
