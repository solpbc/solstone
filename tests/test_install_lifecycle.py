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
                service_log = self.work_dir / "service.log"
                env["SOLSTONE_TEST_SERVICE_LOG"] = str(service_log)

                # 1. Install journal and desktop
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "journal,desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc.returncode, 0, f"Install failed: {proc.stderr}\n{proc.stdout}")
                self.assertEqual(
                    service_log.read_text(encoding="utf-8").splitlines(),
                    ["desktop 2.0.3 install-service"],
                )

                # Check journal binary exists
                journal_bin = self.prefix / "bin" / "journal"
                self.assertTrue(journal_bin.is_file())

                # A native service refusal leaves custody intact for an identical retry.
                receipt = data_home / "solstone" / "install.conf"
                receipt_before = receipt.read_bytes()
                service_fail_env = {
                    **env,
                    "SOLSTONE_TEST_SERVICE_FAIL": "desktop:2.0.3:uninstall-service",
                }
                refused = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=service_fail_env,
                )
                self.assertNotEqual(refused.returncode, 0)
                self.assertEqual(json.loads(refused.stdout)["root_code"], "handler-failed")
                self.assertEqual(receipt.read_bytes(), receipt_before)
                self.assertTrue((self.prefix / "bin" / "solstone-linux").is_symlink())

                # 2. Retry uninstall desktop
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
                self.assertEqual(
                    service_log.read_text(encoding="utf-8").splitlines()[-1],
                    "desktop 2.0.3 uninstall-service",
                )

                # Assert lock file still present (never unlinked)
                lock_file = self.prefix / ".solstone-platform.lock"
                self.assertTrue(lock_file.exists(), "Lock file inode was deleted")

                # Assert journal payload still present
                self.assertTrue(journal_bin.is_file(), "Journal payload was deleted by desktop uninstall")
                owner_data = self.work_dir / "journal-owner-data" / "entry.db"
                owner_data.parent.mkdir()
                owner_data.write_bytes(b"journal-owner-data")
                self.assertIn("[component:journal]", receipt.read_text(encoding="utf-8"))
                self.assertNotIn("[component:desktop]", receipt.read_text(encoding="utf-8"))

                # A completed identical rerun is an unchanged success.
                rerun = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(rerun.returncode, 0, rerun.stderr + rerun.stdout)
                self.assertEqual(json.loads(rerun.stdout)["components"]["desktop"]["status"], "unchanged")

                # A partial removal keeps the receipt for a safe identical retry.
                reinstall = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "desktop", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(reinstall.returncode, 0, reinstall.stderr + reinstall.stdout)
                partial_env = {**env, "SOLSTONE_TEST_FAIL_TREE_UNINSTALL_AFTER_PUBLIC": "desktop"}
                partial = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=partial_env,
                )
                self.assertNotEqual(partial.returncode, 0)
                self.assertEqual(json.loads(partial.stdout)["root_code"], "remove-failed")
                self.assertFalse((self.prefix / "bin" / "solstone-linux").exists())
                current = self.prefix / "opt" / "solstone" / "desktop" / "current"
                self.assertTrue(current.is_symlink())
                self.assertIn("[component:desktop]", receipt.read_text(encoding="utf-8"))
                remaining_root = self.prefix / "opt" / "solstone" / "desktop" / os.readlink(current)
                remaining_binary = next(path for path in remaining_root.rglob("solstone-linux") if path.is_file())
                remaining_bytes = remaining_binary.read_bytes()
                remaining_binary.write_bytes(b"substituted")
                substituted = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertNotEqual(substituted.returncode, 0)
                self.assertEqual(json.loads(substituted.stdout)["root_code"], "ownership-unknown")
                self.assertTrue(current.is_symlink())
                remaining_binary.write_bytes(remaining_bytes)
                remaining_binary.chmod(0o755)
                recovered = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(recovered.returncode, 0, recovered.stderr + recovered.stdout)
                self.assertNotIn("[component:desktop]", receipt.read_text(encoding="utf-8"))
                self.assertEqual(owner_data.read_bytes(), b"journal-owner-data")

                # Removal followed by receipt publication failure also converges.
                reinstall2 = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "desktop", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(reinstall2.returncode, 0, reinstall2.stderr + reinstall2.stdout)
                receipt_fail_env = {**env, "SOLSTONE_TEST_FAIL_UNINSTALL_RECEIPT": "desktop"}
                receipt_fail = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=receipt_fail_env,
                )
                self.assertNotEqual(receipt_fail.returncode, 0)
                self.assertEqual(json.loads(receipt_fail.stdout)["root_code"], "receipt-write-failed")
                self.assertIn("[component:desktop]", receipt.read_text(encoding="utf-8"))
                receipt_recovery = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(receipt_recovery.returncode, 0, receipt_recovery.stderr + receipt_recovery.stdout)
                self.assertNotIn("[component:desktop]", receipt.read_text(encoding="utf-8"))
                self.assertEqual(owner_data.read_bytes(), b"journal-owner-data")
            finally:
                server.stop()

    def test_multi_component_uninstall_reports_completed_and_partial_state(self):
        with ephemeral_keypair("test partial uninstall") as (sec, pub, pin):
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
                data_home = self.work_dir / "partial-data"
                env = {**os.environ, "XDG_DATA_HOME": str(data_home)}
                installed = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "desktop,tmux", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(installed.returncode, 0, installed.stderr + installed.stdout)
                receipt = data_home / "solstone" / "install.conf"
                failing_env = {**env, "SOLSTONE_TEST_FAIL_TREE_UNINSTALL_AFTER_PUBLIC": "tmux"}
                failed = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "desktop,tmux", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=failing_env,
                )
                self.assertNotEqual(failed.returncode, 0)
                result = json.loads(failed.stdout)
                self.assertEqual(result["components"]["desktop"]["status"], "removed")
                self.assertEqual(result["components"]["tmux"]["status"], "failed")
                receipt_bytes = receipt.read_bytes()
                self.assertNotIn(b"[component:desktop]\n", receipt_bytes)
                self.assertIn(b"[component:tmux]\n", receipt_bytes)
                self.assertFalse((self.prefix / "bin" / "solstone-linux").exists())
                self.assertFalse((self.prefix / "bin" / "solstone-tmux").exists())
                self.assertTrue((self.prefix / "opt" / "solstone" / "tmux" / "current").is_symlink())

                recovered = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "tmux", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(recovered.returncode, 0, recovered.stderr + recovered.stdout)
                self.assertFalse(receipt.exists())
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
