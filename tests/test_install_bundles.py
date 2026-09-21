# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Families 7 & 8: Bundles, architecture filtering, --json schema, non-tty no-selection."""

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


class TestInstallBundles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"
        self.prefix.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bundle_all_and_json_document(self):
        with ephemeral_keypair("test bundle all") as (sec, pub, pin):
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

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "all", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")

                # Verify single JSON document on stdout
                lines = [l for l in proc.stdout.splitlines() if l.strip()]
                self.assertEqual(len(lines), 1, f"Expected exactly 1 JSON line on stdout, got: {lines}")
                res = json.loads(lines[0])
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["root_code"], "installed")
                self.assertEqual(res["snapshot_consistency"], "converged")
                self.assertEqual(res["verification_layers"], "digest-matched; signatures skipped")
                self.assertIn("journal", res["components"])
                self.assertIn("desktop", res["components"])
                self.assertIn("tmux", res["components"])
            finally:
                server.stop()

    def test_bundle_capture_x86_and_aarch64(self):
        with ephemeral_keypair("test bundle capture") as (sec, pub, pin):
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

                # 1. capture on x86_64 -> cli, desktop, tmux
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "capture", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Capture failed: {proc.stderr}")
                res = json.loads(proc.stdout.strip())
                self.assertIn("cli", res["components"])
                self.assertIn("desktop", res["components"])
                self.assertIn("tmux", res["components"])

                # 2. capture on aarch64 -> cli, tmux + notice
                env = os.environ.copy()
                env["SOLSTONE_TEST_HOST_ARCH"] = "aarch64"

                prefix_aarch64 = self.work_dir / "prefix_aarch64"
                prefix_aarch64.mkdir(parents=True, exist_ok=True)

                proc2 = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "capture", "--prefix", str(prefix_aarch64), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc2.returncode, 0, f"Capture on aarch64 failed: {proc2.stderr}")
                res2 = json.loads(proc2.stdout.strip())
                self.assertIn("cli", res2["components"])
                self.assertIn("tmux", res2["components"])
                self.assertNotIn("desktop", res2["components"])
                self.assertTrue(len(res2["notices"]) > 0)
                self.assertIn("Desktop is not available", res2["notices"][0])
            finally:
                server.stop()

    def test_piped_non_tty_no_selection_defaults_to_journal(self):
        with ephemeral_keypair("test piped non tty") as (sec, pub, pin):
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

                # Run with empty stdin (not a TTY) and no --components
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--prefix", str(self.prefix), "--no-start", "--json"],
                    input="",
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["root_code"], "installed")
                self.assertEqual(set(res["components"]), {"journal"})
            finally:
                server.stop()

    def test_desktop_aarch64_manifest_capability_refusal(self):
        with ephemeral_keypair("test desktop aarch64") as (sec, pub, pin):
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
                env["SOLSTONE_TEST_HOST_ARCH"] = "aarch64"

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "desktop", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertNotEqual(proc.returncode, 0)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "refusal")
                self.assertEqual(res["root_code"], "desktop-aarch64")
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
