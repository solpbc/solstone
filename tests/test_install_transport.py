# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Families 4 & 5: Transport security, URL validation, and installer revision floors."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_test_release_server
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallTransport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"
        self.prefix.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_insecure_remote_http_refuses(self):
        with ephemeral_keypair("test transport") as (sec, pub, pin):
            installer = self.work_dir / "install.sh"
            build_installer(
                repo_root=REPO_ROOT,
                output_path=installer,
                platform_pub_path=pub,
                platform_key_id=pin.key_id,
            )

            proc = subprocess.run(
                [str(installer), "--origin", "http://example.com", "--components", "journal", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            res = json.loads(proc.stdout.strip())
            self.assertEqual(res["status"], "refusal")
            self.assertEqual(res["root_code"], "url-insecure")

    def test_unapproved_https_host_refuses(self):
        with ephemeral_keypair("test transport https") as (sec, pub, pin):
            installer = self.work_dir / "install.sh"
            build_installer(
                repo_root=REPO_ROOT,
                output_path=installer,
                platform_pub_path=pub,
                platform_key_id=pin.key_id,
            )

            proc = subprocess.run(
                [str(installer), "--origin", "https://unauthorized.solstone.app", "--components", "journal", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            res = json.loads(proc.stdout.strip())
            self.assertEqual(res["status"], "refusal")
            self.assertEqual(res["root_code"], "url-insecure")

    def test_url_userinfo_refuses(self):
        with ephemeral_keypair("test transport userinfo") as (sec, pub, pin):
            installer = self.work_dir / "install.sh"
            build_installer(
                repo_root=REPO_ROOT,
                output_path=installer,
                platform_pub_path=pub,
                platform_key_id=pin.key_id,
            )

            proc = subprocess.run(
                [str(installer), "--origin", "http://user:pass@127.0.0.1:8080", "--components", "journal", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            res = json.loads(proc.stdout.strip())
            self.assertEqual(res["status"], "refusal")
            self.assertEqual(res["root_code"], "url-userinfo")

    def test_url_fragment_refuses(self):
        with ephemeral_keypair("test transport frag") as (sec, pub, pin):
            installer = self.work_dir / "install.sh"
            build_installer(
                repo_root=REPO_ROOT,
                output_path=installer,
                platform_pub_path=pub,
                platform_key_id=pin.key_id,
            )

            proc = subprocess.run(
                [str(installer), "--origin", "http://127.0.0.1:8080/path#frag", "--components", "journal", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            res = json.loads(proc.stdout.strip())
            self.assertEqual(res["status"], "refusal")
            self.assertEqual(res["root_code"], "url-fragment")

    def test_installer_revision_floors(self):
        with ephemeral_keypair("test rev floor") as (sec, pub, pin):
            # Manifest requires minimum_installer_revision = 2
            server, _ = setup_test_release_server(self.work_dir, sec, pin, min_installer_revision=2)
            try:
                # 1. Installer with revision 1 against min 2 -> Refuses revision-too-old
                installer_v1 = self.work_dir / "install_v1.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer_v1,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                    override_installer_revision=1,
                )

                proc = subprocess.run(
                    [str(installer_v1), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "refusal")
                self.assertEqual(res["root_code"], "revision-too-old")

                # 2. Installer with revision 2 against min 2 -> Allowed
                installer_v2 = self.work_dir / "install_v2.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer_v2,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                    override_installer_revision=2,
                )

                proc2 = subprocess.run(
                    [str(installer_v2), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc2.returncode, 0, f"Failed: {proc2.stderr}")
                res2 = json.loads(proc2.stdout.strip())
                self.assertEqual(res2["status"], "success")

                # 3. Installer with revision 3 against min 2 -> Allowed
                installer_v3 = self.work_dir / "install_v3.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer_v3,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                    override_installer_revision=3,
                )

                proc3 = subprocess.run(
                    [str(installer_v3), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc3.returncode, 0, f"Failed: {proc3.stderr}")
                res3 = json.loads(proc3.stdout.strip())
                self.assertEqual(res3["status"], "success")
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
