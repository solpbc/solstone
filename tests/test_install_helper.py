# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Families 10 & 11: Privileged helper opcode protocols and package phase recovery."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_fake_sudo, setup_test_release_server
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER_SCRIPT = REPO_ROOT / "helpers" / "solstone-pkg-helper.sh"


class TestInstallHelper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.lock_dir = self.work_dir / "lock"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.etc_root = self.work_dir / "etc"
        self.etc_root.mkdir(parents=True, exist_ok=True)
        self.fake_pkg_db = self.work_dir / "fake_db"
        self.fake_pkg_db.mkdir(parents=True, exist_ok=True)
        self.bin_dir = self.work_dir / "bin"
        setup_fake_sudo(self.bin_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def run_helper(self, stdin_text: str, extra_flags: list[str] | None = None) -> subprocess.CompletedProcess:
        flags = [
            "--lock-dir", str(self.lock_dir),
            "--etc-root", str(self.etc_root),
            "--fake-pkg-db", str(self.fake_pkg_db),
        ]
        if extra_flags:
            flags.extend(extra_flags)
        cmd = [str(HELPER_SCRIPT)] + flags
        return subprocess.run(cmd, input=stdin_text, capture_output=True, text=True)

    def test_helper_ping_pong(self):
        proc = self.run_helper("PING\n")
        self.assertEqual(proc.returncode, 0, f"Helper failed: {proc.stderr}")
        self.assertEqual(proc.stdout.strip(), "PONG")

    def test_helper_unknown_opcode_refused(self):
        proc = self.run_helper("FOOBAR\n")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ERROR:unknown-opcode", proc.stderr)

    def test_helper_metacharacters_refused_before_mutation(self):
        canary = self.work_dir / "canary.txt"
        proc = self.run_helper(f"INSTALL_PKG deb /tmp/pkg.deb; touch {canary}\n")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ERROR:metacharacters-rejected", proc.stderr)
        self.assertFalse(canary.exists(), "Command injection executed!")

    def test_helper_newline_and_extra_operands_refused(self):
        proc = self.run_helper("PING extra operand\n")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ERROR:invalid-ping-operands", proc.stderr)

    def test_helper_relative_and_escaped_path_refused(self):
        proc = self.run_helper("INSTALL_PKG deb relative/path.deb\n")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ERROR:path-not-absolute", proc.stderr)

        proc2 = self.run_helper("INSTALL_PKG deb /tmp/../etc/passwd\n")
        self.assertNotEqual(proc2.returncode, 0)
        self.assertIn("ERROR:path-unsafe", proc2.stderr)

    def test_helper_write_etc_receipt_and_lock_preservation(self):
        body = "[solstone]\nphase=complete\n"
        body_len = len(body.encode("utf-8"))
        proc = self.run_helper(f"WRITE_ETC_RECEIPT {body_len}\n{body}")
        self.assertEqual(proc.returncode, 0, f"Write receipt failed: {proc.stderr}")

        receipt_file = self.etc_root / "solstone" / "install.conf"
        self.assertTrue(receipt_file.is_file())
        self.assertEqual(receipt_file.read_text(encoding="utf-8"), body)

        # Ensure lock file is intact and not unlinked
        lock_file = self.lock_dir / "lock"
        self.assertTrue(lock_file.exists(), "Lock file inode was deleted")

    def test_family_11_package_crash_recovery_and_downgrade(self):
        with ephemeral_keypair("test pkg helper family 11") as (sec, pub, pin):
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
                env["PATH"] = f"{self.bin_dir}:{env['PATH']}"
                env["SOLSTONE_LOCK_DIR"] = str(self.lock_dir)
                env["SOLSTONE_ETC_ROOT"] = str(self.etc_root)
                env["SOLSTONE_FAKE_PKG_DB"] = str(self.fake_pkg_db)
                env["SOLSTONE_HELPER"] = str(HELPER_SCRIPT)

                # 1. First installation via --route package
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--route", "package", "--components", "journal", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc.returncode, 0, f"Package install failed: {proc.stderr}\n{proc.stdout}")
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "success")

                # Verify install.log has 1 entry
                log_file = self.fake_pkg_db / "install.log"
                self.assertTrue(log_file.is_file())
                installs = [l for l in log_file.read_text().splitlines() if l.strip()]
                self.assertEqual(len(installs), 1)

                # 2. Simulate death at payload phase: modify receipt phase to 'payload'
                receipt_file = self.etc_root / "solstone" / "install.conf"
                content = receipt_file.read_text(encoding="utf-8")
                receipt_file.write_text(content.replace("phase=complete", "phase=payload"), encoding="utf-8")

                # Rerun installer --route package --components journal
                proc2 = subprocess.run(
                    [str(installer), "--skip-signature", "--route", "package", "--components", "journal", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc2.returncode, 0, f"Recovery install failed: {proc2.stderr}\n{proc2.stdout}")

                # Verify recovery completed setup without re-installing package
                content_after = receipt_file.read_text(encoding="utf-8")
                self.assertIn("phase=complete", content_after)
                installs_after = [l for l in log_file.read_text().splitlines() if l.strip()]
                self.assertEqual(len(installs_after), 1, f"Expected exactly 1 install line in install.log, got {len(installs_after)}")

                # 3. Test downgrade refusal: seed fake DB with higher version 3.0.0
                pkg_file = self.fake_pkg_db / "deb" / "solstone-journal"
                pkg_file.write_text("install ok installed 3.0.0\n", encoding="utf-8")

                proc3 = subprocess.run(
                    [str(installer), "--skip-signature", "--route", "package", "--components", "journal", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertNotEqual(proc3.returncode, 0)
                res3 = json.loads(proc3.stdout.strip())
                self.assertEqual(res3["status"], "refusal")
                self.assertEqual(res3["root_code"], "downgrade-route-unsupported")
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
