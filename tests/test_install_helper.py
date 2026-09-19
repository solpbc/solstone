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
from tests.install_test_helpers import (
    receipt_section,
    setup_fake_sudo,
    setup_package_launcher_spy,
    setup_test_release_server,
    write_path_stub,
)
from tools.fixture_builder import create_tiny_deb, create_tiny_synthetic_rpm
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

    def run_helper(
        self,
        stdin_text: str,
        extra_flags: list[str] | None = None,
        *,
        fake_db: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        flags = ["--lock-dir", str(self.lock_dir), "--etc-root", str(self.etc_root)]
        if fake_db:
            flags.extend(["--fake-pkg-db", str(self.fake_pkg_db)])
        if extra_flags:
            flags.extend(extra_flags)
        cmd = [str(HELPER_SCRIPT)] + flags
        return subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, env=env)

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

    def test_helper_query_protocol_and_fake_archive_identity(self):
        query_bin = self.work_dir / "query-bin"
        query_env = {**os.environ, "PATH": f"{query_bin}:{os.environ['PATH']}"}
        write_path_stub(
            query_bin,
            "dpkg-query",
            "printf 'install ok installed\\tsolstone-journal\\t2.0.6\\tamd64\\n'\n",
        )
        installed = self.run_helper("QUERY_PKG deb solstone-journal\n", fake_db=False, env=query_env)
        self.assertEqual(installed.returncode, 0)
        self.assertEqual(installed.stdout, "INSTALLED solstone-journal 2.0.6 amd64\n")

        write_path_stub(
            query_bin,
            "dpkg-query",
            "printf 'deinstall ok config-files\\tsolstone-journal\\t2.0.6\\tamd64\\n'\n",
        )
        config_files = self.run_helper("QUERY_PKG deb solstone-journal\n", fake_db=False, env=query_env)
        self.assertEqual(config_files.returncode, 0)
        self.assertEqual(config_files.stdout, "CONFIG_FILES solstone-journal 2.0.6 amd64\n")

        write_path_stub(query_bin, "dpkg-query", "printf 'hold ok installed\\tsolstone-journal\\t2.0.6\\tamd64\\n'\n")
        malformed = self.run_helper("QUERY_PKG deb solstone-journal\n", fake_db=False, env=query_env)
        self.assertNotEqual(malformed.returncode, 0)
        self.assertEqual(malformed.stdout, "ERROR:query-malformed\n")

        write_path_stub(query_bin, "dpkg-query", "exit 1\n")
        absent = self.run_helper("QUERY_PKG deb solstone-journal\n", fake_db=False, env=query_env)
        self.assertEqual(absent.returncode, 0)
        self.assertEqual(absent.stdout, "ABSENT\n")

        write_path_stub(query_bin, "dpkg-query", "exit 2\n")
        failed = self.run_helper("QUERY_PKG deb solstone-journal\n", fake_db=False, env=query_env)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(failed.stdout, "ERROR:query-failed\n")

        write_path_stub(
            query_bin,
            "rpm",
            "printf 'solstone-journal\\t2.0.6-1\\tx86_64\\n'\n",
        )
        rpm = self.run_helper("QUERY_PKG rpm solstone-journal\n", fake_db=False, env=query_env)
        self.assertEqual(rpm.returncode, 0)
        self.assertEqual(rpm.stdout, "INSTALLED solstone-journal 2.0.6-1 x86_64\n")

        archive = self.work_dir / "solstone-journal_2.0.6_amd64.deb"
        create_tiny_deb(archive, "solstone-journal", "2.0.6", "amd64", "journal", b"#!/bin/sh\nexit 0\n")
        fake = self.run_helper(f"INSTALL_PKG deb {archive}\n")
        self.assertEqual(fake.returncode, 0, fake.stderr)
        self.assertEqual(fake.stdout, "OK\n")
        record = self.fake_pkg_db / "deb" / "solstone-journal"
        self.assertEqual(record.read_text(encoding="utf-8"), "INSTALLED solstone-journal 2.0.6 amd64\n")

        epoch_archive = self.work_dir / "solstone-journal_epoch_amd64.deb"
        create_tiny_deb(
            epoch_archive,
            "solstone-journal",
            "1:2.0.6",
            "amd64",
            "journal",
            b"#!/bin/sh\nexit 0\n",
        )
        epoch_fake = self.run_helper(f"INSTALL_PKG deb {epoch_archive}\n")
        self.assertEqual(epoch_fake.returncode, 0, epoch_fake.stderr)
        self.assertEqual(epoch_fake.stdout, "OK\n")
        self.assertEqual(record.read_text(encoding="utf-8"), "INSTALLED solstone-journal 1:2.0.6 amd64\n")

        rpm_archive = self.work_dir / "solstone-journal-2.0.6-1.x86_64.rpm"
        create_tiny_synthetic_rpm(rpm_archive, "solstone-journal", "2.0.6", "x86_64", "journal", b"#!/bin/sh\nexit 0\n")
        fake_rpm = self.run_helper(f"INSTALL_PKG rpm {rpm_archive}\n")
        self.assertEqual(fake_rpm.returncode, 0, fake_rpm.stderr)
        self.assertEqual(fake_rpm.stdout, "OK\n")
        rpm_record = self.fake_pkg_db / "rpm" / "solstone-journal"
        self.assertEqual(rpm_record.read_text(encoding="utf-8"), "INSTALLED solstone-journal 2.0.6-1 x86_64\n")

        record.write_text("INSTALLED solstone-journal 2.0.6 arm64\n", encoding="utf-8")
        wrong_arch = self.run_helper("QUERY_PKG deb solstone-journal\n")
        self.assertEqual(wrong_arch.returncode, 0)
        self.assertEqual(wrong_arch.stdout, "INSTALLED solstone-journal 2.0.6 arm64\n")

    def test_helper_missing_query_tool_and_real_install_stdout_isolated(self):
        minimal_bin = self.work_dir / "minimal-bin"
        for command in ("mkdir", "chmod", "flock"):
            write_path_stub(minimal_bin, command, f"exec /usr/bin/{command} \"$@\"\n")
        missing_env = {**os.environ, "PATH": str(minimal_bin)}
        missing = self.run_helper("QUERY_PKG deb solstone-journal\n", fake_db=False, env=missing_env)
        self.assertNotEqual(missing.returncode, 0)
        self.assertEqual(missing.stdout, "ERROR:query-unavailable\n")

        archive = self.work_dir / "solstone-journal_2.0.6_amd64.deb"
        create_tiny_deb(archive, "solstone-journal", "2.0.6", "amd64", "journal", b"#!/bin/sh\nexit 0\n")
        manager_bin = self.work_dir / "manager-bin"
        state_file = self.work_dir / "manager-state"
        write_path_stub(
            manager_bin,
            "dpkg",
            "printf 'ordinary package-manager warning\\n'\n"
            ": > \"$SOLSTONE_MANAGER_STATE\"\n"
            "exit 0\n",
        )
        write_path_stub(
            manager_bin,
            "dpkg-query",
            "if [ -f \"$SOLSTONE_MANAGER_STATE\" ]; then\n"
            "  printf 'install ok installed\\tsolstone-journal\\t2.0.6\\tamd64\\n'\n"
            "  exit 0\n"
            "fi\n"
            "exit 1\n",
        )
        manager_env = {
            **os.environ,
            "PATH": f"{manager_bin}:{os.environ['PATH']}",
            "SOLSTONE_MANAGER_STATE": str(state_file),
        }
        installed = self.run_helper(f"INSTALL_PKG deb {archive}\n", fake_db=False, env=manager_env)
        self.assertEqual(installed.returncode, 0, installed.stderr)
        self.assertEqual(installed.stdout, "OK\n")

    def test_helper_receipt_failures_preserve_existing_file(self):
        receipt_file = self.etc_root / "solstone" / "install.conf"
        receipt_file.parent.mkdir()
        receipt_file.write_bytes(b"old-receipt\n")

        truncated = self.run_helper("WRITE_ETC_RECEIPT 12\nshort")
        self.assertNotEqual(truncated.returncode, 0)
        self.assertEqual(truncated.stdout, "ERROR:receipt-truncated\n")
        self.assertEqual(receipt_file.read_bytes(), b"old-receipt\n")

        empty = self.run_helper("WRITE_ETC_RECEIPT 0\n\n")
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self.assertEqual(empty.stdout, "OK\n")
        self.assertTrue(receipt_file.is_file())
        self.assertEqual(receipt_file.read_bytes(), b"")

        receipt_file.unlink()
        receipt_file.mkdir()
        directory = self.run_helper("WRITE_ETC_RECEIPT 1\nx")
        self.assertNotEqual(directory.returncode, 0)
        self.assertEqual(directory.stdout, "ERROR:receipt-dest-invalid\n")
        self.assertTrue(receipt_file.is_dir())

        receipt_file.rmdir()
        receipt_file.symlink_to(self.work_dir / "other")
        symlink = self.run_helper("WRITE_ETC_RECEIPT 1\nx")
        self.assertNotEqual(symlink.returncode, 0)
        self.assertEqual(symlink.stdout, "ERROR:receipt-dest-invalid\n")

        receipt_file.unlink()
        receipt_file.mkdir()
        unreadable = self.run_helper("READ_ETC_RECEIPT\n")
        self.assertNotEqual(unreadable.returncode, 0)
        self.assertEqual(unreadable.stdout, "ERROR:receipt-read-failed\n")

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
                fake_root = self.work_dir / "fake-root"
                launcher = setup_package_launcher_spy(self.bin_dir)
                setup_log = self.work_dir / "setup.log"
                env["SOLSTONE_FAKE_ROOT"] = str(fake_root)
                env["SOLSTONE_FAKE_JOURNAL_LAUNCHER"] = str(launcher)
                env["SOLSTONE_PACKAGE_SETUP_LOG"] = str(setup_log)

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

                # 2. Simulate death at package payload phase.
                receipt_file = self.etc_root / "solstone" / "install.conf"
                content = receipt_file.read_text(encoding="utf-8")
                journal_section = receipt_section(content.encode("utf-8"), "component:journal").decode("utf-8")
                payload_section = journal_section.replace("phase=complete", "phase=payload").replace("status=installed", "status=payload")
                receipt_file.write_text(content.replace(journal_section, payload_section), encoding="utf-8")

                # Rerun installer --route package --components journal
                proc2 = subprocess.run(
                    [str(installer), "--skip-signature", "--route", "package", "--components", "journal", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc2.returncode, 0, f"Recovery install failed: {proc2.stderr}\n{proc2.stdout}")

                # Verify recovery completed setup without re-installing package.
                content_after = receipt_file.read_text(encoding="utf-8")
                self.assertIn("phase=complete", receipt_section(content_after.encode("utf-8"), "component:journal").decode("utf-8"))
                installs_after = [l for l in log_file.read_text().splitlines() if l.strip()]
                self.assertEqual(len(installs_after), 1, f"Expected exactly 1 install line in install.log, got {len(installs_after)}")
                setup_lines = setup_log.read_text(encoding="utf-8").splitlines()
                self.assertEqual(sum(line == "arg=setup" for line in setup_lines), 2)

                # 3. Test exact-version downgrade refusal with a matching owned record.
                pkg_file = self.fake_pkg_db / "deb" / "solstone-journal"
                pkg_file.write_text("INSTALLED solstone-journal 3.0.0 amd64\n", encoding="utf-8")
                content = receipt_file.read_text(encoding="utf-8")
                journal_section = receipt_section(content.encode("utf-8"), "component:journal").decode("utf-8")
                owned_newer = journal_section.replace("package_version=2.0.6", "package_version=3.0.0")
                receipt_file.write_text(content.replace(journal_section, owned_newer), encoding="utf-8")

                receipt_before = receipt_file.read_bytes()
                db_before = pkg_file.read_bytes()
                installs_before = log_file.read_bytes()
                server.request_paths.clear()
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
                self.assertEqual(receipt_file.read_bytes(), receipt_before)
                self.assertEqual(pkg_file.read_bytes(), db_before)
                self.assertEqual(log_file.read_bytes(), installs_before)
                self.assertFalse(any(path.endswith((".deb", ".rpm")) for path in server.request_paths))
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
