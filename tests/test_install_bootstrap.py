# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Family 6: Journal bootstrap revision checks and v2 delegation."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_test_release_server, write_path_stub
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix space"
        self.prefix.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_recovery_verifies_bootstrap_and_retains_policy(self):
        with ephemeral_keypair("native recovery") as (sec, pub, pin):
            server, root = setup_test_release_server(self.work_dir, sec, pin)
            try:
                installer = build_installer(REPO_ROOT, self.work_dir / "install.sh", platform_pub_path=pub, platform_key_id=pin.key_id, origin=server.origin)
                data = self.work_dir / "data"
                args_log = self.work_dir / "args"
                env = {**os.environ, "HOME": str(self.work_dir / "home"), "XDG_DATA_HOME": str(data), "SOLSTONE_BOOTSTRAP_ARGS_LOG": str(args_log)}
                self.prefix = self.work_dir / "owner's install"
                args = [str(installer), "--components", "cli", "--prefix", str(self.prefix), "--no-start", "--no-path", "--skip-signature", "--json"]
                failed = subprocess.run(args, env={**env, "SOLSTONE_BOOTSTRAP_FAIL": "1"}, capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0)
                message = json.loads(failed.stdout)["message"]
                self.assertIn("native setup may be incomplete", message)
                command = message.split("native recovery: ", 1)[1]
                recovered = subprocess.run(["sh", "-c", command], env=env, cwd=self.work_dir, capture_output=True, text=True)
                self.assertEqual(recovered.returncode, 0, recovered.stderr + recovered.stdout)
                flags = args_log.read_text().splitlines()
                for flag in ("--no-start", "--no-path", "--upgrade"):
                    self.assertIn(flag, flags)
                self.assertIn(str(self.prefix), flags)
                self.assertFalse((data / "solstone/install.conf").exists())
                self.assertTrue((self.prefix / "install-receipt").exists())
                # Recovery cannot execute changed bootstrap bytes at the same URL.
                native = root / "solstone-journal/release/2.0.6/solstone-journal-2.0.6-install.sh"
                native.write_bytes(native.read_bytes() + b"\n# changed after publication\n")
                args_log.unlink()
                bad = subprocess.run(["sh", "-c", command], env=env, cwd=self.work_dir, capture_output=True, text=True)
                self.assertNotEqual(bad.returncode, 0)
                self.assertFalse(args_log.exists())
            finally:
                server.stop()

    def test_native_success_survives_platform_receipt_failure(self):
        with ephemeral_keypair("native receipt failure") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin)
            try:
                installer = build_installer(REPO_ROOT, self.work_dir / "install.sh", platform_pub_path=pub, platform_key_id=pin.key_id, origin=server.origin)
                bin_dir = self.work_dir / "bin"
                write_path_stub(bin_dir, "mv", 'for arg do case "$arg" in */solstone/install.conf) exit 8 ;; esac; done\nexec /usr/bin/mv "$@"\n')
                env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "XDG_DATA_HOME": str(self.work_dir / "data")}
                result = subprocess.run([str(installer), "--components", "cli", "--prefix", str(self.prefix), "--no-start", "--no-path", "--skip-signature", "--json"], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                report = json.loads(result.stdout)
                self.assertEqual(report["root_code"], "receipt-write-failed")
                self.assertIn("native installation completed", report["message"])
                self.assertIn("sha256sum -c -", report["message"])
                self.assertTrue((self.prefix / "current/bin/journal").exists())
                self.assertTrue((self.prefix / "install-receipt").exists())
            finally:
                server.stop()

    def test_v1_bootstrap_refuses(self):
        # Server with v1 bootstrap (BOOTSTRAP_REVISION=1)
        with ephemeral_keypair("test v1 boot") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin, bootstrap_revision=1)
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
                    [str(installer), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "refusal")
                self.assertEqual(res["root_code"], "v1-bootstrap-unsupported")
            finally:
                server.stop()

    def test_v2_bootstrap_journal_and_cli_delegation(self):
        # Server with v2 bootstrap (BOOTSTRAP_REVISION=2)
        with ephemeral_keypair("test v2 boot") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin, bootstrap_revision=2)
            try:
                installer = self.work_dir / "install.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                )

                args_log = self.work_dir / "bootstrap-args.log"
                env = {**os.environ, "SOLSTONE_BOOTSTRAP_ARGS_LOG": str(args_log)}

                def expected_args(role: str, *, upgrade: bool = False, prefix: Path | None = None) -> list[str]:
                    args = [
                        "--role", role,
                        "--prefix", str(prefix or self.prefix),
                        "--origin", server.origin,
                        "--lane", "release",
                        "--version", "2.0.6",
                        "--no-start",
                        "--no-path",
                        "--skip-signature",
                    ]
                    if upgrade:
                        args.append("--upgrade")
                    return args

                # 1. Install journal role
                proc_j = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--no-start", "--no-path", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc_j.returncode, 0, f"Failed: {proc_j.stderr}")
                res_j = json.loads(proc_j.stdout.strip())
                self.assertEqual(res_j["status"], "success")
                self.assertEqual(res_j["components"]["journal"]["role"], "journal")
                self.assertEqual(args_log.read_text(encoding="utf-8").splitlines(), expected_args("journal"))

                proc_upgrade = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--prefix", str(self.prefix), "--no-start", "--no-path", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc_upgrade.returncode, 0, proc_upgrade.stderr + proc_upgrade.stdout)
                self.assertEqual(args_log.read_text(encoding="utf-8").splitlines(), expected_args("journal", upgrade=True))

                # Verify installed binary
                j_bin = self.prefix / "current" / "bin" / "journal"
                self.assertTrue(j_bin.is_file())
                proc_out = subprocess.run([str(j_bin)], capture_output=True, text=True)
                self.assertIn("journal 2.0.6", proc_out.stdout)

                # 2. Install cli role
                cli_prefix = self.work_dir / "cli prefix"
                proc_c = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "cli", "--prefix", str(cli_prefix), "--no-start", "--no-path", "--json"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc_c.returncode, 0, f"Failed: {proc_c.stderr}")
                res_c = json.loads(proc_c.stdout.strip())
                self.assertEqual(res_c["status"], "success")
                self.assertEqual(res_c["components"]["cli"]["role"], "cli")
                self.assertEqual(args_log.read_text(encoding="utf-8").splitlines(), expected_args("cli", prefix=cli_prefix))

                proc_out_cli = subprocess.run([str(cli_prefix / "current" / "bin" / "journal")], capture_output=True, text=True)
                self.assertIn("cli 2.0.6", proc_out_cli.stdout)
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
