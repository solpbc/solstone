# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Family 6: Journal bootstrap revision checks and v2 delegation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.install_test_helpers import HermeticInstallerTestCase

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_test_release_server, write_path_stub
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallBootstrap(HermeticInstallerTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix space"
        self.prefix.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _release(self, label):
        keys = ephemeral_keypair(label)
        sec, pub, pin = keys.__enter__()
        self.addCleanup(keys.__exit__, None, None, None)
        server, root = setup_test_release_server(self.work_dir, sec, pin)
        self.addCleanup(server.stop)
        installer = build_installer(REPO_ROOT, self.work_dir / "install.sh", platform_pub_path=pub, platform_key_id=pin.key_id, origin=server.origin)
        env = {**os.environ, "HOME": str(self.work_dir / "home"), "XDG_DATA_HOME": str(self.work_dir / "data")}
        return installer, root, env

    def _journal_section(self) -> str:
        receipt = (self.work_dir / "data/solstone/install.conf").read_text()
        start = receipt.index("[component:journal]\n")
        rest = receipt[start + 1:]
        cut = rest.find("\n[")
        return receipt[start:] if cut == -1 else receipt[start:start + 1 + cut + 1]

    def test_a_failed_journal_install_resumes_with_the_same_command(self):
        installer, _, env = self._release("resume")
        args = [str(installer), "--components", "journal", "--prefix", str(self.prefix), "--no-start", "--skip-signature", "--json"]
        for fail_env, left in (({"SOLSTONE_BOOTSTRAP_FAIL": "1"}, "tree"), ({"SOLSTONE_BOOTSTRAP_FAIL_EARLY": "1"}, "nothing")):
            with self.subTest(left=left):
                shutil.rmtree(self.prefix, ignore_errors=True)
                shutil.rmtree(self.work_dir / "data", ignore_errors=True)
                failed = subprocess.run(args, env={**env, **fail_env}, capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0)
                message = json.loads(failed.stdout)["message"]
                if left == "tree":
                    self.assertIn("Once the problem above is fixed, run the same install.sh command again, and it will finish installing the journal.", message)
                else:
                    self.assertIn("The journal was not installed, so once the problem above is fixed, run the same install.sh command again.", message)
                self.assertNotIn("sha256sum", message)
                self.assertIn("status=pending\n", self._journal_section())
                resumed = subprocess.run(args, env=env, capture_output=True, text=True)
                self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
                self.assertIn("status=installed\n", self._journal_section())
                self.assertNotIn("prior_version", self._journal_section())

    def test_an_interrupted_journal_install_says_to_run_the_same_command(self):
        installer, _, env = self._release("interrupt")
        args = [str(installer), "--components", "journal", "--prefix", str(self.prefix), "--no-start", "--skip-signature", "--json"]
        interrupted = subprocess.run(args, env={**env, "SOLSTONE_BOOTSTRAP_INTERRUPT": "1"}, capture_output=True, text=True)
        self.assertNotEqual(interrupted.returncode, 0)
        report = json.loads(interrupted.stdout)
        self.assertEqual(report["root_code"], "interrupted")
        self.assertTrue(report["message"].startswith("installation was interrupted. The journal was not installed, so run the same install.sh command again."), report["message"])
        self.assertNotIn("preserve", report["message"])
        resumed = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)

    def test_a_failed_update_resumes_and_keeps_the_prior_version_until_it_finishes(self):
        installer, _, env = self._release("resume update")
        args = [str(installer), "--components", "journal", "--prefix", str(self.prefix), "--no-start", "--skip-signature", "--json"]
        first = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        # Record an older journal consistently, so the next run is an update.
        receipt = self.work_dir / "data/solstone/install.conf"
        receipt.write_text(receipt.read_text().replace("version=2.0.6", "version=2.0.5"))
        native = self.prefix / "install-receipt"
        native.write_text(native.read_text().replace("journal_version=2.0.6", "journal_version=2.0.5"))
        (self.prefix / "current/bin/journal").write_text("#!/bin/sh\necho journal 2.0.5\n")
        failed = subprocess.run(args, env={**env, "SOLSTONE_BOOTSTRAP_FAIL_EARLY": "1"}, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        section = self._journal_section()
        self.assertIn("version=2.0.6\nprior_version=2.0.5\nstatus=pending\n", section)
        resumed = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
        self.assertIn("version=2.0.6\nstatus=installed\n", self._journal_section())

    def test_unfinished_model_installation_names_the_journal_command_then_resumes(self):
        installer, _, env = self._release("models unfinished")
        args = [str(installer), "--components", "journal", "--prefix", str(self.prefix), "--no-start", "--skip-signature", "--json"]
        models = subprocess.run(args, env={**env, "SOLSTONE_BOOTSTRAP_FAIL": "1", "SOLSTONE_BOOTSTRAP_FAIL_STATUS": "80"}, capture_output=True, text=True)
        self.assertEqual(models.returncode, 1, models.stderr + models.stdout)
        report = json.loads(models.stdout)
        self.assertEqual(report["root_code"], "setup-failed")
        journal = f"{self.prefix}/current/bin/journal"
        self.assertIn(f"model installation did not finish. Nothing in your journal was removed. To finish it, run: {journal} install-models --variant auto, then run the same install.sh command again", report["message"])
        other = subprocess.run(args, env={**env, "SOLSTONE_BOOTSTRAP_FAIL": "1"}, capture_output=True, text=True)
        self.assertNotIn("model installation", json.loads(other.stdout)["message"])
        resumed = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)

    def test_a_journal_without_this_installers_record_gets_a_runnable_recovery(self):
        installer, root, env = self._release("stranded recovery")
        args_log = self.work_dir / "args"
        env = {**env, "SOLSTONE_BOOTSTRAP_ARGS_LOG": str(args_log)}
        self.prefix = self.work_dir / "owner's install"
        args = [str(installer), "--components", "cli", "--prefix", str(self.prefix), "--no-start", "--no-path", "--skip-signature", "--json"]
        failed = subprocess.run(args, env={**env, "SOLSTONE_BOOTSTRAP_FAIL": "1"}, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        # A journal left by an installer before revision 6 has no pending record.
        (self.work_dir / "data/solstone/install.conf").unlink()
        rerun = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(rerun.returncode, 1, rerun.stderr + rerun.stdout)
        report = json.loads(rerun.stdout)
        self.assertEqual(report["root_code"], "ownership-unknown")
        command = report["message"].split("To finish or update that journal, run: ", 1)[1]
        self.assertNotIn("\n", command, "a recovery command an owner copies must be one line")
        args_log.unlink()
        recovered = subprocess.run(["sh", "-c", command], env=env, cwd=self.work_dir, capture_output=True, text=True)
        self.assertEqual(recovered.returncode, 0, recovered.stderr + recovered.stdout)
        flags = args_log.read_text().splitlines()
        for flag in ("--no-start", "--no-path", "--upgrade"):
            self.assertIn(flag, flags)
        self.assertIn(str(self.prefix), flags)
        self.assertIn("cli", flags)
        # Recovery cannot execute changed bootstrap bytes at the same URL.
        native = root / "solstone-journal/release/2.0.6/solstone-journal-2.0.6-install.sh"
        native.write_bytes(native.read_bytes() + b"\n# changed after publication\n")
        args_log.unlink()
        bad = subprocess.run(["sh", "-c", command], env=env, cwd=self.work_dir, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertFalse(args_log.exists())

    def test_native_success_survives_platform_receipt_failure(self):
        installer, _, env = self._release("native receipt failure")
        bin_dir = self.work_dir / "bin"
        write_path_stub(bin_dir, "mv", 'for arg do case "$arg" in */solstone/install.conf.tmp.*) exit 8 ;; esac; done\nexec /usr/bin/mv "$@"\n')
        args = [str(installer), "--components", "cli", "--prefix", str(self.prefix), "--no-start", "--no-path", "--skip-signature", "--json"]
        result = subprocess.run(args, env={**env, "PATH": f"{bin_dir}:{os.environ['PATH']}"}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertEqual(report["root_code"], "receipt-write-failed")
        self.assertIn("the journal installed and is ready to use, but this installer could not record that it did. Once the problem above is fixed, run the same install.sh command again to record it.", report["message"])
        self.assertTrue((self.prefix / "current/bin/journal").exists())
        recorded = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(recorded.returncode, 0, recorded.stderr + recorded.stdout)

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
