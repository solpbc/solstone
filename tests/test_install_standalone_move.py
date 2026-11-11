# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""A journal from the journal's own installer is taken over where it is."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.install_test_helpers import HermeticInstallerTestCase, receipt_section, setup_test_release_server, snapshot_paths
from solstone_platform.sign import ephemeral_keypair
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION = "2.0.6"
DIGEST = hashlib.sha256(b"standalone journal tree").hexdigest()


class TestInstallStandaloneMove(HermeticInstallerTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.home = self.work_dir / "home"
        self.home.mkdir()
        self.data = self.work_dir / "data"
        self.standalone = self.home / ".local" / "solstone-journal"
        self.args_log = self.work_dir / "bootstrap-args"
        keys = ephemeral_keypair("standalone move")
        sec, pub, pin = keys.__enter__()
        self.addCleanup(keys.__exit__, None, None, None)
        self.server, self.root = setup_test_release_server(self.work_dir, sec, pin)
        self.addCleanup(self.server.stop)
        self.installer = build_installer(
            REPO_ROOT, self.work_dir / "install.sh", platform_pub_path=pub, platform_key_id=pin.key_id,
            origin=self.server.origin, native_test_pins=self.server.native_pins,
        )
        self.env = {
            **os.environ, "HOME": str(self.home), "XDG_DATA_HOME": str(self.data),
            "SOLSTONE_BOOTSTRAP_ARGS_LOG": str(self.args_log),
        }
        # The owner's journal: the takeover never touches it.
        self.journal = self.home / "journal"
        (self.journal / "20261101").mkdir(parents=True)
        (self.journal / "20261101" / "note.md").write_bytes(b"owner bytes\n")

    def tearDown(self):
        self.tmp.cleanup()

    def make_standalone(self, *, role="journal", policy="start", setup="complete", signature="verified",
                        lane="release", origin=None, path_block=True, **overrides):
        entry = f"{VERSION}-{DIGEST[:12]}"
        dest = self.standalone / "versions" / entry
        (dest / "bin").mkdir(parents=True)
        (dest / ".release").write_text(f"product=solstone-journal\nversion={VERSION}\n", encoding="utf-8")
        (dest / ".archive-sha256").write_text(DIGEST + "\n", encoding="utf-8")
        launcher = dest / "bin" / "journal"
        launcher.write_text(f"#!/bin/sh\necho journal {VERSION}\n", encoding="utf-8")
        launcher.chmod(0o755)
        (self.standalone / "current").symlink_to(f"versions/{entry}")
        fields = {
            "schema_version": "1", "journal_version": VERSION, "lane": lane,
            "origin": origin or self.server.origin, "architecture": "x86_64", "installer_revision": "1",
            "bootstrap_revision": "4", "route": "tree", "signature_verification": signature,
            "role": role, "journal_state": "present", "service_policy": policy, "setup_status": setup,
        }
        fields.update(overrides)
        receipt = "".join(f"{key}={value}\n" for key, value in fields.items() if value is not None)
        (self.standalone / "install-receipt").write_text(receipt, encoding="utf-8")
        if path_block:
            (self.home / ".profile").write_text(
                "# owner line\n# BEGIN solstone-journal PATH\n"
                f'PATH="{self.standalone}/current/bin${{PATH:+:$PATH}}"\nexport PATH\n'
                "# END solstone-journal PATH\n",
                encoding="utf-8",
            )

    @property
    def receipt(self) -> Path:
        return self.data / "solstone" / "install.conf"

    def run_installer(self, *args, extra_env=None):
        return subprocess.run(
            [str(self.installer), *args, "--json"], capture_output=True, text=True,
            env={**self.env, **(extra_env or {})},
        )

    def bootstrap_args(self) -> list[str]:
        return self.args_log.read_text(encoding="utf-8").splitlines() if self.args_log.exists() else []

    def assert_journal_untouched(self):
        self.assertEqual((self.journal / "20261101" / "note.md").read_bytes(), b"owner bytes\n")

    def test_a_standalone_journal_is_taken_over_where_it_is(self):
        for args in (("--components", "journal"), ("--upgrade",)):
            with self.subTest(args=args):
                shutil.rmtree(self.standalone, ignore_errors=True)
                shutil.rmtree(self.data, ignore_errors=True)
                self.make_standalone()
                result = self.run_installer(*args)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertIn(f"your journal now updates with this installer, from where it is: {self.standalone}", report["notices"])
                bootstrap = self.bootstrap_args()
                self.assertEqual(bootstrap[bootstrap.index("--prefix") + 1], str(self.standalone))
                self.assertEqual(bootstrap[bootstrap.index("--role") + 1], "journal")
                self.assertIn("--upgrade", bootstrap)
                self.assertNotIn("--no-start", bootstrap)
                self.assertNotIn("--no-path", bootstrap, "the existing PATH block is kept and rewritten in place")
                receipt = self.receipt.read_bytes()
                self.assertIn(f"prefix={self.home / '.local'}\n".encode(), receipt_section(receipt, "solstone"))
                section = receipt_section(receipt, "component:journal")
                self.assertIn(f"prefix={self.standalone}\n".encode(), section)
                self.assertIn(b"status=installed\n", section)
                self.assertFalse((self.home / ".local" / "current").exists(), "nothing is installed beside it")
                self.assert_journal_untouched()

    def test_service_policy_and_missing_path_block_are_kept(self):
        self.make_standalone(policy="no-start", path_block=False)
        result = self.run_installer("--components", "journal")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        bootstrap = self.bootstrap_args()
        self.assertIn("--no-start", bootstrap)
        self.assertIn("--no-path", bootstrap)
        self.assertFalse((self.home / ".profile").exists())

    def test_a_cli_install_is_taken_over_as_cli_and_a_role_change_refuses(self):
        self.make_standalone(role="cli", policy="none", setup="not-applicable")
        before = snapshot_paths(self.home)
        wrong = self.run_installer("--components", "journal")
        self.assertEqual(wrong.returncode, 1, wrong.stdout + wrong.stderr)
        self.assertEqual(json.loads(wrong.stdout)["root_code"], "role-conflict")
        self.assertEqual(snapshot_paths(self.home), before)
        self.assertFalse(self.receipt.exists())
        result = self.run_installer("--upgrade")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"prefix={self.standalone}\n".encode(), receipt_section(self.receipt.read_bytes(), "component:cli"))

    def test_a_failed_takeover_leaves_the_journal_with_its_own_installer(self):
        self.make_standalone()
        # An earlier desktop install already wrote this installer's receipt.
        self.receipt.parent.mkdir(parents=True)
        prior = (
            f"[solstone]\nschema_version=1\nplatform_version=2.0.0\nlane=release\norigin={self.server.origin}\n"
            f"route=tree\nprefix={self.home / '.local'}\narch=x86_64\nverification=minisign\ninstaller_revision=1\nphase=complete\n"
        ).encode()
        self.receipt.write_bytes(prior)
        before = snapshot_paths(self.standalone)
        for fault in ("SOLSTONE_BOOTSTRAP_FAIL_EARLY", "SOLSTONE_BOOTSTRAP_INTERRUPT"):
            with self.subTest(fault=fault):
                failed = self.run_installer("--components", "journal", extra_env={fault: "1"})
                self.assertNotEqual(failed.returncode, 0)
                report = json.loads(failed.stdout)
                self.assertIn(f"This installer did not take over your journal, and it works as it did before, in {self.standalone}.", report["message"])
                self.assertEqual(self.receipt.read_bytes(), prior)
                self.assertEqual(snapshot_paths(self.standalone), before)
        self.receipt.unlink()
        failed = self.run_installer("--components", "journal", extra_env={"SOLSTONE_BOOTSTRAP_FAIL_EARLY": "1"})
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(self.receipt.exists())
        retried = self.run_installer("--components", "journal")
        self.assertEqual(retried.returncode, 0, retried.stdout + retried.stderr)
        self.assertIn(f"prefix={self.standalone}\n".encode(), receipt_section(self.receipt.read_bytes(), "component:journal"))

    def test_after_the_takeover_the_journal_updates_and_uninstalls_where_it_is(self):
        self.make_standalone()
        first = self.run_installer("--components", "journal")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        again = self.run_installer("--upgrade")
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertEqual(json.loads(again.stdout)["notices"], [])
        bootstrap = self.bootstrap_args()
        self.assertEqual(bootstrap[bootstrap.index("--prefix") + 1], str(self.standalone))
        removed = self.run_installer("--uninstall", "--components", "journal", "--no-start")
        self.assertEqual(removed.returncode, 0, removed.stdout + removed.stderr)
        self.assertFalse((self.standalone / "current").exists())
        self.assertFalse((self.standalone / "install-receipt").exists())
        self.assert_journal_untouched()

    def test_an_unidentified_standalone_journal_refuses_with_nothing_changed(self):
        cases = {
            "unverified": dict(signature="unknown"),
            "foreign origin": dict(origin="https://example.invalid"),
            "no role": dict(role=None),
            "bad setup state": dict(setup="broken"),
            "wrong version": dict(journal_version="2.0.7"),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                shutil.rmtree(self.standalone, ignore_errors=True)
                self.make_standalone(**overrides)
                before = snapshot_paths(self.home)
                result = self.run_installer("--components", "journal")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["root_code"], "standalone-unverified")
                self.assertIn("Nothing was changed. For help, see", report["message"])
                self.assertEqual(snapshot_paths(self.home), before)
                self.assertFalse(self.receipt.exists())
                self.assertEqual(self.bootstrap_args(), [])

    def test_a_tree_that_does_not_match_its_receipt_refuses(self):
        self.make_standalone()
        (self.standalone / "versions" / f"{VERSION}-{DIGEST[:12]}" / ".archive-sha256").write_text("0" * 64 + "\n")
        result = self.run_installer("--components", "journal")
        self.assertEqual(json.loads(result.stdout)["root_code"], "standalone-unverified")
        self.assertEqual(self.bootstrap_args(), [])

    def test_another_lane_needs_the_lane_named(self):
        self.make_standalone(lane="staging")
        before = snapshot_paths(self.home)
        result = self.run_installer("--components", "journal")
        report = json.loads(result.stdout)
        self.assertEqual(report["root_code"], "lane-mismatch")
        self.assertIn("--lane staging", report["message"])
        self.assertEqual(snapshot_paths(self.home), before)

    def test_a_release_whose_journal_cannot_find_it_keeps_the_old_answer(self):
        # Before the delivered journal reads a recorded journal prefix, a
        # takeover would leave solstone update unable to identify the journal.
        keys = ephemeral_keypair("standalone move without takeover")
        sec, pub, pin = keys.__enter__()
        self.addCleanup(keys.__exit__, None, None, None)
        older = self.work_dir / "older"
        older.mkdir()
        server, _ = setup_test_release_server(older, sec, pin, bootstrap_takeover=False)
        self.addCleanup(server.stop)
        installer = build_installer(
            REPO_ROOT, older / "install.sh", platform_pub_path=pub, platform_key_id=pin.key_id,
            origin=server.origin, native_test_pins=server.native_pins,
        )
        self.make_standalone(origin=server.origin)
        before = snapshot_paths(self.home)
        result = subprocess.run([str(installer), "--components", "journal", "--json"], capture_output=True, text=True, env=self.env)
        report = json.loads(result.stdout)
        self.assertEqual(report["root_code"], "standalone-install")
        self.assertIn("https://updates.solstone.app/solstone-journal/install.sh --upgrade", report["message"])
        self.assertEqual(snapshot_paths(self.home), before)
        self.assertFalse(self.receipt.exists())

    def test_preview_and_uninstall_change_nothing(self):
        self.make_standalone()
        before = snapshot_paths(self.home)
        preview = self.run_installer("--components", "journal", "--dry-run")
        self.assertEqual(preview.returncode, 0, preview.stdout + preview.stderr)
        removal = self.run_installer("--uninstall", "--components", "journal")
        self.assertEqual(json.loads(removal.stdout)["root_code"], "standalone-install")
        self.assertEqual(snapshot_paths(self.home), before)
        self.assertFalse(self.receipt.exists())

    def test_a_second_journal_beside_a_recorded_one_refuses(self):
        first = self.run_installer("--components", "journal", "--no-start")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.make_standalone(path_block=False)
        before = snapshot_paths(self.home)
        result = self.run_installer("--upgrade")
        self.assertEqual(json.loads(result.stdout)["root_code"], "standalone-install")
        self.assertEqual(snapshot_paths(self.home), before)

    def test_a_version_1_journal_command_does_not_block_a_new_install(self):
        bin_dir = self.home / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        launcher = bin_dir / "journal"
        for content, blocks in (
            ("#!/bin/bash\n# journal — managed by 'journal config'. Edits will be overwritten.\n# managed-version: 7\nexec /x \"$@\"\n", False),
            ("#!/home/owner/.local/share/uv/tools/solstone-journal/bin/python3\nimport sys\n", False),
            ("#!/bin/bash\n# managed-version: 8\nexec /x \"$@\"\n", True),
        ):
            with self.subTest(content=content.splitlines()[0:3]):
                shutil.rmtree(self.data, ignore_errors=True)
                for leftover in ("current", "versions", "install-receipt"):
                    path = self.home / ".local" / leftover
                    if path.is_symlink() or path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                launcher.write_text(content, encoding="utf-8")
                launcher.chmod(0o755)
                result = self.run_installer("--components", "journal", "--no-start")
                if blocks:
                    self.assertEqual(json.loads(result.stdout)["root_code"], "ownership-unknown")
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
