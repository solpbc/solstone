# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Package-route identity, ownership, phase, and launcher lifecycle coverage."""

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
from tools.build_installer import build_installer


REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER_SCRIPT = REPO_ROOT / "helpers" / "solstone-pkg-helper.sh"


class TestInstallPackageLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.bin_dir = self.work_dir / "bin"
        setup_fake_sudo(self.bin_dir)
        self.lock_dir = self.work_dir / "lock"
        self.etc_root = self.work_dir / "etc"
        self.fake_db = self.work_dir / "db"
        self.fake_root = self.work_dir / "fake-root"
        for path in (self.lock_dir, self.etc_root, self.fake_db):
            path.mkdir()
        self.setup_log = self.work_dir / "setup.log"
        self.shadow_log = self.work_dir / "shadow.log"
        self.launcher = setup_package_launcher_spy(self.bin_dir)
        write_path_stub(self.bin_dir, "journal", "printf 'journal\\n' >> \"$SOLSTONE_SHADOW_LOG\"\nexit 99\n")
        write_path_stub(self.bin_dir, "solstone-journal", "printf 'solstone-journal\\n' >> \"$SOLSTONE_SHADOW_LOG\"\nexit 99\n")
        self.env = {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "SOLSTONE_LOCK_DIR": str(self.lock_dir),
            "SOLSTONE_ETC_ROOT": str(self.etc_root),
            "SOLSTONE_FAKE_PKG_DB": str(self.fake_db),
            "SOLSTONE_FAKE_ROOT": str(self.fake_root),
            "SOLSTONE_FAKE_JOURNAL_LAUNCHER": str(self.launcher),
            "SOLSTONE_HELPER": str(HELPER_SCRIPT),
            "SOLSTONE_PACKAGE_SETUP_LOG": str(self.setup_log),
            "SOLSTONE_SHADOW_LOG": str(self.shadow_log),
        }

    def tearDown(self):
        self.tmp.cleanup()

    @property
    def receipt(self) -> Path:
        return self.etc_root / "solstone" / "install.conf"

    def build_release(self, sec: Path, pub: Path, pin):
        server, _ = setup_test_release_server(self.work_dir, sec, pin)
        installer = self.work_dir / "install.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=installer,
            platform_pub_path=pub,
            platform_key_id=pin.key_id,
            origin=server.origin,
        )
        return server, installer

    def run_install(self, installer: Path, components: str, *, no_start: bool = False):
        args = [
            str(installer), "--skip-signature", "--route", "deb", "--components", components, "--json",
        ]
        if no_start:
            args.append("--no-start")
        return subprocess.run(args, capture_output=True, text=True, env=self.env)

    def installs(self) -> list[str]:
        log = self.fake_db / "install.log"
        return [] if not log.exists() else [line for line in log.read_text(encoding="utf-8").splitlines() if line]

    def setup_records(self) -> list[list[str]]:
        if not self.setup_log.exists():
            return []
        records: list[list[str]] = []
        current: list[str] = []
        for line in self.setup_log.read_text(encoding="utf-8").splitlines():
            if line == "--":
                records.append(current)
                current = []
            else:
                current.append(line)
        return records

    @staticmethod
    def package_requests(paths: list[str]) -> list[str]:
        return [path for path in paths if path.endswith((".deb", ".rpm"))]

    def test_fresh_exact_identity_is_unchanged_without_fetch_or_setup(self):
        with ephemeral_keypair("package lifecycle exact") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin)
            try:
                first = self.run_install(installer, "journal", no_start=True)
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                self.assertEqual(len(self.installs()), 1)
                section = receipt_section(self.receipt.read_bytes(), "component:journal")
                for value in (
                    b"phase=complete\n", b"status=installed\n", b"role=journal\n", b"route=deb\n",
                    b"service_policy=skip-service\n", b"package_name=solstone-journal\n",
                    b"package_version=2.0.6\n", b"package_arch=amd64\n",
                ):
                    self.assertIn(value, section)
                section_text = section.decode("utf-8")
                self.assertRegex(section_text, r"(?m)^artifact_sha256=.+$")
                self.assertRegex(section_text, r"(?m)^payload_build_id=.+$")
                self.assertNotIn(b"prior_", section)
                first_receipt = self.receipt.read_bytes()
                records = self.setup_records()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0][0], f"argv0={self.fake_root}/usr/bin/journal")
                self.assertIn("arg=setup", records[0])
                self.assertIn("arg=--skip-service", records[0])

                server.request_paths.clear()
                second = self.run_install(installer, "journal", no_start=True)
                self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
                self.assertEqual(json.loads(second.stdout)["components"]["journal"]["status"], "unchanged")
                self.assertEqual(len(self.installs()), 1)
                self.assertEqual(self.setup_records(), records)
                self.assertEqual(self.receipt.read_bytes(), first_receipt)
                self.assertEqual(self.package_requests(server.request_paths), [])
                self.assertFalse(self.shadow_log.exists())
            finally:
                server.stop()

    def test_installed_unknown_owner_and_wrong_arch_refuse_without_fetch(self):
        with ephemeral_keypair("package lifecycle unknown") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin)
            try:
                db_file = self.fake_db / "deb" / "solstone-journal"
                db_file.parent.mkdir()
                db_file.write_text("INSTALLED solstone-journal 2.0.6 amd64\n", encoding="utf-8")
                unknown = self.run_install(installer, "journal")
                self.assertNotEqual(unknown.returncode, 0)
                self.assertEqual(json.loads(unknown.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(self.package_requests(server.request_paths), [])
                self.assertEqual(self.installs(), [])

                server.request_paths.clear()
                db_file.write_text("INSTALLED solstone-journal 2.0.6 arm64\n", encoding="utf-8")
                wrong_arch = self.run_install(installer, "journal")
                self.assertNotEqual(wrong_arch.returncode, 0)
                self.assertEqual(json.loads(wrong_arch.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(self.package_requests(server.request_paths), [])
            finally:
                server.stop()

    def test_wrong_route_section_incomplete_prior_and_epoch_downgrade_refuse(self):
        with ephemeral_keypair("package lifecycle ownership binding") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin)
            try:
                desktop = self.run_install(installer, "desktop")
                self.assertEqual(desktop.returncode, 0, desktop.stderr + desktop.stdout)
                receipt_complete = self.receipt.read_bytes()
                db_before = (self.fake_db / "deb" / "solstone-linux").read_bytes()
                installs_before = self.installs()

                desktop_section = receipt_section(receipt_complete, "component:desktop")
                wrong_route = desktop_section.replace(b"route=deb\n", b"route=rpm\n")
                self.receipt.write_bytes(receipt_complete.replace(desktop_section, wrong_route))
                server.request_paths.clear()
                refused_route = self.run_install(installer, "desktop")
                self.assertNotEqual(refused_route.returncode, 0)
                self.assertEqual(json.loads(refused_route.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(self.installs(), installs_before)
                self.assertEqual(self.package_requests(server.request_paths), [])
                self.assertEqual((self.fake_db / "deb" / "solstone-linux").read_bytes(), db_before)

                wrong_section = desktop_section.replace(b"[component:desktop]\n", b"[component:journal]\n")
                self.receipt.write_bytes(receipt_complete.replace(desktop_section, wrong_section))
                server.request_paths.clear()
                refused_section = self.run_install(installer, "desktop")
                self.assertNotEqual(refused_section.returncode, 0)
                self.assertEqual(json.loads(refused_section.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(self.installs(), installs_before)
                self.assertEqual(self.package_requests(server.request_paths), [])

                incomplete_prior = desktop_section.replace(
                    b"phase=complete\nstatus=installed\n",
                    b"phase=intended\nstatus=intended\n"
                    b"prior_role=\nprior_route=\nprior_service_policy=\n"
                    b"prior_package_name=solstone-linux\nprior_package_version=2.0.3-1\n"
                    b"prior_package_arch=amd64\nprior_artifact_sha256=\nprior_payload_build_id=\n",
                )
                self.receipt.write_bytes(receipt_complete.replace(desktop_section, incomplete_prior))
                server.request_paths.clear()
                refused_prior = self.run_install(installer, "desktop")
                self.assertNotEqual(refused_prior.returncode, 0)
                self.assertEqual(json.loads(refused_prior.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(self.installs(), installs_before)
                self.assertEqual(self.package_requests(server.request_paths), [])

                missing_prior_role = incomplete_prior.replace(b"prior_role=\n", b"")
                self.receipt.write_bytes(receipt_complete.replace(desktop_section, missing_prior_role))
                server.request_paths.clear()
                refused_missing = self.run_install(installer, "desktop")
                self.assertNotEqual(refused_missing.returncode, 0)
                self.assertEqual(json.loads(refused_missing.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(self.installs(), installs_before)
                self.assertEqual(self.package_requests(server.request_paths), [])

                colon_hash = incomplete_prior.replace(
                    b"prior_role=\nprior_route=\nprior_service_policy=\n",
                    b"prior_role=desktop\nprior_route=deb\nprior_service_policy=start\n",
                ).replace(
                    b"prior_artifact_sha256=\nprior_payload_build_id=\n",
                    b"prior_artifact_sha256=:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
                    b"prior_payload_build_id=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n",
                )
                self.receipt.write_bytes(receipt_complete.replace(desktop_section, colon_hash))
                server.request_paths.clear()
                refused_hash = self.run_install(installer, "desktop")
                self.assertNotEqual(refused_hash.returncode, 0)
                self.assertEqual(json.loads(refused_hash.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(self.installs(), installs_before)
                self.assertEqual(self.package_requests(server.request_paths), [])

                self.receipt.unlink()
                (self.fake_db / "deb" / "solstone-linux").unlink()
                journal = self.run_install(installer, "journal")
                self.assertEqual(journal.returncode, 0, journal.stderr + journal.stdout)
                journal_receipt = self.receipt.read_bytes()
                journal_section = receipt_section(journal_receipt, "component:journal")
                epoch_section = journal_section.replace(b"package_version=2.0.6\n", b"package_version=1:1.0\n")
                self.receipt.write_bytes(journal_receipt.replace(journal_section, epoch_section))
                journal_db = self.fake_db / "deb" / "solstone-journal"
                journal_db.write_text("INSTALLED solstone-journal 1:1.0 amd64\n", encoding="utf-8")
                receipt_before = self.receipt.read_bytes()
                db_epoch_before = journal_db.read_bytes()
                epoch_installs = self.installs()
                server.request_paths.clear()
                downgrade = self.run_install(installer, "journal")
                self.assertNotEqual(downgrade.returncode, 0)
                self.assertEqual(json.loads(downgrade.stdout)["root_code"], "downgrade-route-unsupported")
                self.assertEqual(self.receipt.read_bytes(), receipt_before)
                self.assertEqual(journal_db.read_bytes(), db_epoch_before)
                self.assertEqual(self.installs(), epoch_installs)
                self.assertEqual(self.package_requests(server.request_paths), [])
            finally:
                server.stop()

    def test_intended_and_payload_recovery_skip_package_fetch(self):
        with ephemeral_keypair("package lifecycle recovery") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin)
            try:
                first = self.run_install(installer, "journal")
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                complete = receipt_section(self.receipt.read_bytes(), "component:journal")

                target = {
                    key: value
                    for key, value in (line.split(b"=", 1) for line in complete.splitlines() if b"=" in line)
                }
                prior = b"".join((
                    b"prior_role=journal\n",
                    b"prior_route=deb\n",
                    b"prior_service_policy=skip-service\n",
                    b"prior_package_name=" + target[b"package_name"] + b"\n",
                    b"prior_package_version=" + target[b"package_version"] + b"\n",
                    b"prior_package_arch=" + target[b"package_arch"] + b"\n",
                    b"prior_artifact_sha256=" + target[b"artifact_sha256"] + b"\n",
                    b"prior_payload_build_id=" + target[b"payload_build_id"] + b"\n",
                ))
                intended = complete.replace(
                    b"phase=complete\nstatus=installed\n", b"phase=intended\nstatus=intended\n"
                ) + prior
                intended_receipt = self.receipt.read_bytes().replace(complete, intended)
                self.receipt.write_bytes(intended_receipt)
                server.request_paths.clear()
                intended_recovery = self.run_install(installer, "journal")
                self.assertEqual(intended_recovery.returncode, 0, intended_recovery.stderr + intended_recovery.stdout)
                self.assertEqual(len(self.installs()), 1)
                self.assertEqual(self.package_requests(server.request_paths), [])
                recovered_receipt = self.receipt.read_bytes()
                self.assertNotIn(
                    b"prior_", receipt_section(recovered_receipt, "component:journal")
                )

                db_file = self.fake_db / "deb" / "solstone-journal"
                self.receipt.write_bytes(intended_receipt)
                db_file.write_text("INSTALLED solstone-journal 2.0.9 amd64\n", encoding="utf-8")
                twin_receipt = self.receipt.read_bytes()
                twin_db = db_file.read_bytes()
                server.request_paths.clear()
                intended_twin = self.run_install(installer, "journal")
                self.assertNotEqual(intended_twin.returncode, 0)
                self.assertEqual(json.loads(intended_twin.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(self.receipt.read_bytes(), twin_receipt)
                self.assertEqual(db_file.read_bytes(), twin_db)
                self.assertEqual(len(self.installs()), 1)
                self.assertEqual(self.package_requests(server.request_paths), [])

                self.receipt.write_bytes(recovered_receipt)
                db_file.write_text("INSTALLED solstone-journal 2.0.6 amd64\n", encoding="utf-8")

                current = receipt_section(self.receipt.read_bytes(), "component:journal")
                payload = current.replace(b"phase=complete\nstatus=installed\n", b"phase=payload\nstatus=payload\n")
                self.receipt.write_bytes(self.receipt.read_bytes().replace(current, payload))
                server.request_paths.clear()
                payload_recovery = self.run_install(installer, "journal")
                self.assertEqual(payload_recovery.returncode, 0, payload_recovery.stderr + payload_recovery.stdout)
                self.assertEqual(len(self.installs()), 1)
                self.assertEqual(self.package_requests(server.request_paths), [])
                self.assertEqual(len(self.setup_records()), 3)

                current = receipt_section(self.receipt.read_bytes(), "component:journal")
                intended_again = current.replace(b"phase=complete\nstatus=installed\n", b"phase=intended\nstatus=intended\n")
                self.receipt.write_bytes(self.receipt.read_bytes().replace(current, intended_again))
                (self.fake_db / "deb" / "solstone-journal").unlink()
                server.request_paths.clear()
                absent_recovery = self.run_install(installer, "journal")
                self.assertEqual(absent_recovery.returncode, 0, absent_recovery.stderr + absent_recovery.stdout)
                self.assertEqual(len(self.installs()), 2)
                self.assertEqual(len(self.package_requests(server.request_paths)), 1)
            finally:
                server.stop()

    def test_role_conflict_and_unselected_receipt_custody(self):
        with ephemeral_keypair("package lifecycle custody") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin)
            try:
                desktop = self.run_install(installer, "desktop")
                self.assertEqual(desktop.returncode, 0, desktop.stderr + desktop.stdout)
                desktop_section = receipt_section(self.receipt.read_bytes(), "component:desktop")
                cli = self.run_install(installer, "cli")
                self.assertEqual(cli.returncode, 0, cli.stderr + cli.stdout)
                self.assertEqual(receipt_section(self.receipt.read_bytes(), "component:desktop"), desktop_section)
                before = len(self.installs())
                receipt_before = self.receipt.read_bytes()
                db_file = self.fake_db / "deb" / "solstone-journal"
                db_before = db_file.read_bytes()
                server.request_paths.clear()
                journal = self.run_install(installer, "journal")
                self.assertNotEqual(journal.returncode, 0)
                self.assertEqual(json.loads(journal.stdout)["root_code"], "role-conflict")
                self.assertEqual(self.receipt.read_bytes(), receipt_before)
                self.assertEqual(db_file.read_bytes(), db_before)
                self.assertEqual(len(self.installs()), before)
                self.assertEqual(self.package_requests(server.request_paths), [])
                self.assertEqual(self.setup_records(), [])
                self.assertFalse(self.shadow_log.exists())
            finally:
                server.stop()

    def test_owned_policy_replacement_and_negative_twin(self):
        with ephemeral_keypair("package lifecycle replacement") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin)
            try:
                first = self.run_install(installer, "journal")
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                server.request_paths.clear()
                replacement = self.run_install(installer, "journal", no_start=True)
                self.assertEqual(replacement.returncode, 0, replacement.stderr + replacement.stdout)
                self.assertEqual(len(self.installs()), 2)
                self.assertEqual(len(self.package_requests(server.request_paths)), 1)
                section = receipt_section(self.receipt.read_bytes(), "component:journal")
                self.assertIn(b"service_policy=skip-service\n", section)
                self.assertNotIn(b"prior_", section)
                records = self.setup_records()
                self.assertEqual(len(records), 2)
                self.assertNotIn("arg=--skip-service", records[0])
                self.assertIn("arg=--skip-service", records[1])

                db_file = self.fake_db / "deb" / "solstone-journal"
                db_file.write_text("INSTALLED solstone-journal 2.0.9 amd64\n", encoding="utf-8")
                server.request_paths.clear()
                twin = self.run_install(installer, "journal", no_start=True)
                self.assertNotEqual(twin.returncode, 0)
                self.assertEqual(json.loads(twin.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(len(self.installs()), 2)
                self.assertEqual(self.package_requests(server.request_paths), [])
            finally:
                server.stop()

    def test_mixed_package_failure_keeps_completed_earlier_component(self):
        with ephemeral_keypair("package lifecycle mixed") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin)
            try:
                tmux_section = (
                    b"[component:tmux]\n"
                    b"phase=complete\nstatus=installed\nrole=tmux\nroute=deb\nservice_policy=start\n"
                    b"package_name=solstone-tmux\npackage_version=seed\npackage_arch=amd64\n"
                    b"artifact_sha256=seed\npayload_build_id=seed\n"
                )
                self.receipt.parent.mkdir(parents=True)
                self.receipt.write_bytes(b"[solstone]\nseed=1\n" + tmux_section)
                db_file = self.fake_db / "deb" / "solstone-journal"
                db_file.parent.mkdir()
                db_file.write_text("INSTALLED solstone-journal 2.0.6 amd64\n", encoding="utf-8")
                server.request_paths.clear()

                mixed = self.run_install(installer, "desktop,journal")
                self.assertNotEqual(mixed.returncode, 0)
                mixed_json = json.loads(mixed.stdout)
                self.assertEqual(mixed_json["root_code"], "ownership-unknown")
                self.assertEqual(mixed_json["components"]["desktop"]["status"], "succeeded")
                self.assertEqual(mixed_json["components"]["desktop"]["phase"], "complete")
                self.assertEqual(mixed_json["components"]["journal"]["status"], "failed")
                desktop_section = receipt_section(self.receipt.read_bytes(), "component:desktop")
                self.assertIn(b"phase=complete\nstatus=installed\n", desktop_section)
                self.assertEqual(receipt_section(self.receipt.read_bytes(), "component:tmux"), tmux_section)
                self.assertNotIn(b"[component:journal]\n", self.receipt.read_bytes())
                installs = self.installs()
                self.assertEqual(len(installs), 1)
                self.assertIn("solstone-linux", installs[0])
                self.assertEqual(db_file.read_text(encoding="utf-8"), "INSTALLED solstone-journal 2.0.6 amd64\n")

                receipt_after_mixed = self.receipt.read_bytes()
                server.request_paths.clear()
                retry = self.run_install(installer, "desktop")
                self.assertEqual(retry.returncode, 0, retry.stderr + retry.stdout)
                self.assertEqual(json.loads(retry.stdout)["components"]["desktop"]["status"], "unchanged")
                self.assertEqual(self.installs(), installs)
                self.assertEqual(self.receipt.read_bytes(), receipt_after_mixed)
                self.assertEqual(self.package_requests(server.request_paths), [])
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
