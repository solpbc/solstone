# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""A new journal updates itself, an unchanged app keeps its run state, and old app versions go."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.install_test_helpers import HermeticInstallerTestCase

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_test_release_server
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent
JOURNAL_ONLY = "your journal updates itself:"
WITH_APP = "your journal and the solstone app update themselves:"


class TestInstallAutomaticUpdates(HermeticInstallerTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"
        self.journal_log = self.work_dir / "journal.log"
        self.service_log = self.work_dir / "service.log"
        self.env = {
            **os.environ,
            "XDG_DATA_HOME": str(self.work_dir / "data"),
            "XDG_CONFIG_HOME": str(self.work_dir / "config"),
            "SOLSTONE_TEST_JOURNAL_LOG": str(self.journal_log),
            "SOLSTONE_TEST_SERVICE_LOG": str(self.service_log),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def build_release(self, root: str, sec: Path, pub: Path, pin, platform_version: str, native_version: str):
        server, _ = setup_test_release_server(
            self.work_dir / root, sec, pin, version=platform_version, native_version=native_version,
            bootstrap_journal_cli=True,
        )
        installer = self.work_dir / root / "install.sh"
        build_installer(repo_root=REPO_ROOT, output_path=installer, platform_pub_path=pub,
                        platform_key_id=pin.key_id, origin=server.origin, native_test_pins=server.native_pins)
        return server, installer

    def merge_release(self, source, destination, version: str, pub: Path, pin) -> Path:
        lane = destination.root_dir / "solstone" / "release"
        shutil.copytree(source.root_dir / "solstone" / "release" / version, lane / version)
        (lane / "latest").write_text(f"{version}\n", encoding="utf-8")
        installer = self.work_dir / f"install-{version}.sh"
        build_installer(repo_root=REPO_ROOT, output_path=installer, platform_pub_path=pub,
                        platform_key_id=pin.key_id, origin=destination.origin, native_test_pins=destination.native_pins)
        return installer

    def run_install(self, installer: Path, components: str, *extra: str, env: dict | None = None):
        proc = subprocess.run(
            [str(installer), "--components", components, "--prefix", str(self.prefix), "--json", *extra],
            capture_output=True, text=True, env=env or self.env,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return json.loads(proc.stdout)

    def journal_calls(self) -> list[str]:
        return self.journal_log.read_text().splitlines() if self.journal_log.exists() else []

    def service_calls(self) -> list[str]:
        return self.service_log.read_text().splitlines() if self.service_log.exists() else []

    def update_notices(self, result: dict) -> list[str]:
        return [notice for notice in result["notices"] if "update" in notice and "themselves" in notice or "updates itself" in notice]

    def test_a_new_journal_turns_automatic_updates_on_and_says_so(self):
        with ephemeral_keypair("automatic updates fresh") as (sec, pub, pin):
            server, installer = self.build_release("fresh", sec, pub, pin, "2.0.0", "2.0.3")
            try:
                result = self.run_install(installer, "journal")
                self.assertEqual(self.journal_calls(), ["update --automatic on"])
                notices = self.update_notices(result)
                self.assertEqual(len(notices), 1)
                self.assertTrue(notices[0].startswith(JOURNAL_ONLY), notices)
                self.assertTrue(notices[0].endswith("to turn this off: solstone update --automatic off"), notices)

                # Running the installer again over an existing journal never switches it.
                again = self.run_install(installer, "journal")
                self.assertEqual(self.journal_calls(), ["update --automatic on"])
                self.assertEqual(self.update_notices(again), [])
            finally:
                server.stop()

    def test_the_owner_can_opt_out_at_install_time(self):
        with ephemeral_keypair("automatic updates opt out") as (sec, pub, pin):
            server, installer = self.build_release("optout", sec, pub, pin, "2.0.0", "2.0.3")
            try:
                result = self.run_install(installer, "journal", "--no-auto-update")
                self.assertEqual(self.journal_calls(), [])
                self.assertEqual(self.update_notices(result), [])
            finally:
                server.stop()

    def test_a_journal_that_does_not_know_the_switch_is_left_alone_and_nothing_is_said(self):
        with ephemeral_keypair("automatic updates old journal") as (sec, pub, pin):
            server, installer = self.build_release("old", sec, pub, pin, "2.0.0", "2.0.3")
            try:
                result = self.run_install(installer, "journal", env={**self.env, "SOLSTONE_TEST_JOURNAL_STATUS": "2"})
                self.assertEqual(self.journal_calls(), ["update --automatic on"])
                self.assertEqual(self.update_notices(result), [])
                self.assertEqual(result["status"], "success")
            finally:
                server.stop()

    def test_with_the_app_the_notice_names_it(self):
        with ephemeral_keypair("automatic updates with app") as (sec, pub, pin):
            server, installer = self.build_release("withapp", sec, pub, pin, "2.0.0", "2.0.3")
            try:
                result = self.run_install(installer, "journal,desktop")
                notices = self.update_notices(result)
                self.assertEqual(len(notices), 1)
                self.assertTrue(notices[0].startswith(WITH_APP), notices)
            finally:
                server.stop()

    def test_an_unchanged_app_with_its_service_in_place_is_not_touched(self):
        with ephemeral_keypair("unchanged app run state") as (sec, pub, pin):
            server, installer = self.build_release("unchanged", sec, pub, pin, "2.0.0", "2.0.3")
            try:
                self.run_install(installer, "desktop")
                self.assertEqual(self.service_calls(), ["desktop 2.0.3 install-service"])
                unit = Path(os.environ["HOME"]) / ".config" / "systemd" / "user" / "solstone-linux.service"
                unit.parent.mkdir(parents=True)
                unit.write_text("[Unit]\n")

                self.run_install(installer, "desktop")
                self.assertEqual(self.service_calls(), ["desktop 2.0.3 install-service"])

                # A missing service is set up again.
                unit.unlink()
                self.run_install(installer, "desktop")
                self.assertEqual(self.service_calls(), ["desktop 2.0.3 install-service"] * 2)
            finally:
                server.stop()

    def test_an_app_update_keeps_the_current_version_and_the_two_before_it(self):
        with ephemeral_keypair("app version prune") as (sec, pub, pin):
            base, installer = self.build_release("base", sec, pub, pin, "2.0.0", "2.0.3")
            releases = [self.build_release(f"r{n}", sec, pub, pin, f"2.0.{n}", f"2.0.{n + 3}")[0] for n in (1, 2, 3)]
            try:
                root = self.prefix / "opt" / "solstone" / "desktop"
                self.run_install(installer, "desktop", "--no-start")
                oldest = next(path for path in root.iterdir() if path.name != "current")
                # A version a browser registration still names stays, whatever its age.
                registration = Path(os.environ["HOME"]) / ".mozilla" / "native-messaging-hosts" / "org.solpbc.solstone.json"
                registration.parent.mkdir(parents=True)
                registration.write_text(json.dumps({"path": f"{oldest}/solstone-linux/bin/solstone-linux"}))
                for n, release in zip((1, 2, 3), releases):
                    self.run_install(self.merge_release(release, base, f"2.0.{n}", pub, pin), "desktop", "--no-start")
                versions = sorted(path.name.split("-")[0] for path in root.iterdir() if path.name != "current")
                self.assertEqual(versions, ["2.0.3", "2.0.4", "2.0.5", "2.0.6"])

                registration.unlink()
                newest = self.build_release("r4", sec, pub, pin, "2.0.4", "2.0.7")[0]
                releases.append(newest)
                self.run_install(self.merge_release(newest, base, "2.0.4", pub, pin), "desktop", "--no-start")
                versions = sorted(path.name.split("-")[0] for path in root.iterdir() if path.name != "current")
                self.assertEqual(versions, ["2.0.5", "2.0.6", "2.0.7"])
                self.assertTrue(os.readlink(root / "current").startswith("2.0.7-"))
            finally:
                base.stop()
                for release in releases:
                    release.stop()


if __name__ == "__main__":
    unittest.main()
