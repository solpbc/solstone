# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""A journal or solstone app installed from a package by hand is authenticated, then recorded."""

import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from tests.install_test_helpers import (
    HermeticInstallerTestCase,
    receipt_section,
    setup_fake_sudo,
    setup_test_release_server,
    snapshot_paths,
)
from solstone_platform.sign import ephemeral_keypair
from tools.build_installer import build_installer
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER_SCRIPT = REPO_ROOT / "helpers" / "solstone-pkg-helper.sh"


def deb_members(deb: bytes) -> dict[str, bytes]:
    """The regular files in a .deb's data archive."""
    assert deb.startswith(b"!<arch>\n")
    offset = 8
    while offset < len(deb):
        header = deb[offset:offset + 60]
        name = header[:16].decode().strip()
        size = int(header[48:58].decode().strip())
        body = deb[offset + 60:offset + 60 + size]
        offset += 60 + size + (size % 2)
        if name.startswith("data.tar"):
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as archive:
                return {member.name.lstrip("./"): archive.extractfile(member).read() for member in archive if member.isfile()}
    raise AssertionError("no data archive")


class TestInstallPackageAdopt(HermeticInstallerTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.bin_dir = self.work_dir / "bin"
        setup_fake_sudo(self.bin_dir)
        self.etc_root = self.work_dir / "etc"
        self.fake_db = self.work_dir / "db"
        self.fake_root = self.work_dir / "fake-root"
        for path in (self.work_dir / "lock", self.etc_root, self.fake_db, self.fake_root):
            path.mkdir()
        self.service_log = self.work_dir / "service.log"
        keys = ephemeral_keypair("package adoption")
        sec, pub, pin = keys.__enter__()
        self.addCleanup(keys.__exit__, None, None, None)
        self.server, self.root = setup_test_release_server(self.work_dir, sec, pin)
        self.addCleanup(self.server.stop)
        self.installer = build_installer(
            REPO_ROOT, self.work_dir / "install.sh", platform_pub_path=pub, platform_key_id=pin.key_id,
            origin=self.server.origin, native_test_pins=self.server.native_pins,
        )
        self.platform = json.loads((self.root / "solstone/release/2.0.0/platform.json").read_text())
        self.env = {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "XDG_DATA_HOME": str(self.work_dir / "tree-data"),
            "SOLSTONE_LOCK_DIR": str(self.work_dir / "lock"),
            "SOLSTONE_ETC_ROOT": str(self.etc_root),
            "SOLSTONE_FAKE_PKG_DB": str(self.fake_db),
            "SOLSTONE_FAKE_ROOT": str(self.fake_root),
            "SOLSTONE_HELPER": str(HELPER_SCRIPT),
            "SOLSTONE_TEST_SERVICE_LOG": str(self.service_log),
        }

    def tearDown(self):
        self.tmp.cleanup()

    @property
    def receipt(self) -> Path:
        return self.etc_root / "solstone" / "install.conf"

    def variant(self, component: str, route: str = "deb") -> dict:
        return self.platform["components"][component]["arches"]["x86_64"][route]

    def hand_install(self, component: str, *, route: str = "deb", version: str | None = None) -> dict[str, bytes]:
        """Put a package's files and database record in place, as a hand install leaves them."""
        variant = self.variant(component, route)
        identity = variant["package_identity"]
        deb = (self.root / "solstone/release/2.0.0" / self.variant(component, "deb")["filename"]).read_bytes()
        files = deb_members(deb)
        for path, data in files.items():
            target = self.fake_root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755)
        db = self.fake_db / route
        db.mkdir(exist_ok=True)
        (db / identity["name"]).write_text(f"INSTALLED {identity['name']} {version or identity['version']} {identity['arch']}\n")
        listed = ["/.", "/usr", "/usr/bin", *(f"/{path}" for path in sorted(files))]
        (db / f"{identity['name']}.files").write_text("\n".join(listed) + "\n")
        return files

    def run_installer(self, *args):
        return subprocess.run([str(self.installer), *args, "--json"], capture_output=True, text=True, env=self.env)

    def installs(self) -> list[str]:
        log = self.fake_db / "install.log"
        return [] if not log.exists() else [line for line in log.read_text().splitlines() if line]

    def unchanged_state(self):
        return snapshot_paths(self.fake_root, self.fake_db, self.etc_root)

    def test_a_hand_installed_journal_with_the_signed_bytes_is_recorded_as_a_package_install(self):
        for args in (("--components", "journal"), ("--components", "journal", "--route", "package"), ("--upgrade",)):
            with self.subTest(args=args):
                self.receipt.unlink(missing_ok=True)
                self.hand_install("journal")
                result = self.run_installer(*args)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["route"], "deb")
                self.assertIn("the solstone-journal package on this computer now updates with this installer.", report["notices"])
                self.assertEqual(self.installs(), [], "the authenticated package is kept, not reinstalled")
                section = receipt_section(self.receipt.read_bytes(), "component:journal")
                self.assertIn(b"phase=complete\nstatus=installed\nrole=journal\nroute=deb\nservice_policy=start\n", section)
                self.assertIn(f"package_version={self.variant('journal')['package_identity']['version']}\n".encode(), section)
                self.assertIn(b"verification=minisign\n", receipt_section(self.receipt.read_bytes(), "solstone"))

    def test_a_hand_installed_solstone_app_is_recorded_and_its_service_set_up(self):
        self.hand_install("desktop")
        result = self.run_installer("--components", "desktop")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.installs(), [])
        self.assertIn(b"route=deb\n", receipt_section(self.receipt.read_bytes(), "component:desktop"))
        self.assertIn("install-service", self.service_log.read_text())

    def test_an_rpm_is_authenticated_the_same_way(self):
        self.hand_install("journal", route="rpm")
        # rpm's nodocs leaves a packaged file out on purpose and says so.
        listing = self.fake_db / "rpm" / f"{self.variant('journal', 'rpm')['package_identity']['name']}.files"
        listing.write_text(listing.read_text() + "- /usr/share/doc/solstone-journal/LICENSE\n")
        result = self.run_installer("--components", "journal", "--route", "rpm")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(b"route=rpm\n", receipt_section(self.receipt.read_bytes(), "component:journal"))
        self.assertEqual(self.installs(), [])

    def test_a_changed_missing_or_extra_file_refuses_with_nothing_changed(self):
        name = self.variant("journal")["package_identity"]["name"]

        def changed(files):
            path = self.fake_root / next(iter(files))
            path.write_bytes(path.read_bytes() + b"# changed\n")

        def missing(files):
            (self.fake_root / next(iter(files))).unlink()

        def extra(_files):
            extra_file = self.fake_root / "usr/bin/extra"
            extra_file.write_text("#!/bin/sh\n")
            listing = self.fake_db / "deb" / f"{name}.files"
            listing.write_text(listing.read_text() + "/usr/bin/extra\n")

        def linked(files):
            path = self.fake_root / next(iter(files))
            real = path.with_name("real")
            path.rename(real)
            path.symlink_to(real)

        for fault in (changed, missing, extra, linked):
            with self.subTest(fault=fault.__name__):
                files = self.hand_install("journal")
                fault(files)
                before = self.unchanged_state()
                result = self.run_installer("--components", "journal")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["root_code"], "digest-mismatch")
                self.assertTrue(report["message"].endswith("This installer did not take it over. Nothing was changed."), report["message"])
                self.assertEqual(self.unchanged_state(), before)
                self.assertFalse(self.receipt.exists())
                self.assertEqual(self.installs(), [])
                for leftover in (self.fake_root / "usr/bin/extra", self.fake_root / "usr/bin/real"):
                    leftover.unlink(missing_ok=True)

    def test_an_older_package_is_replaced_with_the_verified_one(self):
        files = self.hand_install("journal", version="2.0.1")
        launcher = self.work_dir / "packaged-journal"
        launcher.write_bytes(next(iter(files.values())))
        launcher.chmod(0o755)
        self.env["SOLSTONE_FAKE_JOURNAL_LAUNCHER"] = str(launcher)
        result = self.run_installer("--components", "journal")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.installs()), 1)
        self.assertIn(self.variant("journal")["filename"], self.installs()[0])
        self.assertIn(b"phase=complete\n", receipt_section(self.receipt.read_bytes(), "component:journal"))

    def test_a_newer_or_foreign_package_refuses_with_nothing_changed(self):
        name = self.variant("journal")["package_identity"]["name"]
        for line, code in (
            (f"INSTALLED {name} 9.0.0 amd64", "downgrade-route-unsupported"),
            (f"INSTALLED {name} 2.0.6 arm64", "ownership-unknown"),
            (f"UNCONFIGURED {name} 2.0.6 amd64", "package-unconfigured"),
        ):
            with self.subTest(line=line):
                self.hand_install("journal")
                (self.fake_db / "deb" / name).write_text(line + "\n")
                before = self.unchanged_state()
                result = self.run_installer("--components", "journal")
                self.assertEqual(json.loads(result.stdout)["root_code"], code, result.stdout)
                self.assertEqual(self.unchanged_state(), before)
                self.assertEqual(self.server.request_paths and [p for p in self.server.request_paths if p.endswith(".deb")], [])

    def test_an_explicit_tree_route_names_the_package_route(self):
        self.hand_install("journal")
        before = self.unchanged_state()
        result = self.run_installer("--components", "journal", "--route", "tree")
        report = json.loads(result.stdout)
        self.assertEqual(report["root_code"], "ownership-unknown")
        self.assertIn("run it with --route package instead", report["message"])
        self.assertEqual(self.unchanged_state(), before)

    def test_a_preview_authenticates_and_changes_nothing(self):
        files = self.hand_install("journal")
        before = self.unchanged_state()
        preview = self.run_installer("--components", "journal", "--dry-run")
        self.assertEqual(preview.returncode, 0, preview.stdout + preview.stderr)
        self.assertEqual(self.unchanged_state(), before)
        path = self.fake_root / next(iter(files))
        path.write_bytes(b"changed\n")
        refused = self.run_installer("--components", "journal", "--dry-run")
        self.assertEqual(json.loads(refused.stdout)["root_code"], "digest-mismatch")


if __name__ == "__main__":
    unittest.main()
