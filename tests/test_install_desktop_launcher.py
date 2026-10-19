# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""The desktop tree handler publishes, refreshes and removes its app-menu launcher."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.install_test_helpers import HermeticInstallerTestCase

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_test_release_server
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_HANDLER = REPO_ROOT / "handlers" / "desktop" / "v1" / "install-desktop"
UNINSTALL_HANDLER = REPO_ROOT / "handlers" / "desktop" / "v1" / "uninstall-desktop-service"
TMUX_INSTALL_HANDLER = REPO_ROOT / "handlers" / "tmux" / "v1" / "install-tmux"
LAUNCHER_NAME = "org.solpbc.solstone.Window.desktop"
MARK = "X-Solstone-Installer=platform-v1"


class TestDesktopLauncherHandler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.root = Path(self.tmp.name)
        self.prefix = self.root / "prefix"
        self.data_home = self.root / "data"
        self.env = {
            **os.environ,
            "HOME": str(self.root / "home"),
            "XDG_DATA_HOME": str(self.data_home),
            "XDG_CONFIG_HOME": str(self.root / "config"),
        }
        self.tree = self.prefix / "opt" / "solstone" / "desktop" / "1.0.0-abc" / "solstone-linux-1.0.0-linux-x86_64"
        (self.tree / "bin").mkdir(parents=True)
        self.binary = self.tree / "bin" / "solstone-linux"
        self.binary.write_text("#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)
        apps = self.tree / "share" / "applications"
        apps.mkdir(parents=True)
        (apps / LAUNCHER_NAME).write_text(
            "[Desktop Entry]\nType=Application\nName=solstone\nExec=solstone-linux window\n"
            "Icon=solstone-observer\nCategories=Utility;\n"
        )
        self.icon = self.tree / "share" / "icons" / "hicolor" / "256x256" / "apps" / "solstone-observer.png"
        self.icon.parent.mkdir(parents=True)
        self.icon.write_bytes(b"png")
        self.public = self.prefix / "bin" / "solstone-linux"
        self.public.parent.mkdir(parents=True)
        self.public.symlink_to(self.binary)

    def tearDown(self):
        self.tmp.cleanup()

    @property
    def launcher(self) -> Path:
        return self.data_home / "applications" / LAUNCHER_NAME

    def run_handler(self, handler: Path, binary: Path, role: str = "desktop", *extra: str):
        result = subprocess.run(
            [str(handler), "--prefix", str(self.prefix), "--binary", str(binary), "--route", "tree", "--role", role, *extra],
            capture_output=True, text=True, env=self.env,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fresh_install_publishes_launcher_with_public_exec(self):
        self.run_handler(INSTALL_HANDLER, self.binary, "desktop", "--no-start", "--no-path")
        lines = self.launcher.read_text().splitlines()
        self.assertIn(f'Exec="{self.public}" window', lines)
        self.assertIn(f"Icon={self.icon}", lines)
        self.assertEqual(lines.count(MARK), 1)

    def test_converged_install_refreshes_its_own_launcher_once(self):
        self.run_handler(INSTALL_HANDLER, self.binary, "desktop", "--no-start", "--no-path")
        self.run_handler(INSTALL_HANDLER, self.public, "desktop", "--no-start", "--no-path")
        lines = self.launcher.read_text().splitlines()
        self.assertIn(f'Exec="{self.public}" window', lines)
        self.assertEqual(lines.count(MARK), 1)

    def test_launcher_not_written_by_the_installer_is_left_alone(self):
        self.launcher.parent.mkdir(parents=True)
        self.launcher.write_text("[Desktop Entry]\nName=mine\n")
        self.run_handler(INSTALL_HANDLER, self.binary, "desktop", "--no-start", "--no-path")
        self.assertEqual(self.launcher.read_text(), "[Desktop Entry]\nName=mine\n")
        self.run_handler(UNINSTALL_HANDLER, self.public)
        self.assertEqual(self.launcher.read_text(), "[Desktop Entry]\nName=mine\n")

    def test_uninstall_removes_the_installer_launcher(self):
        self.run_handler(INSTALL_HANDLER, self.binary, "desktop", "--no-start", "--no-path")
        self.assertTrue(self.launcher.is_file())
        self.run_handler(UNINSTALL_HANDLER, self.public)
        self.assertFalse(self.launcher.exists())

    def test_unwritable_applications_folder_does_not_fail_the_install(self):
        self.data_home.mkdir(parents=True)
        (self.data_home / "applications").write_text("not a folder\n")
        self.run_handler(INSTALL_HANDLER, self.binary, "desktop", "--no-start", "--no-path")
        self.assertEqual((self.data_home / "applications").read_text(), "not a folder\n")

    def test_tmux_role_writes_no_launcher(self):
        self.run_handler(TMUX_INSTALL_HANDLER, self.binary, "tmux", "--no-start", "--no-path")
        self.assertFalse(self.launcher.exists())


class TestBundleWithoutTmux(HermeticInstallerTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"

    def tearDown(self):
        self.tmp.cleanup()

    def run_all(self, *extra: str, tmux_present: str):
        with ephemeral_keypair("bundle without tmux") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin)
            try:
                installer = self.work_dir / "install.sh"
                build_installer(
                    repo_root=REPO_ROOT, output_path=installer, platform_pub_path=pub,
                    platform_key_id=pin.key_id, origin=server.origin,
                )
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--dry-run", "--components", "all", "--prefix", str(self.prefix), "--json", *extra],
                    capture_output=True, text=True,
                    env={**os.environ, "SOLSTONE_TEST_TMUX_PRESENT": tmux_present},
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                return json.loads(proc.stdout.strip())
            finally:
                server.stop()

    def test_all_leaves_out_tmux_when_tmux_is_missing(self):
        res = self.run_all(tmux_present="0")
        self.assertIn("journal", res["components"])
        self.assertIn("desktop", res["components"])
        self.assertNotIn("tmux", res["components"])
        self.assertEqual(len(res["notices"]), 1)

    def test_all_keeps_tmux_when_present(self):
        res = self.run_all(tmux_present="1")
        self.assertIn("tmux", res["components"])
        self.assertEqual(res["notices"], [])

    def test_no_start_keeps_tmux_even_when_missing(self):
        res = self.run_all("--no-start", tmux_present="0")
        self.assertIn("tmux", res["components"])


if __name__ == "__main__":
    unittest.main()
