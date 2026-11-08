# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for Linux catalogue drop of tmux and installer revision 20 move to desktop."""

import hashlib
import json
import os
from pathlib import Path
import pty
import select
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import unittest

from tests.install_test_helpers import (
    HermeticInstallerTestCase,
    receipt_section,
    setup_fake_sudo,
    setup_package_launcher_spy,
    setup_test_release_server,
    write_path_stub,
)
from solstone_platform.sign import ephemeral_keypair
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER_SCRIPT = REPO_ROOT / "helpers" / "solstone-pkg-helper.sh"


def safe_receipt_section(receipt: bytes, name: str) -> bytes:
    try:
        return receipt_section(receipt, name)
    except ValueError:
        return b""


class TestInstallTmuxMove(HermeticInstallerTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.bin_dir = self.work_dir / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        setup_fake_sudo(self.bin_dir)
        self.lock_dir = self.work_dir / "lock"
        self.etc_root = self.work_dir / "etc"
        self.fake_db = self.work_dir / "db"
        self.fake_root = self.work_dir / "fake-root"
        self.tree_prefix = self.work_dir / "prefix"
        for path in (self.lock_dir, self.etc_root, self.fake_db, self.tree_prefix):
            path.mkdir(parents=True, exist_ok=True)

        self.service_log = self.work_dir / "service.log"
        self.adopt_log = self.work_dir / "adopt.log"
        self.launcher = setup_package_launcher_spy(self.bin_dir)

        desktop_stub = (
            "#!/bin/sh\n"
            "case \"${1:-}\" in\n"
            "  install-service|uninstall-service)\n"
            "    marker_status=\"marker-missing\"\n"
            "    if [ -f \"$HOME/.local/share/solstone-linux/desktop-capture-off\" ]; then\n"
            "      marker_status=\"marker-exists\"\n"
            "    fi\n"
            "    [ -z \"${SOLSTONE_TEST_SERVICE_LOG:-}\" ] || printf 'desktop %s %s\\n' \"$1\" \"$marker_status\" >> \"$SOLSTONE_TEST_SERVICE_LOG\"\n"
            "    exit 0 ;;\n"
            "esac\n"
            "if [ \"${1:-}\" = \"terminal\" ] && [ \"${2:-}\" = \"adopt-tmux\" ]; then\n"
            "  [ -z \"${SOLSTONE_ADOPT_LOG:-}\" ] || printf '%s\\n' \"$*\" >> \"$SOLSTONE_ADOPT_LOG\"\n"
            "  [ -z \"${SOLSTONE_ADOPT_STDERR:-}\" ] || printf '%s\\n' \"$SOLSTONE_ADOPT_STDERR\" >&2\n"
            "  exit \"${SOLSTONE_ADOPT_EXIT:-0}\"\n"
            "fi\n"
            "echo 2.0.37\n"
        )
        write_path_stub(self.fake_root / "usr/bin", "solstone-linux", desktop_stub)

        tmux_stub = (
            "#!/bin/sh\n"
            "case \"${1:-}\" in\n"
            "  install-service|uninstall-service)\n"
            "    [ -z \"${SOLSTONE_TEST_SERVICE_LOG:-}\" ] || printf 'tmux %s\\n' \"$1\" >> \"$SOLSTONE_TEST_SERVICE_LOG\"\n"
            "    exit 0 ;;\n"
            "esac\n"
            "echo 0.1.0\n"
        )
        write_path_stub(self.fake_root / "usr/bin", "solstone-tmux", tmux_stub)

        self.env = {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "SOLSTONE_LOCK_DIR": str(self.lock_dir),
            "SOLSTONE_ETC_ROOT": str(self.etc_root),
            "SOLSTONE_FAKE_PKG_DB": str(self.fake_db),
            "SOLSTONE_FAKE_ROOT": str(self.fake_root),
            "SOLSTONE_FAKE_JOURNAL_LAUNCHER": str(self.launcher),
            "SOLSTONE_HELPER": str(HELPER_SCRIPT),
            "SOLSTONE_TEST_SERVICE_LOG": str(self.service_log),
            "SOLSTONE_ADOPT_LOG": str(self.adopt_log),
        }

    def tearDown(self):
        self.tmp.cleanup()

    @property
    def tree_receipt(self) -> Path:
        return Path(os.environ["XDG_DATA_HOME"]) / "solstone" / "install.conf"

    @property
    def package_receipt(self) -> Path:
        return self.etc_root / "solstone" / "install.conf"

    @property
    def marker_path(self) -> Path:
        return Path(os.environ["HOME"]) / ".local" / "share" / "solstone-linux" / "desktop-capture-off"

    def seed_package_tmux(self, version: str = "0.1.0", lane: str = "staging") -> bytes:
        self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
        tmux_section = (
            f"[component:tmux]\n"
            f"phase=complete\nstatus=installed\nrole=tmux\nroute=deb\nservice_policy=start\n"
            f"package_name=solstone-tmux\npackage_version={version}\npackage_arch=amd64\n"
            f"artifact_sha256={'0'*64}\npayload_build_id={'0'*64}\n"
        ).encode("utf-8")
        header = (
            f"[solstone]\nschema_version=1\nplatform_version=1.9.0\nlane={lane}\n"
        ).encode("utf-8")
        self.package_receipt.write_bytes(header + tmux_section)
        db_file = self.fake_db / "deb" / "solstone-tmux"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db_file.write_text(f"INSTALLED solstone-tmux {version} amd64\n", encoding="utf-8")
        return tmux_section

    def seed_tree_tmux(
        self,
        prefix: Path | None = None,
        version: str = "0.1.0",
        lane: str = "staging",
        no_start: int = 0,
        no_path: int = 0,
        arch: str = "x86_64",
    ) -> bytes:
        if prefix is None:
            prefix = Path(os.environ["HOME"]) / ".local"
        self.tree_receipt.parent.mkdir(parents=True, exist_ok=True)
        prefix_str = str(prefix)
        current_target = f"{version}-000000000000"
        relpath = "bin/solstone-tmux"
        opt_tmux = prefix / "opt" / "solstone" / "tmux"
        dest = opt_tmux / current_target
        dest.mkdir(parents=True, exist_ok=True)
        exec_file = dest / relpath
        exec_file.parent.mkdir(parents=True, exist_ok=True)
        exec_content = b"#!/bin/sh\nexit 0\n"
        exec_file.write_bytes(exec_content)
        exec_file.chmod(0o755)
        sha = hashlib.sha256(exec_content).hexdigest()

        current = opt_tmux / "current"
        if current.is_symlink() or current.exists():
            current.unlink()
        current.symlink_to(current_target)

        bin_dir = prefix / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        public_path = bin_dir / "solstone-tmux"
        if public_path.is_symlink() or public_path.exists():
            public_path.unlink()
        public_path.symlink_to(f"{prefix_str}/opt/solstone/tmux/current/{relpath}")

        tmux_section = (
            f"[component:tmux]\n"
            f"role=tmux\nroute=tree\narch={arch}\nversion={version}\nstatus=installed\n"
            f"phase=complete\nprefix={prefix_str}\n"
            f"no_start={no_start}\nno_path={no_path}\n"
            f"public_path={prefix_str}/bin/solstone-tmux\n"
            f"current={prefix_str}/opt/solstone/tmux/current\n"
            f"current_target={current_target}\n"
            f"executable_relpath={relpath}\n"
            f"executable_sha256={sha}\n"
        ).encode("utf-8")
        header = (
            f"[solstone]\nschema_version=1\nplatform_version=1.9.0\nlane={lane}\n"
        ).encode("utf-8")
        self.tree_receipt.write_bytes(header + tmux_section)
        return tmux_section

    def build_release(
        self,
        sec: Path,
        pub: Path,
        pin,
        *,
        lane: str = "release",
        min_installer_revision: int = 20,
        platform_version: str = "2.0.0",
        desktop_version: str | None = None,
        desktop_arches: tuple[str, ...] = ("x86_64",),
        override_installer_revision: int | None = None,
    ):
        server, _ = setup_test_release_server(
            self.work_dir,
            sec,
            pin,
            lane=lane,
            version=platform_version,
            min_installer_revision=min_installer_revision,
            desktop_version=desktop_version,
            desktop_arches=desktop_arches,
        )
        installer = self.work_dir / "install.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=installer,
            platform_pub_path=pub,
            platform_key_id=pin.key_id,
            origin=server.origin,
            override_installer_revision=override_installer_revision,
        )
        return server, installer

    def run_pty_interaction(self, command: str, env: dict[str, str], interactions: list[tuple[bytes, bytes]]):
        stdout_path = self.work_dir / f"pty-stdout-{time.monotonic_ns()}"
        stderr_path = self.work_dir / f"pty-stderr-{time.monotonic_ns()}"
        pid, master = pty.fork()
        if pid == 0:
            out_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            err_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            os.execve("/bin/sh", ["sh", "-c", command], env)

        tty_output = bytearray()
        step = 0
        last_idx = 0
        status = None
        deadline = time.monotonic() + 30
        try:
            while time.monotonic() < deadline:
                ready, _, _ = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        chunk = b""
                    tty_output.extend(chunk)
                    if step < len(interactions):
                        trigger, reply = interactions[step]
                        idx = tty_output.find(trigger, last_idx)
                        if idx != -1:
                            os.write(master, reply)
                            last_idx = idx + len(trigger)
                            step += 1
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited == pid:
                    break
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                self.fail("PTY installer invocation timed out")
        finally:
            os.close(master)
        exit_code = os.waitstatus_to_exitcode(status) if status is not None else 1
        stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        return exit_code, bytes(tty_output), stdout, stderr

    def test_tmux_move_declined_consent_keeps_tmux_and_advances_headers(self):
        with ephemeral_keypair("test declined consent") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.tree_receipt.parent.mkdir(parents=True, exist_ok=True)
                self.tree_receipt.write_text(
                    "[solstone]\nschema_version=1\nplatform_version=1.9.0\nlane=release\n",
                    encoding="utf-8",
                )
                pkg_tmux_sec = self.seed_package_tmux(version="0.1.0", lane="release")

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--non-interactive", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["root_code"], "installed")
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "declined")

                self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), pkg_tmux_sec)
                self.assertEqual(safe_receipt_section(self.tree_receipt.read_bytes(), "component:tmux"), b"")

                tree_text = self.tree_receipt.read_text(encoding="utf-8")
                pkg_text = self.package_receipt.read_text(encoding="utf-8")
                self.assertIn("platform_version=2.0.0", tree_text)
                self.assertIn("platform_version=2.0.0", pkg_text)
                stderr_nonempty = [l for l in proc.stderr.splitlines() if l.strip()]
                self.assertEqual(len(stderr_nonempty), 1)
            finally:
                server.stop()

    def test_tmux_move_approved_with_yes_flag_tree_and_package(self):
        with ephemeral_keypair("test approved with yes") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                # Tree move
                self.seed_tree_tmux(version="0.1.0", lane="staging")
                proc_tree = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_tree.returncode, 0, f"Failed: {proc_tree.stderr}\n{proc_tree.stdout}")
                doc_tree = json.loads(proc_tree.stdout)
                self.assertEqual(doc_tree["root_code"], "installed")
                self.assertEqual(doc_tree["components"]["tmux"]["status"], "moved")
                self.assertEqual(doc_tree["components"]["tmux"]["reason"], "moved")
                self.assertEqual(doc_tree["components"]["desktop"]["status"], "succeeded")
                self.assertEqual(safe_receipt_section(self.tree_receipt.read_bytes(), "component:tmux"), b"")
                self.assertIn(b"[component:desktop]\n", self.tree_receipt.read_bytes())
                home_prefix = Path(os.environ["HOME"]) / ".local"
                self.assertFalse((home_prefix / "bin" / "solstone-tmux").exists())
                self.assertTrue((home_prefix / "bin" / "solstone-linux").exists())

                # Package move
                shutil.rmtree(self.fake_db)
                self.fake_db.mkdir()
                if self.package_receipt.exists():
                    self.package_receipt.unlink()
                if self.tree_receipt.exists():
                    self.tree_receipt.unlink()
                self.seed_package_tmux(version="0.1.0", lane="staging")

                proc_pkg = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_pkg.returncode, 0, f"Failed: {proc_pkg.stderr}\n{proc_pkg.stdout}")
                doc_pkg = json.loads(proc_pkg.stdout)
                self.assertEqual(doc_pkg["root_code"], "installed")
                self.assertEqual(doc_pkg["components"]["tmux"]["status"], "moved")
                self.assertEqual(doc_pkg["components"]["tmux"]["reason"], "moved")
                self.assertEqual(doc_pkg["components"]["desktop"]["status"], "succeeded")
                self.assertEqual(safe_receipt_section(self.package_receipt.read_bytes(), "component:tmux"), b"")
                self.assertIn(b"[component:desktop]\n", self.package_receipt.read_bytes())
                self.assertFalse((self.fake_db / "deb" / "solstone-tmux").exists())
                self.assertTrue((self.fake_db / "deb" / "solstone-linux").exists())
            finally:
                server.stop()

    def test_tmux_move_marker_created_before_first_desktop_receipt_write(self):
        with ephemeral_keypair("test marker created") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                # Non-confirm path
                self.seed_package_tmux(version="0.1.0", lane="staging")
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                self.assertIn("desktop install-service marker-exists", self.service_log.read_text(encoding="utf-8"))
                self.assertTrue(self.marker_path.exists())
                self.assertEqual(self.marker_path.stat().st_uid, os.getuid())

                # Confirm path
                shutil.rmtree(self.fake_db)
                self.fake_db.mkdir()
                if self.package_receipt.exists():
                    self.package_receipt.unlink()
                if self.tree_receipt.exists():
                    self.tree_receipt.unlink()
                if self.marker_path.exists():
                    self.marker_path.unlink()
                self.service_log.unlink(missing_ok=True)
                self.seed_package_tmux(version="0.1.0", lane="staging")

                proc_conf = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--confirm-package-command"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_conf.returncode, 0, f"Failed: {proc_conf.stderr}\n{proc_conf.stdout}")
                self.assertIn("desktop install-service marker-exists", self.service_log.read_text(encoding="utf-8"))
                self.assertTrue(self.marker_path.exists())
                self.assertEqual(self.marker_path.stat().st_uid, os.getuid())
            finally:
                server.stop()

    def test_tmux_move_marker_path_ignores_xdg_environment_variables(self):
        with ephemeral_keypair("test marker path ignores xdg") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                # Absolute XDG_CONFIG_HOME
                abs_config = self.work_dir / "abs-custom-config"
                test_env = self.env.copy()
                test_env["XDG_CONFIG_HOME"] = str(abs_config)
                self.seed_package_tmux(version="0.1.0", lane="staging")
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                self.assertTrue(self.marker_path.exists())
                self.assertFalse((abs_config / "solstone-linux" / "desktop-capture-off").exists())

                # Relative XDG_CONFIG_HOME
                if self.package_receipt.exists():
                    self.package_receipt.unlink()
                if self.tree_receipt.exists():
                    self.tree_receipt.unlink()
                if self.marker_path.exists():
                    self.marker_path.unlink()
                shutil.rmtree(self.fake_db)
                self.fake_db.mkdir()
                test_env["XDG_CONFIG_HOME"] = "relative-custom-config"
                self.seed_package_tmux(version="0.1.0", lane="staging")
                proc_rel = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_rel.returncode, 0, f"Failed: {proc_rel.stderr}\n{proc_rel.stdout}")
                self.assertTrue(self.marker_path.exists())
            finally:
                server.stop()

    def test_tmux_move_unwritable_marker_directory_installs_nothing(self):
        with ephemeral_keypair("test unwritable marker dir") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                marker_dir = self.marker_path.parent
                marker_dir.parent.mkdir(parents=True, exist_ok=True)
                marker_dir.write_text("blocked", encoding="utf-8")

                pkg_sec = self.seed_package_tmux(version="0.1.0", lane="staging")
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "desktop-failed")
                self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), pkg_sec)
                self.assertNotIn(b"[component:desktop]\n", self.package_receipt.read_bytes())
                self.assertFalse((self.fake_db / "deb" / "solstone-linux").exists())
            finally:
                server.stop()

    def test_tmux_move_adopt_exit_75_leaves_tmux_intact_with_retry_reason(self):
        with ephemeral_keypair("test adopt exit 75") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                test_env = self.env.copy()
                test_env["SOLSTONE_ADOPT_EXIT"] = "75"
                pkg_sec = self.seed_package_tmux(version="0.1.0", lane="staging")

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "retry")
                self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), pkg_sec)
                self.assertTrue(proc.stderr.strip())
            finally:
                server.stop()

    def test_tmux_move_adopt_exit_3_leaves_tmux_intact_with_failed_reason(self):
        with ephemeral_keypair("test adopt exit 3") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                test_env = self.env.copy()
                test_env["SOLSTONE_ADOPT_EXIT"] = "3"
                test_env["SOLSTONE_ADOPT_STDERR"] = "custom adopt error output"
                pkg_sec = self.seed_package_tmux(version="0.1.0", lane="staging")

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["components"]["tmux"]["status"], "failed")
                self.assertEqual(doc["components"]["tmux"]["reason"], "failed")
                self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), pkg_sec)
                self.assertIn("custom adopt error output", proc.stderr)
            finally:
                server.stop()

    def test_tmux_move_adopt_nonzero_leaves_tmux_intact_and_desktop_stands(self):
        with ephemeral_keypair("test adopt nonzero") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                test_env = self.env.copy()
                test_env["SOLSTONE_ADOPT_EXIT"] = "9"
                pkg_sec = self.seed_package_tmux(version="0.1.0", lane="staging")

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["components"]["tmux"]["status"], "failed")
                self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), pkg_sec)
                self.assertIn(b"[component:desktop]\n", self.package_receipt.read_bytes())
            finally:
                server.stop()

    def test_tmux_move_removal_failure_leaves_section_with_removal_failed_reason(self):
        with ephemeral_keypair("test removal failure") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                pkg_sec = self.seed_package_tmux(version="0.1.0", lane="staging")

                code, tty_bytes, stdout, stderr = self.run_pty_interaction(
                    f"{installer} --skip-signature --upgrade --confirm-package-command",
                    self.env,
                    [
                        (b"move to solstone app?", b"yes\n"),
                        (b"run this package command?", b"yes\n"),
                        (b"run this package command?", b"no\n"),
                    ],
                )
                self.assertEqual(code, 0, f"Failed with code {code}: {stderr}\n{stdout}")
                self.assertIn("curl -fsSL https://solstone.app/install.sh", stderr)
                self.assertIn("--uninstall --components tmux", stderr)
                self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), pkg_sec)
                self.assertIn(b"[component:desktop]\n", self.package_receipt.read_bytes())
            finally:
                server.stop()

    def test_tmux_move_as_root_keeps_tmux_with_root_reason_and_no_remedy(self):
        with ephemeral_keypair("test move as root") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                id_stub = (
                    "#!/bin/sh\n"
                    "case \"${1:-}\" in\n"
                    "  -u) echo 0 ;;\n"
                    "  -un) echo root ;;\n"
                    "  *) exec /usr/bin/id \"$@\" ;;\n"
                    "esac\n"
                )
                write_path_stub(self.bin_dir, "id", id_stub)
                pkg_sec = self.seed_package_tmux(version="0.1.0", lane="staging")

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "root")
                self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), pkg_sec)
                self.assertNotIn("update --lane", proc.stderr)
                self.assertNotIn("curl -fsSL https://solstone.app", proc.stderr)
            finally:
                server.stop()

    def test_tmux_move_unavailable_architecture_keeps_tmux_with_no_remedy(self):
        with ephemeral_keypair("test unavailable architecture") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20, desktop_arches=("x86_64",))
            try:
                test_env = self.env.copy()
                test_env["SOLSTONE_TEST_HOST_ARCH"] = "aarch64"
                tree_sec = self.seed_tree_tmux(version="0.1.0", lane="staging", arch="aarch64")

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "unavailable")
                self.assertEqual(receipt_section(self.tree_receipt.read_bytes(), "component:tmux"), tree_sec)
                self.assertNotIn("update --lane", proc.stderr)
                self.assertNotIn("curl -fsSL https://solstone.app", proc.stderr)
            finally:
                server.stop()

    def test_fresh_install_components_tmux_refused_on_tmux_less_catalogue(self):
        with ephemeral_keypair("test fresh tmux refused") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "tmux", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["root_code"], "invalid-selection")
            finally:
                server.stop()

    def test_uninstall_recorded_tmux_on_tmux_less_catalogue_succeeds_from_receipt(self):
        with ephemeral_keypair("test uninstall recorded tmux") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.seed_package_tmux(version="0.1.0", lane="staging")

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "tmux", "--yes", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["root_code"], "uninstalled")
                self.assertFalse((self.fake_db / "deb" / "solstone-tmux").exists())
                self.assertFalse(self.package_receipt.exists())
            finally:
                server.stop()

    def test_help_menu_all_list_omit_tmux_on_tmux_less_catalogue(self):
        with ephemeral_keypair("test omit tmux floor 20") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                # 1. --help
                proc_help = subprocess.run(
                    [str(installer), "--help", "--skip-signature"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_help.returncode, 0)
                self.assertNotIn("desktop, tmux", proc_help.stdout)

                # 2. interactive menu
                code, tty_bytes, _, _ = self.run_pty_interaction(
                    f"{installer} --skip-signature",
                    self.env,
                    [(b"Select components by number", b"2\n")],
                )
                self.assertEqual(code, 0)
                self.assertIn(b"(1-3)", tty_bytes)
                self.assertNotIn(b"(1-4)", tty_bytes)
                self.assertNotIn(b"Solstone Tmux", tty_bytes)

                # 3. --components all --dry-run --json
                proc_all = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "all", "--dry-run", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_all.returncode, 0)
                doc_all = json.loads(proc_all.stdout)
                self.assertNotIn("tmux", doc_all["components"])
                self.assertIn("desktop", doc_all["components"])
                self.assertIn("journal", doc_all["components"])

                # 4. --list --json
                proc_list = subprocess.run(
                    [str(installer), "--skip-signature", "--list", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_list.returncode, 0)
                doc_list = json.loads(proc_list.stdout)
                self.assertNotIn("tmux", doc_list["components"])
            finally:
                server.stop()

    def test_help_menu_all_list_include_tmux_on_three_component_catalogue(self):
        with ephemeral_keypair("test include tmux floor 1") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=1)
            try:
                # 1. --help
                proc_help = subprocess.run(
                    [str(installer), "--help", "--skip-signature"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_help.returncode, 0)
                self.assertIn("Linux: journal, cli, desktop, tmux", proc_help.stdout)

                # 2. interactive menu
                code, tty_bytes, _, _ = self.run_pty_interaction(
                    f"{installer} --skip-signature",
                    self.env,
                    [(b"Select components by number", b"2\n")],
                )
                self.assertEqual(code, 0)
                self.assertIn(b"(1-4)", tty_bytes)
                self.assertIn(b"tmux", tty_bytes)

                # 3. --components all --dry-run --json
                proc_all = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "all", "--dry-run", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_all.returncode, 0)
                doc_all = json.loads(proc_all.stdout)
                self.assertIn("tmux", doc_all["components"])

                # 4. --list --json
                proc_list = subprocess.run(
                    [str(installer), "--skip-signature", "--list", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_list.returncode, 0)
                doc_list = json.loads(proc_list.stdout)
                self.assertIn("tmux", doc_list["components"])
            finally:
                server.stop()

    def test_revision_19_behaviour_preserved_on_three_component_catalogue(self):
        with ephemeral_keypair("test rev 19 on 3-comp") as (sec, pub, pin):
            server, installer = self.build_release(
                sec,
                pub,
                pin,
                min_installer_revision=1,
                override_installer_revision=19,
            )
            try:
                self.seed_tree_tmux(version="0.0.9", lane="staging")

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["root_code"], "installed")
                self.assertEqual(doc["components"]["tmux"]["status"], "succeeded")
                self.assertNotIn("reason", doc["components"]["tmux"])
                self.assertNotIn("update --lane", proc.stderr)
            finally:
                server.stop()

    def test_headless_fresh_package_move_adds_weak_dep_flag_and_skips_app_hint(self):
        with ephemeral_keypair("test headless fresh package move") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                test_env = self.env.copy()
                test_env.pop("DISPLAY", None)
                test_env.pop("WAYLAND_DISPLAY", None)
                write_path_stub(self.bin_dir, "loginctl", "exit 1\n")

                # Confirm path
                self.seed_package_tmux(version="0.1.0", lane="staging")
                proc_conf = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--confirm-package-command"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_conf.returncode, 0, f"Failed: {proc_conf.stderr}\n{proc_conf.stdout}")
                self.assertIn("--no-install-recommends", proc_conf.stderr)
                self.assertNotIn("app menu", proc_conf.stderr)

                # Non-confirm path
                shutil.rmtree(self.fake_db)
                self.fake_db.mkdir()
                if self.package_receipt.exists():
                    self.package_receipt.unlink()
                if self.tree_receipt.exists():
                    self.tree_receipt.unlink()
                if self.marker_path.exists():
                    self.marker_path.unlink()
                self.seed_package_tmux(version="0.1.0", lane="staging")

                proc_nonconf = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_nonconf.returncode, 0, f"Failed: {proc_nonconf.stderr}\n{proc_nonconf.stdout}")
                log_lines = (self.fake_db / "install.log").read_text(encoding="utf-8").splitlines()
                desk_lines = [l for l in log_lines if "solstone-linux" in l]
                self.assertTrue(desk_lines)
                self.assertTrue(desk_lines[-1].endswith("no-weak-deps"))
            finally:
                server.stop()

    def test_graphical_package_move_omits_weak_dep_flag_and_keeps_app_hint(self):
        with ephemeral_keypair("test graphical package move") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                test_env = self.env.copy()
                test_env["DISPLAY"] = ":1"
                write_path_stub(self.bin_dir, "loginctl", 'printf ":1\\n"\nexit 0\n')

                # Confirm path
                self.seed_package_tmux(version="0.1.0", lane="staging")
                proc_conf = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--confirm-package-command"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_conf.returncode, 0, f"Failed: {proc_conf.stderr}\n{proc_conf.stdout}")
                self.assertNotIn("--no-install-recommends", proc_conf.stderr)
                self.assertIn("app menu", proc_conf.stderr)

                # Non-confirm path
                shutil.rmtree(self.fake_db)
                self.fake_db.mkdir()
                if self.package_receipt.exists():
                    self.package_receipt.unlink()
                if self.tree_receipt.exists():
                    self.tree_receipt.unlink()
                if self.marker_path.exists():
                    self.marker_path.unlink()
                self.seed_package_tmux(version="0.1.0", lane="staging")

                proc_nonconf = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_nonconf.returncode, 0, f"Failed: {proc_nonconf.stderr}\n{proc_nonconf.stdout}")
                log_lines = (self.fake_db / "install.log").read_text(encoding="utf-8").splitlines()
                desk_lines = [l for l in log_lines if "solstone-linux" in l]
                self.assertTrue(desk_lines)
                self.assertFalse(desk_lines[-1].endswith("no-weak-deps"))
            finally:
                server.stop()

    def test_plan_pkg_arity_preserved_for_ordinary_confirm_install(self):
        with ephemeral_keypair("test ordinary confirm plan arity") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                plan_log = self.work_dir / "plan_arity.log"
                wrapper_bin = self.bin_dir / "wrapper-helper-arity"
                wrapper_bin.write_text(
                    f"#!/bin/sh\n"
                    f"real_helper='{HELPER_SCRIPT}'\n"
                    f"\"$real_helper\" \"$@\" | while IFS= read -r line || [ -n \"$line\" ]; do\n"
                    f"    case \"$line\" in PLAN*) printf '%s\\n' \"$line\" >> '{plan_log}' ;; esac\n"
                    f"    printf '%s\\n' \"$line\"\n"
                    f"done\n",
                    encoding="utf-8",
                )
                wrapper_bin.chmod(0o755)

                test_env = self.env.copy()
                test_env["SOLSTONE_HELPER"] = str(wrapper_bin)

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--route", "deb", "--components", "desktop", "--confirm-package-command", "--yes"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                plan_line = plan_log.read_text(encoding="utf-8").strip()
                fields = plan_line.split()
                self.assertEqual(len(fields), 11, f"Expected 11 fields, got {len(fields)}: {fields}")
                self.assertTrue(fields[-1].endswith(".deb"))
                self.assertNotIn("--no-install-recommends", fields)
            finally:
                server.stop()

    def test_zero_byte_package_receipt_deleted_when_helper_already_used(self):
        with ephemeral_keypair("test zero byte receipt deleted") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
                self.package_receipt.touch()
                self.assertTrue(self.package_receipt.exists())

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--route", "deb", "--uninstall", "--components", "desktop", "--yes", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                self.assertFalse(self.package_receipt.exists())
            finally:
                server.stop()

    def test_zero_byte_package_receipt_left_intact_when_helper_not_in_use(self):
        with ephemeral_keypair("test zero byte receipt left intact") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
                self.package_receipt.touch()
                self.assertTrue(self.package_receipt.exists())

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--route", "tree", "--components", "desktop", "--prefix", str(self.tree_prefix), "--yes", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                self.assertTrue(self.package_receipt.exists())
                self.assertEqual(self.package_receipt.stat().st_size, 0)
            finally:
                server.stop()

    def test_delete_etc_receipt_used_when_uninstall_empties_package_receipt(self):
        with ephemeral_keypair("test delete etc receipt on empty") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.seed_package_tmux(version="0.1.0", lane="staging")

                helper_in_log = self.work_dir / "helper_in.log"
                wrapper_bin = self.bin_dir / "wrapper-helper-in"
                wrapper_bin.write_text(
                    f"#!/bin/sh\n"
                    f"real_helper='{HELPER_SCRIPT}'\n"
                    f"tee -a '{helper_in_log}' | \"$real_helper\" \"$@\"\n",
                    encoding="utf-8",
                )
                wrapper_bin.chmod(0o755)

                test_env = self.env.copy()
                test_env["SOLSTONE_HELPER"] = str(wrapper_bin)

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--uninstall", "--components", "tmux", "--yes", "--json"],
                    env=test_env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                log_text = helper_in_log.read_text(encoding="utf-8")
                self.assertIn("DELETE_ETC_RECEIPT", log_text)
                self.assertNotIn("WRITE_ETC_RECEIPT 0", log_text)
                self.assertFalse(self.package_receipt.exists())
            finally:
                server.stop()

    def test_remedy_line_tree_journal_argv_solstone_update(self):
        with ephemeral_keypair("test remedy tree journal") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, lane="staging", min_installer_revision=20)
            try:
                journal_ver = "2.0.5"
                versions_dir = self.tree_prefix / "versions" / f"{journal_ver}-000000000000" / "bin"
                versions_dir.mkdir(parents=True, exist_ok=True)
                for b_name in ("journal", "solstone"):
                    b_file = versions_dir / b_name
                    b_file.write_text(f"#!/bin/sh\necho {journal_ver}\n", encoding="utf-8")
                    b_file.chmod(0o755)
                cur = self.tree_prefix / "current"
                if cur.is_symlink() or cur.exists():
                    cur.unlink()
                cur.symlink_to(f"versions/{journal_ver}-000000000000")
                (self.tree_prefix / "install-receipt").write_text(
                    f"route=tree\nrole=journal\njournal_version={journal_ver}\n", encoding="utf-8"
                )

                self.seed_tree_tmux(prefix=self.tree_prefix, version="0.1.0", lane="staging")

                header = b"[solstone]\nschema_version=1\nplatform_version=1.9.0\nlane=staging\n"
                journal_sec = (
                    f"[component:journal]\n"
                    f"role=journal\nroute=tree\narch=x86_64\nversion={journal_ver}\nstatus=installed\n"
                    f"phase=complete\nprefix={self.tree_prefix}\n"
                ).encode("utf-8")
                tmux_sec = receipt_section(self.tree_receipt.read_bytes(), "component:tmux")
                self.tree_receipt.write_bytes(header + journal_sec + tmux_sec)

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--lane", "staging", "--non-interactive", "--json", "--prefix", str(self.tree_prefix)],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                remedy_lines = [l for l in proc.stderr.splitlines() if "update --lane" in l]
                self.assertEqual(len(remedy_lines), 1, f"Expected 1 remedy line, got: {proc.stderr}")
                self.assertEqual(shlex.split(remedy_lines[0]), [f"{self.tree_prefix}/bin/solstone", "update", "--lane", "staging"])
            finally:
                server.stop()

    def test_remedy_line_package_journal_argv_solstone_update(self):
        with ephemeral_keypair("test remedy package journal") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, lane="staging", min_installer_revision=20)
            try:
                self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
                header = b"[solstone]\nschema_version=1\nplatform_version=1.9.0\nlane=staging\n"
                journal_sec = (
                    f"[component:journal]\n"
                    f"phase=complete\nstatus=installed\nrole=journal\nroute=deb\nservice_policy=start\n"
                    f"package_name=solstone-journal\npackage_version=2.0.6\npackage_arch=amd64\n"
                    f"artifact_sha256={'0'*64}\npayload_build_id={'0'*64}\n"
                ).encode("utf-8")
                tmux_sec = (
                    f"[component:tmux]\n"
                    f"phase=complete\nstatus=installed\nrole=tmux\nroute=deb\nservice_policy=start\n"
                    f"package_name=solstone-tmux\npackage_version=0.1.0\npackage_arch=amd64\n"
                    f"artifact_sha256={'0'*64}\npayload_build_id={'0'*64}\n"
                ).encode("utf-8")
                self.package_receipt.write_bytes(header + journal_sec + tmux_sec)
                db_dir = self.fake_db / "deb"
                db_dir.mkdir(parents=True, exist_ok=True)
                (db_dir / "solstone-journal").write_text("INSTALLED solstone-journal 2.0.6 amd64\n", encoding="utf-8")
                (db_dir / "solstone-tmux").write_text("INSTALLED solstone-tmux 0.1.0 amd64\n", encoding="utf-8")

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--lane", "staging", "--non-interactive", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                remedy_lines = [l for l in proc.stderr.splitlines() if "update --lane" in l]
                self.assertEqual(len(remedy_lines), 1, f"Expected 1 remedy line, got: {proc.stderr}")
                self.assertEqual(shlex.split(remedy_lines[0]), ["solstone", "update", "--lane", "staging"])
            finally:
                server.stop()

    def test_remedy_line_direct_upgrade_handles_space_in_prefix(self):
        with ephemeral_keypair("test remedy direct upgrade space") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, lane="staging", min_installer_revision=20)
            try:
                space_prefix = self.work_dir / "my space prefix"
                self.seed_tree_tmux(prefix=space_prefix, version="0.1.0", lane="staging", no_start=1, no_path=1)

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--lane", "staging", "--non-interactive", "--json", "--prefix", str(space_prefix)],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}\n{proc.stdout}")
                remedy_lines = [l for l in proc.stderr.splitlines() if l.startswith("curl -fsSL")]
                self.assertEqual(len(remedy_lines), 1, f"Expected 1 curl remedy line, got: {proc.stderr}")
                remedy_line = remedy_lines[0]
                tokens = shlex.split(remedy_line)
                sh_idx = tokens.index("--")
                sub_args = tokens[sh_idx + 1 :]
                self.assertIn("--prefix", sub_args)
                self.assertEqual(sub_args[sub_args.index("--prefix") + 1], str(space_prefix))

                proc_up = subprocess.run(
                    [str(installer), *sub_args, "--yes", "--skip-signature", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_up.returncode, 0, f"Failed: {proc_up.stderr}\n{proc_up.stdout}")
                doc = json.loads(proc_up.stdout)
                self.assertEqual(doc["root_code"], "installed")
                self.assertEqual(doc["components"]["tmux"]["status"], "moved")
                desktop_sec = receipt_section(self.tree_receipt.read_bytes(), "component:desktop")
                self.assertIn(b"no_start=1", desktop_sec)
                self.assertIn(b"no_path=1", desktop_sec)
            finally:
                server.stop()

    def test_installer_revision_19_fails_floor_20_with_revision_too_old(self):
        with ephemeral_keypair("test rev 19 fails floor 20") as (sec, pub, pin):
            server, installer = self.build_release(
                sec,
                pub,
                pin,
                min_installer_revision=20,
                override_installer_revision=19,
            )
            try:
                proc = subprocess.run(
                    [str(installer), "--components", "desktop", "--json", "--skip-signature", "--prefix", str(self.tree_prefix)],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                doc = json.loads(proc.stdout)
                self.assertEqual(doc["root_code"], "revision-too-old")
                self.assertFalse(self.tree_receipt.exists())
                self.assertFalse(self.package_receipt.exists())
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
