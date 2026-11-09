# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""A recorded tmux app on a catalogue without tmux: kept, or moved into the desktop app."""

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
    snapshot_paths,
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


def receipt_header(receipt: bytes) -> dict[str, str]:
    values: dict[str, str] = {}
    section = None
    for line in receipt.decode("utf-8").splitlines():
        if line.startswith("["):
            section = line
            continue
        if section == "[solstone]" and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def receipt_sections(receipt: bytes) -> list[str]:
    return [line for line in receipt.decode("utf-8").splitlines() if line.startswith("[")]


def remedy_commands(stderr: str) -> list[list[str]]:
    """Every stderr line that parses to a command reaching the move again."""
    found = []
    for line in stderr.splitlines():
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if len(tokens) == 4 and tokens[1:3] == ["update", "--lane"] and tokens[0].endswith("solstone"):
            found.append(tokens)
        elif tokens[:2] == ["curl", "-fsSL"] and "--" in tokens and "--upgrade" in tokens[tokens.index("--") + 1:]:
            found.append(tokens)
    return found


def installer_arguments(command: list[str]) -> list[str]:
    return command[command.index("--") + 1:]


def uninstall_commands(stderr: str) -> list[list[str]]:
    found = []
    for line in stderr.splitlines():
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if tokens[:2] == ["curl", "-fsSL"] and "--" in tokens and "--uninstall" in installer_arguments(tokens):
            found.append(installer_arguments(tokens))
    return found


class TestInstallTmuxMove(HermeticInstallerTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.bin_dir = self.work_dir / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        setup_fake_sudo(self.bin_dir)
        # The user manager's environment comes from SOLSTONE_TEST_MANAGER_ENV;
        # every other systemctl call still reaches the hermetic guard.
        guard = shutil.which("systemctl", path=os.environ["PATH"])
        write_path_stub(
            self.bin_dir,
            "systemctl",
            "if [ \"$*\" = \"--user show-environment\" ]; then\n"
            "  [ -z \"${SOLSTONE_TEST_MANAGER_ENV:-}\" ] || printf '%s\\n' \"$SOLSTONE_TEST_MANAGER_ENV\"\n"
            "  exit 0\n"
            "fi\n"
            f"exec {shlex.quote(str(guard))} \"$@\"\n",
        )
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
            "    [ \"${1:-}\" != install-service ] || [ -z \"${SOLSTONE_DESKTOP_SERVICE_FAIL:-}\" ] || exit 9\n"
            "    exit 0 ;;\n"
            "esac\n"
            "if [ \"${1:-}\" = \"terminal\" ] && [ \"${2:-}\" = \"adopt-tmux\" ]; then\n"
            "  [ -z \"${SOLSTONE_ADOPT_LOG:-}\" ] || printf '%s\\n' \"$*\" >> \"$SOLSTONE_ADOPT_LOG\"\n"
            "  [ -z \"${SOLSTONE_ADOPT_STDOUT:-}\" ] || printf '%s\\n' \"$SOLSTONE_ADOPT_STDOUT\"\n"
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

    def run_installer(self, installer: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        # A new session has no controlling terminal: the run sees no tty.
        return subprocess.run(
            [str(installer), "--skip-signature", *args],
            env=self.env if env is None else env,
            capture_output=True,
            text=True,
            start_new_session=True,
        )

    def service_log_text(self) -> str:
        return self.service_log.read_text(encoding="utf-8") if self.service_log.exists() else ""

    def assert_ran(self, proc: subprocess.CompletedProcess, code: int = 0) -> dict:
        self.assertEqual(proc.returncode, code, f"{proc.stderr}\n{proc.stdout}")
        return json.loads(proc.stdout)

    def add_catalogue(self, sec: Path, pin, *, platform_version: str, desktop_version: str, lane: str = "release") -> None:
        """Publish one more floor-20 catalogue beside the first, and point latest at it."""
        # The first catalogue's objects are already published; build fresh natives.
        shutil.rmtree(self.work_dir / "natives", ignore_errors=True)
        server, _ = setup_test_release_server(
            self.work_dir,
            sec,
            pin,
            lane=lane,
            version=platform_version,
            min_installer_revision=20,
            desktop_version=desktop_version,
        )
        # Its journal bootstrap URL names this server, which serves the same files.
        self.addCleanup(server.stop)

    def append_package_section(self, section: bytes) -> None:
        self.package_receipt.write_bytes(self.package_receipt.read_bytes() + section)

    def seed_package_tmux(self, version: str = "0.1.0", lane: str = "staging", policy: str = "start") -> bytes:
        self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
        tmux_section = (
            f"[component:tmux]\n"
            f"phase=complete\nstatus=installed\nrole=tmux\nroute=deb\nservice_policy={policy}\n"
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
        append: bool = False,
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
        exec_content = (
            b"#!/bin/sh\n"
            b"[ -z \"${SOLSTONE_TEST_SERVICE_LOG:-}\" ] || printf 'tmux %s\\n' \"$1\" >> \"$SOLSTONE_TEST_SERVICE_LOG\"\n"
            b"exit 0\n"
        )
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
        if append:
            self.tree_receipt.write_bytes(self.tree_receipt.read_bytes() + tmux_section)
            return tmux_section
        header = (
            f"[solstone]\nschema_version=1\nplatform_version=1.9.0\nlane={lane}\n"
        ).encode("utf-8")
        self.tree_receipt.write_bytes(header + tmux_section)
        return tmux_section

    def tmux_tree_paths(self, prefix: Path | None = None) -> list[Path]:
        prefix = prefix or Path(os.environ["HOME"]) / ".local"
        return [prefix / "opt" / "solstone" / "tmux", prefix / "bin" / "solstone-tmux"]

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

                tree_header = receipt_header(self.tree_receipt.read_bytes())
                package_header = receipt_header(self.package_receipt.read_bytes())
                for header in (tree_header, package_header):
                    self.assertEqual(header["platform_version"], "2.0.0")
                    self.assertEqual(header["lane"], "release")
                    self.assertEqual(header["origin"], server.origin)
                    self.assertEqual(header["installer_revision"], (REPO_ROOT / "compat" / "installer_revision").read_text(encoding="utf-8").strip())
                self.assertEqual(tree_header["route"], "tree")
                self.assertEqual(tree_header["phase"], "complete")
                self.assertIn("prefix", tree_header)
                self.assertNotIn("route", package_header)
                stderr_nonempty = [l for l in proc.stderr.splitlines() if l.strip()]
                self.assertEqual(len(stderr_nonempty), 1)
                self.assertEqual(len(remedy_commands(proc.stderr)), 1)
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
                self.assertEqual(receipt_sections(self.package_receipt.read_bytes()), ["[solstone]", "[component:desktop]"])
                self.assertEqual(receipt_header(self.package_receipt.read_bytes())["platform_version"], "2.0.0")
                self.assertEqual(self.adopt_log.read_text(encoding="utf-8").splitlines()[-1], "terminal adopt-tmux --yes")
            finally:
                server.stop()

    def test_tree_move_with_an_app_that_cannot_load_changes_nothing_and_can_be_retried(self):
        with ephemeral_keypair("test unloadable desktop") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                tmux_section = self.seed_tree_tmux(version="0.1.0", lane="staging")
                home_prefix = Path(os.environ["HOME"]) / ".local"
                blocked = self.env.copy()
                blocked["SOLSTONE_DESKTOP_UNLOADABLE"] = "1"
                refused = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=blocked,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(refused.returncode, 0, refused.stdout)
                self.assertEqual(json.loads(refused.stdout)["root_code"], "runtime-libraries-missing")
                self.assertEqual(safe_receipt_section(self.tree_receipt.read_bytes(), "component:tmux"), tmux_section)
                self.assertFalse((home_prefix / "opt" / "solstone" / "desktop").exists())
                self.assertTrue((home_prefix / "bin" / "solstone-tmux").exists())

                moved = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(moved.returncode, 0, f"Failed: {moved.stderr}\n{moved.stdout}")
                self.assertEqual(json.loads(moved.stdout)["components"]["tmux"]["status"], "moved")
                self.assertTrue((home_prefix / "bin" / "solstone-linux").exists())
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

    def test_terminal_only_line_shows_only_when_no_desktop_config_exists(self):
        with ephemeral_keypair("test terminal only line") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.seed_tree_tmux(version="0.1.0", lane="release")

                def consent_lines(env: dict[str, str]) -> int:
                    code, tty_bytes, stdout, stderr = self.run_pty_interaction(
                        f"{installer} --skip-signature --upgrade", env, [(b"yes or no", b"no\n")],
                    )
                    self.assertEqual(code, 0, f"{stderr}\n{stdout}")
                    before = tty_bytes[: tty_bytes.index(b"yes or no")].decode("utf-8")
                    return sum(1 for line in before.splitlines() if line.strip())

                def config_under(root: Path) -> Path:
                    config = root / "solstone-linux" / "config.json"
                    config.parent.mkdir(parents=True, exist_ok=True)
                    config.write_text("{}\n", encoding="utf-8")
                    return config

                none_anywhere = consent_lines(self.env)

                shell_xdg = self.work_dir / "shell-config"
                config_under(shell_xdg)
                in_shell_xdg = consent_lines({**self.env, "XDG_CONFIG_HOME": str(shell_xdg)})

                manager_xdg = self.work_dir / "manager-config"
                config_under(manager_xdg)
                manager_env = f"HOME={os.environ['HOME']}\nXDG_CONFIG_HOME={manager_xdg}"
                in_manager_xdg = consent_lines({**self.env, "SOLSTONE_TEST_MANAGER_ENV": manager_env})
                relative_manager = consent_lines({**self.env, "SOLSTONE_TEST_MANAGER_ENV": "XDG_CONFIG_HOME=manager-config"})

                home_config = config_under(Path(os.environ["HOME"]) / ".config")
                in_home = consent_lines(self.env)
                home_config.unlink()

                self.assertEqual(none_anywhere, in_shell_xdg + 1)
                self.assertEqual(in_manager_xdg, in_shell_xdg)
                self.assertEqual(in_home, in_shell_xdg)
                self.assertEqual(relative_manager, none_anywhere)
                self.assertEqual(self.tree_receipt.read_bytes().count(b"[component:tmux]\n"), 1)
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
                self.assertEqual(doc["status"], "success")
                self.assertNotEqual(doc["root_code"], "installed")
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "desktop-failed")
                self.assertNotIn("desktop", doc["components"])
                self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), pkg_sec)
                self.assertNotIn(b"[component:desktop]\n", self.package_receipt.read_bytes())
                self.assertFalse((self.fake_db / "deb" / "solstone-linux").exists())
                self.assertFalse(self.adopt_log.exists())
                self.assertEqual(sum(str(self.marker_path) in line for line in proc.stderr.splitlines()), 1)
                self.assertEqual(len(remedy_commands(proc.stderr)), 1)
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
                # Removal waits for adopt's answer: the tmux package is still installed.
                self.assertTrue((self.fake_db / "deb" / "solstone-tmux").exists())
                self.assertNotIn("tmux uninstall-service", self.service_log.read_text(encoding="utf-8"))
                self.assertIn("desktop install-service marker-exists", self.service_log.read_text(encoding="utf-8"))
                self.assertEqual(self.adopt_log.read_text(encoding="utf-8").splitlines(), ["terminal adopt-tmux --yes"])
                remedies = remedy_commands(proc.stderr)
                self.assertEqual(len(remedies), 1)
                self.assertEqual(installer_arguments(remedies[0])[:5], ["--upgrade", "--components", "tmux", "--lane", "release"])
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
                test_env["SOLSTONE_ADOPT_STDERR"] = "first diagnostic line\n\nthe last diagnostic line\n"
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
                self.assertIn(b"[component:desktop]\n", self.package_receipt.read_bytes())
                self.assertTrue((self.fake_db / "deb" / "solstone-tmux").exists())
                self.assertIn("desktop install-service marker-exists", self.service_log.read_text(encoding="utf-8"))
                self.assertEqual(sum("the last diagnostic line" in line for line in proc.stderr.splitlines()), 1)
                self.assertNotIn("first diagnostic line", proc.stderr)
                self.assertEqual(len(remedy_commands(proc.stderr)), 1)
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
                        (b"yes or no", b"yes\n"),
                        (b"yes or no", b"yes\n"),
                        (b"yes or no", b"no\n"),
                    ],
                )
                self.assertEqual(code, 0, f"Failed with code {code}: {stderr}\n{stdout}")
                self.assertEqual(uninstall_commands(stderr), [["--uninstall", "--components", "tmux", "--lane", "release"]])
                self.assertEqual(remedy_commands(stderr), [])
                self.assertTrue((self.fake_db / "deb" / "solstone-tmux").exists())
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
                self.assertTrue((self.fake_db / "deb" / "solstone-tmux").exists())
                self.assertEqual(remedy_commands(proc.stderr), [])
                self.assertEqual(len([l for l in proc.stderr.splitlines() if l.strip()]), 1)
                self.assertFalse(self.marker_path.exists())

                # The tree route, the other way a root run reaches the move.
                self.package_receipt.unlink()
                tree_sec = self.seed_tree_tmux(version="0.1.0", lane="staging")
                before = snapshot_paths(*self.tmux_tree_paths())
                proc_tree = subprocess.run(
                    [str(installer), "--skip-signature", "--upgrade", "--yes", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                doc_tree = self.assert_ran(proc_tree)
                self.assertEqual(doc_tree["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc_tree["components"]["tmux"]["reason"], "root")
                self.assertEqual(receipt_section(self.tree_receipt.read_bytes(), "component:tmux"), tree_sec)
                self.assertEqual(snapshot_paths(*self.tmux_tree_paths()), before)
                self.assertEqual(remedy_commands(proc_tree.stderr), [])
                self.assertEqual(len([l for l in proc_tree.stderr.splitlines() if l.strip()]), 1)
                self.assertNotIn(b"[component:desktop]", self.tree_receipt.read_bytes())
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
                self.assertEqual(remedy_commands(proc.stderr), [])
                self.assertEqual(len([l for l in proc.stderr.splitlines() if l.strip()]), 1)
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
                # 1. interactive menu
                code, tty_bytes, _, _ = self.run_pty_interaction(
                    f"{installer} --skip-signature",
                    self.env,
                    [(b"Select components by number", b"2\n")],
                )
                self.assertEqual(code, 0)
                self.assertIn(b"(1-3)", tty_bytes)
                self.assertNotIn(b"(1-4)", tty_bytes)
                self.assertNotIn(b"Solstone Tmux", tty_bytes)

                # 2. --components all --dry-run --json, with tmux on PATH
                write_path_stub(self.bin_dir, "tmux", "exit 0\n")
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

                # 3. --list --json
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

    def test_help_prints_without_reaching_the_catalogue(self):
        with ephemeral_keypair("test help offline") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            server.stop()
            requests_before = len(server.request_paths)
            for args in (["--help"], ["-h"], ["--json", "--help"], ["--origin", "http://127.0.0.1:9", "--help"]):
                with self.subTest(args=args):
                    proc = subprocess.run([str(installer), *args], env=self.env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    text = json.loads(proc.stdout)["help"] if "--json" in args else proc.stdout
                    linux_line = next(line for line in text.splitlines() if "--components LIST" in line)
                    for component in ("journal", "cli", "desktop"):
                        self.assertIn(component, linux_line)
                    # tmux appears only where it is removed from a receipt.
                    tmux_lines = [line for line in text.splitlines() if "tmux" in line]
                    self.assertTrue(tmux_lines)
                    self.assertTrue(all("--uninstall" in line for line in tmux_lines), tmux_lines)
            self.assertEqual(len(server.request_paths), requests_before)

    def test_help_menu_all_list_include_tmux_on_three_component_catalogue(self):
        with ephemeral_keypair("test include tmux floor 1") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=1)
            try:
                # 1. interactive menu
                code, tty_bytes, _, _ = self.run_pty_interaction(
                    f"{installer} --skip-signature",
                    self.env,
                    [(b"Select components by number", b"2\n")],
                )
                self.assertEqual(code, 0)
                self.assertIn(b"(1-4)", tty_bytes)
                self.assertIn(b"tmux", tty_bytes)

                # 2. --components all --dry-run --json, with tmux on PATH
                write_path_stub(self.bin_dir, "tmux", "exit 0\n")
                proc_all = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "all", "--dry-run", "--json"],
                    env=self.env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc_all.returncode, 0)
                doc_all = json.loads(proc_all.stdout)
                self.assertIn("tmux", doc_all["components"])

                # 3. --list --json
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

    def test_tmux_carrying_catalogue_upgrades_and_installs_tmux_without_asking(self):
        with ephemeral_keypair("test tmux catalogue") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=8)
            try:
                self.seed_tree_tmux(version="0.0.9", lane="release")
                # A terminal is attached and nothing answers: an installer that
                # asked would wait here until the harness gave up.
                code, tty_bytes, stdout, stderr = self.run_pty_interaction(
                    f"{installer} --skip-signature --upgrade --json",
                    self.env,
                    [],
                )
                self.assertEqual(code, 0, f"{stderr}\n{stdout}")
                doc = json.loads(stdout)
                self.assertEqual(doc["root_code"], "installed")
                self.assertEqual(doc["components"]["tmux"]["status"], "succeeded")
                self.assertNotIn("reason", doc["components"]["tmux"])
                self.assertEqual(remedy_commands(stderr), [])
                self.assertFalse(self.marker_path.exists())
                tmux_section = receipt_section(self.tree_receipt.read_bytes(), "component:tmux")
                self.assertNotIn(b"version=0.0.9\n", tmux_section)

                fresh_env = {**self.env, "XDG_DATA_HOME": str(self.work_dir / "fresh-data")}
                proc = self.run_installer(installer, "--components", "tmux", "--prefix", str(self.tree_prefix), "--yes", "--json", env=fresh_env)
                doc_fresh = self.assert_ran(proc)
                self.assertEqual(doc_fresh["components"]["tmux"]["status"], "succeeded")
                fresh_receipt = self.work_dir / "fresh-data" / "solstone" / "install.conf"
                self.assertIn(b"[component:tmux]\n", fresh_receipt.read_bytes())
                self.assertTrue((self.tree_prefix / "bin" / "solstone-tmux").is_symlink())
                self.assertEqual(remedy_commands(proc.stderr), [])
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

    def test_zero_byte_package_receipt_reads_as_absent_for_a_package_install(self):
        with ephemeral_keypair("test zero byte receipt replaced") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
                self.package_receipt.touch()

                proc = self.run_installer(installer, "--route", "deb", "--components", "desktop", "--yes", "--json")
                doc = self.assert_ran(proc)
                self.assertEqual(doc["root_code"], "installed")
                self.assertEqual(receipt_sections(self.package_receipt.read_bytes()), ["[solstone]", "[component:desktop]"])
                self.assertEqual(receipt_header(self.package_receipt.read_bytes())["platform_version"], "2.0.0")
            finally:
                server.stop()

    def test_zero_byte_package_receipt_left_with_one_line_while_the_run_continues(self):
        with ephemeral_keypair("test zero byte receipt left intact") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
                self.package_receipt.touch()

                for extra in ([], ["--upgrade"]):
                    with self.subTest(extra=extra):
                        proc = self.run_installer(
                            installer, *extra, "--route", "tree", "--components", "desktop",
                            "--prefix", str(self.tree_prefix), "--yes", "--json",
                        )
                        doc = self.assert_ran(proc)
                        self.assertEqual(doc["root_code"], "installed")
                        self.assertIn(doc["components"]["desktop"]["status"], ("succeeded", "unchanged"))
                        self.assertIn(b"[component:desktop]\n", self.tree_receipt.read_bytes())
                        self.assertTrue(self.package_receipt.exists())
                        self.assertEqual(self.package_receipt.stat().st_size, 0)
                        self.assertEqual(sum(str(self.package_receipt) in line for line in proc.stderr.splitlines()), 1)
            finally:
                server.stop()

    HEADER_ONLY_RECEIPT = b"[solstone]\nschema_version=1\nplatform_version=1.9.0\nlane=release\n"

    def test_header_only_package_receipt_reads_as_absent_for_a_package_install(self):
        with ephemeral_keypair("test header only receipt replaced") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
                self.package_receipt.write_bytes(self.HEADER_ONLY_RECEIPT)

                proc = self.run_installer(installer, "--route", "deb", "--components", "desktop", "--yes", "--json")
                doc = self.assert_ran(proc)
                self.assertEqual(doc["root_code"], "installed")
                self.assertEqual(receipt_sections(self.package_receipt.read_bytes()), ["[solstone]", "[component:desktop]"])
                self.assertEqual(receipt_header(self.package_receipt.read_bytes())["platform_version"], "2.0.0")
            finally:
                server.stop()

    def test_header_only_package_receipt_left_with_one_line_during_a_tree_move(self):
        with ephemeral_keypair("test header only receipt left intact") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.seed_tree_tmux(version="0.1.0", lane="release")
                self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
                self.package_receipt.write_bytes(self.HEADER_ONLY_RECEIPT)

                proc = self.run_installer(installer, "--upgrade", "--yes", "--json")
                doc = self.assert_ran(proc)
                self.assertEqual(doc["root_code"], "installed")
                self.assertEqual(doc["components"]["tmux"]["status"], "moved")
                self.assertEqual(doc["components"]["tmux"]["reason"], "moved")
                self.assertIn(b"[component:desktop]\n", self.tree_receipt.read_bytes())
                self.assertEqual(self.package_receipt.read_bytes(), self.HEADER_ONLY_RECEIPT)
                self.assertEqual(sum(str(self.package_receipt) in line for line in proc.stderr.splitlines()), 1)

                proc_again = self.run_installer(installer, "--upgrade", "--yes", "--json")
                doc_again = self.assert_ran(proc_again)
                self.assertEqual(doc_again["root_code"], "installed")
                self.assertEqual(self.package_receipt.read_bytes(), self.HEADER_ONLY_RECEIPT)
                self.assertEqual(sum(str(self.package_receipt) in line for line in proc_again.stderr.splitlines()), 1)
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
                log_lines = helper_in_log.read_text(encoding="utf-8").splitlines()
                self.assertIn("DELETE_ETC_RECEIPT", log_lines)
                self.assertNotIn("WRITE_ETC_RECEIPT 0", log_lines)
                # The emptied receipt is first written as its header alone, then deleted.
                delete_at = log_lines.index("DELETE_ETC_RECEIPT")
                writes = [i for i, line in enumerate(log_lines) if line.startswith("WRITE_ETC_RECEIPT ")]
                self.assertTrue(writes and writes[-1] < delete_at)
                self.assertNotIn("[component:", "\n".join(log_lines[writes[-1]:delete_at]))
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

    def install_tree_desktop(self, installer: Path, *args: str) -> bytes:
        proc = self.run_installer(installer, "--components", "desktop", "--yes", "--json", *args)
        self.assert_ran(proc)
        return receipt_section(self.tree_receipt.read_bytes(), "component:desktop")

    def install_tree_journal(self, installer: Path, prefix: Path) -> None:
        proc = self.run_installer(installer, "--components", "journal", "--prefix", str(prefix), "--no-start", "--yes", "--json")
        self.assert_ran(proc)

    def test_adopt_stdout_reaches_stderr_and_json_stays_parseable(self):
        with ephemeral_keypair("test adopt stdout") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.seed_package_tmux(version="0.1.0", lane="staging")
                test_env = {**self.env, "SOLSTONE_ADOPT_STDOUT": "adopt wrote this to stdout"}
                proc = self.run_installer(installer, "--upgrade", "--yes", "--json", env=test_env)
                doc = self.assert_ran(proc)
                self.assertEqual(doc["components"]["tmux"]["status"], "moved")
                self.assertNotIn("adopt wrote this to stdout", proc.stdout)
                self.assertIn("adopt wrote this to stdout", proc.stderr)
            finally:
                server.stop()

    def test_marker_directories_are_created_private_and_existing_ones_are_left_alone(self):
        with ephemeral_keypair("test marker modes") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                home = Path(os.environ["HOME"])
                local, share, marker_dir = home / ".local", home / ".local" / "share", self.marker_path.parent
                cases = (
                    (None, {local: 0o700, share: 0o700, marker_dir: 0o700}),
                    (share, {share: 0o755, marker_dir: 0o700}),
                    (marker_dir, {marker_dir: 0o755}),
                )
                for existing, expected_modes in cases:
                    with self.subTest(existing=str(existing)):
                        shutil.rmtree(local, ignore_errors=True)
                        shutil.rmtree(self.fake_db)
                        self.fake_db.mkdir()
                        self.package_receipt.unlink(missing_ok=True)
                        if existing is not None:
                            existing.mkdir(parents=True)
                            existing.chmod(0o755)
                        self.seed_package_tmux(version="0.1.0", lane="release")
                        self.assert_ran(self.run_installer(installer, "--upgrade", "--yes", "--json"))
                        self.assertTrue(self.marker_path.is_file())
                        self.assertEqual(self.marker_path.stat().st_size, 0)
                        for directory, mode in expected_modes.items():
                            self.assertEqual(directory.stat().st_mode & 0o777, mode, directory)
            finally:
                server.stop()

    def test_no_start_desktop_one_version_behind_moves_in_a_start_invocation(self):
        with ephemeral_keypair("test recorded no-start desktop") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                prefix = Path(os.environ["HOME"]) / ".local"
                self.install_tree_journal(installer, prefix)
                self.install_tree_desktop(installer, "--prefix", str(prefix), "--no-start")
                self.seed_tree_tmux(prefix=prefix, version="0.1.0", append=True)
                self.add_catalogue(sec, pin, platform_version="2.0.1", desktop_version="2.0.38")
                self.service_log.unlink(missing_ok=True)

                bootstrap_args = self.work_dir / "bootstrap-args.log"
                test_env = {**self.env, "SOLSTONE_BOOTSTRAP_ARGS_LOG": str(bootstrap_args)}
                proc = self.run_installer(
                    installer, "--upgrade", "--components", "journal,tmux", "--prefix", str(prefix),
                    "--no-path", "--yes", "--json", env=test_env,
                )
                doc = self.assert_ran(proc)
                self.assertEqual(doc["components"]["tmux"]["status"], "moved")
                self.assertEqual(doc["components"]["desktop"]["status"], "succeeded")
                receipt = self.tree_receipt.read_bytes()
                desktop = receipt_section(receipt, "component:desktop")
                self.assertIn(b"version=2.0.38\n", desktop)
                self.assertIn(b"no_start=1\n", desktop)
                self.assertIn(b"no_path=0\n", desktop)
                self.assertEqual(safe_receipt_section(receipt, "component:tmux"), b"")
                # The journal ran with the invocation's own Start policy.
                journal_args = bootstrap_args.read_text(encoding="utf-8").splitlines()
                self.assertIn("--upgrade", journal_args)
                self.assertNotIn("--no-start", journal_args)
                self.assertIn("--no-path", journal_args)
                # adopt ran against the desktop the move had just made current.
                self.assertEqual(self.adopt_log.read_text(encoding="utf-8").splitlines(), ["2.0.38 terminal adopt-tmux --yes"])
                self.assertNotIn("desktop 2.0.38 install-service", self.service_log_text())
                self.assertEqual(receipt_header(receipt)["platform_version"], "2.0.1")
                self.assertEqual(receipt_header(receipt)["prefix"], str(prefix))
            finally:
                server.stop()

    def test_no_start_tmux_moves_into_a_started_desktop_without_changing_its_policy(self):
        with ephemeral_keypair("test recorded start desktop") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.install_tree_desktop(installer)
                self.seed_tree_tmux(version="0.1.0", no_start=1, append=True)
                self.add_catalogue(sec, pin, platform_version="2.0.1", desktop_version="2.0.38")
                self.service_log.unlink(missing_ok=True)

                proc = self.run_installer(installer, "--upgrade", "--components", "tmux", "--no-start", "--yes", "--json")
                doc = self.assert_ran(proc)
                self.assertEqual(doc["components"]["tmux"]["status"], "moved")
                desktop = receipt_section(self.tree_receipt.read_bytes(), "component:desktop")
                self.assertIn(b"version=2.0.38\n", desktop)
                self.assertIn(b"no_start=0\n", desktop)
                self.assertIn("desktop 2.0.38 install-service", self.service_log_text())
            finally:
                server.stop()

    def test_tree_desktop_under_its_own_prefix_takes_over_a_package_tmux(self):
        with ephemeral_keypair("test tree desktop package tmux") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                desktop_prefix = self.work_dir / "desktop prefix"
                self.install_tree_desktop(installer, "--prefix", str(desktop_prefix))
                self.seed_package_tmux(version="0.1.0", lane="release")
                self.add_catalogue(sec, pin, platform_version="2.0.1", desktop_version="2.0.38")

                proc = self.run_installer(installer, "--upgrade", "--components", "tmux", "--yes", "--json")
                doc = self.assert_ran(proc)
                self.assertEqual(doc["components"]["tmux"]["status"], "moved")
                self.assertEqual(doc["components"]["tmux"]["reason"], "moved")
                self.assertEqual(doc["components"]["desktop"]["route"], "tree")
                tree = self.tree_receipt.read_bytes()
                desktop = receipt_section(tree, "component:desktop")
                self.assertIn(f"prefix={desktop_prefix}\n".encode(), desktop)
                self.assertIn(b"version=2.0.38\n", desktop)
                self.assertTrue((desktop_prefix / "bin" / "solstone-linux").is_symlink())
                self.assertFalse((Path(os.environ["HOME"]) / ".local" / "bin" / "solstone-linux").exists())
                self.assertFalse((self.fake_db / "deb" / "solstone-linux").exists())
                self.assertFalse((self.fake_db / "deb" / "solstone-tmux").exists())
                # The package receipt, left with no component, is gone; the tree
                # receipt alone records every component, on the new coordinate.
                self.assertFalse(self.package_receipt.exists())
                self.assertEqual(receipt_sections(tree), ["[solstone]", "[component:desktop]"])
                header = receipt_header(tree)
                self.assertEqual(header["platform_version"], "2.0.1")
                self.assertEqual(header["lane"], "release")
                self.assertEqual(header["prefix"], str(desktop_prefix))
                self.assertEqual(self.adopt_log.read_text(encoding="utf-8").splitlines(), ["2.0.38 terminal adopt-tmux --yes"])
            finally:
                server.stop()

    def test_declined_move_keeps_tmux_bytes_while_other_components_update(self):
        with ephemeral_keypair("test declined keeps tmux bytes") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                prefix = Path(os.environ["HOME"]) / ".local"
                self.install_tree_desktop(installer)
                tmux_section = self.seed_tree_tmux(version="0.1.0", append=True)
                before = snapshot_paths(*self.tmux_tree_paths())
                self.add_catalogue(sec, pin, platform_version="2.0.1", desktop_version="2.0.38")

                proc = self.run_installer(installer, "--upgrade", "--json")
                doc = self.assert_ran(proc)
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "declined")
                self.assertEqual(doc["components"]["desktop"]["status"], "succeeded")
                tree = self.tree_receipt.read_bytes()
                self.assertEqual(receipt_section(tree, "component:tmux"), tmux_section)
                self.assertEqual(snapshot_paths(*self.tmux_tree_paths()), before)
                self.assertIn(b"version=2.0.38\n", receipt_section(tree, "component:desktop"))
                self.assertEqual(receipt_header(tree)["platform_version"], "2.0.1")
                self.assertFalse(self.adopt_log.exists())
                self.assertFalse(self.marker_path.exists())
                remedies = remedy_commands(proc.stderr)
                self.assertEqual(len(remedies), 1)
                self.assertEqual(
                    installer_arguments(remedies[0]),
                    ["--upgrade", "--components", "tmux", "--lane", "release", "--prefix", str(prefix)],
                )
            finally:
                server.stop()

    def test_declined_move_with_a_tree_journal_and_a_package_tmux_advances_both_headers(self):
        with ephemeral_keypair("test declined mixed routes") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.install_tree_journal(installer, self.tree_prefix)
                tmux_section = self.seed_package_tmux(version="0.1.0", lane="release")
                self.add_catalogue(sec, pin, platform_version="2.0.1", desktop_version="2.0.38")

                bootstrap_args = self.work_dir / "bootstrap-args.log"
                test_env = {**self.env, "SOLSTONE_BOOTSTRAP_ARGS_LOG": str(bootstrap_args)}
                proc = self.run_installer(
                    installer, "--upgrade", "--non-interactive", "--json", "--prefix", str(self.tree_prefix), env=test_env,
                )
                doc = self.assert_ran(proc)
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["journal"]["status"], "succeeded")
                self.assertIn("--upgrade", bootstrap_args.read_text(encoding="utf-8").splitlines())
                package = self.package_receipt.read_bytes()
                tree = self.tree_receipt.read_bytes()
                self.assertEqual(receipt_section(package, "component:tmux"), tmux_section)
                self.assertTrue((self.fake_db / "deb" / "solstone-tmux").exists())
                for key in ("platform_version", "lane"):
                    self.assertEqual(receipt_header(tree)[key], receipt_header(package)[key])
                self.assertEqual(receipt_header(package)["platform_version"], "2.0.1")
                self.assertEqual(remedy_commands(proc.stderr), [[f"{self.tree_prefix}/bin/solstone", "update", "--lane", "release"]])
            finally:
                server.stop()

    def test_fresh_desktop_download_failure_leaves_no_marker_and_keeps_tmux(self):
        with ephemeral_keypair("test fresh desktop download failure") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                for archive in (self.work_dir / "www" / "solstone" / "release" / "2.0.0").glob("solstone-linux-*.tar.gz"):
                    archive.unlink()
                tmux_section = self.seed_tree_tmux(version="0.1.0", lane="release")
                before = snapshot_paths(*self.tmux_tree_paths())

                proc = self.run_installer(installer, "--upgrade", "--yes", "--json")
                doc = self.assert_ran(proc, code=1)
                self.assertEqual(doc["status"], "refusal")
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "desktop-failed")
                self.assertFalse(self.marker_path.exists())
                self.assertFalse(self.adopt_log.exists())
                self.assertEqual(receipt_section(self.tree_receipt.read_bytes(), "component:tmux"), tmux_section)
                self.assertNotIn(b"[component:desktop]", self.tree_receipt.read_bytes())
                self.assertEqual(snapshot_paths(*self.tmux_tree_paths()), before)
                self.assertEqual(len(remedy_commands(proc.stderr)), 1)
            finally:
                server.stop()

    def test_ordinary_desktop_install_writes_no_marker(self):
        with ephemeral_keypair("test ordinary desktop install") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.install_tree_desktop(installer, "--prefix", str(self.tree_prefix))
                self.assertFalse(self.marker_path.exists())
                self.assertFalse(self.marker_path.parent.exists())
            finally:
                server.stop()

    def test_interrupted_package_desktop_resumes_on_its_own_route_during_the_move(self):
        with ephemeral_keypair("test interrupted package desktop") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.seed_tree_tmux(version="0.1.0", lane="release")
                self.package_receipt.parent.mkdir(parents=True, exist_ok=True)
                self.package_receipt.write_bytes(
                    b"[solstone]\nschema_version=1\nplatform_version=1.9.0\nlane=release\n"
                    b"[component:desktop]\nphase=intended\nstatus=intended\nrole=desktop\nroute=deb\nservice_policy=start\n"
                    b"package_name=solstone-linux\npackage_version=2.0.36\npackage_arch=amd64\n"
                    + f"artifact_sha256={'1'*64}\npayload_build_id={'1'*64}\n".encode()
                    + b"prior_role=desktop\nprior_route=deb\nprior_service_policy=start\n"
                    b"prior_package_name=solstone-linux\nprior_package_version=2.0.30\nprior_package_arch=amd64\n"
                    + f"prior_artifact_sha256={'2'*64}\nprior_payload_build_id={'2'*64}\n".encode()
                )
                (self.fake_db / "deb").mkdir(parents=True, exist_ok=True)
                (self.fake_db / "deb" / "solstone-linux").write_text("INSTALLED solstone-linux 2.0.30 amd64\n", encoding="utf-8")

                proc = self.run_installer(installer, "--upgrade", "--components", "tmux", "--yes", "--json")
                doc = self.assert_ran(proc)
                self.assertEqual(doc["components"]["tmux"]["status"], "moved")
                self.assertEqual(doc["components"]["desktop"]["route"], "deb")
                desktop = receipt_section(self.package_receipt.read_bytes(), "component:desktop")
                self.assertIn(b"phase=complete\n", desktop)
                self.assertNotIn(b"prior_", desktop)
                self.assertNotIn(b"package_version=2.0.30\n", desktop)
                self.assertNotIn("2.0.30", (self.fake_db / "deb" / "solstone-linux").read_text(encoding="utf-8"))
                # Never reinstalled on tmux's tree route; the tmux tree and its receipt are gone.
                self.assertFalse((Path(os.environ["HOME"]) / ".local" / "bin" / "solstone-linux").exists())
                self.assertFalse(self.tree_receipt.exists())
                self.assertTrue(self.marker_path.exists())
            finally:
                server.stop()

    def test_fresh_desktop_failure_after_the_other_components_exits_failed(self):
        with ephemeral_keypair("test fresh desktop failure") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.install_tree_journal(installer, self.tree_prefix)
                tmux_section = self.seed_tree_tmux(prefix=self.tree_prefix, version="0.1.0", append=True)
                before = snapshot_paths(*self.tmux_tree_paths(self.tree_prefix))
                self.add_catalogue(sec, pin, platform_version="2.0.1", desktop_version="2.0.38")

                bootstrap_args = self.work_dir / "bootstrap-args.log"
                test_env = {
                    **self.env,
                    "SOLSTONE_BOOTSTRAP_ARGS_LOG": str(bootstrap_args),
                    "SOLSTONE_TEST_SERVICE_FAIL": "desktop:2.0.38:install-service",
                }
                proc = self.run_installer(installer, "--upgrade", "--prefix", str(self.tree_prefix), "--yes", "--json", env=test_env)
                doc = self.assert_ran(proc, code=1)
                self.assertEqual(doc["components"]["journal"]["status"], "succeeded")
                self.assertIn("--upgrade", bootstrap_args.read_text(encoding="utf-8").splitlines())
                self.assertEqual(doc["components"]["desktop"]["status"], "failed")
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "desktop-failed")
                self.assertFalse(self.adopt_log.exists())
                self.assertFalse(self.marker_path.exists())
                tree = self.tree_receipt.read_bytes()
                self.assertEqual(receipt_section(tree, "component:tmux"), tmux_section)
                self.assertNotIn(b"[component:desktop]", tree)
                self.assertEqual(snapshot_paths(*self.tmux_tree_paths(self.tree_prefix)), before)
                self.assertEqual(remedy_commands(proc.stderr), [[f"{self.tree_prefix}/bin/solstone", "update", "--lane", "release"]])
            finally:
                server.stop()

    def test_recorded_desktop_whose_upgrade_fails_is_left_as_it_was(self):
        with ephemeral_keypair("test recorded desktop failure") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                prefix = Path(os.environ["HOME"]) / ".local"
                desktop_section = self.install_tree_desktop(installer)
                desktop_paths = [prefix / "bin" / "solstone-linux", prefix / "opt" / "solstone" / "desktop" / "current"]
                old_target = prefix / "opt" / "solstone" / "desktop" / os.readlink(desktop_paths[1])
                before = snapshot_paths(*desktop_paths, old_target)
                tmux_section = self.seed_tree_tmux(version="0.1.0", append=True)
                self.add_catalogue(sec, pin, platform_version="2.0.1", desktop_version="2.0.38")

                test_env = {**self.env, "SOLSTONE_TEST_SERVICE_FAIL": "desktop:2.0.38:install-service"}
                proc = self.run_installer(installer, "--upgrade", "--components", "tmux", "--yes", "--json", env=test_env)
                doc = self.assert_ran(proc, code=1)
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "desktop-failed")
                self.assertFalse(self.adopt_log.exists())
                tree = self.tree_receipt.read_bytes()
                self.assertEqual(receipt_section(tree, "component:desktop"), desktop_section)
                self.assertEqual(receipt_section(tree, "component:tmux"), tmux_section)
                self.assertEqual(snapshot_paths(*desktop_paths, old_target), before)
                self.assertEqual(len(remedy_commands(proc.stderr)), 1)
            finally:
                server.stop()

    def test_declined_desktop_package_command_declines_the_move(self):
        with ephemeral_keypair("test declined desktop package command") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                tmux_section = self.seed_package_tmux(version="0.1.0", lane="staging")
                code, _, stdout, stderr = self.run_pty_interaction(
                    f"{installer} --skip-signature --upgrade --confirm-package-command",
                    self.env,
                    [(b"yes or no", b"yes\n"), (b"yes or no", b"no\n")],
                )
                self.assertEqual(code, 0, f"{stderr}\n{stdout}")
                self.assertFalse((self.fake_db / "deb" / "solstone-linux").exists())
                package = self.package_receipt.read_bytes()
                self.assertEqual(receipt_section(package, "component:tmux"), tmux_section)
                self.assertNotIn(b"[component:desktop]", package)
                self.assertFalse(self.marker_path.exists())
                self.assertFalse(self.adopt_log.exists())
                remedies = remedy_commands(stderr)
                self.assertEqual(len(remedies), 1)
                self.assertEqual(installer_arguments(remedies[0])[:5], ["--upgrade", "--components", "tmux", "--lane", "release"])

                # The same decline with --json reports the move as declined.
                # With --json the package question goes to stderr, so both answers
                # are typed ahead on the terminal.
                proc_code, _, json_out, json_err = self.run_pty_interaction(
                    f"{installer} --skip-signature --upgrade --confirm-package-command --json",
                    self.env,
                    [(b"yes or no", b"yes\nno\n")],
                )
                self.assertEqual(proc_code, 0, f"{json_err}\n{json_out}")
                doc = json.loads(json_out)
                self.assertEqual(doc["components"]["tmux"]["status"], "kept")
                self.assertEqual(doc["components"]["tmux"]["reason"], "declined")
                self.assertNotIn("desktop", doc["components"])
                self.assertFalse(self.marker_path.exists())
            finally:
                server.stop()

    def test_incomplete_package_desktop_points_the_remedy_at_the_direct_upgrade(self):
        with ephemeral_keypair("test incomplete package desktop") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, lane="staging", min_installer_revision=20)
            try:
                tmux_section = self.seed_package_tmux(version="0.1.0", lane="staging")
                self.append_package_section(
                    b"[component:journal]\nphase=complete\nstatus=installed\nrole=journal\nroute=deb\nservice_policy=start\n"
                    b"package_name=solstone-journal\npackage_version=2.0.6\npackage_arch=amd64\n"
                    + f"artifact_sha256={'0'*64}\npayload_build_id={'0'*64}\n".encode()
                )
                (self.fake_db / "deb" / "solstone-journal").write_text("INSTALLED solstone-journal 2.0.6 amd64\n", encoding="utf-8")

                test_env = {**self.env, "SOLSTONE_DESKTOP_SERVICE_FAIL": "1"}
                proc = self.run_installer(installer, "--upgrade", "--lane", "staging", "--components", "tmux", "--yes", "--json", env=test_env)
                doc = self.assert_ran(proc, code=1)
                self.assertEqual(doc["components"]["tmux"]["reason"], "desktop-failed")
                package = self.package_receipt.read_bytes()
                self.assertIn(b"phase=payload\n", receipt_section(package, "component:desktop"))
                self.assertEqual(receipt_section(package, "component:tmux"), tmux_section)
                self.assertFalse(self.adopt_log.exists())
                # A desktop section was written, so the marker stays for that app's first start.
                self.assertTrue(self.marker_path.exists())
                remedies = remedy_commands(proc.stderr)
                self.assertEqual(len(remedies), 1)
                self.assertEqual(remedies[0][:2], ["curl", "-fsSL"])
                self.assertEqual(installer_arguments(remedies[0])[:5], ["--upgrade", "--components", "tmux", "--lane", "staging"])
            finally:
                server.stop()

    def test_failed_removal_reports_removal_failed_and_a_rerun_removes_tmux(self):
        with ephemeral_keypair("test removal failure rerun") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                wrapper = self.bin_dir / "helper-failing-removal"
                wrapper.write_text(
                    "#!/bin/sh\n"
                    "request=$(mktemp)\n"
                    "cat > \"$request\"\n"
                    "if [ -n \"${SOLSTONE_TEST_FAIL_REMOVE:-}\" ] && head -n 1 \"$request\" | grep -q '^REMOVE_PKG '; then\n"
                    "  rm -f \"$request\"; echo ERROR:remove-failed; exit 0\n"
                    "fi\n"
                    f"'{HELPER_SCRIPT}' \"$@\" < \"$request\"\n"
                    "status=$?\n"
                    "rm -f \"$request\"\n"
                    "exit $status\n",
                    encoding="utf-8",
                )
                wrapper.chmod(0o755)
                tmux_section = self.seed_package_tmux(version="0.1.0", lane="staging")

                failing_env = {**self.env, "SOLSTONE_HELPER": str(wrapper), "SOLSTONE_TEST_FAIL_REMOVE": "1"}
                proc = self.run_installer(installer, "--upgrade", "--yes", "--json", env=failing_env)
                doc = self.assert_ran(proc)
                self.assertEqual(doc["components"]["tmux"]["status"], "moved")
                self.assertEqual(doc["components"]["tmux"]["reason"], "removal-failed")
                self.assertEqual(uninstall_commands(proc.stderr), [["--uninstall", "--components", "tmux", "--lane", "release"]])
                self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), tmux_section)
                self.assertTrue((self.fake_db / "deb" / "solstone-tmux").exists())
                self.assertIn(b"[component:desktop]\n", self.package_receipt.read_bytes())

                retry_env = {**self.env, "SOLSTONE_HELPER": str(wrapper)}
                proc_retry = self.run_installer(installer, "--upgrade", "--yes", "--json", env=retry_env)
                doc_retry = self.assert_ran(proc_retry)
                self.assertEqual(doc_retry["components"]["tmux"]["status"], "moved")
                self.assertEqual(doc_retry["components"]["tmux"]["reason"], "moved")
                self.assertEqual(safe_receipt_section(self.package_receipt.read_bytes(), "component:tmux"), b"")
                self.assertFalse((self.fake_db / "deb" / "solstone-tmux").exists())
                self.assertEqual(len(self.adopt_log.read_text(encoding="utf-8").splitlines()), 2)
            finally:
                server.stop()

    def test_package_query_failure_during_removal_reports_removal_failed_and_exits_zero(self):
        with ephemeral_keypair("test removal query failure") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                wrapper = self.bin_dir / "helper-failing-tmux-query"
                wrapper.write_text(
                    "#!/bin/sh\n"
                    "request=$(mktemp)\n"
                    "cat > \"$request\"\n"
                    "if [ \"$(head -n 1 \"$request\")\" = \"QUERY_PKG deb solstone-tmux\" ]; then\n"
                    "  rm -f \"$request\"; printf '%s\\n' \"$SOLSTONE_TEST_QUERY_ANSWER\"; exit \"$SOLSTONE_TEST_QUERY_EXIT\"\n"
                    "fi\n"
                    f"'{HELPER_SCRIPT}' \"$@\" < \"$request\"\n"
                    "status=$?\n"
                    "rm -f \"$request\"\n"
                    "exit $status\n",
                    encoding="utf-8",
                )
                wrapper.chmod(0o755)
                tmux_section = self.seed_package_tmux(version="0.1.0", lane="staging")

                for answer, exit_code in (("ERROR:query-failed", "1"), ("NOT-A-STATE", "0")):
                    with self.subTest(answer=answer):
                        env = {
                            **self.env,
                            "SOLSTONE_HELPER": str(wrapper),
                            "SOLSTONE_TEST_QUERY_ANSWER": answer,
                            "SOLSTONE_TEST_QUERY_EXIT": exit_code,
                        }
                        proc = self.run_installer(installer, "--upgrade", "--yes", "--json", env=env)
                        doc = self.assert_ran(proc)
                        self.assertEqual(doc["status"], "success")
                        self.assertEqual(doc["root_code"], "installed")
                        self.assertEqual(doc["components"]["tmux"]["status"], "moved")
                        self.assertEqual(doc["components"]["tmux"]["reason"], "removal-failed")
                        self.assertEqual(uninstall_commands(proc.stderr), [["--uninstall", "--components", "tmux", "--lane", "release"]])
                        self.assertEqual(receipt_section(self.package_receipt.read_bytes(), "component:tmux"), tmux_section)
                        self.assertIn(b"[component:desktop]\n", self.package_receipt.read_bytes())
            finally:
                server.stop()

    def test_uninstall_tree_tmux_from_the_receipt_removes_its_service(self):
        with ephemeral_keypair("test uninstall tree tmux") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                for handler_root in (None, self.work_dir / "no-handlers"):
                    with self.subTest(bundled_handler=handler_root is None):
                        test_env = dict(self.env)
                        if handler_root is not None:
                            handler_root.mkdir(exist_ok=True)
                            test_env["SOLSTONE_HANDLER_ROOT"] = str(handler_root)
                        self.service_log.unlink(missing_ok=True)
                        self.seed_tree_tmux(version="0.1.0", lane="release")
                        proc = self.run_installer(installer, "--uninstall", "--components", "tmux", "--yes", "--json", env=test_env)
                        doc = self.assert_ran(proc)
                        self.assertEqual(doc["root_code"], "uninstalled")
                        self.assertEqual(doc["components"]["tmux"]["status"], "removed")
                        self.assertIn("tmux uninstall-service", self.service_log.read_text(encoding="utf-8"))
                        self.assertFalse(self.tree_receipt.exists())
                        for path in self.tmux_tree_paths():
                            self.assertFalse(path.exists() or path.is_symlink(), path)
            finally:
                server.stop()

    def test_uninstall_package_tmux_keeps_the_remaining_section_under_a_valid_header(self):
        with ephemeral_keypair("test uninstall package tmux remaining") as (sec, pub, pin):
            server, installer = self.build_release(sec, pub, pin, min_installer_revision=20)
            try:
                self.seed_package_tmux(version="0.1.0", lane="release")
                desktop_section = (
                    b"[component:desktop]\nphase=complete\nstatus=installed\nrole=desktop\nroute=deb\nservice_policy=start\n"
                    b"package_name=solstone-linux\npackage_version=2.0.37\npackage_arch=amd64\n"
                    + f"artifact_sha256={'3'*64}\npayload_build_id={'3'*64}\n".encode()
                )
                self.append_package_section(desktop_section)
                (self.fake_db / "deb" / "solstone-linux").write_text("INSTALLED solstone-linux 2.0.37 amd64\n", encoding="utf-8")

                proc = self.run_installer(installer, "--uninstall", "--components", "tmux", "--yes", "--json")
                doc = self.assert_ran(proc)
                self.assertEqual(doc["components"]["tmux"]["status"], "removed")
                package = self.package_receipt.read_bytes()
                self.assertEqual(receipt_sections(package), ["[solstone]", "[component:desktop]"])
                self.assertEqual(receipt_section(package, "component:desktop"), desktop_section)
                header = receipt_header(package)
                self.assertEqual(header["schema_version"], "1")
                self.assertEqual(header["lane"], "release")
                self.assertIn("platform_version", header)
                self.assertFalse((self.fake_db / "deb" / "solstone-tmux").exists())
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
