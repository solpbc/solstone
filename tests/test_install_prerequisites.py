# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Prerequisite refusals, hosts without a package tool, and the Linux component menu."""

import hashlib
import json
import os
from pathlib import Path
import pty
import select
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import setup_test_release_server
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent
HINT_HELPER = REPO_ROOT / "helpers" / "install-hint.sh"

# The same digest is pinned by solstone-journal core/distribution/install.test.sh
# for its copy of this block. Change both copies and both pins together.
SHARED_INSTALL_HINT_SHA256 = "a39261fb9cef1e5ac12de862a686368141911442823a89954425e1f8a8990318"

OS_RELEASES = {
    "fedora": 'NAME="Fedora Linux"\nID=fedora\nVERSION_ID=43\n',
    "almalinux": 'ID="almalinux"\nID_LIKE="rhel centos fedora"\n',
    "rhel": 'ID="rhel"\nID_LIKE="fedora"\n',
    "ubuntu": "ID=ubuntu\nID_LIKE=debian\n",
    "debian": "ID=debian\n",
    "mint": 'ID=linuxmint\nID_LIKE="ubuntu debian"\n',
    "arch": "ID=arch\n",
    "manjaro": "ID=manjaro\nID_LIKE=arch\n",
    "tumbleweed": 'ID="opensuse-tumbleweed"\nID_LIKE="opensuse suse"\n',
    "nixos": "ID=nixos\n",
}

MINISIGN_HINTS = {
    "fedora": "sudo dnf install minisign",
    "almalinux": "sudo dnf install epel-release && sudo dnf install minisign",
    "rhel": "enable EPEL (https://docs.fedoraproject.org/en-US/epel/) and run sudo dnf install minisign",
    "ubuntu": "sudo apt install minisign",
    "debian": "sudo apt install minisign",
    "mint": "sudo apt install minisign",
    "arch": "sudo pacman -S minisign",
    "manjaro": "sudo pacman -S minisign",
    "tumbleweed": "sudo zypper install minisign",
    "nixos": "install minisign from your distribution's packages",
}

PACKAGE_TOOL_PREFIXES = ("dpkg", "rpm", "apt", "dnf", "yum", "zypper")


def restricted_path(bin_dir: Path, drop: tuple[str, ...] = (), drop_prefixes: tuple[str, ...] = ()) -> str:
    """A PATH holding every host command except the named ones."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("/usr/local/bin", "/usr/bin", "/bin", str(Path(shutil.which("minisign") or "/").parent)):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name in drop or name.startswith(drop_prefixes):
                continue
            target = Path(directory) / name
            link = bin_dir / name
            if link.exists() or link.is_symlink() or not os.access(target, os.X_OK) or target.is_dir():
                continue
            link.symlink_to(target)
    return str(bin_dir)


class TestSharedInstallHint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def hint(self, package: str, os_release: str | None) -> str:
        path = self.work_dir / "os-release"
        if os_release is None:
            path = self.work_dir / "missing"
        else:
            path.write_text(os_release, encoding="utf-8")
        proc = subprocess.run(
            ["sh", "-c", 'set -eu; . "$1"; install_hint "$2" "$3"', "sh", str(HINT_HELPER), package, str(path)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_shared_block_digest_is_pinned_with_the_journal_bootstrap(self):
        self.assertEqual(hashlib.sha256(HINT_HELPER.read_bytes()).hexdigest(), SHARED_INSTALL_HINT_SHA256)

    def test_production_installer_embeds_the_shared_block_verbatim(self):
        out = self.work_dir / "install.sh"
        build_installer(repo_root=REPO_ROOT, output_path=out, is_production=True)
        self.assertIn(HINT_HELPER.read_text(encoding="utf-8"), out.read_text(encoding="utf-8"))

    def test_minisign_hint_for_each_distribution_family(self):
        for distro, os_release in OS_RELEASES.items():
            with self.subTest(distro=distro):
                self.assertEqual(self.hint("minisign", os_release), MINISIGN_HINTS[distro])
        self.assertEqual(self.hint("minisign", None), "install minisign from your distribution's packages")

    def test_epel_applies_to_minisign_only(self):
        self.assertEqual(self.hint("util-linux", OS_RELEASES["almalinux"]), "sudo dnf install util-linux")
        self.assertEqual(self.hint("util-linux", OS_RELEASES["rhel"]), "sudo dnf install util-linux")
        self.assertEqual(self.hint("curl", OS_RELEASES["arch"]), "sudo pacman -S curl")


class TestInstallPrerequisites(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        cls.work_dir = Path(cls.tmp.name)
        cls.keys = ephemeral_keypair("prerequisites")
        sec, pub, pin = cls.keys.__enter__()
        cls.server, _ = setup_test_release_server(cls.work_dir / "release", sec, pin)
        cls.installer = cls.work_dir / "install.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=cls.installer,
            platform_pub_path=pub,
            platform_key_id=pin.key_id,
            origin=cls.server.origin,
        )
        cls.os_release = cls.work_dir / "os-release"
        cls.os_release.write_text(OS_RELEASES["fedora"], encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.keys.__exit__(None, None, None)
        cls.tmp.cleanup()

    def setUp(self):
        self.case_dir = Path(tempfile.mkdtemp(dir=self.work_dir))
        self.server.request_paths.clear()

    def run_installer(self, *args: str, path: str | None = None, extra_env: dict[str, str] | None = None):
        env = os.environ.copy()
        env["XDG_DATA_HOME"] = str(self.case_dir / "data")
        env["XDG_CONFIG_HOME"] = str(self.case_dir / "config")
        env["SOLSTONE_TEST_OS_RELEASE"] = str(self.os_release)
        if path is not None:
            env["PATH"] = path
        env.update(extra_env or {})
        proc = subprocess.run(
            [str(self.installer), *args, "--prefix", str(self.case_dir / "prefix"), "--json"],
            capture_output=True,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        return proc, list(self.server.request_paths)

    def assert_prerequisite_refusal(self, proc, requests, code: str, command: str):
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["root_code"], code)
        self.assertIn(f"To fix: {command}, then run the installer again.", result["message"])
        self.assertIn("Nothing was changed.", result["message"])
        self.assertEqual(requests, [], "a prerequisite refusal must come before any download")
        self.assertFalse((self.case_dir / "prefix").exists())

    def test_missing_minisign_refuses_before_download_with_the_install_command(self):
        path = restricted_path(self.case_dir / "bin", drop=("minisign",))
        proc, requests = self.run_installer("--components", "journal", "--dry-run", path=path)
        self.assert_prerequisite_refusal(proc, requests, "verifier-missing", "sudo dnf install minisign")

    def test_missing_minisign_hint_is_printed_to_a_person(self):
        env = os.environ.copy()
        env.update({
            "PATH": restricted_path(self.case_dir / "bin", drop=("minisign",)),
            "SOLSTONE_TEST_OS_RELEASE": str(self.os_release),
            "HOME": str(self.case_dir / "home"),
        })
        proc = subprocess.run(
            [str(self.installer), "--components", "journal", "--non-interactive"],
            capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(
            proc.stderr,
            "ERROR: verifier-missing: minisign was not found, and it is needed to verify what the "
            "installer downloads. Nothing was changed. To fix: sudo dnf install minisign, then run the installer again.\n",
        )

    def test_skip_signature_does_not_require_minisign(self):
        path = restricted_path(self.case_dir / "bin", drop=("minisign",))
        proc, requests = self.run_installer("--components", "journal", "--dry-run", "--skip-signature", path=path)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(requests)

    def test_missing_flock_refuses_before_download_and_a_preview_does_not_need_it(self):
        path = restricted_path(self.case_dir / "bin", drop=("flock",))
        proc, requests = self.run_installer("--components", "journal", "--no-start", path=path)
        self.assert_prerequisite_refusal(proc, requests, "lock-tool-missing", "sudo dnf install util-linux")
        proc, _ = self.run_installer("--components", "journal", "--dry-run", "--skip-signature", path=path)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_missing_fetch_tool_refuses_with_the_install_command(self):
        path = restricted_path(self.case_dir / "bin", drop=("curl", "wget"))
        proc, requests = self.run_installer("--components", "journal", "--dry-run", path=path)
        self.assert_prerequisite_refusal(proc, requests, "missing-fetch-tool", "sudo dnf install curl")

    def test_missing_tar_refuses_an_install_but_not_a_preview(self):
        path = restricted_path(self.case_dir / "bin", drop=("tar",))
        proc, requests = self.run_installer("--components", "journal", "--no-start", path=path)
        self.assert_prerequisite_refusal(proc, requests, "missing-tool", "sudo dnf install tar")
        proc, _ = self.run_installer("--components", "journal", "--dry-run", "--skip-signature", path=path)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_host_without_a_package_tool_installs_on_the_tree_route(self):
        path = restricted_path(self.case_dir / "bin", drop_prefixes=PACKAGE_TOOL_PREFIXES)
        proc, _ = self.run_installer(
            "--components", "journal", "--skip-signature", "--no-start", "--non-interactive",
            path=path, extra_env={"SOLSTONE_TEST_REAL_PKG_PROBE": "1"},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["components"]["journal"]["route"], "tree")
        self.assertEqual(result["components"]["journal"]["status"], "succeeded")
        self.assertTrue((self.case_dir / "data" / "solstone" / "install.conf").is_file())

    def test_real_package_probe_still_sees_a_package_the_host_reports(self):
        path = restricted_path(self.case_dir / "bin", drop_prefixes=PACKAGE_TOOL_PREFIXES)
        for name, body in (
            ("rpm", "printf '2.0.3-1 x86_64\\n'\n"),
            ("dpkg-query", "printf 'ii 2.0.3 amd64\\n'\n"),
        ):
            stub = Path(path) / name
            stub.write_text("#!/bin/sh\n" + body, encoding="utf-8")
            stub.chmod(0o755)
        proc, _ = self.run_installer(
            "--components", "journal", "--skip-signature", "--no-start", "--non-interactive",
            path=path, extra_env={"SOLSTONE_TEST_REAL_PKG_PROBE": "1"},
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "ownership-unknown")

    def test_explicit_package_route_without_package_tools_refuses_before_download(self):
        path = restricted_path(self.case_dir / "bin", drop_prefixes=PACKAGE_TOOL_PREFIXES)
        proc, requests = self.run_installer(
            "--components", "journal", "--route", "rpm", "--no-start",
            path=path, extra_env={"SOLSTONE_TEST_REAL_PKG_PROBE": "1"},
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["root_code"], "missing-package-tool")
        self.assertIn("leave out --route", result["message"])
        self.assertEqual(requests, [])


class TestLinuxComponentMenu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        cls.work_dir = Path(cls.tmp.name)
        cls.keys = ephemeral_keypair("menu")
        sec, pub, pin = cls.keys.__enter__()
        cls.server, _ = setup_test_release_server(cls.work_dir / "release", sec, pin)
        cls.installer = cls.work_dir / "install.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=cls.installer,
            platform_pub_path=pub,
            platform_key_id=pin.key_id,
            origin=cls.server.origin,
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.keys.__exit__(None, None, None)
        cls.tmp.cleanup()

    def run_menu(self, answers: list[bytes]):
        case_dir = Path(tempfile.mkdtemp(dir=self.work_dir))
        env = os.environ.copy()
        env["XDG_DATA_HOME"] = str(case_dir / "data")
        env["XDG_CONFIG_HOME"] = str(case_dir / "config")
        stdout_path = case_dir / "stdout"
        stderr_path = case_dir / "stderr"
        command = (
            f"{shlex.quote(str(self.installer))} --skip-signature --dry-run --no-start "
            f"--prefix {shlex.quote(str(case_dir / 'prefix'))} </dev/null"
        )
        pid, master = pty.fork()
        if pid == 0:
            out_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            err_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            os.execve("/bin/sh", ["sh", "-c", command], env)
        tty_output = bytearray()
        prompts_answered = 0
        status = None
        deadline = time.monotonic() + 30
        try:
            while time.monotonic() < deadline:
                ready, _, _ = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        tty_output.extend(os.read(master, 4096))
                    except OSError:
                        pass
                    while prompts_answered < len(answers) and tty_output.count(b"Select components") > prompts_answered:
                        os.write(master, answers[prompts_answered])
                        prompts_answered += 1
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited == pid:
                    break
            else:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
                self.fail("menu run timed out: " + tty_output.decode(errors="replace"))
        finally:
            os.close(master)
        return os.waitstatus_to_exitcode(status), bytes(tty_output), stderr_path.read_text()

    def test_enter_takes_the_journal_default(self):
        code, tty_output, stderr = self.run_menu([b"\n"])
        self.assertEqual(code, 0, stderr + tty_output.decode(errors="replace"))
        self.assertIn(b"press Enter for the journal only", tty_output)
        self.assertIn("preview complete", stderr)
        self.assertEqual(tty_output.count(b"Select components"), 1)

    def test_a_mistype_asks_again(self):
        code, tty_output, stderr = self.run_menu([b"x\n", b"3\n"])
        self.assertEqual(code, 0, stderr + tty_output.decode(errors="replace"))
        self.assertIn(b"'x' is not one of the choices.", tty_output)
        self.assertEqual(tty_output.count(b"Select components"), 2)

    def test_three_mistypes_refuse(self):
        code, tty_output, stderr = self.run_menu([b"x\n", b"9\n", b"all\n"])
        self.assertEqual(code, 1, stderr + tty_output.decode(errors="replace"))
        self.assertIn("invalid-selection: Invalid component selection 'all'", stderr)


if __name__ == "__main__":
    unittest.main()
