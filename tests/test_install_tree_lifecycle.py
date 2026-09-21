# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tree lifecycle convergence, atomic pointer, receipt, lock, and PTY coverage."""

import fcntl
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

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import receipt_section, setup_test_release_server
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallTreeLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"
        self.data_home = self.work_dir / "data"
        self.config_home = self.work_dir / "config"
        self.env = os.environ.copy()
        self.env["XDG_DATA_HOME"] = str(self.data_home)
        self.env["XDG_CONFIG_HOME"] = str(self.config_home)

    def tearDown(self):
        self.tmp.cleanup()

    @property
    def receipt(self) -> Path:
        return self.data_home / "solstone" / "install.conf"

    def build_release(
        self,
        root: str,
        sec: Path,
        pub: Path,
        pin,
        platform_version: str,
        native_version: str,
        *,
        desktop_version: str | None = None,
        tmux_version: str | None = None,
        desktop_runtime_version: str | None = None,
        native_build_marker: str = "",
    ):
        release_dir = self.work_dir / root
        server, _ = setup_test_release_server(
            release_dir,
            sec,
            pin,
            version=platform_version,
            native_version=native_version,
            desktop_version=desktop_version,
            tmux_version=tmux_version,
            desktop_runtime_version=desktop_runtime_version,
            native_build_marker=native_build_marker,
        )
        installer = release_dir / "install.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=installer,
            platform_pub_path=pub,
            platform_key_id=pin.key_id,
            origin=server.origin,
        )
        return server, installer

    def merge_release(self, source_server, destination_server, version: str, pub: Path, pin) -> Path:
        source = source_server.root_dir / "solstone" / "release" / version
        destination_lane = destination_server.root_dir / "solstone" / "release"
        shutil.copytree(source, destination_lane / version)
        (destination_lane / "latest").write_text(f"{version}\n", encoding="utf-8")
        installer = self.work_dir / f"install-{version}.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=installer,
            platform_pub_path=pub,
            platform_key_id=pin.key_id,
            origin=destination_server.origin,
        )
        return installer

    def run_install(
        self,
        installer: Path,
        components: str,
        extra: list[str] | None = None,
        *,
        start: bool = False,
        env: dict[str, str] | None = None,
    ):
        args = [
            str(installer),
            "--skip-signature",
            "--components",
            components,
            "--prefix",
            str(self.prefix),
            "--json",
        ]
        if not start:
            args.append("--no-start")
        if extra:
            args.extend(extra)
        return subprocess.run(args, capture_output=True, text=True, env=env or self.env)

    def test_wrapped_desktop_rejects_symlinked_installed_path(self):
        with ephemeral_keypair("wrapped desktop") as (sec, pub, pin):
            server, installer = self.build_release("wrapped", sec, pub, pin, "2.0.0", "2.0.3")
            try:
                result = self.run_install(installer, "desktop")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                public = self.prefix / "bin/solstone-linux"
                executable = public.resolve()
                self.assertEqual(executable.parent.parent.name, "solstone-linux-2.0.3-linux-x86_64")
                self.assertIn(b"executable_relpath=solstone-linux-2.0.3-linux-x86_64/bin/solstone-linux", self.receipt.read_bytes())
                self.assertEqual(subprocess.run([str(public), "--version"], capture_output=True, text=True, check=True).stdout.strip(), "2.0.3")
                before = self.receipt.read_bytes()
                for path in (executable.parent.parent, executable.parent, executable):
                    with self.subTest(path=path.name):
                        outside = self.work_dir / "outside"
                        path.rename(outside)
                        path.symlink_to(outside, target_is_directory=outside.is_dir())
                        try:
                            for operation in ("--upgrade", "--uninstall"):
                                refused = self.run_install(installer, "desktop", [operation])
                                self.assertNotEqual(refused.returncode, 0)
                                self.assertEqual(json.loads(refused.stdout)["root_code"], "ownership-unknown")
                                self.assertEqual(self.receipt.read_bytes(), before)
                                self.assertTrue(outside.exists())
                        finally:
                            path.unlink()
                            outside.rename(path)
                again = self.run_install(installer, "desktop", ["--upgrade"])
                self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
                self.assertEqual(json.loads(again.stdout)["components"]["desktop"]["status"], "unchanged")
            finally:
                server.stop()

    def test_noncanonical_desktop_wrapper_preserves_prior_install(self):
        from unittest.mock import patch
        from tools.fixture_builder import create_tiny_tar

        def noncanonical_tar(path, files):
            if path.name.startswith("solstone-linux-"):
                files = {"unexpected/" + name.split("/", 1)[1]: data for name, data in files.items()}
            return create_tiny_tar(path, files)

        with ephemeral_keypair("unexpected desktop wrapper") as (sec, pub, pin):
            old_server, installer = self.build_release("old-wrapper", sec, pub, pin, "2.0.0", "2.0.3")
            with patch("tools.fixture_builder.create_tiny_tar", side_effect=noncanonical_tar):
                new_server, _ = self.build_release("new-wrapper", sec, pub, pin, "2.0.1", "2.0.4")
            try:
                installed = self.run_install(installer, "desktop")
                self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)
                public = self.prefix / "bin/solstone-linux"
                before = self.receipt.read_bytes(), public.read_bytes(), os.readlink(public)
                installer = self.merge_release(new_server, old_server, "2.0.1", pub, pin)
                refused = self.run_install(installer, "desktop", ["--upgrade"])
                self.assertNotEqual(refused.returncode, 0)
                self.assertEqual(json.loads(refused.stdout)["root_code"], "executable-missing")
                self.assertEqual((self.receipt.read_bytes(), public.read_bytes(), os.readlink(public)), before)
            finally:
                old_server.stop()
                new_server.stop()

    def test_uninstall_preview_preserves_installed_tree(self):
        with ephemeral_keypair("preview tree") as (sec, pub, pin):
            server, installer = self.build_release("preview", sec, pub, pin, "2.0.0", "2.0.3")
            try:
                installed = self.run_install(installer, "cli,desktop,tmux")
                self.assertEqual(installed.returncode, 0, installed.stderr + installed.stdout)
                sentinel = self.work_dir / "owner-data"
                sentinel.write_text("keep my data")
                from tests.install_test_helpers import snapshot_paths
                before = snapshot_paths(self.prefix, self.data_home, self.config_home, sentinel)
                preview = self.run_install(installer, "cli,desktop,tmux", ["--uninstall", "--dry-run"])
                self.assertEqual(preview.returncode, 0, preview.stderr + preview.stdout)
                result = json.loads(preview.stdout)
                self.assertEqual(result["root_code"], "dry-run-completed")
                self.assertTrue(all(c["status"] == "planned" and c["phase"] == "planned" for c in result["components"].values()))
                self.assertEqual(snapshot_paths(self.prefix, self.data_home, self.config_home, sentinel), before)
                removed = self.run_install(installer, "cli,desktop,tmux", ["--uninstall"])
                self.assertEqual(removed.returncode, 0, removed.stderr + removed.stdout)
                self.assertFalse(self.receipt.exists())
                self.assertEqual(sentinel.read_text(), "keep my data")
            finally:
                server.stop()

    def test_fresh_handler_failure_is_retryable_and_space_prefix_is_usable(self):
        self.prefix = self.work_dir / "install space"
        with ephemeral_keypair("fresh retry") as (sec, pub, pin):
            server, installer = self.build_release("retry", sec, pub, pin, "2.0.0", "2.0.3")
            try:
                failed = self.run_install(installer, "tmux", start=True, env={**self.env, "SOLSTONE_TEST_SERVICE_FAIL": "tmux:2.0.3-x86_64:install-service"})
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(json.loads(failed.stdout)["root_code"], "handler-failed")
                self.assertFalse((self.prefix / "opt/solstone/tmux").exists())
                self.assertFalse(self.receipt.exists())
                retry = self.run_install(installer, "tmux")
                self.assertEqual(retry.returncode, 0, retry.stderr + retry.stdout)
                # Execute the generated shell code, including its space-containing PATH.
                sourced = subprocess.run(["sh", "-c", '. "$1"; command -v solstone-tmux; solstone-tmux --version', "sh", str(self.config_home / "solstone/env")], env=self.env, capture_output=True, text=True)
                self.assertEqual(sourced.returncode, 0, sourced.stderr)
                self.assertEqual(sourced.stdout.splitlines()[0], str(self.prefix / "bin/solstone-tmux"))
                self.assertIn("2.0.3", sourced.stdout)
            finally:
                server.stop()

    def test_native_service_delegation_and_transaction_rollback(self):
        with ephemeral_keypair("tree lifecycle services") as (sec, pub, pin):
            old_server, old_installer = self.build_release(
                "service-old", sec, pub, pin, "2.0.0", "2.0.3"
            )
            new_server, _ = self.build_release(
                "service-new", sec, pub, pin, "2.0.1", "2.0.4"
            )
            service_log = self.work_dir / "service.log"
            service_env = {**self.env, "SOLSTONE_TEST_SERVICE_LOG": str(service_log)}
            try:
                installed = self.run_install(
                    old_installer, "desktop,tmux", start=True, env=service_env
                )
                self.assertEqual(installed.returncode, 0, installed.stderr + installed.stdout)
                self.assertEqual(
                    service_log.read_text().splitlines(),
                    [
                        "desktop 2.0.3 install-service",
                        "tmux 2.0.3-x86_64 install-service",
                    ],
                )

                rerun = self.run_install(
                    old_installer, "desktop,tmux", start=True, env=service_env
                )
                self.assertEqual(rerun.returncode, 0, rerun.stderr + rerun.stdout)
                self.assertEqual(
                    service_log.read_text().splitlines()[-2:],
                    [
                        "desktop 2.0.3 install-service",
                        "tmux 2.0.3-x86_64 install-service",
                    ],
                )

                current_before = {
                    component: os.readlink(
                        self.prefix / "opt" / "solstone" / component / "current"
                    )
                    for component in ("desktop", "tmux")
                }
                receipt_before = self.receipt.read_bytes()
                upgrade_installer = self.merge_release(
                    new_server, old_server, "2.0.1", pub, pin
                )
                failing_env = {
                    **service_env,
                    "SOLSTONE_TEST_SERVICE_FAIL": "tmux:2.0.4-x86_64:install-service",
                }
                failed = self.run_install(
                    upgrade_installer,
                    "desktop,tmux",
                    start=True,
                    env=failing_env,
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(json.loads(failed.stdout)["root_code"], "handler-failed")
                result = json.loads(failed.stdout)
                self.assertEqual(result["components"]["desktop"]["status"], "succeeded")
                self.assertEqual(result["components"]["tmux"]["status"], "failed")
                self.assertIn(str(self.receipt), result["receipt_paths"])
                self.assertEqual(receipt_section(self.receipt.read_bytes(), "component:tmux"), receipt_section(receipt_before, "component:tmux"))
                self.assertIn(b"version=2.0.4\n", receipt_section(self.receipt.read_bytes(), "component:desktop"))
                self.assertEqual(os.readlink(self.prefix / "opt/solstone/tmux/current"), current_before["tmux"])
                self.assertNotEqual(os.readlink(self.prefix / "opt/solstone/desktop/current"), current_before["desktop"])
                self.assertEqual(
                    service_log.read_text().splitlines()[-4:],
                    [
                        "desktop 2.0.4 install-service",
                        "tmux 2.0.4-x86_64 install-service",
                        "tmux 2.0.4-x86_64 uninstall-service",
                        "tmux 2.0.3-x86_64 install-service",
                    ],
                )

                no_start_prefix = self.work_dir / "no-start-prefix"
                self.prefix = no_start_prefix
                no_start_log = self.work_dir / "no-start.log"
                no_start_env = {
                    **self.env,
                    "XDG_DATA_HOME": str(self.work_dir / "no-start-data"),
                    "XDG_CONFIG_HOME": str(self.work_dir / "no-start-config"),
                    "SOLSTONE_TEST_SERVICE_LOG": str(no_start_log),
                }
                no_start = self.run_install(old_installer, "desktop,tmux", env=no_start_env)
                self.assertEqual(no_start.returncode, 0, no_start.stderr + no_start.stdout)
                self.assertFalse(no_start_log.exists())
            finally:
                old_server.stop()
                new_server.stop()

    def test_same_version_noop_current_and_handler_repair(self):
        with ephemeral_keypair("tree lifecycle noop") as (sec, pub, pin):
            server, installer = self.build_release("release", sec, pub, pin, "2.0.0", "2.0.3")
            platform_only_server, _ = self.build_release("platform-only", sec, pub, pin, "2.0.1", "2.0.3")
            try:
                first = self.run_install(installer, "desktop,tmux")
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                for component, binary, expected in (
                    ("desktop", "solstone-linux", "2.0.3"),
                    ("tmux", "solstone-tmux", "2.0.3-x86_64"),
                ):
                    current = self.prefix / "opt" / "solstone" / component / "current"
                    public = self.prefix / "bin" / binary
                    self.assertTrue(current.is_symlink())
                    self.assertRegex(os.readlink(current), r"^2\.0\.3-[0-9a-f]{12}$")
                    self.assertTrue(public.is_symlink())
                    self.assertIn("/current/", os.readlink(public))
                    observed = subprocess.run([str(public), "--version"], capture_output=True, text=True, check=True)
                    self.assertEqual(observed.stdout.strip(), expected)

                receipt_before = self.receipt.read_bytes()
                tree_before = {
                    str(path.relative_to(self.prefix)): (path.read_bytes() if path.is_file() and not path.is_symlink() else os.readlink(path))
                    for path in self.prefix.rglob("*")
                    if path.is_file() or path.is_symlink()
                }
                server.request_paths.clear()
                second = self.run_install(installer, "desktop,tmux")
                self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
                result = json.loads(second.stdout)
                self.assertEqual(result["lane"], "release")
                self.assertEqual(result["platform_version"], "2.0.0")
                self.assertEqual(result["components"]["desktop"]["status"], "unchanged")
                self.assertEqual(result["components"]["tmux"]["status"], "unchanged")
                self.assertEqual(self.receipt.read_bytes(), receipt_before)
                tree_after = {
                    str(path.relative_to(self.prefix)): (path.read_bytes() if path.is_file() and not path.is_symlink() else os.readlink(path))
                    for path in self.prefix.rglob("*")
                    if path.is_file() or path.is_symlink()
                }
                self.assertEqual(tree_after, tree_before)
                self.assertFalse(any(path.endswith((".tar.gz", ".deb", ".rpm")) for path in server.request_paths))

                component_sections_before = {
                    component: receipt_section(self.receipt.read_bytes(), f"component:{component}")
                    for component in ("desktop", "tmux")
                }
                platform_only_installer = self.merge_release(platform_only_server, server, "2.0.1", pub, pin)
                server.request_paths.clear()
                platform_only = self.run_install(platform_only_installer, "desktop,tmux")
                self.assertEqual(platform_only.returncode, 0, platform_only.stderr + platform_only.stdout)
                platform_result = json.loads(platform_only.stdout)
                self.assertEqual(platform_result["components"]["desktop"]["status"], "unchanged")
                self.assertEqual(platform_result["components"]["tmux"]["status"], "unchanged")
                self.assertIn(b"platform_version=2.0.1\n", receipt_section(self.receipt.read_bytes(), "solstone"))
                for component, section in component_sections_before.items():
                    self.assertEqual(receipt_section(self.receipt.read_bytes(), f"component:{component}"), section)
                self.assertFalse(any(path.endswith((".tar.gz", ".deb", ".rpm")) for path in server.request_paths))

                with self.receipt.open("ab") as receipt_stream:
                    receipt_stream.write(b"[component:desktop]\n")
                server.request_paths.clear()
                duplicate = self.run_install(platform_only_installer, "desktop")
                self.assertNotEqual(duplicate.returncode, 0)
                self.assertEqual(json.loads(duplicate.stdout)["root_code"], "ownership-unknown")
                self.assertEqual(self.receipt.read_bytes().count(b"[component:desktop]\n"), 2)
                self.assertFalse(any(path.endswith("solstone-linux-2.0.3-linux-x86_64.tar.gz") for path in server.request_paths))
                self.receipt.write_bytes(self.receipt.read_bytes().rsplit(b"[component:desktop]\n", 1)[0])

                env_file = self.config_home / "solstone" / "env"
                for state_file in (env_file,):
                    canonical = state_file.read_bytes()
                    state_file.write_bytes(canonical + b"corrupt=true\n")
                    server.request_paths.clear()
                    repair = self.run_install(platform_only_installer, "desktop")
                    self.assertEqual(repair.returncode, 0, repair.stderr + repair.stdout)
                    self.assertEqual(json.loads(repair.stdout)["components"]["desktop"]["status"], "succeeded")
                    self.assertEqual(state_file.read_bytes(), canonical)
                    self.assertTrue(any(path.endswith("solstone-linux-2.0.3-linux-x86_64.tar.gz") for path in server.request_paths))
            finally:
                server.stop()
                platform_only_server.stop()

    def test_wrong_runtime_version_never_reports_unchanged(self):
        with ephemeral_keypair("tree lifecycle runtime version") as (sec, pub, pin):
            server, installer = self.build_release(
                "wrong-runtime",
                sec,
                pub,
                pin,
                "2.0.0",
                "2.0.3",
                desktop_runtime_version="2.0.30",
            )
            try:
                first = self.run_install(installer, "desktop")
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                server.request_paths.clear()
                rerun = self.run_install(installer, "desktop")
                self.assertEqual(rerun.returncode, 0, rerun.stderr + rerun.stdout)
                self.assertEqual(json.loads(rerun.stdout)["components"]["desktop"]["status"], "succeeded")
                self.assertTrue(any(path.endswith("solstone-linux-2.0.3-linux-x86_64.tar.gz") for path in server.request_paths))
            finally:
                server.stop()

    def test_upgrade_preserves_unselected_and_mixed_component_noop(self):
        with ephemeral_keypair("tree lifecycle upgrade") as (sec, pub, pin):
            old_server, old_installer = self.build_release("old", sec, pub, pin, "2.0.0", "2.0.3")
            mixed_server, _ = self.build_release(
                "mixed",
                sec,
                pub,
                pin,
                "2.0.1",
                "2.0.3",
                desktop_version="2.0.3",
                tmux_version="2.0.4",
            )
            upgrade_server, _ = self.build_release("upgrade", sec, pub, pin, "2.0.2", "2.0.4")
            try:
                first = self.run_install(old_installer, "desktop,tmux")
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                old_tmux_section = receipt_section(self.receipt.read_bytes(), "component:tmux")
                old_receipt = self.receipt.read_bytes()
                desktop_tree_before = {
                    str(path.relative_to(self.prefix)): (path.read_bytes() if path.is_file() and not path.is_symlink() else os.readlink(path))
                    for path in (self.prefix / "opt" / "solstone" / "desktop").rglob("*")
                    if path.is_file() or path.is_symlink()
                }
                desktop_section_before = receipt_section(old_receipt, "component:desktop")
                desktop = self.prefix / "bin" / "solstone-linux"
                desktop_link_before = os.readlink(desktop)

                mixed_installer = self.merge_release(mixed_server, old_server, "2.0.1", pub, pin)
                handler_root = self.work_dir / "mixed-failing-handlers"
                desktop_handlers = handler_root / "desktop" / "v1"
                desktop_handlers.mkdir(parents=True)
                desktop_handler = desktop_handlers / "install-desktop"
                desktop_handler.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                desktop_handler.chmod(0o755)
                failing_handlers = handler_root / "tmux" / "v1"
                failing_handlers.mkdir(parents=True)
                failing_handler = failing_handlers / "install-tmux"
                failing_handler.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
                failing_handler.chmod(0o755)
                failing_env = self.env.copy()
                failing_env["SOLSTONE_HANDLER_ROOT"] = str(handler_root)
                mixed_failed = subprocess.run(
                    [str(mixed_installer), "--skip-signature", "--components", "desktop,tmux", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=failing_env,
                )
                self.assertNotEqual(mixed_failed.returncode, 0)
                failed_result = json.loads(mixed_failed.stdout)
                self.assertEqual(failed_result["components"]["desktop"]["status"], "unchanged")
                self.assertEqual(failed_result["components"]["tmux"]["status"], "failed")
                self.assertEqual(receipt_section(self.receipt.read_bytes(), "component:desktop"), desktop_section_before)
                self.assertEqual(receipt_section(self.receipt.read_bytes(), "component:tmux"), old_tmux_section)

                old_server.request_paths.clear()
                mixed = self.run_install(mixed_installer, "desktop,tmux")
                self.assertEqual(mixed.returncode, 0, mixed.stderr + mixed.stdout)
                mixed_result = json.loads(mixed.stdout)
                self.assertEqual(mixed_result["components"]["desktop"]["status"], "unchanged")
                self.assertEqual(mixed_result["components"]["tmux"]["status"], "succeeded")
                self.assertEqual(receipt_section(self.receipt.read_bytes(), "component:desktop"), desktop_section_before)
                self.assertNotEqual(receipt_section(self.receipt.read_bytes(), "component:tmux"), old_tmux_section)
                self.assertIn(b"platform_version=2.0.1\n", receipt_section(self.receipt.read_bytes(), "solstone"))
                self.assertEqual(os.readlink(desktop), desktop_link_before)
                desktop_tree_after = {
                    str(path.relative_to(self.prefix)): (path.read_bytes() if path.is_file() and not path.is_symlink() else os.readlink(path))
                    for path in (self.prefix / "opt" / "solstone" / "desktop").rglob("*")
                    if path.is_file() or path.is_symlink()
                }
                self.assertEqual(desktop_tree_after, desktop_tree_before)
                self.assertFalse(any(path.endswith("solstone-linux-2.0.3-linux-x86_64.tar.gz") for path in old_server.request_paths))
                self.assertTrue(any(path.endswith("solstone-tmux-2.0.4-x86_64-linux.tar.gz") for path in old_server.request_paths))
                tmux = self.prefix / "bin" / "solstone-tmux"
                self.assertEqual(subprocess.run([str(tmux), "--version"], capture_output=True, text=True, check=True).stdout.strip(), "2.0.4-x86_64")
                tmux_section_after_mixed = receipt_section(self.receipt.read_bytes(), "component:tmux")

                upgrade_installer = self.merge_release(upgrade_server, old_server, "2.0.2", pub, pin)
                desktop_only = self.run_install(upgrade_installer, "desktop")
                self.assertEqual(desktop_only.returncode, 0, desktop_only.stderr + desktop_only.stdout)
                self.assertEqual(subprocess.run([str(desktop), "--version"], capture_output=True, text=True, check=True).stdout.strip(), "2.0.4")
                desktop_targets = [path.name for path in (self.prefix / "opt" / "solstone" / "desktop").iterdir() if path.is_dir() and not path.is_symlink()]
                self.assertTrue(any(name.startswith("2.0.3-") for name in desktop_targets))
                self.assertTrue(os.readlink(self.prefix / "opt" / "solstone" / "desktop" / "current").startswith("2.0.4-"))
                self.assertEqual(receipt_section(self.receipt.read_bytes(), "component:tmux"), tmux_section_after_mixed)
            finally:
                old_server.stop()
                mixed_server.stop()
                upgrade_server.stop()

    def test_failure_before_current_and_lock_cannot_be_bypassed(self):
        with ephemeral_keypair("tree lifecycle atomic") as (sec, pub, pin):
            old_server, old_installer = self.build_release("old", sec, pub, pin, "2.0.0", "2.0.3", native_build_marker="first")
            new_server, _ = self.build_release("new", sec, pub, pin, "2.0.1", "2.0.3", native_build_marker="respin")
            try:
                first = self.run_install(old_installer, "desktop")
                self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
                current = self.prefix / "opt" / "solstone" / "desktop" / "current"
                public = self.prefix / "bin" / "solstone-linux"
                current_before = os.readlink(current)
                public_before = os.readlink(public)
                receipt_before = self.receipt.read_bytes()
                binary_before = public.read_bytes()
                new_installer = self.merge_release(new_server, old_server, "2.0.1", pub, pin)

                failing_handlers = self.work_dir / "failing-handlers" / "desktop" / "v1"
                failing_handlers.mkdir(parents=True)
                failing_handler = failing_handlers / "install-desktop"
                failing_handler.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
                failing_handler.chmod(0o755)
                handler_env = self.env.copy()
                handler_env["SOLSTONE_HANDLER_ROOT"] = str(self.work_dir / "failing-handlers")
                handler_failed = subprocess.run(
                    [str(new_installer), "--skip-signature", "--components", "desktop", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=handler_env,
                )
                self.assertNotEqual(handler_failed.returncode, 0)
                self.assertEqual(json.loads(handler_failed.stdout)["root_code"], "handler-failed")
                self.assertEqual(os.readlink(current), current_before)
                self.assertEqual(public.read_bytes(), binary_before)
                self.assertEqual(self.receipt.read_bytes(), receipt_before)

                fail_env = self.env.copy()
                fail_env["SOLSTONE_TEST_FAIL_BEFORE_CURRENT"] = "1"
                failed = subprocess.run(
                    [str(new_installer), "--skip-signature", "--components", "desktop", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=fail_env,
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(json.loads(failed.stdout)["root_code"], "test-before-current")
                self.assertEqual(os.readlink(current), current_before)
                self.assertEqual(os.readlink(public), public_before)
                self.assertEqual(public.read_bytes(), binary_before)
                self.assertEqual(self.receipt.read_bytes(), receipt_before)
                self.assertEqual(subprocess.run([str(public), "--version"], capture_output=True, text=True, check=True).stdout.strip(), "2.0.3")

                fake_bin = self.work_dir / "fake-bin"
                fake_bin.mkdir()
                fake_mv = fake_bin / "mv"
                fake_mv.write_text(
                    "#!/bin/sh\n"
                    "case \"$2\" in */.current.tmp.*) exit 41 ;; esac\n"
                    "exec /usr/bin/mv \"$@\"\n",
                    encoding="utf-8",
                )
                fake_mv.chmod(0o755)
                rename_env = self.env.copy()
                rename_env["PATH"] = f"{fake_bin}:{rename_env['PATH']}"
                rename_failed = subprocess.run(
                    [str(new_installer), "--skip-signature", "--components", "desktop", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=rename_env,
                )
                self.assertNotEqual(rename_failed.returncode, 0)
                self.assertEqual(json.loads(rename_failed.stdout)["root_code"], "pointer-publish-failed")
                self.assertEqual(os.readlink(current), current_before)
                self.assertEqual(public.read_bytes(), binary_before)
                self.assertEqual(self.receipt.read_bytes(), receipt_before)

                fake_mv.write_text(
                    "#!/bin/sh\n"
                    f"if [ \"${{3:-}}\" = {shlex.quote(str(self.receipt))} ]; then exit 42; fi\n"
                    "exec /usr/bin/mv \"$@\"\n",
                    encoding="utf-8",
                )
                receipt_publish_failed = subprocess.run(
                    [str(new_installer), "--skip-signature", "--components", "desktop", "--prefix", str(self.prefix), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env=rename_env,
                )
                self.assertNotEqual(receipt_publish_failed.returncode, 0)
                self.assertEqual(json.loads(receipt_publish_failed.stdout)["root_code"], "receipt-write-failed")
                self.assertEqual(os.readlink(current), current_before)
                self.assertEqual(os.readlink(public), public_before)
                self.assertEqual(public.read_bytes(), binary_before)
                self.assertEqual(self.receipt.read_bytes(), receipt_before)

                self.receipt.unlink()
                self.receipt.mkdir()
                receipt_directory = self.run_install(new_installer, "desktop")
                self.assertNotEqual(receipt_directory.returncode, 0)
                self.assertEqual(json.loads(receipt_directory.stdout)["root_code"], "receipt-write-failed")
                self.assertTrue(self.receipt.is_dir())
                self.assertEqual(os.readlink(current), current_before)
                self.assertEqual(public.read_bytes(), binary_before)
                shutil.rmtree(self.receipt)
                self.receipt.write_bytes(receipt_before)

                succeeded = self.run_install(new_installer, "desktop")
                self.assertEqual(succeeded.returncode, 0, succeeded.stderr + succeeded.stdout)
                self.assertNotEqual(os.readlink(current), current_before)
                targets = [path.name for path in current.parent.iterdir() if path.is_dir() and not path.is_symlink()]
                self.assertEqual(len([name for name in targets if name.startswith("2.0.3-")]), 2)
                receipt_after_success = self.receipt.read_bytes()

                lock_path = self.prefix / ".solstone-platform.lock"
                with lock_path.open("a") as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = self.run_install(new_installer, "desktop")
                self.assertNotEqual(locked.returncode, 0)
                self.assertEqual(json.loads(locked.stdout)["root_code"], "tree-locked")
                self.assertEqual(self.receipt.read_bytes(), receipt_after_success)
            finally:
                old_server.stop()
                new_server.stop()

    def run_pty_command(self, command: str, env: dict[str, str], response: bytes | None = None):
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
        sent = response is None
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
                    if response is not None and not sent and b"Select components" in tty_output:
                        os.write(master, response)
                        sent = True
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited == pid:
                    break
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                self.fail("PTY installer invocation timed out")
        finally:
            os.close(master)
        return os.waitstatus_to_exitcode(status), bytes(tty_output), stdout_path.read_text(), stderr_path.read_text()

    def test_piped_menu_and_machine_safe_pty_paths(self):
        with ephemeral_keypair("tree lifecycle pty") as (sec, pub, pin):
            server, installer = self.build_release("pty", sec, pub, pin, "2.0.0", "2.0.3")
            try:
                command = (
                    f"{{ cat {shlex.quote(str(installer))}; printf '\\n# trailing-pipe-marker\\n'; }} | "
                    f"/bin/sh -s -- --skip-signature "
                    f"--prefix {shlex.quote(str(self.prefix))} --no-start"
                )
                code, tty_bytes, stdout, stderr = self.run_pty_command(command, self.env, b"3\n")
                self.assertEqual(code, 0, stderr + stdout + tty_bytes.decode(errors="replace"))
                self.assertIn(b"Solstone Platform Component Selection", tty_bytes)
                self.assertEqual(stdout, "")
                self.assertTrue((self.prefix / "bin" / "solstone-linux").is_symlink())

                json_prefix = self.work_dir / "json-prefix"
                json_command = (
                    f"{shlex.quote(str(installer))} --skip-signature --no-start "
                    f"--prefix {shlex.quote(str(json_prefix))} --json"
                )
                code, tty_bytes, stdout, stderr = self.run_pty_command(json_command, self.env)
                self.assertEqual(code, 0, stderr + stdout)
                self.assertNotIn(b"Component Selection", tty_bytes)
                self.assertEqual(stderr, "")
                lines = [line for line in stdout.splitlines() if line]
                self.assertEqual(len(lines), 1)
                self.assertEqual(set(json.loads(lines[0])["components"]), {"journal"})

                no_tty = subprocess.run(
                    [str(installer), "--skip-signature", "--no-start", "--prefix", str(self.work_dir / "no-tty"), "--json"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    env={**self.env, "XDG_DATA_HOME": str(self.work_dir / "no-tty-data")},
                )
                self.assertEqual(no_tty.returncode, 0, no_tty.stderr + no_tty.stdout)
                no_tty_result = json.loads(no_tty.stdout)
                self.assertEqual(set(no_tty_result["components"]), {"journal"})

                noninteractive = subprocess.run(
                    [str(installer), "--skip-signature", "--non-interactive", "--components", "desktop", "--prefix", str(self.work_dir / "noninteractive"), "--no-start", "--json"],
                    capture_output=True,
                    text=True,
                    env={**self.env, "XDG_DATA_HOME": str(self.work_dir / "noninteractive-data")},
                )
                self.assertEqual(noninteractive.returncode, 0, noninteractive.stderr + noninteractive.stdout)
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
