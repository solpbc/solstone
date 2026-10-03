# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""The downloadable script carries its own exact, trusted runtime."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.install_test_helpers import HermeticInstallerTestCase

from solstone_platform.sign import ephemeral_keypair
from tests.test_install_tmux_authority import TmuxFixture
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallRuntime(HermeticInstallerTestCase):
    def test_pending_tree_uses_selected_owned_version_for_readback_and_cleanup(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
            root = Path(tmp)
            script = build_installer(REPO_ROOT, root / "install.sh", is_production=True)
            content = script.read_text()
            entry = ('OPT_PREFIX="$1"; HOST_ARCH=x86_64; '
                     'tree_component_authority journal removal; '
                     'run_installed_journal "$OPT_PREFIX/current/bin" "$TREE_JOURNAL_VERSION" '
                     'setup --clean-uninstall --yes --installer-transaction')
            script.write_text(content.replace('main "$@"', entry))
            for selected in ("2.0.29", "2.0.30"):
                with self.subTest(selected=selected):
                    prefix = root / f"owned prefix {selected}"
                    binaries = prefix / "versions" / f"{selected}-abcdef" / "bin"
                    binaries.mkdir(parents=True)
                    (prefix / "current").symlink_to(f"versions/{selected}-abcdef")
                    (prefix / "install-receipt").write_text(
                        "route=tree\nrole=journal\njournal_version=2.0.29\n")
                    data_home = prefix / "data"
                    receipt = data_home / "solstone" / "install.conf"
                    receipt.parent.mkdir(parents=True)
                    receipt.write_text(
                        f"[component:journal]\nrole=journal\nroute=tree\nprefix={prefix}\n"
                        "arch=x86_64\nversion=2.0.30\nprior_version=2.0.29\nstatus=pending\n")
                    log = root / f"argv-{selected}.jsonl"
                    for name in ("journal", "solstone"):
                        executable = binaries / name
                        executable.write_text(
                            '#!/usr/bin/env python3\nimport json, os, sys\n'
                            'from pathlib import Path\n'
                            'with Path(os.environ["ARGUMENT_LOG"]).open("a") as f: '
                            'f.write(json.dumps(sys.argv) + "\\n")\n'
                            f'print("journal (solstone) {selected}")\n')
                        executable.chmod(0o755)
                    result = subprocess.run(
                        [str(script), str(prefix)],
                        env={**os.environ, "XDG_DATA_HOME": str(data_home), "ARGUMENT_LOG": str(log)},
                        capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    canonical = selected == "2.0.30"
                    command = [str(prefix / "current" / "bin" / ("solstone" if canonical else "journal"))]
                    if canonical:
                        command.append("journal")
                    self.assertEqual([json.loads(line) for line in log.read_text().splitlines()], [
                        [*command, "--version"],
                        [*command, "setup", "--clean-uninstall", "--yes", "--installer-transaction"],
                    ])
                    if canonical:
                        # A second entry point cannot escape the ownership proof.
                        launcher = binaries / "solstone"
                        launcher.rename(root / "foreign-solstone")
                        launcher.symlink_to(root / "foreign-solstone")
                        log.unlink()
                        result = subprocess.run(
                            [str(script), str(prefix)],
                            env={**os.environ, "XDG_DATA_HOME": str(data_home), "ARGUMENT_LOG": str(log)},
                            capture_output=True, text=True, timeout=10)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("ownership-unknown", result.stderr)
                        self.assertFalse(log.exists())

    def test_current_and_historical_journal_invocations_preserve_argv_and_exit(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
            root = Path(tmp)
            script = build_installer(REPO_ROOT, root / "install.sh", is_production=True)
            content = script.read_text()
            script.write_text(content.replace('main "$@"', 'run_installed_journal "$@"'))
            binaries = root / "owned bin with spaces"
            binaries.mkdir()
            log = root / "argv.json"
            for name in ("journal", "solstone"):
                executable = binaries / name
                executable.write_text(
                    '#!/usr/bin/env python3\nimport json, os, sys\n'
                    'from pathlib import Path\n'
                    'Path(os.environ["ARGUMENT_LOG"]).write_text(json.dumps(sys.argv))\n'
                    'sys.exit(int(os.environ["COMMAND_EXIT"]))\n'
                )
                executable.chmod(0o755)
            tail = ["setup", "--journal", str(root / "journal with spaces"), "--skip-service"]
            for version in ("1.0.22", "2.0.9", "2.0.29", "2.0.30", "2.0.31-abcdef", "2.1.0"):
                canonical = version in ("2.0.30", "2.0.31-abcdef", "2.1.0")
                expected = [str(binaries / ("solstone" if canonical else "journal"))]
                if canonical:
                    expected.append("journal")
                for exit_code in (0, 80, 9):
                    with self.subTest(version=version, exit_code=exit_code):
                        result = subprocess.run(
                            [str(script), str(binaries), version, *tail],
                            env={**os.environ, "ARGUMENT_LOG": str(log), "COMMAND_EXIT": str(exit_code)},
                            capture_output=True, text=True, timeout=10,
                        )
                        self.assertEqual(result.returncode, exit_code, result.stderr)
                        self.assertEqual(json.loads(log.read_text()), [*expected, *tail])

    def test_materialized_runtime_preserves_source_bytes(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
            root = Path(tmp)
            script = build_installer(REPO_ROOT, root / "install.sh", is_production=True)
            content = script.read_text()
            script.write_text(content.replace('main "$@"', 'SCRATCH_DIR="$1"; init_bundled_runtime'))
            staged = root / "staged"
            staged.mkdir()
            result = subprocess.run([str(script), str(staged)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            for source in [*REPO_ROOT.glob("handlers/*/v1/*"), REPO_ROOT / "helpers/solstone-pkg-helper.sh"]:
                target = staged / "runtime" / source.relative_to(REPO_ROOT)
                self.assertEqual(target.read_bytes(), source.read_bytes())
                self.assertEqual(target.stat().st_mode & 0o777, 0o700)
            # An unwritable staging destination must fail, never continue.
            bad = root / "not-a-directory"
            bad.write_text("sentinel")
            failure = subprocess.run([str(script), str(bad)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("runtime-unavailable", failure.stderr)
            self.assertEqual(bad.read_text(), "sentinel")

    def test_signed_install_outside_checkout_ignores_ambient_handlers(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp, ephemeral_keypair("runtime platform") as (sec, pub, pin), ephemeral_keypair("runtime tmux") as (native_sec, _native_pub, native_pin):
            root = Path(tmp)
            fixture = TmuxFixture(root, sec, pub, pin, native_sec, native_pin)
            try:
                cwd = root / "empty"
                cwd.mkdir()
                poison = cwd / "handlers/tmux/v1/install-tmux"
                poison.parent.mkdir(parents=True)
                poison.write_text("#!/bin/sh\nexit 93\n")
                poison.chmod(0o755)
                env = {**os.environ, "HOME": str(root / "home"), "XDG_DATA_HOME": str(root / "data"), "XDG_CONFIG_HOME": str(root / "config")}
                args = [str(fixture.installer), "--components", "tmux", "--prefix", str(root / "prefix"), "--no-start", "--json"]
                for extra in ([], [], ["--uninstall"]):
                    result = subprocess.run([*args, *extra], cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    self.assertEqual(json.loads(result.stdout)["verification_layers"], "minisign+digest")
                fixture.signature.write_text("tampered\n")
                result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(json.loads(result.stdout)["root_code"], ("signature-invalid", "digest-mismatch"))
            finally:
                fixture.close()
