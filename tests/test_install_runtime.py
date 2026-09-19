# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""The downloadable script carries its own exact, trusted runtime."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.test_install_tmux_authority import TmuxFixture
from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallRuntime(unittest.TestCase):
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
