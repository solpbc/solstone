# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Family 1: Parse guard and truncation resilience tests."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.build_installer import build_installer
from solstone_platform.sign import ephemeral_keypair

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallParseGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory(dir="/var/tmp")
        cls.installer_path = Path(cls.tmp_dir.name) / "install.sh"
        with ephemeral_keypair("test parse guard") as (sec, pub, pin):
            build_installer(
                repo_root=REPO_ROOT,
                output_path=cls.installer_path,
                platform_pub_path=pub,
                platform_key_id=pin.key_id,
            )

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_dash_syntax(self):
        proc = subprocess.run(["dash", "-n", str(self.installer_path)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"dash -n failed: {proc.stderr}")

    def test_shellcheck(self):
        proc = subprocess.run(["shellcheck", "-s", "sh", str(self.installer_path)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"shellcheck failed: {proc.stdout}\n{proc.stderr}")

    def test_truncation_resilience(self):
        content = self.installer_path.read_bytes()
        total_len = len(content)

        # Test truncation at various percentage points
        percentages = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        for pct in percentages:
            cutoff = int(total_len * pct)
            truncated = content[:cutoff]
            with tempfile.NamedTemporaryFile(dir="/var/tmp", mode="wb", delete=True) as tf:
                tf.write(truncated)
                tf.flush()
                # dash -n must fail syntax check
                proc = subprocess.run(["dash", "-n", tf.name], capture_output=True)
                self.assertNotEqual(proc.returncode, 0, f"Truncated script at {pct*100}% unexpectedly passed syntax check")

        # Test right after token 'main'
        main_idx = content.rfind(b"main")
        if main_idx != -1:
            truncated = content[:main_idx + 4]
            with tempfile.NamedTemporaryFile(dir="/var/tmp", mode="wb", delete=True) as tf:
                tf.write(truncated)
                tf.flush()
                proc = subprocess.run(["dash", "-n", tf.name], capture_output=True)
                self.assertNotEqual(proc.returncode, 0, "Truncated right after 'main' unexpectedly passed syntax check")


if __name__ == "__main__":
    unittest.main()
