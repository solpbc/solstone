# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for packaged pin resources, synchronization, and isolated verification."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from solstone_platform.pins import (
    DESKTOP_KEY_ID,
    JOURNAL_KEY_ID,
    PLATFORM_KEY_ID,
    TMUX_KEY_ID,
    _verify_constant_synchronization,
    embedded_pins,
    load_packaged_pin,
    parse_minisign_pub,
    require_production_platform_pin,
)


class TestPinsPackaging(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent

    def test_packaged_pins_load_correctly(self):
        pins = embedded_pins()
        self.assertEqual(pins.journal.key_id, JOURNAL_KEY_ID)
        self.assertEqual(pins.desktop.key_id, DESKTOP_KEY_ID)
        self.assertEqual(pins.tmux.key_id, TMUX_KEY_ID)
        self.assertEqual(pins.platform.key_id, PLATFORM_KEY_ID)

    def test_require_production_platform_pin_without_repo_root(self):
        prod_pin = require_production_platform_pin()
        self.assertEqual(prod_pin.key_id, PLATFORM_KEY_ID)

    def test_constant_synchronization_check(self):
        # Must pass without raising Refusal
        _verify_constant_synchronization()

    def test_load_all_packaged_pins(self):
        for name in ["journal", "desktop", "tmux", "platform"]:
            pin = load_packaged_pin(name)
            self.assertTrue(len(pin.key_id) == 16)
            self.assertTrue(len(pin.pubkey) > 10)

    def test_isolated_subprocess_packaging_execution(self):
        """Verify package can load pins in an isolated environment with NO checkout on sys.path."""
        pins_on_disk = {}
        for name in ["journal", "desktop", "tmux", "platform"]:
            pub_path = self.repo_root / "pins" / f"{name}.pub"
            pins_on_disk[name] = parse_minisign_pub(pub_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp_prefix:
            prefix_path = Path(tmp_prefix)
            src_pkg = self.repo_root / "src" / "solstone_platform"
            dest_pkg = prefix_path / "solstone_platform"
            shutil.copytree(src_pkg, dest_pkg)

            empty_cwd = prefix_path / "empty_cwd"
            empty_cwd.mkdir()

            script = (
                "import json\n"
                "from solstone_platform.pins import embedded_pins, require_production_platform_pin\n"
                "pins = embedded_pins()\n"
                "prod = require_production_platform_pin()\n"
                "res = {\n"
                "    'journal': {'key_id': pins.journal.key_id, 'pubkey': pins.journal.pubkey},\n"
                "    'desktop': {'key_id': pins.desktop.key_id, 'pubkey': pins.desktop.pubkey},\n"
                "    'tmux': {'key_id': pins.tmux.key_id, 'pubkey': pins.tmux.pubkey},\n"
                "    'platform': {'key_id': prod.key_id, 'pubkey': prod.pubkey},\n"
                "}\n"
                "print(json.dumps(res))\n"
            )

            env = {"PYTHONPATH": str(prefix_path), "PATH": "/usr/bin:/bin"}
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(empty_cwd),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"Subprocess failed: {proc.stderr}")
            result = json.loads(proc.stdout)

            for name in ["journal", "desktop", "tmux", "platform"]:
                self.assertEqual(result[name]["key_id"], pins_on_disk[name].key_id)
                self.assertEqual(result[name]["pubkey"], pins_on_disk[name].pubkey)

    def test_drift_detection_resource_mutation_fails(self):
        """Mutating pin_resources/*.pub in a copy causes import-time synchronization failure."""
        with tempfile.TemporaryDirectory() as tmp_prefix:
            prefix_path = Path(tmp_prefix)
            src_pkg = self.repo_root / "src" / "solstone_platform"
            dest_pkg = prefix_path / "solstone_platform"
            shutil.copytree(src_pkg, dest_pkg)

            # Mutate packaged journal pin
            tampered = (
                "untrusted comment: minisign public key: FFFFFFFFFFFFFFFF\n"
                "RWQeioghnBwEeKLzHwyJWiKKd8r4KYILul+Mc7ZeN/acOtGb4I4QGutz\n"
            )
            (dest_pkg / "pin_resources" / "journal.pub").write_text(tampered, encoding="utf-8")

            script = "import solstone_platform.pins\n"
            env = {"PYTHONPATH": str(prefix_path), "PATH": "/usr/bin:/bin"}
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(prefix_path),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("pin-mismatch", proc.stderr)

    def test_drift_detection_constant_mutation_fails(self):
        """Mutating a constant in pins.py in a copy causes import-time synchronization failure."""
        with tempfile.TemporaryDirectory() as tmp_prefix:
            prefix_path = Path(tmp_prefix)
            src_pkg = self.repo_root / "src" / "solstone_platform"
            dest_pkg = prefix_path / "solstone_platform"
            shutil.copytree(src_pkg, dest_pkg)

            # Mutate constant in pins.py
            pins_py = dest_pkg / "pins.py"
            content = pins_py.read_text(encoding="utf-8")
            mutated = content.replace(JOURNAL_KEY_ID, "AAAAAAAAAAAAAAAA")
            pins_py.write_text(mutated, encoding="utf-8")

            script = "import solstone_platform.pins\n"
            env = {"PYTHONPATH": str(prefix_path), "PATH": "/usr/bin:/bin"}
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(prefix_path),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("pin-mismatch", proc.stderr)

    def test_sync_pins_tool_updates_both_resources_and_constants(self):
        """tools/sync_pins.py updates packaged resources and constants together."""
        from tools.sync_pins import sync_pins
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_repo = Path(tmp_dir)
            # Copy pins/
            shutil.copytree(self.repo_root / "pins", mock_repo / "pins")
            # Copy src/solstone_platform
            shutil.copytree(self.repo_root / "src", mock_repo / "src")

            # Run sync_pins on the mock repo
            sync_pins(mock_repo)

            # Test subprocess import from mock_repo/src
            script = (
                "import solstone_platform.pins as pins\n"
                "pins._verify_constant_synchronization()\n"
            )
            env = {"PYTHONPATH": str(mock_repo / "src"), "PATH": "/usr/bin:/bin"}
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(mock_repo),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"Sync check failed: {proc.stderr}")


if __name__ == "__main__":
    unittest.main()
