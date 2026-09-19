# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Offline CLI errors stay readable to humans and machine consumers."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.build_installer import build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallInterface(unittest.TestCase):
    def test_offline_help_and_argument_errors(self):
        with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
            root = Path(tmp)
            script = build_installer(REPO_ROOT, root / "install.sh", is_production=True)
            env = {**os.environ, "HOME": str(root / "home"), "XDG_DATA_HOME": str(root / "data")}
            for option in ("--prefix", "--origin", "--components", "--lane", "--version", "--route"):
                for args in ([option, "--json"], ["--json", option], [option + "=", "--json"]):
                    with self.subTest(args=args):
                        result = subprocess.run([str(script), *args], env=env, capture_output=True, text=True)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(json.loads(result.stdout)["root_code"], "missing-value")
            for prefix in ("relative", "/tmp/line\nbreak", "/tmp/trailing\n", "/tmp/a:b", "/tmp/../escape"):
                result = subprocess.run([str(script), "--prefix", prefix, "--json"], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(json.loads(result.stdout)["root_code"], "prefix-invalid")
            strange = '--unknown"\\\t\nend\n'
            result = subprocess.run([str(script), strange, "--json"], env=env, capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(report["message"], "Unknown argument: " + strange)
            self.assertEqual(report["verification_layers"], "not-completed")
            for args in (["--help"], ["--help", "--json"]):
                result = subprocess.run([str(script), *args], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                help_text = json.loads(result.stdout)["help"] if "--json" in args else result.stdout
                for flag in ("--upgrade", "--uninstall", "--no-start", "--dry-run"):
                    self.assertIn(flag, help_text)
            self.assertFalse((root / "home").exists())
            self.assertFalse((root / "data").exists())
