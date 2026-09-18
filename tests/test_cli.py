# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import tempfile
import unittest

from solstone_platform.cli import main, read_passphrase_file
from solstone_platform.refusals import PASSPHRASE_SOURCE_INVALID, Refusal


class TestPassphraseFile(unittest.TestCase):
    def test_reads_one_private_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "key.pass"
            path.write_text("correct horse battery staple\n", encoding="utf-8")
            os.chmod(path, 0o600)
            self.assertEqual(read_passphrase_file(path), "correct horse battery staple")

    def test_refuses_group_readable_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "key.pass"
            path.write_text("secret\n", encoding="utf-8")
            os.chmod(path, 0o640)
            with self.assertRaises(Refusal) as ctx:
                read_passphrase_file(path)
            self.assertEqual(ctx.exception.name, PASSPHRASE_SOURCE_INVALID)

    def test_refuses_multiple_lines(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "key.pass"
            path.write_text("first\nsecond\n", encoding="utf-8")
            os.chmod(path, 0o600)
            with self.assertRaises(Refusal) as ctx:
                read_passphrase_file(path)
            self.assertEqual(ctx.exception.name, PASSPHRASE_SOURCE_INVALID)

    def test_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            real_path = Path(tmp_dir) / "real.pass"
            link_path = Path(tmp_dir) / "link.pass"
            real_path.write_text("secret\n", encoding="utf-8")
            os.chmod(real_path, 0o600)
            link_path.symlink_to(real_path)
            with self.assertRaises(Refusal) as ctx:
                read_passphrase_file(link_path)
            self.assertEqual(ctx.exception.name, PASSPHRASE_SOURCE_INVALID)

    def test_refuses_oversized_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "key.pass"
            path.write_bytes(b"x" * 4097)
            os.chmod(path, 0o600)
            with self.assertRaises(Refusal) as ctx:
                read_passphrase_file(path)
            self.assertEqual(ctx.exception.name, PASSPHRASE_SOURCE_INVALID)


class TestProductionCliGate(unittest.TestCase):
    def test_explicit_production_pin_still_requires_ack(self):
        repo_root = Path(__file__).parent.parent
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = main([
                "sign",
                "--manifest", str(repo_root / "examples" / "platform.json"),
                "--secret-key", str(repo_root / "missing.key"),
                "--platform-pub", str(repo_root / "pins" / "platform.pub"),
            ])
        self.assertEqual(result, 1)
        self.assertIn("--acknowledge-production required", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
