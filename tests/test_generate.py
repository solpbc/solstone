# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path
import tempfile
import unittest

from solstone_platform.canonical import parse_json_strict
from solstone_platform.generate import generate_platform_manifest
from solstone_platform.schema import validate_platform_manifest
from tools.fixture_builder import build_tiny_natives


class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent

    def test_end_to_end_generate_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pinset, dirs = build_tiny_natives(tmp_path)

            manifest_bytes = generate_platform_manifest(
                version="2.0.3",
                lane="release",
                created_unix=1773792000,
                source_commit="0" * 40,
                platform_key_id="0000000000000000",
                repo_root=self.repo_root,
                journal_dir=dirs["journal"],
                desktop_dir=dirs["desktop"],
                tmux_dir=dirs["tmux"],
                journal_origin="https://127.0.0.1",
                pins=pinset,
            )

            manifest_obj = parse_json_strict(manifest_bytes)
            validate_platform_manifest(manifest_obj)
            self.assertEqual(manifest_obj["version"], "2.0.3")
            self.assertEqual(manifest_obj["components"]["journal"]["version"], "2.0.6")
            self.assertEqual(manifest_obj["components"]["desktop"]["version"], "2.0.3")
            self.assertEqual(manifest_obj["components"]["tmux"]["version"], "2.0.3")

    def test_generate_determinism(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pinset, dirs = build_tiny_natives(tmp_path)

            m1 = generate_platform_manifest(
                version="2.0.3",
                lane="release",
                created_unix=1773792000,
                source_commit="a" * 40,
                platform_key_id="1111222233334444",
                repo_root=self.repo_root,
                journal_dir=dirs["journal"],
                desktop_dir=dirs["desktop"],
                tmux_dir=dirs["tmux"],
                journal_origin="https://127.0.0.1",
                pins=pinset,
            )

            m2 = generate_platform_manifest(
                version="2.0.3",
                lane="release",
                created_unix=1773792000,
                source_commit="a" * 40,
                platform_key_id="1111222233334444",
                repo_root=self.repo_root,
                journal_dir=dirs["journal"],
                desktop_dir=dirs["desktop"],
                tmux_dir=dirs["tmux"],
                journal_origin="https://127.0.0.1",
                pins=pinset,
            )

            self.assertEqual(m1, m2)

    def test_output_contains_no_secrets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pinset, dirs = build_tiny_natives(tmp_path)

            manifest_bytes = generate_platform_manifest(
                version="2.0.3",
                lane="release",
                created_unix=1773792000,
                source_commit="0" * 40,
                platform_key_id="0000000000000000",
                repo_root=self.repo_root,
                journal_dir=dirs["journal"],
                desktop_dir=dirs["desktop"],
                tmux_dir=dirs["tmux"],
                journal_origin="https://127.0.0.1",
                pins=pinset,
            )
            text = manifest_bytes.decode("utf-8")
            self.assertNotIn("AWS4-HMAC-SHA256", text)
            secret_header = "untrusted comment: minisign " + "secret key"
            self.assertNotIn(secret_header, text)
            self.assertNotIn("passphrase", text.lower())

    def test_cli_always_uses_embedded_native_pins(self):
        cli_py = (self.repo_root / "src" / "solstone_platform" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("pins=embedded_pins()", cli_py)
        # Ensure no native pin override flags in generate
        self.assertNotIn("--journal-pin", cli_py)
        self.assertNotIn("--desktop-pin", cli_py)
        self.assertNotIn("--tmux-pin", cli_py)


if __name__ == "__main__":
    unittest.main()
