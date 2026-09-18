# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path
import unittest

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.compat import installer_revision_allows, load_minimum_installer_revision
from solstone_platform.refusals import (
    DESKTOP_AARCH64,
    DUPLICATE_KEY,
    INCOMPLETE_VARIANTS,
    LANE_INVALID,
    SCHEMA_INVALID,
    VERSION_INVALID,
    Refusal,
)
from solstone_platform.schema import validate_platform_manifest


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        self.example_path = self.repo_root / "examples" / "platform.json"
        self.example_bytes = self.example_path.read_bytes()
        self.example_obj = parse_json_strict(self.example_bytes)

    def test_example_manifest_valid(self):
        validate_platform_manifest(self.example_obj)

    def test_installer_revision_allows(self):
        min_rev_path = self.repo_root / "compat" / "minimum_installer_revision"
        min_rev = load_minimum_installer_revision(min_rev_path)
        self.assertEqual(min_rev, 1)

        # Actual 0 vs min 1 -> False
        self.assertFalse(installer_revision_allows(actual=0, minimum=1))
        # Actual 1 vs min 1 -> True
        self.assertTrue(installer_revision_allows(actual=1, minimum=1))
        # Actual 2 vs min 1 -> True
        self.assertTrue(installer_revision_allows(actual=2, minimum=1))
        # Actual 1 vs min 2 -> False
        self.assertFalse(installer_revision_allows(actual=1, minimum=2))

    def test_non_positive_minimum_installer_revision_refused(self):
        obj = parse_json_strict(self.example_bytes)
        obj["minimum_installer_revision"] = 0
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(obj)
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

    def test_duplicate_key_rejection(self):
        raw = b'{"schema_version": 1, "protocol_version": 1, "version": "1.0.0", "version": "2.0.0"}'
        with self.assertRaises(Refusal) as ctx:
            parse_json_strict(raw)
        self.assertEqual(ctx.exception.name, DUPLICATE_KEY)

    def test_duplicate_key_nested(self):
        raw = b'{"components": {"journal": {"version": "1.0.0"}, "journal": {"version": "2.0.0"}}}'
        with self.assertRaises(Refusal) as ctx:
            parse_json_strict(raw)
        self.assertEqual(ctx.exception.name, DUPLICATE_KEY)

    def test_desktop_aarch64_refusal(self):
        obj = parse_json_strict(self.example_bytes)
        obj["components"]["desktop"]["arches"]["aarch64"] = dict(obj["components"]["desktop"]["arches"]["x86_64"])
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(obj)
        self.assertEqual(ctx.exception.name, DESKTOP_AARCH64)

    def test_incomplete_variants_refusal(self):
        obj = parse_json_strict(self.example_bytes)
        del obj["components"]["desktop"]["arches"]["x86_64"]["rpm"]
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(obj)
        self.assertEqual(ctx.exception.name, INCOMPLETE_VARIANTS)

    def test_invalid_semver_refusal(self):
        obj = parse_json_strict(self.example_bytes)
        obj["version"] = "2.0.3-beta"
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(obj)
        self.assertEqual(ctx.exception.name, VERSION_INVALID)

    def test_invalid_lane_refusal(self):
        obj = parse_json_strict(self.example_bytes)
        obj["lane"] = "production"
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(obj)
        self.assertEqual(ctx.exception.name, LANE_INVALID)

    def test_unexpected_keys_refusal(self):
        obj = parse_json_strict(self.example_bytes)
        obj["extra_field"] = "malicious"
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(obj)
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

    def test_canonical_json_roundtrip(self):
        canon1 = canonical_json_bytes(self.example_obj)
        reparsed = parse_json_strict(canon1)
        canon2 = canonical_json_bytes(reparsed)
        self.assertEqual(canon1, canon2)


if __name__ == "__main__":
    unittest.main()
