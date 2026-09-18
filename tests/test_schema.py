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
from solstone_platform.schema import (
    COMPONENT_CONTRACTS,
    MAX_PLATFORM_BYTES,
    load_platform_manifest_bytes,
    validate_platform_manifest,
)


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

    def assert_schema_refusal(self, obj):
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(obj)
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

    def assert_loader_schema_refusal(self, obj):
        with self.assertRaises(Refusal) as ctx:
            load_platform_manifest_bytes(canonical_json_bytes(obj))
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

    def test_platform_loader_requires_exact_canonical_bytes(self):
        self.assertEqual(load_platform_manifest_bytes(self.example_bytes), self.example_obj)
        for raw in (
            b"\xef\xbb\xbf" + self.example_bytes,
            self.example_bytes + b"\x00",
            self.example_bytes + b"\n",
            b'{"schema_version":1,"schema_version":1}',
            b'{"schema_version":NaN}',
            b'{"schema_version":1.0}',
            b"\xff",
        ):
            with self.subTest(raw=raw[:40]):
                with self.assertRaises(Refusal):
                    load_platform_manifest_bytes(raw)

    def test_platform_loader_enforces_byte_depth_and_integer_bounds(self):
        invalid = [
            b" " * (MAX_PLATFORM_BYTES + 1),
            (b"[" * 65) + b"null" + (b"]" * 65),
            b'{"created_unix":123456789012345678901}',
        ]
        for raw in invalid:
            with self.subTest(length=len(raw)):
                with self.assertRaises(Refusal) as ctx:
                    load_platform_manifest_bytes(raw)
                self.assertEqual(ctx.exception.name, SCHEMA_INVALID)
        self.assertEqual(canonical_json_bytes(10**19), b"10000000000000000000")

    def test_canonical_serializer_is_total_and_uses_utf16_key_order(self):
        self.assertEqual(
            canonical_json_bytes({"\ue000": 1, "\U00010000": 2}),
            '{"\U00010000":2,"\ue000":1}'.encode("utf-8"),
        )
        cyclic = []
        cyclic.append(cyclic)
        nested = None
        for _ in range(66):
            nested = [nested]
        invalid_values = [
            {1: "non-string-key"},
            cyclic,
            "\ud800",
            10**20,
            1.5,
            nested,
        ]
        for value in invalid_values:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(Refusal) as ctx:
                    canonical_json_bytes(value)
                self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

    def test_semver_is_ascii_bounded_and_boolean_discriminators_refuse(self):
        for field in ("schema_version", "protocol_version"):
            obj = parse_json_strict(self.example_bytes)
            obj[field] = True
            self.assert_schema_refusal(obj)
        for value in ("1.2.3١", "1.2." + "3" * 21):
            for path in ("platform", "component", "reader"):
                obj = parse_json_strict(self.example_bytes)
                if path == "platform":
                    obj["version"] = value
                elif path == "component":
                    obj["components"]["journal"]["version"] = value
                else:
                    obj["components"]["journal"]["provenance"]["state_reader_min"] = value
                with self.assertRaises(Refusal) as ctx:
                    validate_platform_manifest(obj)
                expected = VERSION_INVALID if path in {"platform", "component"} else SCHEMA_INVALID
                self.assertEqual(ctx.exception.name, expected)

    def test_component_cross_field_contracts_refuse_swaps(self):
        mutations = []
        obj = parse_json_strict(self.example_bytes)
        obj["components"]["journal"]["arches"]["x86_64"]["tree"]["native_target"] = "linux-aarch64"
        mutations.append(obj)
        obj = parse_json_strict(self.example_bytes)
        obj["components"]["journal"]["arches"]["x86_64"]["tree"]["authority"] = dict(
            obj["components"]["desktop"]["arches"]["x86_64"]["tree"]["authority"]
        )
        mutations.append(obj)
        obj = parse_json_strict(self.example_bytes)
        obj["components"]["journal"]["install_entrypoint"] = "install-desktop"
        mutations.append(obj)
        obj = parse_json_strict(self.example_bytes)
        obj["components"]["journal"]["arches"]["x86_64"]["deb"]["package_identity"]["arch"] = "x86_64"
        mutations.append(obj)
        obj = parse_json_strict(self.example_bytes)
        obj["components"]["journal"]["arches"]["x86_64"]["tree"]["executable"]["version_command"] = ["journal", "-V"]
        mutations.append(obj)
        for obj in mutations:
            self.assert_schema_refusal(obj)

    def test_journal_provenance_contract_and_reader_window(self):
        for contract in (1, 3):
            obj = parse_json_strict(self.example_bytes)
            obj["components"]["journal"]["provenance"]["bootstrap"]["contract_version"] = contract
            self.assert_loader_schema_refusal(obj)
        obj = parse_json_strict(self.example_bytes)
        prov = obj["components"]["journal"]["provenance"]
        prov["state_reader_min"] = "3.0.0"
        prov["state_reader_max"] = "2.0.0"
        self.assert_loader_schema_refusal(obj)
        obj = parse_json_strict(self.example_bytes)
        obj["components"]["journal"]["version"] = "9.0.0"
        self.assert_loader_schema_refusal(obj)

    def test_inventory_order_uniqueness_and_executable_binding(self):
        base = parse_json_strict(self.example_bytes)
        variant = base["components"]["journal"]["arches"]["x86_64"]["tree"]
        executable = variant["executable"]

        obj = parse_json_strict(self.example_bytes)
        inv = obj["components"]["journal"]["arches"]["x86_64"]["tree"]["archive_inventory"]
        inv.append({"path": "opt/other/readme", "kind": "file", "size": 1, "sha256": "0" * 64})
        inv.sort(key=lambda item: item["path"])
        inv.reverse()
        self.assert_schema_refusal(obj)

        obj = parse_json_strict(self.example_bytes)
        inv = obj["components"]["journal"]["arches"]["x86_64"]["tree"]["archive_inventory"]
        inv.append({"path": "opt/other/journal", "kind": "file", "size": 1, "sha256": executable["sha256"]})
        inv.sort(key=lambda item: item["path"])
        self.assert_schema_refusal(obj)

        obj = parse_json_strict(self.example_bytes)
        variant = obj["components"]["journal"]["arches"]["x86_64"]["tree"]
        variant["executable"]["sha256"] = "f" * 64
        self.assert_schema_refusal(obj)

        obj = parse_json_strict(self.example_bytes)
        variant = obj["components"]["journal"]["arches"]["x86_64"]["tree"]
        variant["archive_inventory"].extend([
            {"path": "opt/literal%2Fname", "kind": "file", "size": 0, "sha256": "0" * 64},
            {"path": "usr/bin/link", "kind": "symlink", "size": 0, "link_target": "../lib/foo"},
        ])
        variant["archive_inventory"].sort(key=lambda item: item["path"])
        validate_platform_manifest(obj)

    def test_variant_filenames_are_globally_unique(self):
        obj = parse_json_strict(self.example_bytes)
        journal_name = obj["components"]["journal"]["arches"]["x86_64"]["tree"]["filename"]
        obj["components"]["desktop"]["arches"]["x86_64"]["tree"]["filename"] = journal_name
        self.assert_schema_refusal(obj)

    def test_semantic_validator_translates_mixed_keys(self):
        obj = parse_json_strict(self.example_bytes)
        obj[7] = "bad key"
        self.assert_schema_refusal(obj)

    def test_component_contract_constants_match_checked_in_contracts(self):
        for component, expected in COMPONENT_CONTRACTS.items():
            contract = parse_json_strict((self.repo_root / "contracts" / f"{component}.v1.json").read_bytes())
            self.assertEqual(expected["handler_contract_version"], contract["contract_version"])
            self.assertEqual(expected["install_entrypoint"], contract["install_entrypoint"])
            self.assertEqual(expected["uninstall_service_entrypoint"], contract["uninstall_service_entrypoint"])
            self.assertEqual(expected["executable_name"], contract["executable_name"])
            self.assertEqual(expected["version_command"], contract["version_command"])

    def test_json_schema_direct_constraints_match_python_contract(self):
        schema = parse_json_strict((self.repo_root / "schema" / "platform.v1.json").read_bytes())
        self.assertIn("Structural subset only", schema["$comment"])
        for component, expected in COMPONENT_CONTRACTS.items():
            definition = schema["definitions"][f"{component}_component"]["properties"]
            self.assertEqual(definition["handler_contract_version"]["const"], expected["handler_contract_version"])
            self.assertEqual(definition["install_entrypoint"]["const"], expected["install_entrypoint"])
            self.assertEqual(
                definition["uninstall_service_entrypoint"]["const"],
                expected["uninstall_service_entrypoint"],
            )


if __name__ == "__main__":
    unittest.main()
