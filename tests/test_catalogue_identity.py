# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import copy
import tempfile
from pathlib import Path
import unittest

from solstone_platform.canonical import parse_json_strict
from solstone_platform.refusals import (
    DUPLICATE_KEY,
    RELEASE_COHERENCE,
    SCHEMA_INVALID,
    VERSION_INVALID,
    Refusal,
)
from solstone_platform.schema import (
    MAX_SEMVER_DIGITS,
    SCHEMA_1_KEYS,
    SCHEMA_2_KEYS,
    allocate_catalogue_revision,
    catalogue_coordinate,
    compare_catalogue_coordinates,
    parse_catalogue_coordinate,
    render_platform_schema,
    render_posix_catalogue_keys,
    render_posix_identity_fragment,
    validate_catalogue_transition,
    validate_platform_manifest,
)


class TestCatalogueIdentity(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        self.example_path = self.repo_root / "examples" / "platform.json"
        self.example_bytes = self.example_path.read_bytes()
        self.example_obj = parse_json_strict(self.example_bytes)

    def test_parse_catalogue_coordinate(self):
        self.assertEqual(parse_catalogue_coordinate("2.0.32"), ("bare", (2, 0, 32), None))
        self.assertEqual(parse_catalogue_coordinate("2.0.32-r1"), ("revised", (2, 0, 32), 1))
        self.assertEqual(parse_catalogue_coordinate("0.0.0-r99"), ("revised", (0, 0, 0), 99))

        # Invalid formats raise VERSION_INVALID
        for invalid in (
            "2.0.32-r0",
            "2.0.32-r01",
            "2.0.32-r-1",
            "2.0.32-rc1",
            "2.0.32-r",
            "2.0.32.1",
            "v2.0.32",
            "2.0.32-r1extra",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(Refusal) as ctx:
                    parse_catalogue_coordinate(invalid)
                self.assertEqual(ctx.exception.name, VERSION_INVALID)

    def test_identity_edge_corpus(self):
        for coordinate in ("2.0.33\n", "2.0.33-r1\n", "2.0.33-r01", "2.0.33-r١",
                           "2.0.33-r" + "1" * 21, "1" * 21 + ".0.0-r1", True, None):
            with self.subTest(coordinate=coordinate), self.assertRaises(Refusal):
                parse_catalogue_coordinate(coordinate)
        for value in (True, "1", 1.0, 0, -1, 10**MAX_SEMVER_DIGITS):
            obj = copy.deepcopy(self.example_obj)
            obj.update(schema_version=2, version=obj["components"]["journal"]["version"], catalogue_revision=value)
            with self.subTest(value=value), self.assertRaises(Refusal):
                validate_platform_manifest(obj)
            with self.subTest(allocation=value), self.assertRaises(Refusal):
                allocate_catalogue_revision("2.0.33-r1", "2.0.33", value)
        self.assertEqual(compare_catalogue_coordinates("9.0.0", "2.0.34-r1"), -1)
        schema = __import__("json").loads(render_platform_schema())
        for branch in schema["oneOf"]:
            for property_schema in branch["properties"].values():
                self.assertIsInstance(property_schema, (dict, bool))

    def test_compare_catalogue_coordinates(self):
        # Bare vs Bare
        self.assertEqual(compare_catalogue_coordinates("2.0.31", "2.0.32"), -1)
        self.assertEqual(compare_catalogue_coordinates("2.0.32", "2.0.32"), 0)
        self.assertEqual(compare_catalogue_coordinates("2.0.33", "2.0.32"), 1)

        # Bare vs Revised: Revised is strictly newer across eras
        self.assertEqual(compare_catalogue_coordinates("2.0.32", "2.0.32-r1"), -1)
        self.assertEqual(compare_catalogue_coordinates("2.0.32-r1", "2.0.32"), 1)
        self.assertEqual(compare_catalogue_coordinates("2.0.31", "2.0.32-r1"), -1)
        self.assertEqual(compare_catalogue_coordinates("2.0.33", "2.0.32-r10"), -1)
        self.assertEqual(compare_catalogue_coordinates("2.0.32-r10", "2.0.33"), 1)

        # Revised vs Revised (equal triple) -> Revision decides
        self.assertEqual(compare_catalogue_coordinates("2.0.32-r1", "2.0.32-r2"), -1)
        self.assertEqual(compare_catalogue_coordinates("2.0.32-r2", "2.0.32-r2"), 0)
        self.assertEqual(compare_catalogue_coordinates("2.0.32-r3", "2.0.32-r2"), 1)

        # Revised vs Revised (different triple) -> Triple dominates
        self.assertEqual(compare_catalogue_coordinates("2.0.31-r9", "2.0.32-r1"), -1)
        self.assertEqual(compare_catalogue_coordinates("2.0.33-r1", "2.0.32-r99"), 1)

    def test_allocate_catalogue_revision(self):
        # From bare base
        self.assertEqual(allocate_catalogue_revision("2.0.32", "2.0.32"), 1)
        self.assertEqual(allocate_catalogue_revision("2.0.32", "2.0.33"), 1)
        self.assertEqual(allocate_catalogue_revision("2.0.32", "2.0.32", explicit_revision=5), 5)

        # From revised base - same journal triple
        self.assertEqual(allocate_catalogue_revision("2.0.32-r1", "2.0.32"), 2)
        self.assertEqual(allocate_catalogue_revision("2.0.32-r1", "2.0.32", explicit_revision=3), 3)
        with self.assertRaises(Refusal) as ctx:
            allocate_catalogue_revision("2.0.32-r2", "2.0.32", explicit_revision=2)
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

        # From revised base - higher journal triple
        self.assertEqual(allocate_catalogue_revision("2.0.32-r5", "2.0.33"), 1)
        self.assertEqual(allocate_catalogue_revision("2.0.32-r5", "2.0.33", explicit_revision=2), 2)

        # Older journal triple
        with self.assertRaises(Refusal) as ctx:
            allocate_catalogue_revision("2.0.33-r1", "2.0.32")
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

        # Decimal exhaustion
        max_rev = 10**MAX_SEMVER_DIGITS - 1
        with self.assertRaises(Refusal) as ctx:
            allocate_catalogue_revision(f"2.0.32-r{max_rev}", "2.0.32")
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_validate_catalogue_transition(self):
        base_s1 = copy.deepcopy(self.example_obj)
        cand_s2 = copy.deepcopy(self.example_obj)
        cand_s2["schema_version"] = 2
        cand_s2["catalogue_revision"] = 1
        cand_s2["minimum_installer_revision"] = 8
        cand_s2["version"] = cand_s2["components"]["journal"]["version"]

        # Valid s1 -> s2
        validate_catalogue_transition(base_s1, cand_s2, required_schema2_floor=8)

        # s1 -> s2 with wrong floor
        cand_s2_bad = copy.deepcopy(cand_s2)
        cand_s2_bad["minimum_installer_revision"] = 7
        with self.assertRaises(Refusal) as ctx:
            validate_catalogue_transition(base_s1, cand_s2_bad, required_schema2_floor=8)
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

        # s2 -> s2 valid
        base_s2 = copy.deepcopy(cand_s2)
        cand_s2_next = copy.deepcopy(cand_s2)
        cand_s2_next["catalogue_revision"] = 2
        validate_catalogue_transition(base_s2, cand_s2_next, required_schema2_floor=8)

        # s2 -> s2 protected field changed
        cand_s2_changed = copy.deepcopy(cand_s2_next)
        cand_s2_changed["lane"] = "staging"
        with self.assertRaises(Refusal) as ctx:
            validate_catalogue_transition(base_s2, cand_s2_changed, required_schema2_floor=8)
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

        # s2 -> s1 rollback refused
        with self.assertRaises(Refusal) as ctx:
            validate_catalogue_transition(base_s2, base_s1, required_schema2_floor=8)
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_schema_2_manifest_validation(self):
        s2_obj = copy.deepcopy(self.example_obj)
        s2_obj["schema_version"] = 2
        s2_obj["catalogue_revision"] = 1
        s2_obj["version"] = s2_obj["components"]["journal"]["version"]

        # Valid schema 2
        validate_platform_manifest(s2_obj)
        self.assertEqual(catalogue_coordinate(s2_obj), f"{s2_obj['version']}-r1")

        # Schema 2 with version != journal version -> Refused
        s2_bad_ver = copy.deepcopy(s2_obj)
        s2_bad_ver["version"] = "9.9.9"
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(s2_bad_ver)
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

        # Schema 2 with extra key -> Refused
        s2_extra = copy.deepcopy(s2_obj)
        s2_extra["unknown_key"] = "value"
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(s2_extra)
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

        # Schema 2 with invalid revision type
        s2_bool_rev = copy.deepcopy(s2_obj)
        s2_bool_rev["catalogue_revision"] = True
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(s2_bool_rev)
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

        s2_str_rev = copy.deepcopy(s2_obj)
        s2_str_rev["catalogue_revision"] = "1"
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(s2_str_rev)
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

    def test_schema_1_manifest_validation(self):
        s1_obj = copy.deepcopy(self.example_obj)
        # Schema 1 version may differ from journal
        s1_obj["version"] = "2.0.0"
        validate_platform_manifest(s1_obj)
        self.assertEqual(catalogue_coordinate(s1_obj), "2.0.0")

        # Schema 1 with catalogue_revision -> Refused
        s1_rev = copy.deepcopy(s1_obj)
        s1_rev["catalogue_revision"] = 1
        with self.assertRaises(Refusal) as ctx:
            validate_platform_manifest(s1_rev)
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

    def test_renderer_preserves_parser_boundaries_and_is_idempotent(self):
        from tools.render_catalogue_identity import render_all
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "install.sh.in").write_bytes((self.repo_root / "install.sh.in").read_bytes())
            original = (root / "install.sh.in").read_bytes()
            render_all(root)
            self.assertEqual((root / "install.sh.in").read_bytes(), original)
            render_all(root)
            self.assertEqual((root / "install.sh.in").read_bytes(), original)
            self.assertEqual((root / "schema/platform.v1.json").read_text(), render_platform_schema())

    def test_rendered_artifacts_match_sources(self):
        schema_path = self.repo_root / "schema" / "platform.v1.json"
        self.assertEqual(schema_path.read_text(encoding="utf-8"), render_platform_schema())

        install_in = (self.repo_root / "install.sh.in").read_text(encoding="utf-8")
        self.assertIn(render_posix_identity_fragment().strip(), install_in)
        self.assertIn(render_posix_catalogue_keys().strip() + "\n", install_in)


if __name__ == "__main__":
    unittest.main()
