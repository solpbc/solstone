# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path
import unittest

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.destination import ResultStatus
from solstone_platform.publish import compare_semver, publish_release
from solstone_platform.refusals import (
    PUBLISH_INDETERMINATE,
    RELEASE_COHERENCE,
    ROLLBACK_REFUSED,
    SAME_VERSION_DIFFERENT_BYTES,
    SIGNATURE_PIN_MISMATCH,
    Refusal,
)
from solstone_platform.sign import ephemeral_keypair, sign_manifest
from solstone_platform.testdest import InMemoryDestination


class TestPublish(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        self.example_path = self.repo_root / "examples" / "platform.json"
        self.raw_manifest = self.example_path.read_bytes()

    def test_compare_semver(self):
        self.assertEqual(compare_semver("1.0.0", "1.0.0"), 0)
        self.assertEqual(compare_semver("1.0.0", "1.0.1"), -1)
        self.assertEqual(compare_semver("1.1.0", "1.0.1"), 1)
        self.assertEqual(compare_semver("2.0.0", "1.99.99"), 1)

    def test_publish_flow_and_idempotence(self):
        dest = InMemoryDestination()
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            # Update example with pin.key_id
            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            manifest_bytes = canonical_json_bytes(obj)

            sig_bytes = sign_manifest(
                manifest_bytes=manifest_bytes,
                secret_key_path=sec_path,
                selected_pin=pin,
                passphrase_callback=lambda: "",
            )

            report = publish_release(
                manifest_bytes=manifest_bytes,
                signature_bytes=sig_bytes,
                selected_pin=pin,
                dest=dest,
            )
            self.assertEqual(report.version, "2.0.3")
            self.assertTrue(report.latest_promoted)

            # Check destination contents and metadata
            latest_get = dest.get("solstone/release/latest")
            self.assertTrue(latest_get.is_ok())
            self.assertEqual(latest_get.body, b"2.0.3\n")
            self.assertEqual(latest_get.content_type, "text/plain; charset=utf-8")
            self.assertEqual(latest_get.cache_control, "no-store, max-age=0")

            m_get = dest.get("solstone/release/2.0.3/platform.json")
            self.assertEqual(m_get.content_type, "application/json")
            self.assertEqual(m_get.cache_control, "public, max-age=31536000, immutable")

            s_get = dest.get("solstone/release/2.0.3/platform.json.minisig")
            self.assertEqual(s_get.content_type, "application/octet-stream")
            self.assertEqual(s_get.cache_control, "public, max-age=31536000, immutable")

            # Republish identical bytes -> must succeed idempotently
            report2 = publish_release(
                manifest_bytes=manifest_bytes,
                signature_bytes=sig_bytes,
                selected_pin=pin,
                dest=dest,
            )
            self.assertTrue(report2.latest_promoted)

    def test_publish_same_version_different_bytes_rejected(self):
        dest = InMemoryDestination()
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            manifest_bytes1 = canonical_json_bytes(obj)
            sig_bytes1 = sign_manifest(manifest_bytes1, sec_path, pin, passphrase_callback=lambda: "")

            publish_release(manifest_bytes1, sig_bytes1, pin, dest)

            # Modify commit hash in manifest for same version
            obj["source_commit"] = "1111111111111111111111111111111111111111"
            manifest_bytes2 = canonical_json_bytes(obj)
            sig_bytes2 = sign_manifest(manifest_bytes2, sec_path, pin, passphrase_callback=lambda: "")

            with self.assertRaises(Refusal) as ctx:
                publish_release(manifest_bytes2, sig_bytes2, pin, dest)
            self.assertEqual(ctx.exception.name, SAME_VERSION_DIFFERENT_BYTES)

    def test_split_pair_conflict(self):
        dest = InMemoryDestination()
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            # Manifest A
            obj_a = parse_json_strict(self.raw_manifest)
            obj_a["platform_key_id"] = pin.key_id
            obj_a["source_commit"] = "0" * 40
            manifest_a = canonical_json_bytes(obj_a)
            sig_a = sign_manifest(manifest_a, sec_path, pin, passphrase_callback=lambda: "")

            # Manifest B (same version, different commit)
            obj_b = parse_json_strict(self.raw_manifest)
            obj_b["platform_key_id"] = pin.key_id
            obj_b["source_commit"] = "1" * 40
            manifest_b = canonical_json_bytes(obj_b)
            sig_b = sign_manifest(manifest_b, sec_path, pin, passphrase_callback=lambda: "")

            # Publisher A claims platform.json
            dest.put_if_absent(
                "solstone/release/2.0.3/platform.json",
                manifest_a,
                "application/json",
                "public, max-age=31536000, immutable",
            )

            # Before A uploads signature, Publisher B attempts to publish B's bytes
            with self.assertRaises(Refusal) as ctx:
                publish_release(manifest_b, sig_b, pin, dest)
            self.assertEqual(ctx.exception.name, SAME_VERSION_DIFFERENT_BYTES)

            # Destination must contain exactly A's platform.json
            m_get = dest.get("solstone/release/2.0.3/platform.json")
            self.assertTrue(m_get.is_ok())
            self.assertEqual(m_get.body, manifest_a)

            # Signature and latest must still be absent
            self.assertTrue(dest.get("solstone/release/2.0.3/platform.json.minisig").is_absent())
            self.assertTrue(dest.get("solstone/release/latest").is_absent())

            # B must not be able to fill B's signature on A's manifest
            with self.assertRaises(Refusal) as ctx:
                publish_release(manifest_a, sig_b, pin, dest)
            self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)
            self.assertTrue(dest.get("solstone/release/2.0.3/platform.json.minisig").is_absent())

            # Now Publisher A completes publish with matching signature
            report_a = publish_release(manifest_a, sig_a, pin, dest)
            self.assertTrue(report_a.latest_promoted)
            self.assertEqual(dest.get("solstone/release/2.0.3/platform.json.minisig").body, sig_a)
            self.assertEqual(dest.get("solstone/release/latest").body, b"2.0.3\n")

    def test_identical_retry_fills_missing_signature(self):
        dest = InMemoryDestination()
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            manifest_bytes = canonical_json_bytes(obj)
            sig_bytes = sign_manifest(manifest_bytes, sec_path, pin, passphrase_callback=lambda: "")

            # Put platform.json directly
            dest.put_if_absent(
                "solstone/release/2.0.3/platform.json",
                manifest_bytes,
                "application/json",
                "public, max-age=31536000, immutable",
            )
            self.assertTrue(dest.get("solstone/release/2.0.3/platform.json.minisig").is_absent())
            self.assertTrue(dest.get("solstone/release/latest").is_absent())

            # Publish retry with identical manifest and signature
            report = publish_release(manifest_bytes, sig_bytes, pin, dest)
            self.assertTrue(report.latest_promoted)
            self.assertEqual(dest.get("solstone/release/2.0.3/platform.json.minisig").body, sig_bytes)
            self.assertEqual(dest.get("solstone/release/latest").body, b"2.0.3\n")

    def test_stale_list_does_not_authorize_overwrite(self):
        dest = InMemoryDestination()
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            obj_a = parse_json_strict(self.raw_manifest)
            obj_a["platform_key_id"] = pin.key_id
            obj_a["source_commit"] = "0" * 40
            manifest_a = canonical_json_bytes(obj_a)
            sig_a = sign_manifest(manifest_a, sec_path, pin, passphrase_callback=lambda: "")

            publish_release(manifest_a, sig_a, pin, dest)

            # Diagnostic list call (must not affect publisher)
            keys = dest.list_keys()
            self.assertIn("solstone/release/2.0.3/platform.json", keys)

            # Attempt publish of same version different bytes
            obj_b = parse_json_strict(self.raw_manifest)
            obj_b["platform_key_id"] = pin.key_id
            obj_b["source_commit"] = "1" * 40
            manifest_b = canonical_json_bytes(obj_b)
            sig_b = sign_manifest(manifest_b, sec_path, pin, passphrase_callback=lambda: "")

            with self.assertRaises(Refusal) as ctx:
                publish_release(manifest_b, sig_b, pin, dest)
            self.assertEqual(ctx.exception.name, SAME_VERSION_DIFFERENT_BYTES)
            self.assertEqual(dest.get("solstone/release/2.0.3/platform.json").body, manifest_a)

            # Assert list_keys or list( does not appear in publish.py
            publish_py = (self.repo_root / "src" / "solstone_platform" / "publish.py").read_text(encoding="utf-8")
            self.assertNotIn("list_keys", publish_py)
            self.assertNotIn("list(", publish_py)

    def test_post_generation_native_change(self):
        dest = InMemoryDestination()
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            manifest_bytes = canonical_json_bytes(obj)
            sig_bytes = sign_manifest(manifest_bytes, sec_path, pin, passphrase_callback=lambda: "")

            def fail_native():
                raise Refusal(RELEASE_COHERENCE, "native changed")

            with self.assertRaises(Refusal) as ctx:
                publish_release(
                    manifest_bytes=manifest_bytes,
                    signature_bytes=sig_bytes,
                    selected_pin=pin,
                    dest=dest,
                    verify_native_callback=fail_native,
                )
            self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)
            # Latest must be absent
            self.assertTrue(dest.get("solstone/release/latest").is_absent())
            # Platform objects may exist
            self.assertTrue(dest.get("solstone/release/2.0.3/platform.json").is_ok())

    def test_rollback_refused(self):
        dest = InMemoryDestination()
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            obj["version"] = "2.0.4"
            m_204 = canonical_json_bytes(obj)
            s_204 = sign_manifest(m_204, sec_path, pin, passphrase_callback=lambda: "")
            publish_release(m_204, s_204, pin, dest)

            # Attempt to publish older version 2.0.3
            obj["version"] = "2.0.3"
            m_203 = canonical_json_bytes(obj)
            s_203 = sign_manifest(m_203, sec_path, pin, passphrase_callback=lambda: "")

            with self.assertRaises(Refusal) as ctx:
                publish_release(m_203, s_203, pin, dest)
            self.assertEqual(ctx.exception.name, ROLLBACK_REFUSED)

    def test_n_vs_n_plus_1_race_retry_success(self):
        dest = InMemoryDestination()
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            # Seed destination with 2.0.2 as latest
            dest.put_if_absent("solstone/release/latest", b"2.0.2\n", "text/plain; charset=utf-8", "no-store, max-age=0")

            # Prepare 2.0.4 for publish
            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            obj["version"] = "2.0.4"
            m_204 = canonical_json_bytes(obj)
            s_204 = sign_manifest(m_204, sec_path, pin, passphrase_callback=lambda: "")

            # Set up race: first CAS attempt on latest will fail, but destination latest becomes 2.0.3
            # We simulate this by mutating destination latest right before CAS
            dest.objects["solstone/release/latest"].body = b"2.0.3\n"
            dest.objects["solstone/release/latest"].etag = '"new-etag-203"'

            report = publish_release(m_204, s_204, pin, dest)
            self.assertEqual(report.version, "2.0.4")
            self.assertTrue(report.latest_promoted)
            self.assertEqual(dest.get("solstone/release/latest").body, b"2.0.4\n")

    def test_n_vs_n_plus_2_race_no_demote(self):
        dest = InMemoryDestination()
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            # Seed destination with 2.0.5
            dest.put_if_absent("solstone/release/latest", b"2.0.5\n", "text/plain; charset=utf-8", "no-store, max-age=0")

            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            obj["version"] = "2.0.3"
            m_203 = canonical_json_bytes(obj)
            s_203 = sign_manifest(m_203, sec_path, pin, passphrase_callback=lambda: "")

            # Simulate Step 6 observing 2.0.2 initially, but CAS fails against 2.0.5
            from solstone_platform.destination import GetResult, ResultStatus
            dest.get_overrides["solstone/release/latest"] = GetResult(status=ResultStatus.OK, body=b"2.0.2\n", etag='"stale-etag-202"')

            report = publish_release(m_203, s_203, pin, dest)
            self.assertEqual(report.version, "2.0.3")
            self.assertFalse(report.latest_promoted)
            self.assertEqual(dest.get("solstone/release/latest").body, b"2.0.5\n")

    def test_fault_injection_put_ok_no_object(self):
        dest = InMemoryDestination()
        dest.fail_next_put_ok_no_object = True
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            m = canonical_json_bytes(obj)
            s = sign_manifest(m, sec_path, pin, passphrase_callback=lambda: "")

            with self.assertRaises(Refusal) as ctx:
                publish_release(m, s, pin, dest)
            self.assertEqual(ctx.exception.name, PUBLISH_INDETERMINATE)
            self.assertTrue(dest.get("solstone/release/latest").is_absent())

    def test_fault_injection_put_corrupted_bytes(self):
        dest = InMemoryDestination()
        dest.fail_next_put_wrong_bytes = b"corrupted payload"
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            m = canonical_json_bytes(obj)
            s = sign_manifest(m, sec_path, pin, passphrase_callback=lambda: "")

            with self.assertRaises(Refusal) as ctx:
                publish_release(m, s, pin, dest)
            self.assertEqual(ctx.exception.name, PUBLISH_INDETERMINATE)
            self.assertTrue(dest.get("solstone/release/latest").is_absent())

    def test_metadata_mismatch_refusal(self):
        dest = InMemoryDestination()
        dest.fail_next_put_wrong_content_type = "text/plain"
        with ephemeral_keypair("publisher test key") as (sec_path, pub_path, pin):
            obj = parse_json_strict(self.raw_manifest)
            obj["platform_key_id"] = pin.key_id
            m = canonical_json_bytes(obj)
            s = sign_manifest(m, sec_path, pin, passphrase_callback=lambda: "")

            with self.assertRaises(Refusal) as ctx:
                publish_release(m, s, pin, dest)
            self.assertEqual(ctx.exception.name, PUBLISH_INDETERMINATE)


if __name__ == "__main__":
    unittest.main()
