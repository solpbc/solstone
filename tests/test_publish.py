# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for atomic publication rail, complete dependency claim, and immutable destination integrity."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.destination import Destination
from solstone_platform.generate import generate_platform_manifest
from solstone_platform.pins import MinisignPin, PinSet, embedded_pins
from solstone_platform.publish import compare_semver, publish_release
from solstone_platform.r2 import R2Config, R2Destination
from solstone_platform.refusals import (
    DUPLICATE_KEY,
    PUBLISH_INDETERMINATE,
    RELEASE_COHERENCE,
    ROLLBACK_REFUSED,
    SAME_VERSION_DIFFERENT_BYTES,
    SCHEMA_INVALID,
    SIGNATURE_PIN_MISMATCH,
    UNSAFE_FILENAME,
    Refusal,
)
from solstone_platform.sign import ephemeral_keypair, sign_manifest
from solstone_platform.testdest import FIXTURE_BUILD_TOKEN, FixtureDestination, StoredObject
from solstone_platform.witness import disable_witness, enable_witness, get_witness
from tools.fixture_builder import build_tiny_natives


class TestPublish(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo_root = Path(__file__).parent.parent
        self.witness = enable_witness()
        self.kp_ctx = ephemeral_keypair("platform test key")
        self.sec_path, self.pub_path, self.platform_pin = self.kp_ctx.__enter__()

    def tearDown(self):
        self.kp_ctx.__exit__(None, None, None)
        disable_witness()
        self.temp_dir.cleanup()

    def _setup_fixture_release(
        self,
        version: str = "2.0.3",
        lane: str = "release",
        bootstrap_script: bytes | None = None,
        min_bootstrap_revision: int = 2,
    ):
        natives_dir = self.root / f"natives-{version}"
        built_pinset, dirs = build_tiny_natives(
            natives_dir,
            bootstrap_script=bootstrap_script,
            min_bootstrap_revision=min_bootstrap_revision,
        )

        p_pin = self.platform_pin
        s_path = self.sec_path

        manifest_bytes = generate_platform_manifest(
            version=version,
            lane=lane,
            created_unix=1700000000,
            source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b",
            platform_key_id=p_pin.key_id,
            repo_root=self.repo_root,
            journal_dir=dirs["journal"],
            desktop_dir=dirs["desktop"],
            tmux_dir=dirs["tmux"],
            journal_origin="https://updates.solstone.app",
            pins=built_pinset,
        )
        manifest_path = self.root / f"platform-{version}.json"
        manifest_path.write_bytes(manifest_bytes)

        sig_bytes = sign_manifest(
            manifest_bytes=manifest_bytes,
            secret_key_path=s_path,
            selected_pin=p_pin,
            passphrase_callback=lambda: "",
            repo_root=self.repo_root,
        )
        sig_path = self.root / f"platform-{version}.json.minisig"
        sig_path.write_bytes(sig_bytes)

        dest = FixtureDestination(pinset=built_pinset, platform_pin=p_pin)

        return {
            "version": version,
            "manifest_path": manifest_path,
            "signature_path": sig_path,
            "manifest_bytes": manifest_bytes,
            "sig_bytes": sig_bytes,
            "journal_dir": dirs["journal"],
            "desktop_dir": dirs["desktop"],
            "tmux_dir": dirs["tmux"],
            "dest": dest,
            "pinset": built_pinset,
            "platform_pin": p_pin,
            "sec_path": s_path,
        }

    def test_compare_semver(self):
        self.assertEqual(compare_semver("1.0.0", "1.0.0"), 0)
        self.assertEqual(compare_semver("1.0.0", "1.0.1"), -1)
        self.assertEqual(compare_semver("1.1.0", "1.0.1"), 1)
        self.assertEqual(compare_semver("2.0.0", "1.99.99"), 1)

    def test_invalid_platform_bytes_refuse_before_destination_use(self):
        rel = self._setup_fixture_release("2.0.3")
        rel["manifest_path"].write_bytes(rel["manifest_bytes"] + b"\n")
        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=rel["dest"],
            )
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)
        self.assertEqual(rel["dest"].ledger, [])
        self.assertEqual(rel["dest"].network_sentinel, [])
        self.assertEqual(rel["dest"].objects, {})

    def test_publish_preserves_loader_refusal_names_before_destination_use(self):
        cases = (
            (b'{"schema_version":1,"schema_version":1}', DUPLICATE_KEY),
            (canonical_json_bytes({"schema_version": 1}), SCHEMA_INVALID),
        )
        for raw, expected_name in cases:
            with self.subTest(expected_name=expected_name):
                rel = self._setup_fixture_release("2.0.3")
                rel["manifest_path"].write_bytes(raw)
                with self.assertRaises(Refusal) as ctx:
                    publish_release(
                        manifest_path=rel["manifest_path"],
                        signature_path=rel["signature_path"],
                        journal_dir=rel["journal_dir"],
                        desktop_dir=rel["desktop_dir"],
                        tmux_dir=rel["tmux_dir"],
                        dest=rel["dest"],
                    )
                self.assertEqual(ctx.exception.name, expected_name)
                self.assertEqual(rel["dest"].ledger, [])
                self.assertEqual(rel["dest"].network_sentinel, [])
                self.assertEqual(rel["dest"].objects, {})

    def test_closed_signature_rejects_lookalike_arguments_before_capture(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        # 1. Reject positionally (8th positional arg)
        with self.assertRaises(Refusal) as ctx:
            publish_release(
                rel["manifest_path"],
                rel["signature_path"],
                rel["journal_dir"],
                rel["desktop_dir"],
                rel["tmux_dir"],
                dest,
                None,  # bootstrap_file (7th)
                "lookalike_8th_positional",  # 8th positional
            )
        self.assertEqual(ctx.exception.name, UNSAFE_FILENAME)
        self.assertEqual(dest.ledger, [])
        self.assertEqual(dest.network_sentinel, [])

        # 2. Reject keyword lookalikes
        lookalikes = [
            {"skip_signature": True},
            {"verify": False},
            {"verify_native_callback": lambda: None},
            {"assertion_object": object()},
            {"selected_pin": rel["platform_pin"]},
            {"key_prefix": "solstone"},
            {"repo_root": self.repo_root},
            {"resource_root": self.repo_root},
            {"pins": rel["pinset"]},
            {"snapshot": None},
        ]

        for kwargs in lookalikes:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(Refusal) as ctx:
                    publish_release(
                        manifest_path=rel["manifest_path"],
                        signature_path=rel["signature_path"],
                        journal_dir=rel["journal_dir"],
                        desktop_dir=rel["desktop_dir"],
                        tmux_dir=rel["tmux_dir"],
                        dest=dest,
                        **kwargs,
                    )
                self.assertEqual(ctx.exception.name, UNSAFE_FILENAME)
                self.assertEqual(dest.ledger, [])
                self.assertEqual(dest.network_sentinel, [])

        # 3. Valid twin passes cleanly
        twin_flag = []
        def post_cap():
            twin_flag.append(True)
        dest._post_capture_hook = post_cap

        report = publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )
        self.assertTrue(report.latest_promoted)
        self.assertEqual(twin_flag, [True])
        self.assertEqual(dest.ledger[0].method, "put_if_absent")
        self.assertFalse(dest.ledger[0].key.endswith("/latest"))
        self.assertFalse(dest.ledger[0].key.endswith("/platform.json"))

    def test_publish_flow_and_idempotence(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        report = publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )
        self.assertEqual(report.version, "2.0.3")
        self.assertEqual(report.lane, "release")
        self.assertTrue(report.latest_promoted)

        # Check ledger order: first op is put_if_absent on native or bootstrap, not latest and not platform.json
        first_op = dest.ledger[0]
        self.assertEqual(first_op.method, "put_if_absent")
        self.assertFalse(first_op.key.endswith("/latest"))
        self.assertFalse(first_op.key.endswith("/platform.json"))
        bootstrap_put = next(
            entry
            for entry in dest.ledger
            if entry.method == "put_if_absent" and entry.key.endswith("-install.sh")
        )
        self.assertEqual(bootstrap_put.content_type, "application/octet-stream")

        # Verify pointer in destination
        latest_get = dest.get("solstone/release/latest")
        self.assertTrue(latest_get.is_ok())
        self.assertEqual(latest_get.body, b"2.0.3\n")

        # Republish identical release -> must succeed idempotently without latest CAS promotion
        dest.ledger.clear()
        report2 = publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )
        self.assertEqual(report2.version, "2.0.3")
        self.assertFalse(report2.latest_promoted)
        # Verify no compare_and_swap in ledger
        cas_ops = [op for op in dest.ledger if op.method == "compare_and_swap"]
        self.assertEqual(len(cas_ops), 0)

    def test_noncanonical_signed_platform_manifest_refuses(self):
        rel = self._setup_fixture_release("2.0.3")
        noncanonical_bytes = rel["manifest_bytes"] + b"\n"
        rel["manifest_path"].write_bytes(noncanonical_bytes)
        subprocess.run(
            [
                "minisign",
                "-S",
                "-W",
                "-s",
                str(rel["sec_path"]),
                "-m",
                str(rel["manifest_path"]),
                "-x",
                str(rel["signature_path"]),
                "-t",
                "noncanonical platform fixture",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=rel["dest"],
            )
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)
        self.assertEqual(rel["dest"].ledger, [])

    def test_latest_cas_success_requires_authoritative_readback(self):
        rel = self._setup_fixture_release("2.0.3")
        latest_key = "solstone/release/latest"

        def corrupt_latest_after_cas():
            rel["dest"].objects[latest_key] = StoredObject(
                body=b"9.9.9\n",
                etag='"corrupt"',
                content_type="text/plain; charset=utf-8",
                cache_control="no-store, max-age=0",
            )

        rel["dest"].cas_post_hook = corrupt_latest_after_cas
        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=rel["dest"],
            )
        self.assertEqual(ctx.exception.name, SAME_VERSION_DIFFERENT_BYTES)

    def test_equal_latest_requires_exact_metadata(self):
        rel = self._setup_fixture_release("2.0.3")
        publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=rel["dest"],
        )
        latest_key = "solstone/release/latest"
        current = rel["dest"].objects[latest_key]
        rel["dest"].objects[latest_key] = StoredObject(
            body=current.body,
            etag=current.etag,
            content_type=current.content_type,
            cache_control="public, max-age=60",
        )

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=rel["dest"],
            )
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_identical_retry_fills_missing_signature(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        # Perform initial publish
        report1 = publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )
        self.assertTrue(report1.latest_promoted)

        # Simulate missing signature in dest
        sig_key = "solstone/release/2.0.3/platform.json.minisig"
        del dest.objects[sig_key]

        dest.ledger.clear()
        report2 = publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )
        # Idempotent retry fills missing signature, does not promote latest again
        self.assertFalse(report2.latest_promoted)
        self.assertTrue(dest.get(sig_key).is_ok())
        self.assertEqual(dest.get(sig_key).body, rel["sig_bytes"])
        cas_ops = [op for op in dest.ledger if op.method == "compare_and_swap"]
        self.assertEqual(len(cas_ops), 0)

    def test_n_vs_n_plus_1_race_retry_success(self):
        """Simulate concurrent publication of identical 2.0.3 where concurrent worker finished CAS first."""
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        # When CAS is about to execute for 2.0.3, simulate concurrent worker already setting latest to 2.0.3 with new etag
        def concurrent_cas():
            dest.objects["solstone/release/latest"] = StoredObject(
                body=b"2.0.3\n",
                etag='"concurrent_worker_etag"',
                content_type="text/plain; charset=utf-8",
                cache_control="no-store, max-age=0",
            )

        dest.cas_pre_hook = concurrent_cas

        report = publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )
        # Succeeded idempotently because re-read SemVer matched
        self.assertFalse(report.latest_promoted)
        self.assertEqual(dest.get("solstone/release/latest").body, b"2.0.3\n")

    def test_selective_claiming_ignores_unrelated_files(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        # Place unrelated files in source trees
        (rel["journal_dir"] / "extra_journal.txt").write_text("ignore me", encoding="utf-8")
        (rel["desktop_dir"] / "extra_desktop.dat").write_bytes(b"ignore desktop")
        (rel["tmux_dir"] / "unrelated_tmux.bin").write_bytes(b"ignore tmux")

        publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )

        # Assert no extra file keys were put into destination
        for key in dest.objects:
            self.assertNotIn("extra_journal.txt", key)
            self.assertNotIn("extra_desktop.dat", key)
            self.assertNotIn("unrelated_tmux.bin", key)

    def test_publish_refuses_manifest_declaring_other_key_before_destination_use(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        # Modify manifest to declare unknown platform_key_id
        m = parse_json_strict(rel["manifest_path"].read_bytes())
        m["platform_key_id"] = "AAAAAAAAAAAAAAAA"
        rel["manifest_path"].write_bytes(canonical_json_bytes(m))

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)
        self.assertEqual(dest.ledger, [])
        self.assertEqual(dest.network_sentinel, [])

    def test_publish_same_version_different_bytes_rejected(self):
        rel1 = self._setup_fixture_release("2.0.3")
        dest = rel1["dest"]

        publish_release(
            manifest_path=rel1["manifest_path"],
            signature_path=rel1["signature_path"],
            journal_dir=rel1["journal_dir"],
            desktop_dir=rel1["desktop_dir"],
            tmux_dir=rel1["tmux_dir"],
            dest=dest,
        )

        # Setup second release with different artifact bytes
        rel2 = self._setup_fixture_release("2.0.3")
        # Reuse same dest
        rel2["dest"] = dest
        dest.pinset = rel2["pinset"]

        # Alter an artifact file in rel2
        for p in rel2["desktop_dir"].glob("*.deb"):
            p.write_bytes(b"different deb bytes than rel1")

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel2["manifest_path"],
                signature_path=rel2["signature_path"],
                journal_dir=rel2["journal_dir"],
                desktop_dir=rel2["desktop_dir"],
                tmux_dir=rel2["tmux_dir"],
                dest=dest,
            )
        self.assertIn(ctx.exception.name, (RELEASE_COHERENCE, SAME_VERSION_DIFFERENT_BYTES))

    def test_split_pair_conflict(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        manifest_key = "solstone/release/2.0.3/platform.json"
        dest.objects[manifest_key] = StoredObject(
            body=b'{"conflicting": true}',
            etag='"conflicttag"',
            content_type="application/json",
            cache_control="public, max-age=31536000, immutable",
        )

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, SAME_VERSION_DIFFERENT_BYTES)

    def test_stale_list_does_not_authorize_overwrite(self):
        """Publisher must never call list_keys."""
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )

        methods_used = [entry.method for entry in dest.ledger]
        self.assertNotIn("list_keys", methods_used)
        self.assertNotIn("LIST", methods_used)

    def test_fault_injection_put_corrupted_bytes(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        dest.fail_next_put_wrong_bytes = b"corrupted bytes on store"

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertIn(ctx.exception.name, (RELEASE_COHERENCE, SAME_VERSION_DIFFERENT_BYTES))

    def test_fault_injection_put_ok_no_object(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        dest.fail_next_put_ok_no_object = True

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, PUBLISH_INDETERMINATE)

    def test_metadata_mismatch_refusal(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        dest.fail_next_put_wrong_content_type = "text/wrong"

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_rollback_refused(self):
        rel_new = self._setup_fixture_release("2.0.3")
        dest = rel_new["dest"]

        publish_release(
            manifest_path=rel_new["manifest_path"],
            signature_path=rel_new["signature_path"],
            journal_dir=rel_new["journal_dir"],
            desktop_dir=rel_new["desktop_dir"],
            tmux_dir=rel_new["tmux_dir"],
            dest=dest,
        )

        rel_old = self._setup_fixture_release("2.0.2")
        dest.pinset = rel_old["pinset"]

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel_old["manifest_path"],
                signature_path=rel_old["signature_path"],
                journal_dir=rel_old["journal_dir"],
                desktop_dir=rel_old["desktop_dir"],
                tmux_dir=rel_old["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, ROLLBACK_REFUSED)

    def test_n_vs_n_plus_2_race_no_demote(self):
        """Simulate a race where concurrent worker publishes 2.0.4 right before 2.0.3 CAS."""
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        def install_newer_version():
            dest.objects["solstone/release/latest"] = StoredObject(
                body=b"2.0.4\n",
                etag='"etag204"',
                content_type="text/plain; charset=utf-8",
                cache_control="no-store, max-age=0",
            )

        dest.cas_pre_hook = install_newer_version

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, ROLLBACK_REFUSED)
        self.assertEqual(dest.get("solstone/release/latest").body, b"2.0.4\n")

    def test_malformed_latest_pointer_refuses(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        dest.objects["solstone/release/latest"] = StoredObject(
            body=b"not-a-semver\n",
            etag='"badetag"',
            content_type="text/plain; charset=utf-8",
            cache_control="no-store, max-age=0",
        )

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_post_capture_hook_live_mutation_succeeds_from_snapshot(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        orig_manifest_bytes = rel["manifest_bytes"]

        # Hook to tamper all 6 classes on live disk after capture
        def live_tamper_hook():
            # In hook, assert witness semantic event count is 0
            self.assertEqual(get_witness().total_semantic_count(), 0)

            rel["manifest_path"].write_bytes(b"corrupted live manifest")
            rel["signature_path"].write_bytes(b"corrupted live signature")
            for p in rel["journal_dir"].rglob("*"):
                if p.is_file():
                    p.write_bytes(b"corrupted live journal")
            for p in rel["desktop_dir"].rglob("*"):
                if p.is_file():
                    p.write_bytes(b"corrupted live desktop")
            for p in rel["tmux_dir"].rglob("*"):
                if p.is_file():
                    p.write_bytes(b"corrupted live tmux")

        dest._post_capture_hook = live_tamper_hook

        get_witness().reset()

        report = publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )
        self.assertEqual(report.version, "2.0.3")
        self.assertTrue(report.latest_promoted)

        # Confirm destination holds original snapshot bytes, not tampered bytes
        stored_manifest = dest.get("solstone/release/2.0.3/platform.json")
        self.assertEqual(stored_manifest.body, orig_manifest_bytes)

    def test_native_crypto_tampering_journal_refuses(self):
        # 1. Tamper journal manifest, refresh platform.json outer digest, re-sign platform.json
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        for p in rel["journal_dir"].rglob("*.manifest.json"):
            tampered_bytes = b'{"tampered": true, "version": "2.0.3"}'
            p.write_bytes(tampered_bytes)
            # Update outer platform.json digest
            import hashlib
            m = parse_json_strict(rel["manifest_path"].read_bytes())
            for arch_data in m["components"]["journal"]["arches"].values():
                for variant_data in arch_data.values():
                    variant_data["authority"]["manifest_sha256"] = hashlib.sha256(tampered_bytes).hexdigest()
            m_bytes = canonical_json_bytes(m)
            rel["manifest_path"].write_bytes(m_bytes)
            # Re-sign platform.json
            s_bytes = sign_manifest(
                manifest_bytes=m_bytes,
                secret_key_path=rel["sec_path"],
                selected_pin=rel["platform_pin"],
                passphrase_callback=lambda: "",
                repo_root=self.repo_root,
            )
            rel["signature_path"].write_bytes(s_bytes)

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)
        self.assertEqual(dest.ledger, [])
        self.assertEqual(dest.network_sentinel, [])

        # 2. Tamper only the journal minisig
        rel_sig = self._setup_fixture_release("2.0.3")
        dest_sig = rel_sig["dest"]
        for p in rel_sig["journal_dir"].rglob("*.manifest.json.minisig"):
            p.write_bytes(b"untrusted comment: tampered\nRWRhbXBlcmVk\n")

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel_sig["manifest_path"],
                signature_path=rel_sig["signature_path"],
                journal_dir=rel_sig["journal_dir"],
                desktop_dir=rel_sig["desktop_dir"],
                tmux_dir=rel_sig["tmux_dir"],
                dest=dest_sig,
            )
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)
        self.assertEqual(dest_sig.ledger, [])

    def test_native_crypto_tampering_desktop_refuses(self):
        # 1. Tamper desktop manifest, refresh platform.json outer digest, re-sign platform.json
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        for p in rel["desktop_dir"].glob("*.rust-release-manifest.json"):
            tampered_bytes = b'{"tampered": true, "version": "2.0.3"}'
            p.write_bytes(tampered_bytes)
            import hashlib
            m = parse_json_strict(rel["manifest_path"].read_bytes())
            for arch_data in m["components"]["desktop"]["arches"].values():
                for variant_data in arch_data.values():
                    variant_data["authority"]["manifest_sha256"] = hashlib.sha256(tampered_bytes).hexdigest()
            m_bytes = canonical_json_bytes(m)
            rel["manifest_path"].write_bytes(m_bytes)
            s_bytes = sign_manifest(
                manifest_bytes=m_bytes,
                secret_key_path=rel["sec_path"],
                selected_pin=rel["platform_pin"],
                passphrase_callback=lambda: "",
                repo_root=self.repo_root,
            )
            rel["signature_path"].write_bytes(s_bytes)

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)
        self.assertEqual(dest.ledger, [])
        self.assertEqual(dest.network_sentinel, [])

        # 2. Tamper only the desktop minisig
        rel_sig = self._setup_fixture_release("2.0.3")
        dest_sig = rel_sig["dest"]
        for p in rel_sig["desktop_dir"].glob("*.minisig"):
            p.write_bytes(b"untrusted comment: tampered\nRWRhbXBlcmVk\n")

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel_sig["manifest_path"],
                signature_path=rel_sig["signature_path"],
                journal_dir=rel_sig["journal_dir"],
                desktop_dir=rel_sig["desktop_dir"],
                tmux_dir=rel_sig["tmux_dir"],
                dest=dest_sig,
            )
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)
        self.assertEqual(dest_sig.ledger, [])

    def test_native_crypto_tampering_tmux_refuses(self):
        # 1. Tamper tmux SHA256SUMS, refresh platform.json outer digest, re-sign platform.json
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        for p in rel["tmux_dir"].glob("SHA256SUMS"):
            tampered_bytes = b"0" * 64 + b"  tampered-file.tar.gz\n"
            p.write_bytes(tampered_bytes)
            import hashlib
            m = parse_json_strict(rel["manifest_path"].read_bytes())
            for arch_data in m["components"]["tmux"]["arches"].values():
                for variant_data in arch_data.values():
                    variant_data["authority"]["sums_sha256"] = hashlib.sha256(tampered_bytes).hexdigest()
            m_bytes = canonical_json_bytes(m)
            rel["manifest_path"].write_bytes(m_bytes)
            s_bytes = sign_manifest(
                manifest_bytes=m_bytes,
                secret_key_path=rel["sec_path"],
                selected_pin=rel["platform_pin"],
                passphrase_callback=lambda: "",
                repo_root=self.repo_root,
            )
            rel["signature_path"].write_bytes(s_bytes)

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel["manifest_path"],
                signature_path=rel["signature_path"],
                journal_dir=rel["journal_dir"],
                desktop_dir=rel["desktop_dir"],
                tmux_dir=rel["tmux_dir"],
                dest=dest,
            )
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)
        self.assertEqual(dest.ledger, [])
        self.assertEqual(dest.network_sentinel, [])

        # 2. Tamper only tmux SHA256SUMS.minisig
        rel_sig = self._setup_fixture_release("2.0.3")
        dest_sig = rel_sig["dest"]
        for p in rel_sig["tmux_dir"].glob("SHA256SUMS.minisig"):
            p.write_bytes(b"untrusted comment: tampered\nRWRhbXBlcmVk\n")

        with self.assertRaises(Refusal) as ctx:
            publish_release(
                manifest_path=rel_sig["manifest_path"],
                signature_path=rel_sig["signature_path"],
                journal_dir=rel_sig["journal_dir"],
                desktop_dir=rel_sig["desktop_dir"],
                tmux_dir=rel_sig["tmux_dir"],
                dest=dest_sig,
            )
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)
        self.assertEqual(dest_sig.ledger, [])

    def test_bootstrap_revision_contract(self):
        # Revision 2 (floor): success
        boot_v2 = b"#!/bin/sh\nBOOTSTRAP_REVISION=2\nBOOTSTRAP_CONTRACT_VERSION=2\necho test\n"
        rel_v2 = self._setup_fixture_release("2.0.3", bootstrap_script=boot_v2)
        report_v2 = publish_release(
            manifest_path=rel_v2["manifest_path"],
            signature_path=rel_v2["signature_path"],
            journal_dir=rel_v2["journal_dir"],
            desktop_dir=rel_v2["desktop_dir"],
            tmux_dir=rel_v2["tmux_dir"],
            dest=rel_v2["dest"],
        )
        self.assertEqual(report_v2.version, "2.0.3")

        # Revision 3 (> floor): success
        boot_v3 = b"#!/bin/sh\nBOOTSTRAP_REVISION=3\nBOOTSTRAP_CONTRACT_VERSION=2\necho test\n"
        rel_v3 = self._setup_fixture_release("2.0.4", bootstrap_script=boot_v3)
        report_v3 = publish_release(
            manifest_path=rel_v3["manifest_path"],
            signature_path=rel_v3["signature_path"],
            journal_dir=rel_v3["journal_dir"],
            desktop_dir=rel_v3["desktop_dir"],
            tmux_dir=rel_v3["tmux_dir"],
            dest=rel_v3["dest"],
        )
        self.assertEqual(report_v3.version, "2.0.4")

        # Revision 1 (< floor): refusal
        boot_v1 = b"#!/bin/sh\nBOOTSTRAP_REVISION=1\nBOOTSTRAP_CONTRACT_VERSION=2\necho test\n"
        natives_dir = self.root / "natives-2.0.5"
        built_pinset, dirs = build_tiny_natives(natives_dir, bootstrap_script=boot_v1, min_bootstrap_revision=2)
        with self.assertRaises(Refusal) as ctx:
            generate_platform_manifest(
                version="2.0.5",
                lane="release",
                created_unix=1700000000,
                source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b",
                platform_key_id=self.platform_pin.key_id,
                repo_root=self.repo_root,
                journal_dir=dirs["journal"],
                desktop_dir=dirs["desktop"],
                tmux_dir=dirs["tmux"],
                journal_origin="https://updates.solstone.app",
                pins=built_pinset,
            )
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_destination_overwrite_denial_and_no_delete(self):
        rel = self._setup_fixture_release("2.0.3")
        dest = rel["dest"]

        # Verify no delete method on Destination classes and protocol
        self.assertIsNone(getattr(FixtureDestination, "delete", None))
        self.assertIsNone(getattr(R2Destination, "delete", None))
        self.assertIsNone(getattr(Destination, "delete", None))
        self.assertIsNone(getattr(dest, "delete", None))

        publish_release(
            manifest_path=rel["manifest_path"],
            signature_path=rel["signature_path"],
            journal_dir=rel["journal_dir"],
            desktop_dir=rel["desktop_dir"],
            tmux_dir=rel["tmux_dir"],
            dest=dest,
        )

        test_keys = [
            "solstone/release/2.0.3/solstone-journal-2.0.6-linux-x86_64.manifest.json",
            "solstone/release/2.0.3/solstone-journal-2.0.6-linux-x86_64.tar.gz",
            "solstone-journal/release/2.0.6/solstone-journal-2.0.6-install.sh",
            "solstone/release/2.0.3/platform.json",
            "solstone/release/2.0.3/platform.json.minisig",
        ]

        for key in test_keys:
            orig_body = dest.get(key).body
            self.assertIsNotNone(orig_body)

            # 1. Attempt put_if_absent on immutable key -> fails precondition
            put_res = dest.put_if_absent(key, b"new tampered bytes", "application/octet-stream", "public")
            self.assertTrue(put_res.is_precondition_failed())
            self.assertEqual(dest.get(key).body, orig_body)

            # 2. Attempt compare_and_swap on immutable key -> raises Refusal
            with self.assertRaises(Refusal):
                dest.compare_and_swap(key, b"new tampered bytes", "*", "application/octet-stream", "public")
            self.assertEqual(dest.get(key).body, orig_body)


if __name__ == "__main__":
    unittest.main()
