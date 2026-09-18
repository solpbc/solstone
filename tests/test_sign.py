# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.pins import (
    PLATFORM_KEY_ID,
    PLATFORM_PUBKEY,
    MinisignPin,
    parse_minisign_pub,
    require_production_platform_pin,
)
from solstone_platform.refusals import (
    FIXTURE_KEY_REFUSED,
    PIN_MISMATCH,
    PRODUCTION_UNAVAILABLE,
    SCHEMA_INVALID,
    Refusal,
)
from solstone_platform.sign import (
    derive_public_key_from_secret,
    ephemeral_keypair,
    sign_manifest,
)


class TestSign(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent

    def test_parse_minisign_generated_public_key_comment(self):
        pin = parse_minisign_pub(
            "untrusted comment: minisign public key: 78041C9C21888A1E\n"
            "RWQeioghnBwEeKLzHwyJWiKKd8r4KYILul+Mc7ZeN/acOtGb4I4QGutz\n"
        )
        self.assertEqual(pin.key_id, "78041C9C21888A1E")

    def manifest_for_pin(self, pin: MinisignPin) -> bytes:
        manifest = parse_json_strict((self.repo_root / "examples" / "platform.json").read_bytes())
        manifest["platform_key_id"] = pin.key_id
        return canonical_json_bytes(manifest)

    def test_ephemeral_sign_and_verify(self):
        saved_dir = None
        with ephemeral_keypair("test key") as (sec_path, pub_path, pin):
            saved_dir = sec_path.parent
            self.assertTrue(saved_dir.exists())
            manifest_bytes = self.manifest_for_pin(pin)
            sig_bytes = sign_manifest(
                manifest_bytes=manifest_bytes,
                secret_key_path=sec_path,
                selected_pin=pin,
                passphrase_callback=lambda: "",
            )
            self.assertTrue(len(sig_bytes) > 0)
            self.assertIn(b"untrusted comment:", sig_bytes)

        # Confirm cleanup of 0700 tempdir
        self.assertFalse(saved_dir.exists())

    def test_derive_public_key_from_secret(self):
        with ephemeral_keypair("derive test") as (sec_path, pub_path, pin):
            derived = derive_public_key_from_secret(sec_path, passphrase="")
            self.assertEqual(derived.key_id, pin.key_id)
            self.assertEqual(derived.pubkey, pin.pubkey)

    def test_sign_with_mismatched_pin_fails(self):
        with ephemeral_keypair("key 1") as (sec_path1, pub_path1, pin1), \
             ephemeral_keypair("key 2") as (sec_path2, pub_path2, pin2):
            manifest_bytes = self.manifest_for_pin(pin1)
            with self.assertRaises(Refusal) as ctx:
                sign_manifest(
                    manifest_bytes=manifest_bytes,
                    secret_key_path=sec_path1,
                    selected_pin=pin2,
                    passphrase_callback=lambda: "",
                )
            self.assertEqual(ctx.exception.name, PIN_MISMATCH)

    def test_production_sign_requires_ack_and_env(self):
        with ephemeral_keypair("fake prod key") as (sec_path, pub_path, pin):
            manifest_bytes = self.manifest_for_pin(pin)
            # Attempting production signing without acknowledge_production
            with self.assertRaises(Refusal) as ctx:
                sign_manifest(
                    manifest_bytes=manifest_bytes,
                    secret_key_path=sec_path,
                    selected_pin=pin,
                    is_production=True,
                    acknowledge_production=False,
                    repo_root=self.repo_root,
                )
            self.assertEqual(ctx.exception.name, PRODUCTION_UNAVAILABLE)

    def test_production_pin_matches_vaulted_identity(self):
        pin = require_production_platform_pin(self.repo_root)
        disk_pin = parse_minisign_pub((self.repo_root / "pins" / "platform.pub").read_text(encoding="utf-8"))
        self.assertEqual(pin.key_id, disk_pin.key_id)
        self.assertEqual(pin.pubkey, disk_pin.pubkey)

    def test_production_sign_refuses_fixture_key_even_with_env(self):
        old_env = os.environ.get("SOLSTONE_PLATFORM_PRODUCTION")
        try:
            os.environ["SOLSTONE_PLATFORM_PRODUCTION"] = "ack"
            with ephemeral_keypair("test key") as (sec_path, pub_path, pin):
                manifest_bytes = self.manifest_for_pin(pin)
                with self.assertRaises(Refusal) as ctx:
                    sign_manifest(
                        manifest_bytes=manifest_bytes,
                        secret_key_path=sec_path,
                        selected_pin=pin,
                        is_production=True,
                        acknowledge_production=True,
                        repo_root=self.repo_root,
                    )
                self.assertIn(ctx.exception.name, (PIN_MISMATCH, FIXTURE_KEY_REFUSED))
        finally:
            if old_env is None:
                os.environ.pop("SOLSTONE_PLATFORM_PRODUCTION", None)
            else:
                os.environ["SOLSTONE_PLATFORM_PRODUCTION"] = old_env

    def test_sign_refuses_manifest_declaring_other_key(self):
        with ephemeral_keypair("selected key") as (sec_path, pub_path, pin):
            manifest = parse_json_strict((self.repo_root / "examples" / "platform.json").read_bytes())
            manifest["platform_key_id"] = "1111222233334444"
            with self.assertRaises(Refusal) as ctx:
                sign_manifest(
                    canonical_json_bytes(manifest),
                    sec_path,
                    pin,
                    passphrase_callback=lambda: "",
                )
            self.assertEqual(ctx.exception.name, PIN_MISMATCH)

    def test_sign_refuses_invalid_bytes_before_passphrase_or_key_derivation(self):
        passphrase_called = []
        with patch("solstone_platform.sign.derive_public_key_from_secret") as derive:
            with self.assertRaises(Refusal) as ctx:
                sign_manifest(
                    manifest_bytes=b'{"schema_version":1}\n',
                    secret_key_path=Path("/not-used"),
                    selected_pin=MinisignPin(key_id=PLATFORM_KEY_ID, pubkey=PLATFORM_PUBKEY),
                    passphrase_callback=lambda: passphrase_called.append(True) or "",
                )
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)
        self.assertEqual(passphrase_called, [])
        derive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
