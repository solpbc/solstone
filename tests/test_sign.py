# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import os
from pathlib import Path
import tempfile
import unittest

from solstone_platform.pins import MinisignPin
from solstone_platform.refusals import (
    FIXTURE_KEY_REFUSED,
    PIN_MISMATCH,
    PRODUCTION_UNAVAILABLE,
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

    def test_ephemeral_sign_and_verify(self):
        saved_dir = None
        with ephemeral_keypair("test key") as (sec_path, pub_path, pin):
            saved_dir = sec_path.parent
            self.assertTrue(saved_dir.exists())
            manifest_bytes = b'{"hello":"world"}'
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
            manifest_bytes = b'{"hello":"world"}'
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
            manifest_bytes = b'{"hello":"world"}'
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

    def test_production_sign_refuses_when_pin_absent_even_with_env(self):
        old_env = os.environ.get("SOLSTONE_PLATFORM_PRODUCTION")
        try:
            os.environ["SOLSTONE_PLATFORM_PRODUCTION"] = "ack"
            with ephemeral_keypair("test key") as (sec_path, pub_path, pin):
                manifest_bytes = b'{"hello":"world"}'
                with self.assertRaises(Refusal) as ctx:
                    sign_manifest(
                        manifest_bytes=manifest_bytes,
                        secret_key_path=sec_path,
                        selected_pin=pin,
                        is_production=True,
                        acknowledge_production=True,
                        repo_root=self.repo_root,
                    )
                self.assertIn(ctx.exception.name, (PRODUCTION_UNAVAILABLE, FIXTURE_KEY_REFUSED))
        finally:
            if old_env is None:
                os.environ.pop("SOLSTONE_PLATFORM_PRODUCTION", None)
            else:
                os.environ["SOLSTONE_PLATFORM_PRODUCTION"] = old_env


if __name__ == "__main__":
    unittest.main()
