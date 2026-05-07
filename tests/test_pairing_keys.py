# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

from solstone.think.pairing import keys


def _openssh_public_key(public_key) -> str:
    return public_key.public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode("utf-8")


def test_validate_public_key_accepts_ssh_ed25519():
    key = ed25519.Ed25519PrivateKey.generate().public_key()
    encoded = _openssh_public_key(key)

    assert keys.validate_public_key(encoded) == encoded


def test_validate_public_key_rejects_non_ed25519_algorithms():
    rsa_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    ).public_key()
    ecdsa_key = ec.generate_private_key(ec.SECP256R1()).public_key()

    for candidate in (_openssh_public_key(rsa_key), _openssh_public_key(ecdsa_key)):
        try:
            keys.validate_public_key(candidate)
        except ValueError as exc:
            assert str(exc) == "public key must be ssh-ed25519"
        else:
            raise AssertionError("expected non-ed25519 key to be rejected")


def test_validate_public_key_rejects_malformed_and_oversized_values():
    for candidate, expected in (
        ("ssh-ed25519 AAAA-not-valid", "public key is invalid"),
        ("ssh-ed25519 " + ("A" * 2049), "public key is too long"),
    ):
        try:
            keys.validate_public_key(candidate)
        except ValueError as exc:
            assert str(exc) == expected
        else:
            raise AssertionError("expected invalid public key to be rejected")


def test_session_key_helpers():
    session_key = keys.generate_session_key()
    session_hash = keys.hash_session_key(session_key)

    assert session_key.startswith("dsk_")
    assert session_hash.startswith("sha256:")
    assert len(session_hash) == len("sha256:") + 64
    assert keys.mask_session_key(session_key) == (
        f"...{session_key[-4:]} (len={len(session_key)})"
    )
