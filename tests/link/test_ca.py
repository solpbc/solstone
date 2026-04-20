# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CA generation, CSR signing, and attestation minting for the solstone fork."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from think.link.ca import (
    cert_fingerprint,
    generate_ca,
    load_ca,
    load_or_generate_ca,
    mint_attestation,
    sign_csr,
)


def test_generate_and_reload(tmp_path: Path) -> None:
    ca_dir = tmp_path / "ca"
    generated = generate_ca(ca_dir)
    reloaded = load_ca(ca_dir)

    assert generated.fingerprint_sha256() == reloaded.fingerprint_sha256()
    assert (ca_dir / "cert.pem").exists()
    assert (ca_dir / "private.pem").exists()
    assert (ca_dir / "private.pem").stat().st_mode & 0o777 == 0o600


def test_load_or_generate_is_idempotent(tmp_path: Path) -> None:
    ca_dir = tmp_path / "ca"

    first = load_or_generate_ca(ca_dir)
    second = load_or_generate_ca(ca_dir)

    assert first.fingerprint_sha256() == second.fingerprint_sha256()


def test_sign_csr_produces_valid_cert_chained_to_ca(tmp_path: Path) -> None:
    ca = generate_ca(tmp_path / "ca")
    key = ec.generate_private_key(ec.SECP256R1())
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, "Jer's iPhone")]),
        )
        .sign(key, hashes.SHA256())
    )
    csr_pem = csr.public_bytes(serialization.Encoding.PEM).decode("ascii")

    cert_pem, fp = sign_csr(ca, csr_pem, "Jer's iPhone")

    assert isinstance(cert_pem, str)
    assert isinstance(fp, str)
    assert fp.startswith("sha256:")
    assert len(fp) == 71

    cert = x509.load_pem_x509_certificate(cert_pem.encode("ascii"))
    assert cert.issuer == ca.cert.subject
    assert cert_fingerprint(cert_pem) == fp


def test_attestation_signed_by_ca_verifies(tmp_path: Path) -> None:
    ca = generate_ca(tmp_path / "ca")
    instance_id = "deadbeef-dead-beef-dead-beefdeadbeef"
    device_fp = "sha256:" + "aa" * 32

    jwt = mint_attestation(ca, instance_id, device_fp, now=1_745_006_400)
    segments = jwt.split(".")

    assert len(segments) == 3
    header_b64, payload_b64, sig_b64 = segments

    header = json.loads(_b64_decode(header_b64))
    payload = json.loads(_b64_decode(payload_b64))
    raw_sig = _b64_decode(sig_b64)

    assert header == {"alg": "ES256", "typ": "home-attest"}
    assert payload["iss"] == f"home:{instance_id}"
    assert payload["aud"] == "spl-relay"
    assert payload["scope"] == "device.enroll"
    assert payload["instance_id"] == instance_id
    assert payload["device_fp"] == device_fp
    assert payload["iat"] == 1_745_006_400
    assert payload["exp"] == 1_745_006_640
    assert payload["exp"] - payload["iat"] == 240
    assert isinstance(payload["jti"], str)
    assert payload["jti"]

    assert len(raw_sig) == 64
    r = int.from_bytes(raw_sig[:32], "big")
    s = int.from_bytes(raw_sig[32:], "big")
    der_sig = encode_dss_signature(r, s)

    public_key = serialization.load_pem_public_key(ca.pubkey_spki_pem.encode("ascii"))
    assert isinstance(public_key, ec.EllipticCurvePublicKey)
    public_key.verify(
        der_sig,
        f"{header_b64}.{payload_b64}".encode("ascii"),
        ec.ECDSA(hashes.SHA256()),
    )


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
