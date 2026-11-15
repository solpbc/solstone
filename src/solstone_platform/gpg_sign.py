# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""OpenPGP signing and verification for APT and RPM repositories."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import tempfile
from typing import Generator, Optional

from solstone_platform.pins import OpenPGPPin
from solstone_platform.refusals import (
    FIXTURE_KEY_REFUSED,
    PIN_MISMATCH,
    PRODUCTION_UNAVAILABLE,
    RELEASE_COHERENCE,
    SCHEMA_INVALID,
    SIGNATURE_PIN_MISMATCH,
    Refusal,
)


def rpm_main_header_offset(data: bytes) -> int:
    """Return the exact byte offset of the RPM general header (following the signature header)."""
    if len(data) < 96:
        raise Refusal(SCHEMA_INVALID, "rpm file too short for lead")
    if data[:4] != b"\xed\xab\xee\xdb":
        raise Refusal(SCHEMA_INVALID, "invalid rpm lead magic")

    offset = 96
    if offset + 16 > len(data):
        raise Refusal(SCHEMA_INVALID, "rpm truncated at signature header")
    if data[offset : offset + 4] != b"\x8e\xad\xe8\x01":
        raise Refusal(SCHEMA_INVALID, "invalid rpm signature header magic")

    _, n_sig_index, sig_data_size = struct.unpack(">4sII", data[offset + 4 : offset + 16])
    offset += 16 + (n_sig_index * 16) + sig_data_size
    if offset % 8 != 0:
        offset += 8 - (offset % 8)

    if offset + 16 > len(data):
        raise Refusal(SCHEMA_INVALID, "rpm truncated at general header")
    if data[offset : offset + 4] != b"\x8e\xad\xe8\x01":
        raise Refusal(SCHEMA_INVALID, "invalid rpm general header magic")
    return offset


def _kill_gpg_agent(gnupghome: Path) -> None:
    try:
        subprocess.run(
            ["/usr/bin/gpgconf", "--homedir", str(gnupghome), "--kill", "all"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


@contextmanager
def isolated_gnupghome(
    secret_key_path: Path,
    passphrase: str,
    selected_pin: OpenPGPPin,
    *,
    is_production: bool = False,
    acknowledge_production: bool = False,
) -> Generator[Path, None, None]:
    """Create a 0700 temporary GNUPGHOME, import secret_key_path, and verify its fingerprint against selected_pin."""
    if is_production:
        if not acknowledge_production:
            raise Refusal(PRODUCTION_UNAVAILABLE, "--acknowledge-production required for production signing")
        if os.environ.get("SOLSTONE_PLATFORM_PRODUCTION") != "ack":
            raise Refusal(PRODUCTION_UNAVAILABLE, "SOLSTONE_PLATFORM_PRODUCTION=ack environment variable required")
        sec_text = secret_key_path.read_text(encoding="utf-8", errors="replace") if secret_key_path.is_file() else ""
        if "fixture" in str(secret_key_path).lower() or "test key" in str(secret_key_path).lower() or "fixture" in sec_text.lower():
            raise Refusal(FIXTURE_KEY_REFUSED, "fixture OpenPGP secret key rejected for production signing")

    if not secret_key_path.is_file():
        raise Refusal(PIN_MISMATCH, f"OpenPGP secret key file not found: {secret_key_path}")

    parent = Path(os.environ.get("SOLSTONE_RELEASE_TMPDIR", "/var/tmp"))
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="solstone-gpg-", dir=parent))
    os.chmod(tmp_dir, 0o700)

    def _sig_handler(signum, frame):
        _kill_gpg_agent(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        signal.default_int_handler(signum, frame)

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        proc = subprocess.run(
            [
                "/usr/bin/gpg",
                "--homedir",
                str(tmp_dir),
                "--batch",
                "--yes",
                "--pinentry-mode",
                "loopback",
                "--passphrase-fd",
                "0",
                "--import",
                str(secret_key_path),
            ],
            input=(passphrase + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(PIN_MISMATCH, f"failed to import OpenPGP secret key: {err}")

        fpr_proc = subprocess.run(
            [
                "/usr/bin/gpg",
                "--homedir",
                str(tmp_dir),
                "--batch",
                "--with-colons",
                "--list-secret-keys",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if fpr_proc.returncode != 0:
            raise Refusal(PIN_MISMATCH, "failed to list imported OpenPGP secret key")

        fingerprints: list[str] = []
        for line in fpr_proc.stdout.decode("utf-8", errors="replace").splitlines():
            parts = line.split(":")
            if len(parts) > 9 and parts[0] == "fpr":
                fingerprints.append(parts[9].strip().upper())

        if selected_pin.fingerprint.upper() not in fingerprints:
            raise Refusal(
                PIN_MISMATCH,
                f"imported OpenPGP secret key fingerprints {fingerprints} do not match selected pin {selected_pin.fingerprint}",
            )

        yield tmp_dir
    finally:
        try:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
        except Exception:
            pass
        _kill_gpg_agent(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def sign_rpm_file(
    *,
    unsigned_rpm: Path,
    signed_rpm: Path,
    gnupghome: Path,
    key_fingerprint: str,
    passphrase: str,
    created_unix: int,
) -> None:
    """Sign an RPM copy with rpmsign --addsign and verify main header + payload byte-identity."""
    unsigned_bytes = unsigned_rpm.read_bytes()
    unsigned_offset = rpm_main_header_offset(unsigned_bytes)

    signed_rpm.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(unsigned_rpm, signed_rpm)

    pass_file = gnupghome / ".rpm-passphrase"
    wrapper = gnupghome / "gpg-rpm-wrapper.sh"
    try:
        fd = os.open(pass_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o600)
        try:
            os.write(fd, (passphrase + "\n").encode("utf-8"))
        finally:
            os.close(fd)

        wrapper_script = (
            "#!/bin/sh\n"
            f'exec /usr/bin/gpg --batch --yes --pinentry-mode loopback --passphrase-file "{pass_file}" "$@"\n'
        )
        wrapper.write_text(wrapper_script, encoding="utf-8")
        os.chmod(wrapper, 0o700)

        proc = subprocess.run(
            [
                "/usr/bin/rpmsign",
                "--addsign",
                "--define",
                f"_gpg_name {key_fingerprint}",
                "--define",
                f"_gpg_path {gnupghome}",
                "--define",
                "_gpg_digest_algo sha512",
                "--define",
                f"__gpg {wrapper}",
                "--define",
                "_gpg_sign_cmd_extra_args --batch --yes",
                str(signed_rpm),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(SIGNATURE_PIN_MISMATCH, f"rpmsign failed for {signed_rpm.name}: {err}")
    finally:
        pass_file.unlink(missing_ok=True)
        wrapper.unlink(missing_ok=True)

    signed_bytes = signed_rpm.read_bytes()
    signed_offset = rpm_main_header_offset(signed_bytes)
    if signed_bytes[signed_offset:] != unsigned_bytes[unsigned_offset:]:
        raise Refusal(
            RELEASE_COHERENCE,
            f"rpmsign mutated general header or payload of {signed_rpm.name}",
        )


def verify_signed_rpm(signed_rpm: Path, public_key_asc: bytes) -> None:
    """Independently verify a signed RPM in an isolated rpmdb holding only public_key_asc."""
    with tempfile.TemporaryDirectory(prefix="solstone-rpmdb-") as tmp_str:
        tmp_dir = Path(tmp_str)
        dbpath = tmp_dir / "rpmdb"
        dbpath.mkdir(parents=True)
        pub_file = tmp_dir / "pubkey.asc"
        pub_file.write_bytes(public_key_asc)

        imp = subprocess.run(
            ["/usr/bin/rpmkeys", "--dbpath", str(dbpath), "--import", str(pub_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if imp.returncode != 0:
            err = imp.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(SIGNATURE_PIN_MISMATCH, f"rpmkeys --import failed: {err}")

        chk = subprocess.run(
            ["/usr/bin/rpmkeys", "--dbpath", str(dbpath), "--checksig", str(signed_rpm)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out = chk.stdout.decode("utf-8", errors="replace").strip()
        if chk.returncode != 0 or "digests signatures OK" not in out:
            err = chk.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(
                SIGNATURE_PIN_MISMATCH,
                f"rpmkeys --checksig failed for {signed_rpm.name}: stdout='{out}' stderr='{err}'",
            )


def sign_apt_release(
    *,
    release_path: Path,
    inrelease_path: Path,
    release_gpg_path: Path,
    gnupghome: Path,
    key_fingerprint: str,
    passphrase: str,
    created_unix: int,
) -> None:
    """Produce clearsigned InRelease and detached ASCII-armored Release.gpg from Release."""
    for mode_args, out_path in (
        (["--clearsign"], inrelease_path),
        (["--armor", "--detach-sign"], release_gpg_path),
    ):
        out_path.unlink(missing_ok=True)
        cmd = [
            "/usr/bin/gpg",
            "--homedir",
            str(gnupghome),
            "--batch",
            "--yes",
            "--pinentry-mode",
            "loopback",
            "--passphrase-fd",
            "0",
            "--digest-algo",
            "SHA512",
            "-u",
            key_fingerprint,
            *mode_args,
            "-o",
            str(out_path),
            str(release_path),
        ]
        proc = subprocess.run(
            cmd,
            input=(passphrase + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not out_path.is_file():
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(SIGNATURE_PIN_MISMATCH, f"gpg failed signing {out_path.name}: {err}")


def sign_detached_asc(
    *,
    data_path: Path,
    sig_asc_path: Path,
    gnupghome: Path,
    key_fingerprint: str,
    passphrase: str,
    created_unix: int,
) -> None:
    """Produce a detached ASCII-armored OpenPGP signature (e.g. repomd.xml.asc)."""
    sig_asc_path.unlink(missing_ok=True)
    proc = subprocess.run(
        [
            "/usr/bin/gpg",
            "--homedir",
            str(gnupghome),
            "--batch",
            "--yes",
            "--pinentry-mode",
            "loopback",
            "--passphrase-fd",
            "0",
            "--digest-algo",
            "SHA512",
            "-u",
            key_fingerprint,
            "--armor",
            "--detach-sign",
            "-o",
            str(sig_asc_path),
            str(data_path),
        ],
        input=(passphrase + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not sig_asc_path.is_file():
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise Refusal(SIGNATURE_PIN_MISMATCH, f"gpg failed signing {sig_asc_path.name}: {err}")


def verify_openpgp_signature(
    *,
    gpg_bytes: bytes,
    expected_fingerprint: str,
    sig_or_inline_path: Path,
    signed_data_path: Optional[Path] = None,
) -> None:
    """Verify an inline (InRelease) or detached (Release.gpg / repomd.xml.asc) signature using gpgv."""
    if not gpg_bytes or (gpg_bytes[0] & 0x80) == 0 or gpg_bytes[:4] == b"KBXf":
        raise Refusal(PIN_MISMATCH, "OpenPGP keyring must be RFC 4880 binary packet format, not KBXf")

    with tempfile.TemporaryDirectory(prefix="solstone-gpgv-") as tmp_str:
        keyring = Path(tmp_str) / "keyring.gpg"
        keyring.write_bytes(gpg_bytes)

        cmd = ["/usr/bin/gpgv", "--status-fd", "2", "--keyring", str(keyring), str(sig_or_inline_path)]
        if signed_data_path is not None:
            cmd.append(str(signed_data_path))

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0 or f"VALIDSIG {expected_fingerprint.upper()}" not in stderr_text.upper():
            raise Refusal(
                SIGNATURE_PIN_MISMATCH,
                f"gpgv verification failed for {sig_or_inline_path.name}: {stderr_text.strip()}",
            )


@contextmanager
def ephemeral_openpgp_keypair(
    passphrase: str = "fixture-openpgp-passphrase",
) -> Generator[tuple[Path, Path, OpenPGPPin], None, None]:
    """Generate an ephemeral RSA OpenPGP keypair in a 0700 tempdir for hermetic testing."""
    parent = Path(os.environ.get("SOLSTONE_RELEASE_TMPDIR", "/var/tmp"))
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="solstone-ephemeral-pgp-", dir=parent))
    os.chmod(tmp_dir, 0o700)
    try:
        param_file = tmp_dir / "params"
        param_file.write_text(
            "%echo Generating ephemeral fixture OpenPGP key\n"
            "Key-Type: RSA\n"
            "Key-Length: 3072\n"
            "Key-Usage: cert sign\n"
            "Name-Real: fixture test key\n"
            "Name-Email: fixture@example.invalid\n"
            "Expire-Date: 0\n"
            f"Passphrase: {passphrase}\n"
            "%commit\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "/usr/bin/gpg",
                "--homedir",
                str(tmp_dir),
                "--batch",
                "--pinentry-mode",
                "loopback",
                "--generate-key",
                str(param_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        colons = subprocess.run(
            ["/usr/bin/gpg", "--homedir", str(tmp_dir), "--batch", "--with-colons", "--list-keys"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.decode("utf-8")
        fpr = ""
        for line in colons.splitlines():
            parts = line.split(":")
            if len(parts) > 9 and parts[0] == "fpr":
                fpr = parts[9].strip().upper()
                break
        if not fpr:
            raise RuntimeError("failed to extract ephemeral OpenPGP fingerprint")

        sec_path = tmp_dir / "fixture.sec.asc"
        pass_path = tmp_dir / "fixture.pass"
        asc_path = tmp_dir / "fixture.asc"
        gpg_path = tmp_dir / "fixture.gpg"

        pass_path.write_text(passphrase + "\n", encoding="utf-8")
        os.chmod(pass_path, 0o600)

        subprocess.run(
            [
                "/usr/bin/gpg",
                "--homedir",
                str(tmp_dir),
                "--batch",
                "--yes",
                "--pinentry-mode",
                "loopback",
                "--passphrase",
                passphrase,
                "--armor",
                "--export-secret-keys",
                "-o",
                str(sec_path),
                fpr,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        os.chmod(sec_path, 0o600)

        subprocess.run(
            ["/usr/bin/gpg", "--homedir", str(tmp_dir), "--batch", "--yes", "--armor", "--export", "-o", str(asc_path), fpr],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        subprocess.run(
            ["/usr/bin/gpg", "--homedir", str(tmp_dir), "--batch", "--yes", "--export", "-o", str(gpg_path), fpr],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        pin = OpenPGPPin(
            fingerprint=fpr,
            key_id=fpr[-16:],
            asc_bytes=asc_path.read_bytes(),
            asc_minisig_bytes=b"untrusted comment: fixture minisig\nRWRmaXh0dXJl\n",
            gpg_bytes=gpg_path.read_bytes(),
        )
        yield sec_path, pass_path, pin
    finally:
        _kill_gpg_agent(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
