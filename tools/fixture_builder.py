# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hermetic builder for tiny native component test fixtures and platform examples."""

import hashlib
import io
from pathlib import Path
import re
import struct

import subprocess
import tarfile
import tempfile
from typing import Optional

from solstone_platform.canonical import canonical_json_bytes
from solstone_platform.pins import MinisignPin, PinSet, parse_minisign_pub
from solstone_platform.sign import ephemeral_keypair


def create_tiny_tar(dest: Path, files: dict[str, bytes]) -> bytes:
    """Create a minimal tar.gz file containing the specified path -> bytes map."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path, data in files.items():
            ti = tarfile.TarInfo(name=path)
            ti.size = len(data)
            ti.mode = 0o755
            tf.addfile(ti, io.BytesIO(data))
    content = buf.getvalue()
    dest.write_bytes(content)
    return content


def create_tiny_deb(dest: Path, pkg_name: str, version: str, arch: str, exe_name: str, exe_bytes: bytes) -> bytes:
    """Create a minimal .deb ar archive containing control.tar.gz and data.tar.gz."""
    # Control tar
    control_text = f"Package: {pkg_name}\nVersion: {version}\nArchitecture: {arch}\nMaintainer: Sol PBC\nDescription: Package for testing platform release pipeline\n"
    cbuf = io.BytesIO()
    with tarfile.open(fileobj=cbuf, mode="w:gz") as ctf:
        ti = tarfile.TarInfo(name="control")
        ti.size = len(control_text.encode("utf-8"))
        ctf.addfile(ti, io.BytesIO(control_text.encode("utf-8")))
    control_bytes = cbuf.getvalue()

    # Data tar
    dbuf = io.BytesIO()
    with tarfile.open(fileobj=dbuf, mode="w:gz") as dtf:
        ti = tarfile.TarInfo(name=f"usr/bin/{exe_name}")
        ti.size = len(exe_bytes)
        ti.mode = 0o755
        dtf.addfile(ti, io.BytesIO(exe_bytes))
    data_bytes = dbuf.getvalue()

    debian_binary = b"2.0\n"

    # Assemble ar
    def ar_entry(name: str, data: bytes) -> bytes:
        # 16s 12s 6s 6s 8s 10s 2s
        hdr = f"{name:<16}{'0':<12}{'0':<6}{'0':<6}{'100644':<8}{len(data):<10}`\n".encode("ascii")
        pad = b"\n" if len(data) % 2 != 0 else b""
        return hdr + data + pad

    ar_data = b"!<arch>\n" + ar_entry("debian-binary", debian_binary) + ar_entry("control.tar.gz", control_bytes) + ar_entry("data.tar.gz", data_bytes)
    dest.write_bytes(ar_data)
    return ar_data


def make_cpio_payload(files: list[tuple[str, bytes]]) -> bytes:
    """Create a minimal new-ascii cpio stream."""
    buf = io.BytesIO()
    for idx, (path, data) in enumerate(files, 1):
        name_bytes = path.encode("utf-8") + b"\x00"
        hdr = (
            f"070701"
            f"{idx:08x}"
            f"{0o100755:08x}"
            f"{0:08x}"
            f"{0:08x}"
            f"{1:08x}"
            f"{0:08x}"
            f"{len(data):08x}"
            f"{0:08x}"
            f"{0:08x}"
            f"{0:08x}"
            f"{0:08x}"
            f"{len(name_bytes):08x}"
            f"{0:08x}"
        ).encode("ascii")
        buf.write(hdr)
        buf.write(name_bytes)
        name_pad = (4 - ((len(hdr) + len(name_bytes)) % 4)) % 4
        buf.write(b"\x00" * name_pad)
        buf.write(data)
        data_pad = (4 - (len(data) % 4)) % 4
        buf.write(b"\x00" * data_pad)

    # Trailer
    trailer_name = b"TRAILER!!!\x00"
    hdr = (
        f"070701"
        f"{0:08x}"
        f"{0:08x}"
        f"{0:08x}"
        f"{0:08x}"
        f"{1:08x}"
        f"{0:08x}"
        f"{0:08x}"
        f"{0:08x}"
        f"{0:08x}"
        f"{0:08x}"
        f"{0:08x}"
        f"{len(trailer_name):08x}"
        f"{0:08x}"
    ).encode("ascii")
    buf.write(hdr)
    buf.write(trailer_name)
    trailer_pad = (4 - ((len(hdr) + len(trailer_name)) % 4)) % 4
    buf.write(b"\x00" * trailer_pad)
    return buf.getvalue()


def create_tiny_synthetic_rpm(dest: Path, pkg_name: str, version: str, arch: str, exe_name: str, exe_bytes: bytes) -> bytes:
    """Create a minimal synthetically valid RPM for testing."""
    # Lead (96 bytes)
    lead = b"\xed\xab\xee\xdb\x03\x00\x00\x00" + b"\x00" * 88

    # General header strings
    str_data = f"{pkg_name}\x00{version}\x001\x00{arch}\x00".encode("utf-8")
    o_name = 0
    o_ver = len(pkg_name) + 1
    o_rel = o_ver + len(version) + 1
    o_arch = o_rel + 2

    # Tags: NAME(1000), VERSION(1001), RELEASE(1002), ARCH(1022)
    entries = [
        struct.pack(">IIII", 1000, 6, o_name, 1),
        struct.pack(">IIII", 1001, 6, o_ver, 1),
        struct.pack(">IIII", 1002, 6, o_rel, 1),
        struct.pack(">IIII", 1022, 6, o_arch, 1),
    ]
    gen_index = b"".join(entries)
    gen_hdr_len = len(str_data)
    gen_hdr_prefix = b"\x8e\xad\xe8\x01\x00\x00\x00\x00" + struct.pack(">II", len(entries), gen_hdr_len)
    gen_header = gen_hdr_prefix + gen_index + str_data

    # Signature header (empty dummy)
    sig_hdr_prefix = b"\x8e\xad\xe8\x01\x00\x00\x00\x00" + struct.pack(">II", 0, 0)
    sig_header = sig_hdr_prefix
    if len(sig_header) % 8 != 0:
        sig_header += b"\x00" * (8 - (len(sig_header) % 8))

    cpio_payload = make_cpio_payload([(f"usr/bin/{exe_name}", exe_bytes)])
    content = lead + sig_header + gen_header + cpio_payload
    dest.write_bytes(content)
    return content


def build_tiny_natives(
    target_dir: Path,
    keypair_comment: str = "fixture native test key",
    bootstrap_script: bytes | None = None,
    min_bootstrap_revision: int = 2,
    native_version: str = "2.0.3",
    desktop_version: str | None = None,
    tmux_version: str | None = None,
    desktop_runtime_version: str | None = None,
    native_build_marker: str = "",
) -> tuple[PinSet, dict[str, Path]]:
    """Build tiny, coherent synthetic natives for journal, desktop, and tmux."""
    dirs = {
        "journal": target_dir / "journal",
        "desktop": target_dir / "desktop",
        "tmux": target_dir / "tmux",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Generate ephemeral keypairs for each
    with ephemeral_keypair(keypair_comment) as (j_sec, j_pub, j_pin), \
         ephemeral_keypair(keypair_comment) as (d_sec, d_pub, d_pin), \
         ephemeral_keypair(keypair_comment) as (t_sec, t_pub, t_pin):

        pinset = PinSet(journal=j_pin, desktop=d_pin, tmux=t_pin)

        # 1. Desktop
        d_version = desktop_version or native_version
        t_version = tmux_version or native_version
        d_runtime_version = desktop_runtime_version or d_version
        marker_line = f"# {native_build_marker}\n" if native_build_marker else ""
        desktop_exe = (
            "#!/bin/sh\n"
            "case \"${1:-}\" in\n"
            "  install-service|uninstall-service)\n"
            f"    [ -z \"${{SOLSTONE_TEST_SERVICE_LOG:-}}\" ] || printf 'desktop {d_runtime_version} %s\\n' \"$1\" >> \"$SOLSTONE_TEST_SERVICE_LOG\"\n"
            f"    case \"${{SOLSTONE_TEST_SERVICE_FAIL:-}}\" in \"desktop:{d_runtime_version}:$1\") exit 9 ;; esac\n"
            "    exit 0 ;;\n"
            "esac\n"
            f"echo {d_runtime_version}\n{marker_line}"
        ).encode("utf-8")
        d_tar = create_tiny_tar(dirs["desktop"] / f"solstone-linux-{d_version}-linux-x86_64.tar.gz", {"usr/bin/solstone-linux": desktop_exe})
        d_deb = create_tiny_deb(dirs["desktop"] / f"solstone-linux_{d_version}-1_amd64.deb", "solstone-linux", d_version, "amd64", "solstone-linux", desktop_exe)
        d_rpm = create_tiny_synthetic_rpm(dirs["desktop"] / f"solstone-linux-{d_version}-1.x86_64.rpm", "solstone-linux", d_version, "x86_64", "solstone-linux", desktop_exe)


        d_manifest = {
            "schema_version": 1,
            "product": "solstone-linux",
            "version": d_version,
            "source_commit": "3a6bded610425b52a8ac875ae364d23f6e70ce7f",
            "source_dirty": False,
            "cargo_lock_sha256": "0" * 64,
            "rust": {"rustc_verbose": "1", "cargo_version": "1"},
            "target": {"kind": "compiled", "triple": "x86_64-unknown-linux-gnu", "profile": "release", "features": []},
            "native_tools": {},
            "dependency_policy": {},
            "active_exceptions": [],
            "artifacts": [
                {"path": f"solstone-linux-{d_version}-linux-x86_64.tar.gz", "sha256": hashlib.sha256(d_tar).hexdigest(), "bytes": len(d_tar)},
                {"path": f"solstone-linux_{d_version}-1_amd64.deb", "sha256": hashlib.sha256(d_deb).hexdigest(), "bytes": len(d_deb)},
                {"path": f"solstone-linux-{d_version}-1.x86_64.rpm", "sha256": hashlib.sha256(d_rpm).hexdigest(), "bytes": len(d_rpm)},
            ],
        }
        d_m_path = dirs["desktop"] / f"solstone-linux-{d_version}-linux-x86_64.rust-release-manifest.json"
        d_m_bytes = canonical_json_bytes(d_manifest)
        d_m_path.write_bytes(d_m_bytes)
        subprocess.run(["minisign", "-S", "-W", "-s", str(d_sec), "-m", str(d_m_path), "-x", str(d_m_path.with_suffix(".json.minisig")), "-t", "solstone-linux release manifest"], check=True)

        # 2. Tmux
        t_sums = []
        for arch, deb_arch, rpm_arch, musl_t in [("x86_64", "amd64", "x86_64", "x86_64-unknown-linux-musl"), ("aarch64", "arm64", "aarch64", "aarch64-unknown-linux-musl")]:
            tmux_exe = (
                "#!/bin/sh\n"
                "case \"${1:-}\" in\n"
                "  install-service|uninstall-service)\n"
                f"    [ -z \"${{SOLSTONE_TEST_SERVICE_LOG:-}}\" ] || printf 'tmux {t_version}-{arch} %s\\n' \"$1\" >> \"$SOLSTONE_TEST_SERVICE_LOG\"\n"
                f"    case \"${{SOLSTONE_TEST_SERVICE_FAIL:-}}\" in \"tmux:{t_version}-{arch}:$1\") exit 9 ;; esac\n"
                "    exit 0 ;;\n"
                "esac\n"
                f"echo {t_version}-{arch}\n{marker_line}"
            ).encode("utf-8")
            t_tar = create_tiny_tar(dirs["tmux"] / f"solstone-tmux-{t_version}-{arch}-linux.tar.gz", {"usr/bin/solstone-tmux": tmux_exe})
            t_deb = create_tiny_deb(dirs["tmux"] / f"solstone-tmux_{t_version}_{deb_arch}.deb", "solstone-tmux", t_version, deb_arch, "solstone-tmux", tmux_exe)
            t_rpm = create_tiny_synthetic_rpm(dirs["tmux"] / f"solstone-tmux-{t_version}-1.{rpm_arch}.rpm", "solstone-tmux", t_version, rpm_arch, "solstone-tmux", tmux_exe)

            tar_sha = hashlib.sha256(t_tar).hexdigest()
            deb_sha = hashlib.sha256(t_deb).hexdigest()
            rpm_sha = hashlib.sha256(t_rpm).hexdigest()

            t_target = {
                "schema_version": 1,
                "product_version": t_version,
                "source_commit": "9a0009469a76977f1bb0a0e0fca762271d34b517",
                "rust_target": musl_t,
                "rustc_vv": "rustc 1.97.1",
                "executable": {"name": "solstone-tmux", "sha256": hashlib.sha256(tmux_exe).hexdigest()},
                "artifacts": [
                    {"name": f"solstone-tmux-{t_version}-{arch}-linux.tar.gz", "sha256": tar_sha},
                    {"name": f"solstone-tmux_{t_version}_{deb_arch}.deb", "sha256": deb_sha},
                    {"name": f"solstone-tmux-{t_version}-1.{rpm_arch}.rpm", "sha256": rpm_sha},
                ],
            }
            target_path = dirs["tmux"] / f"solstone-tmux-{t_version}-{musl_t}.target.json"
            target_bytes = canonical_json_bytes(t_target)
            target_path.write_bytes(target_bytes)
            t_sums.append(f"{hashlib.sha256(target_bytes).hexdigest()}  {target_path.name}")
            t_sums.append(f"{tar_sha}  solstone-tmux-{t_version}-{arch}-linux.tar.gz")
            t_sums.append(f"{deb_sha}  solstone-tmux_{t_version}_{deb_arch}.deb")
            t_sums.append(f"{rpm_sha}  solstone-tmux-{t_version}-1.{rpm_arch}.rpm")

        sums_path = dirs["tmux"] / "SHA256SUMS"
        sums_path.write_text("\n".join(t_sums) + "\n", encoding="utf-8")
        subprocess.run(["minisign", "-S", "-W", "-s", str(t_sec), "-m", str(sums_path), "-x", str(dirs["tmux"] / "SHA256SUMS.minisig"), "-t", f"solstone-tmux {t_version} SHA256SUMS"], check=True)

        if bootstrap_script is not None:
            boot_script = bootstrap_script
        else:
            boot_script = (
                b"#!/bin/sh\n"
                b"BOOTSTRAP_REVISION=2\n"
                b"BOOTSTRAP_CONTRACT_VERSION=2\n"
                b"echo install\n"
            )
        min_boot_rev_val = min_bootstrap_revision
        boot_sha = hashlib.sha256(boot_script).hexdigest()
        bootstrap_name = "solstone-journal-2.0.6-install.sh"

        for arch, target in [("x86_64", "linux-x86_64"), ("aarch64", "linux-aarch64")]:
            arch_dir = dirs["journal"] / target
            arch_dir.mkdir(parents=True, exist_ok=True)
            (arch_dir / bootstrap_name).write_bytes(boot_script)
            j_tar = create_tiny_tar(arch_dir / f"solstone-journal-2.0.6-{target}.tar.gz", {"usr/bin/journal": b"#!/bin/sh\necho 2.0.6\n"})
            j_deb = create_tiny_deb(arch_dir / f"solstone-journal-2.0.6-{target}.deb", "solstone-journal", "2.0.6", "amd64" if arch == "x86_64" else "arm64", "journal", b"#!/bin/sh\necho 2.0.6\n")
            j_rpm = create_tiny_synthetic_rpm(arch_dir / f"solstone-journal-2.0.6-{target}.rpm", "solstone-journal", "2.0.6", "x86_64" if arch == "x86_64" else "aarch64", "journal", b"#!/bin/sh\necho 2.0.6\n")


            tar_sha = hashlib.sha256(j_tar).hexdigest()
            deb_sha = hashlib.sha256(j_deb).hexdigest()
            rpm_sha = hashlib.sha256(j_rpm).hexdigest()

            rel_text = (
                f"product=solstone-journal\nversion=2.0.6\ntarget={target}\ncommit=3075c36b12fad469d4c9c0ab4555908fe8ecca1b\n"
                f"lock_sha256=0000000000000000000000000000000000000000000000000000000000000000\n"
                f"upgrade_epoch=journal-v2\nretention_window=3\nmin_bootstrap_revision={min_boot_rev_val}\n"
                f"bootstrap_contract_version=2\nbootstrap_filename={bootstrap_name}\n"
                f"state_reader_min=2.0.0\nstate_reader_max=2.0.6\n"
            )

            (arch_dir / f"solstone-journal-2.0.6-{target}.release").write_text(rel_text, encoding="utf-8")

            sha256_lines = [
                f"{tar_sha}  solstone-journal-2.0.6-{target}.tar.gz",
                f"{deb_sha}  solstone-journal-2.0.6-{target}.deb",
                f"{rpm_sha}  solstone-journal-2.0.6-{target}.rpm",
                f"{hashlib.sha256(rel_text.encode('utf-8')).hexdigest()}  solstone-journal-2.0.6-{target}.release",
                f"{boot_sha}  {bootstrap_name}",
            ]
            sha256_bytes = ("\n".join(sha256_lines) + "\n").encode("utf-8")
            (arch_dir / f"solstone-journal-2.0.6-{target}.sha256").write_bytes(sha256_bytes)

            j_manifest = {
                "product": "solstone-journal",
                "version": "2.0.6",
                "target": target,
                "files": {
                    f"solstone-journal-2.0.6-{target}.release": hashlib.sha256(rel_text.encode("utf-8")).hexdigest(),
                    bootstrap_name: boot_sha,
                    f"solstone-journal-2.0.6-{target}.tar.gz": tar_sha,
                    f"solstone-journal-2.0.6-{target}.deb": deb_sha,
                    f"solstone-journal-2.0.6-{target}.rpm": rpm_sha,
                    f"solstone-journal-2.0.6-{target}.sha256": hashlib.sha256(sha256_bytes).hexdigest(),
                },
            }
            m_path = arch_dir / f"solstone-journal-2.0.6-{target}.manifest.json"
            m_bytes = canonical_json_bytes(j_manifest)
            m_path.write_bytes(m_bytes)
            subprocess.run(["minisign", "-S", "-W", "-s", str(j_sec), "-m", str(m_path), "-x", str(m_path.with_suffix(".json.minisig")), "-t", "solstone-journal release manifest"], check=True)

        return pinset, dirs
