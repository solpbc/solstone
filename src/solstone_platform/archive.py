# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Safe data-only archive scanning and inventory extraction for tar, deb, and rpm packages."""

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import struct
import subprocess
import tarfile
from typing import Any, Optional

from solstone_platform.canonical import canonical_json_bytes
from solstone_platform.refusals import (
    ARCHIVE_ABSOLUTE_PATH,
    ARCHIVE_DEVICE,
    ARCHIVE_DUPLICATE_MEMBER,
    ARCHIVE_FIFO,
    ARCHIVE_HARDLINK_ESCAPE,
    ARCHIVE_PACKAGE_SCRIPT,
    ARCHIVE_PACKAGE_TRIGGER,
    ARCHIVE_PARENT_TRAVERSAL,
    ARCHIVE_SYMLINK_ESCAPE,
    ARCHIVE_SYMLINK_THEN_CHILD,
    SCHEMA_INVALID,
    Refusal,
)


@dataclass(frozen=True)
class PackageIdentity:
    name: str
    version: str
    arch: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "arch": self.arch}


@dataclass(frozen=True)
class ArchiveScanResult:
    inventory: list[dict[str, Any]]
    payload_build_id: str
    package_identity: Optional[PackageIdentity]
    executable_sha256: dict[str, str]  # filename -> sha256


def normalize_member_path(raw_path: str) -> str:
    """Normalize member path and validate against escapes."""
    if raw_path.startswith("/"):
        raise Refusal(ARCHIVE_ABSOLUTE_PATH, f"archive entry has absolute path '{raw_path}'")

    parts = raw_path.replace("\\", "/").split("/")
    cleaned_parts = []
    for part in parts:
        if part == "..":
            raise Refusal(ARCHIVE_PARENT_TRAVERSAL, f"archive entry contains parent traversal '{raw_path}'")
        if part in ("", "."):
            continue
        cleaned_parts.append(part)

    if not cleaned_parts:
        return ""
    return "/".join(cleaned_parts)


def validate_symlink_target(dest_path: str, link_target: str) -> None:
    """Ensure symlink destination does not escape the unpack directory."""
    if link_target.startswith("/"):
        raise Refusal(ARCHIVE_SYMLINK_ESCAPE, f"symlink '{dest_path}' has absolute target '{link_target}'")

    target_parts = link_target.replace("\\", "/").split("/")
    # Resolve relative to dirname of dest_path
    dest_dir_parts = dest_path.split("/")[:-1]
    combined = list(dest_dir_parts)
    for part in target_parts:
        if part == "..":
            if not combined:
                raise Refusal(ARCHIVE_SYMLINK_ESCAPE, f"symlink '{dest_path}' escapes root via '{link_target}'")
            combined.pop()
        elif part in ("", "."):
            continue
        else:
            combined.append(part)


def validate_hardlink_target(dest_path: str, link_target: str) -> None:
    """Ensure hardlink destination does not escape the unpack directory."""
    if link_target.startswith("/"):
        raise Refusal(ARCHIVE_HARDLINK_ESCAPE, f"hardlink '{dest_path}' has absolute target '{link_target}'")

    target_parts = link_target.replace("\\", "/").split("/")
    dest_dir_parts = dest_path.split("/")[:-1]
    combined = list(dest_dir_parts)
    for part in target_parts:
        if part == "..":
            if not combined:
                raise Refusal(ARCHIVE_HARDLINK_ESCAPE, f"hardlink '{dest_path}' escapes root via '{link_target}'")
            combined.pop()
        elif part in ("", "."):
            continue
        else:
            combined.append(part)


def scan_tar_stream(tf: tarfile.TarFile) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Scan a TarFile object without extracting to disk."""
    inventory: list[dict[str, Any]] = []
    executable_digests: dict[str, str] = {}
    seen_paths: set[str] = set()
    symlink_dirs: set[str] = set()

    for member in tf.getmembers():
        raw_name = member.name
        norm_name = normalize_member_path(raw_name)
        if not norm_name:
            continue

        if norm_name in seen_paths:
            raise Refusal(ARCHIVE_DUPLICATE_MEMBER, f"duplicate member path '{norm_name}'")
        seen_paths.add(norm_name)

        # Check symlink-then-child
        for sym_dir in symlink_dirs:
            if norm_name.startswith(sym_dir + "/"):
                raise Refusal(ARCHIVE_SYMLINK_THEN_CHILD, f"member '{norm_name}' is inside symlink path '{sym_dir}'")

        if member.isblk() or member.ischr():
            raise Refusal(ARCHIVE_DEVICE, f"device node '{norm_name}' in archive")
        if member.isfifo():
            raise Refusal(ARCHIVE_FIFO, f"FIFO node '{norm_name}' in archive")

        if member.isdir():
            inventory.append({"path": norm_name, "kind": "dir", "size": 0})
        elif member.issym():
            target = member.linkname
            validate_symlink_target(norm_name, target)
            inventory.append({
                "path": norm_name,
                "kind": "symlink",
                "size": 0,
                "link_target": target,
            })
            symlink_dirs.add(norm_name)
        elif member.islnk():
            # Hardlink
            target = member.linkname
            validate_hardlink_target(norm_name, target)
            # In platform inventory, treat as file referencing original or duplicate path
            f = tf.extractfile(member)
            content = f.read() if f else b""
            digest = hashlib.sha256(content).hexdigest()
            inventory.append({
                "path": norm_name,
                "kind": "file",
                "size": len(content),
                "sha256": digest,
            })
            exe_name = norm_name.split("/")[-1]
            executable_digests[exe_name] = digest
        elif member.isreg():
            f = tf.extractfile(member)
            content = f.read() if f else b""
            digest = hashlib.sha256(content).hexdigest()
            inventory.append({
                "path": norm_name,
                "kind": "file",
                "size": member.size,
                "sha256": digest,
            })
            exe_name = norm_name.split("/")[-1]
            executable_digests[exe_name] = digest
        else:
            raise Refusal(ARCHIVE_DEVICE, f"unsupported member type for '{norm_name}'")

    # Sort inventory lexicographically by path
    inventory.sort(key=lambda item: item["path"])
    return inventory, executable_digests


def scan_tar_file(path: Path) -> ArchiveScanResult:
    """Scan a .tar.gz archive and produce canonical inventory."""
    try:
        with tarfile.open(path, "r:*") as tf:
            inventory, exe_digests = scan_tar_stream(tf)
    except tarfile.TarError as err:
        raise Refusal(SCHEMA_INVALID, f"failed to read tar archive: {err}") from err

    payload_id = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    return ArchiveScanResult(
        inventory=inventory,
        payload_build_id=payload_id,
        package_identity=None,
        executable_sha256=exe_digests,
    )


def _parse_ar_archive(data: bytes) -> dict[str, bytes]:
    """Parse a standard Unix ar archive."""
    if not data.startswith(b"!<arch>\n"):
        raise Refusal(SCHEMA_INVALID, "not a valid debian ar archive (missing !<arch> magic)")

    members: dict[str, bytes] = {}
    offset = 8
    length = len(data)

    while offset < length:
        if offset + 60 > length:
            break
        header = data[offset : offset + 60]
        offset += 60

        name = header[0:16].decode("ascii", errors="replace").strip()
        size_str = header[48:58].decode("ascii", errors="replace").strip()
        try:
            size = int(size_str)
        except ValueError as err:
            raise Refusal(SCHEMA_INVALID, f"invalid ar header size: {size_str}") from err

        member_data = data[offset : offset + size]
        offset += size
        if size % 2 != 0:
            offset += 1  # 2-byte alignment

        # Handle slash termination in ar member names (e.g. "debian-binary/")
        clean_name = name.rstrip("/")
        members[clean_name] = member_data

    return members


def scan_deb_file(path: Path) -> ArchiveScanResult:
    """Scan a .deb package, inspect control scripts, and inventory data.tar without executing."""
    raw_data = path.read_bytes()
    members = _parse_ar_archive(raw_data)

    if "debian-binary" not in members:
        raise Refusal(SCHEMA_INVALID, "deb package missing debian-binary")

    # Locate control.tar.*
    control_key = next((k for k in members if k.startswith("control.tar")), None)
    if not control_key:
        raise Refusal(SCHEMA_INVALID, "deb package missing control.tar")

    control_bytes = members[control_key]
    pkg_name, pkg_version, pkg_arch = None, None, None

    with tarfile.open(fileobj=io.BytesIO(control_bytes), mode="r:*") as ctf:
        for cmember in ctf.getmembers():
            cname = cmember.name.replace("\\", "/").split("/")[-1]
            if cname in {"preinst", "postinst", "prerm", "postrm", "config"}:
                raise Refusal(ARCHIVE_PACKAGE_SCRIPT, f"deb package contains maintainer script '{cname}'")
            if cname in {"triggers"}:
                raise Refusal(ARCHIVE_PACKAGE_TRIGGER, f"deb package contains trigger '{cname}'")
            if cname == "control" and cmember.isreg():
                cf = ctf.extractfile(cmember)
                if cf:
                    control_text = cf.read().decode("utf-8", errors="replace")
                    for line in control_text.splitlines():
                        if line.startswith("Package:"):
                            pkg_name = line.split(":", 1)[1].strip()
                        elif line.startswith("Version:"):
                            pkg_version = line.split(":", 1)[1].strip()
                        elif line.startswith("Architecture:"):
                            pkg_arch = line.split(":", 1)[1].strip()

    if not pkg_name or not pkg_version or not pkg_arch:
        raise Refusal(SCHEMA_INVALID, "deb package control metadata incomplete (missing Package, Version, or Architecture)")

    # Locate data.tar.*
    data_key = next((k for k in members if k.startswith("data.tar")), None)
    if not data_key:
        raise Refusal(SCHEMA_INVALID, "deb package missing data.tar")

    data_bytes = members[data_key]
    with tarfile.open(fileobj=io.BytesIO(data_bytes), mode="r:*") as dtf:
        inventory, exe_digests = scan_tar_stream(dtf)

    payload_id = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    return ArchiveScanResult(
        inventory=inventory,
        payload_build_id=payload_id,
        package_identity=PackageIdentity(name=pkg_name, version=pkg_version, arch=pkg_arch),
        executable_sha256=exe_digests,
    )


def _parse_rpm_headers(data: bytes) -> tuple[dict[int, Any], int]:
    """Parse lead and header structures from RPM bytes, checking for scriptlet/trigger tags."""
    if len(data) < 96:
        raise Refusal(SCHEMA_INVALID, "rpm file too short for lead")

    lead_magic = data[:4]
    if lead_magic != b"\xed\xab\xee\xdb":
        raise Refusal(SCHEMA_INVALID, "invalid rpm lead magic")

    offset = 96  # Skip lead
    # Signature header
    if offset + 16 > len(data):
        raise Refusal(SCHEMA_INVALID, "rpm truncated at signature header")

    sig_magic = data[offset : offset + 4]
    if sig_magic != b"\x8e\xad\xe8\x01":
        raise Refusal(SCHEMA_INVALID, "invalid rpm signature header magic")

    _, n_sig_index, sig_data_size = struct.unpack(">4sII", data[offset + 4 : offset + 16])
    offset += 16
    sig_index_size = n_sig_index * 16
    offset += sig_index_size + sig_data_size
    # Signature header data is padded to 8-byte boundary
    if offset % 8 != 0:
        offset += 8 - (offset % 8)

    # General header
    if offset + 16 > len(data):
        raise Refusal(SCHEMA_INVALID, "rpm truncated at general header")

    gen_magic = data[offset : offset + 4]
    if gen_magic != b"\x8e\xad\xe8\x01":
        raise Refusal(SCHEMA_INVALID, "invalid rpm general header magic")

    _, n_gen_index, gen_data_size = struct.unpack(">4sII", data[offset + 4 : offset + 16])
    offset += 16
    gen_index_bytes = data[offset : offset + n_gen_index * 16]
    offset += n_gen_index * 16
    gen_data_store = data[offset : offset + gen_data_size]
    offset += gen_data_size

    # Parse index tags
    tags: dict[int, Any] = {}
    for i in range(n_gen_index):
        tag, tag_type, tag_offset, tag_count = struct.unpack(">IIII", gen_index_bytes[i * 16 : (i + 1) * 16])
        if tag_type == 6:  # String
            end = gen_data_store.find(b"\x00", tag_offset)
            tags[tag] = gen_data_store[tag_offset:end].decode("utf-8", errors="replace") if end != -1 else ""
        elif tag_type in (8, 9):  # String array / i18n string
            # Collect strings
            str_list = []
            curr = tag_offset
            for _ in range(tag_count):
                end = gen_data_store.find(b"\x00", curr)
                if end == -1:
                    break
                str_list.append(gen_data_store[curr:end].decode("utf-8", errors="replace"))
                curr = end + 1
            tags[tag] = str_list
        else:
            tags[tag] = tag_offset

    return tags, offset


def _parse_cpio_stream(stream: io.BytesIO) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Parse a new-ASCII cpio payload stream."""
    inventory: list[dict[str, Any]] = []
    executable_digests: dict[str, str] = {}
    seen_paths: set[str] = set()
    symlink_dirs: set[str] = set()

    while True:
        header = stream.read(110)
        if len(header) < 110:
            break
        magic = header[:6]
        if magic not in (b"070701", b"070702"):
            break

        mode = int(header[14:22], 16)
        filesize = int(header[54:62], 16)
        namesize = int(header[94:102], 16)

        name_bytes = stream.read(namesize)
        # 4-byte alignment for name
        name_pad = (4 - ((110 + namesize) % 4)) % 4
        if name_pad:
            stream.read(name_pad)

        filename = name_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
        if filename == "TRAILER!!!":
            break

        norm_name = normalize_member_path(filename)
        if not norm_name:
            # Skip root directory entries
            data = stream.read(filesize)
            data_pad = (4 - (filesize % 4)) % 4
            if data_pad:
                stream.read(data_pad)
            continue

        if norm_name in seen_paths:
            raise Refusal(ARCHIVE_DUPLICATE_MEMBER, f"duplicate member path '{norm_name}' in cpio payload")
        seen_paths.add(norm_name)

        for sym_dir in symlink_dirs:
            if norm_name.startswith(sym_dir + "/"):
                raise Refusal(ARCHIVE_SYMLINK_THEN_CHILD, f"member '{norm_name}' is inside symlink path '{sym_dir}'")

        file_type = mode & 0o170000
        if file_type in (0o020000, 0o060000):  # chr, blk
            raise Refusal(ARCHIVE_DEVICE, f"device node '{norm_name}' in cpio payload")
        if file_type in (0o010000, 0o140000):  # fifo, socket
            raise Refusal(ARCHIVE_FIFO, f"FIFO/socket node '{norm_name}' in cpio payload")

        content = stream.read(filesize)
        data_pad = (4 - (filesize % 4)) % 4
        if data_pad:
            stream.read(data_pad)

        if file_type == 0o040000:  # Directory
            inventory.append({"path": norm_name, "kind": "dir", "size": 0})
        elif file_type == 0o120000:  # Symlink
            link_target = content.decode("utf-8", errors="replace")
            validate_symlink_target(norm_name, link_target)
            inventory.append({
                "path": norm_name,
                "kind": "symlink",
                "size": 0,
                "link_target": link_target,
            })
            symlink_dirs.add(norm_name)
        elif file_type == 0o100000:  # Regular file
            digest = hashlib.sha256(content).hexdigest()
            inventory.append({
                "path": norm_name,
                "kind": "file",
                "size": filesize,
                "sha256": digest,
            })
            exe_name = norm_name.split("/")[-1]
            executable_digests[exe_name] = digest

    inventory.sort(key=lambda item: item["path"])
    return inventory, executable_digests


def scan_rpm_file(path: Path) -> ArchiveScanResult:
    """Scan an RPM file, verify absence of scriptlets/triggers, and unpack payload via rpm2cpio stream."""
    data = path.read_bytes()
    tags, end_offset = _parse_rpm_headers(data)

    # Check scriptlet tags
    # RPMTAG_PREIN = 1023, POSTIN = 1024, PREUN = 1025, POSTUN = 1026, PRETRANS = 1151, POSTTRANS = 1152
    script_tags = [1023, 1024, 1025, 1026, 1151, 1152]
    for st in script_tags:
        if st in tags:
            val = tags[st]
            if isinstance(val, str) and val.strip():
                raise Refusal(ARCHIVE_PACKAGE_SCRIPT, f"rpm contains maintainer script tag {st}")

    # Check trigger tags: RPMTAG_TRIGGERSCRIPTS = 1065, TRIGGERNAME = 1066, TRIGGERSCRIPTPROG = 1092
    trigger_tags = [1065, 1066, 1092]
    for tt in trigger_tags:
        if tt in tags:
            val = tags[tt]
            if val:
                raise Refusal(ARCHIVE_PACKAGE_TRIGGER, f"rpm contains package trigger tag {tt}")

    pkg_name = str(tags.get(1000, ""))
    pkg_ver = str(tags.get(1001, ""))
    pkg_rel = str(tags.get(1002, ""))
    pkg_arch = str(tags.get(1022, ""))

    full_version = f"{pkg_ver}-{pkg_rel}" if pkg_rel else pkg_ver
    if not pkg_name or not pkg_ver or not pkg_arch:
        raise Refusal(SCHEMA_INVALID, "rpm package header metadata incomplete (missing Name, Version, or Arch)")

    # Stream payload via rpm2cpio
    cpio_bytes = b""
    try:
        proc = subprocess.run(
            ["/usr/bin/rpm2cpio", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cpio_bytes = proc.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    if not cpio_bytes or (proc.returncode != 0 and b"TRAILER!!!" not in cpio_bytes):
        # Fall back to uncompressed cpio stream in raw rpm bytes if present
        if end_offset < len(data) and b"070701" in data[end_offset:]:
            idx = data.find(b"070701", end_offset)
            cpio_bytes = data[idx:]
        else:
            err_msg = proc.stderr.decode('utf-8', errors='replace') if 'proc' in locals() else 'rpm2cpio not available'
            raise Refusal(SCHEMA_INVALID, f"failed to extract rpm payload via rpm2cpio: {err_msg}")

    inventory, exe_digests = _parse_cpio_stream(io.BytesIO(cpio_bytes))
    payload_id = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()

    return ArchiveScanResult(
        inventory=inventory,
        payload_build_id=payload_id,
        package_identity=PackageIdentity(name=pkg_name, version=full_version, arch=pkg_arch),
        executable_sha256=exe_digests,
    )


def scan_variant_archive(path: Path) -> ArchiveScanResult:
    """Auto-detect archive type by extension and scan safely."""
    name = path.name
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return scan_tar_file(path)
    if name.endswith(".deb"):
        return scan_deb_file(path)
    if name.endswith(".rpm"):
        return scan_rpm_file(path)
    raise Refusal(SCHEMA_INVALID, f"unsupported archive extension for '{name}'")
