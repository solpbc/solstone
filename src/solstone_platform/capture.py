# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Race-safe, content-blind local source snapshot capture."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Callable, Optional

from solstone_platform.refusals import (
    CAPTURE_DEPTH_EXCEEDED,
    CAPTURE_DUPLICATE_ENTRY,
    CAPTURE_ENTRY_LIMIT_EXCEEDED,
    CAPTURE_FILE_DISAPPEARED,
    CAPTURE_HARDLINK_ALIAS,
    CAPTURE_PATH_ESCAPE,
    CAPTURE_READ_INTERRUPTED,
    CAPTURE_SIZE_LIMIT_EXCEEDED,
    CAPTURE_SPECIAL_FILE,
    CAPTURE_SYMLINK_TRAVERSAL,
    Refusal,
    SOURCE_CHANGED_DURING_CAPTURE,
)

# Test-only injection hooks (production never sets these)
_between_inspect_and_open: Optional[Callable[[str, str], None]] = None
_interrupted_read_hook: Optional[Callable[[str, str], None]] = None

_last_private_parent: Optional[Path] = None


def last_private_parent() -> Optional[Path]:
    """Test accessor to verify cleanup of private snapshot directory."""
    return _last_private_parent


MAX_TOTAL_ENTRIES = 256
MAX_AGGREGATE_BYTES = 64 * 1024 * 1024 * 1024  # 64 GiB
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024   # 16 GiB
MAX_METADATA_BYTES = 4 * 1024 * 1024           # 4 MiB


def _classify_max_size_by_name(name: str) -> int:
    lower = name.lower()
    if lower.endswith(".tar.gz") or lower.endswith(".deb") or lower.endswith(".rpm"):
        return MAX_ARTIFACT_BYTES
    if lower.startswith("solstone-journal-") and lower.endswith("-install.sh"):
        return MAX_METADATA_BYTES
    return MAX_METADATA_BYTES


@dataclass
class CapturedSnapshot:
    snapshot_root: Path
    manifest_path: Path
    signature_path: Path
    journal_dir: Path
    desktop_dir: Path
    tmux_dir: Path
    bootstrap_file: Optional[Path] = None
    captured_files: list[Path] = None

    def cleanup(self) -> None:
        if self.snapshot_root and self.snapshot_root.exists():
            shutil.rmtree(self.snapshot_root, ignore_errors=True)

    def __enter__(self) -> "CapturedSnapshot":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


class _SourceCapturer:
    def __init__(self, snapshot_root: Path) -> None:
        self.snapshot_root = snapshot_root
        self.total_entries = 0
        self.total_bytes = 0
        self.seen_dest_rel_paths: set[str] = set()
        self.captured_paths: list[Path] = []

    def _safe_copy_file(
        self,
        source_class: str,
        parent_fd: int,
        parent_path: Path,
        parent_st: os.stat_result,
        name: str,
        rel_path: str,
        dest_path: Path,
        recorded_stat: os.stat_result | None = None,
    ) -> None:
        if ".." in name or "/" in name or "\\" in name:
            raise Refusal(CAPTURE_PATH_ESCAPE, f"illegal filename: {name}")

        dest_rel = str(dest_path.relative_to(self.snapshot_root))
        if dest_rel in self.seen_dest_rel_paths:
            raise Refusal(CAPTURE_DUPLICATE_ENTRY, f"duplicate entry in capture: {dest_rel}")
        self.seen_dest_rel_paths.add(dest_rel)

        if recorded_stat is None:
            try:
                recorded_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                raise Refusal(CAPTURE_FILE_DISAPPEARED, f"file not found: {rel_path}")

        if _between_inspect_and_open:
            _between_inspect_and_open(source_class, rel_path)

        # Verify parent directory was not replaced/renamed
        try:
            curr_parent_st = os.stat(str(parent_path), follow_symlinks=False)
            if curr_parent_st.st_dev != parent_st.st_dev or curr_parent_st.st_ino != parent_st.st_ino:
                raise Refusal(SOURCE_CHANGED_DURING_CAPTURE, f"parent directory changed during capture: {rel_path}")
        except FileNotFoundError:
            raise Refusal(SOURCE_CHANGED_DURING_CAPTURE, f"parent directory disappeared during capture: {rel_path}")

        # Check stat before open
        try:
            st_before_open = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            raise Refusal(CAPTURE_FILE_DISAPPEARED, f"file disappeared during capture: {rel_path}")
        except OSError as err:
            raise Refusal(CAPTURE_SPECIAL_FILE, f"stat failed for {rel_path}: {err}")

        if stat.S_ISLNK(st_before_open.st_mode):
            raise Refusal(CAPTURE_SYMLINK_TRAVERSAL, f"symlink refused: {rel_path}")
        if not stat.S_ISREG(st_before_open.st_mode):
            raise Refusal(CAPTURE_SPECIAL_FILE, f"non-regular file refused: {rel_path}")

        if st_before_open.st_ino != recorded_stat.st_ino or st_before_open.st_dev != recorded_stat.st_dev:
            raise Refusal(SOURCE_CHANGED_DURING_CAPTURE, f"file inode replaced during capture: {rel_path}")

        try:
            fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            raise Refusal(CAPTURE_FILE_DISAPPEARED, f"file disappeared during capture: {rel_path}")
        except OSError as err:
            try:
                st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if stat.S_ISLNK(st.st_mode):
                    raise Refusal(CAPTURE_SYMLINK_TRAVERSAL, f"symlink refused: {rel_path}")
            except Refusal:
                raise
            except Exception:
                pass
            raise Refusal(CAPTURE_SPECIAL_FILE, f"cannot open regular file {rel_path}: {err}")

        dest_fd: int | None = None
        try:
            st_opened = os.fstat(fd)
            if stat.S_ISLNK(st_opened.st_mode):
                raise Refusal(CAPTURE_SYMLINK_TRAVERSAL, f"symlink refused: {rel_path}")
            if not stat.S_ISREG(st_opened.st_mode):
                raise Refusal(CAPTURE_SPECIAL_FILE, f"non-regular file refused: {rel_path}")
            if st_opened.st_nlink != 1:
                raise Refusal(CAPTURE_HARDLINK_ALIAS, f"hard-linked file with nlink={st_opened.st_nlink} refused: {rel_path}")
            if st_opened.st_ino != st_before_open.st_ino or st_opened.st_dev != st_before_open.st_dev:
                raise Refusal(SOURCE_CHANGED_DURING_CAPTURE, f"file inode changed on open: {rel_path}")

            if self.total_entries + 1 > MAX_TOTAL_ENTRIES:
                raise Refusal(CAPTURE_ENTRY_LIMIT_EXCEEDED, f"capture exceeded maximum entry count {MAX_TOTAL_ENTRIES}")

            max_allowed = _classify_max_size_by_name(name)
            if st_opened.st_size > max_allowed:
                raise Refusal(
                    CAPTURE_SIZE_LIMIT_EXCEEDED,
                    f"file {rel_path} size {st_opened.st_size} exceeds limit {max_allowed}",
                )
            if self.total_bytes + st_opened.st_size > MAX_AGGREGATE_BYTES:
                raise Refusal(
                    CAPTURE_SIZE_LIMIT_EXCEEDED,
                    f"aggregate capture size {self.total_bytes + st_opened.st_size} exceeds limit {MAX_AGGREGATE_BYTES}",
                )

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_fd = os.open(
                dest_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )

            def write_snapshot(chunk: bytes) -> None:
                offset = 0
                while offset < len(chunk):
                    written = os.write(dest_fd, chunk[offset:])
                    if written <= 0:
                        raise Refusal(CAPTURE_READ_INTERRUPTED, f"snapshot write interrupted for {rel_path}")
                    offset += written

            copied_bytes = 0
            hasher1 = hashlib.sha256()

            if st_opened.st_size > 0:
                prefix_len = min(4096, st_opened.st_size)
                first_chunk = os.read(fd, prefix_len)
                if first_chunk:
                    write_snapshot(first_chunk)
                    copied_bytes += len(first_chunk)
                    hasher1.update(first_chunk)

            if _interrupted_read_hook:
                _interrupted_read_hook(source_class, rel_path)

            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                if copied_bytes + len(chunk) > st_opened.st_size:
                    raise Refusal(SOURCE_CHANGED_DURING_CAPTURE, f"file grew during capture: {rel_path}")
                write_snapshot(chunk)
                copied_bytes += len(chunk)
                hasher1.update(chunk)

            # Verification stat
            st_after = os.fstat(fd)
            if (
                st_opened.st_dev != st_after.st_dev
                or st_opened.st_ino != st_after.st_ino
                or st_opened.st_size != st_after.st_size
                or copied_bytes != st_after.st_size
            ):
                raise Refusal(SOURCE_CHANGED_DURING_CAPTURE, f"file metadata mutated during capture: {rel_path}")

            os.fsync(dest_fd)

            # Second read pass to detect in-place rewrite of same size/inode
            os.lseek(fd, 0, os.SEEK_SET)
            hasher2 = hashlib.sha256()
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                hasher2.update(chunk)

            st_verified = os.fstat(fd)
            if (
                st_after.st_dev != st_verified.st_dev
                or st_after.st_ino != st_verified.st_ino
                or st_after.st_size != st_verified.st_size
                or hasher1.digest() != hasher2.digest()
            ):
                raise Refusal(SOURCE_CHANGED_DURING_CAPTURE, f"file content rewritten during capture: {rel_path}")

        finally:
            if dest_fd is not None:
                os.close(dest_fd)
            os.close(fd)

        self.total_entries += 1
        self.total_bytes += copied_bytes
        self.captured_paths.append(dest_path)

    def capture_single_file(self, source_class: str, file_path: Path, dest_path: Path) -> None:
        parent = file_path.parent
        try:
            parent_fd = os.open(
                str(parent),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            )
        except FileNotFoundError:
            raise Refusal(CAPTURE_FILE_DISAPPEARED, f"directory not found: {parent}")

        try:
            parent_st = os.fstat(parent_fd)
            try:
                st = os.stat(file_path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                raise Refusal(CAPTURE_FILE_DISAPPEARED, f"file not found: {file_path}")

            if stat.S_ISLNK(st.st_mode):
                raise Refusal(CAPTURE_SYMLINK_TRAVERSAL, f"symlink refused: {file_path.name}")
            if not stat.S_ISREG(st.st_mode):
                raise Refusal(CAPTURE_SPECIAL_FILE, f"non-regular file refused: {file_path.name}")

            self._safe_copy_file(
                source_class=source_class,
                parent_fd=parent_fd,
                parent_path=parent,
                parent_st=parent_st,
                name=file_path.name,
                rel_path=file_path.name,
                dest_path=dest_path,
                recorded_stat=st,
            )
        finally:
            os.close(parent_fd)

    def capture_directory(self, source_class: str, root_path: Path, dest_dir: Path, allowed_subdirs: Optional[set[str]] = None) -> None:
        try:
            root_fd = os.open(
                str(root_path),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            )
        except FileNotFoundError:
            raise Refusal(CAPTURE_FILE_DISAPPEARED, f"directory not found: {root_path}")

        try:
            root_st = os.fstat(root_fd)
            try:
                names = sorted(os.listdir(root_fd))
            except OSError as err:
                raise Refusal(CAPTURE_FILE_DISAPPEARED, f"cannot list directory {root_path}: {err}")

            for name in names:
                try:
                    entry_st = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                except FileNotFoundError:
                    raise Refusal(CAPTURE_FILE_DISAPPEARED, f"entry disappeared: {name}")

                if stat.S_ISLNK(entry_st.st_mode):
                    raise Refusal(CAPTURE_SYMLINK_TRAVERSAL, f"symlink in source tree refused: {name}")

                if stat.S_ISDIR(entry_st.st_mode):
                    if allowed_subdirs is None or name not in allowed_subdirs:
                        raise Refusal(CAPTURE_DEPTH_EXCEEDED, f"unexpected subdirectory '{name}' in {source_class}")

                    sub_path = root_path / name
                    sub_dest = dest_dir / name
                    try:
                        sub_fd = os.open(
                            name,
                            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                            dir_fd=root_fd,
                        )
                    except FileNotFoundError:
                        raise Refusal(CAPTURE_FILE_DISAPPEARED, f"subdirectory disappeared: {name}")

                    try:
                        sub_st = os.fstat(sub_fd)
                        sub_names = sorted(os.listdir(sub_fd))
                        for sub_name in sub_names:
                            try:
                                sub_entry_st = os.stat(sub_name, dir_fd=sub_fd, follow_symlinks=False)
                            except FileNotFoundError:
                                raise Refusal(CAPTURE_FILE_DISAPPEARED, f"entry disappeared: {name}/{sub_name}")

                            if stat.S_ISLNK(sub_entry_st.st_mode):
                                raise Refusal(CAPTURE_SYMLINK_TRAVERSAL, f"symlink in subdir refused: {name}/{sub_name}")
                            if stat.S_ISDIR(sub_entry_st.st_mode):
                                raise Refusal(CAPTURE_DEPTH_EXCEEDED, f"depth > 2 refused in {source_class}: {name}/{sub_name}")

                            self._safe_copy_file(
                                source_class=source_class,
                                parent_fd=sub_fd,
                                parent_path=sub_path,
                                parent_st=sub_st,
                                name=sub_name,
                                rel_path=f"{name}/{sub_name}",
                                dest_path=sub_dest / sub_name,
                                recorded_stat=sub_entry_st,
                            )
                    finally:
                        os.close(sub_fd)
                else:
                    self._safe_copy_file(
                        source_class=source_class,
                        parent_fd=root_fd,
                        parent_path=root_path,
                        parent_st=root_st,
                        name=name,
                        rel_path=name,
                        dest_path=dest_dir / name,
                        recorded_stat=entry_st,
                    )
        finally:
            os.close(root_fd)



def capture_release_sources(
    manifest_path: Path,
    signature_path: Path,
    journal_dir: Path,
    desktop_dir: Path,
    tmux_dir: Path,
    bootstrap_file: Optional[Path] = None,
) -> CapturedSnapshot:
    """Capture all release sources into a private 0700 snapshot directory."""
    global _last_private_parent
    temp_dir = Path(tempfile.mkdtemp(prefix="solstone-capture-", dir=None))
    _last_private_parent = temp_dir
    os.chmod(temp_dir, 0o700)

    capturer = _SourceCapturer(temp_dir)
    snapshot = CapturedSnapshot(
        snapshot_root=temp_dir,
        manifest_path=temp_dir / "platform" / "platform.json",
        signature_path=temp_dir / "platform" / "platform.json.minisig",
        journal_dir=temp_dir / "journal",
        desktop_dir=temp_dir / "desktop",
        tmux_dir=temp_dir / "tmux",
        bootstrap_file=temp_dir / "bootstrap" / bootstrap_file.name if bootstrap_file else None,
    )

    try:
        # Platform manifest & signature
        capturer.capture_single_file("platform_manifest", manifest_path, snapshot.manifest_path)
        capturer.capture_single_file("platform_signature", signature_path, snapshot.signature_path)

        # Journal tree (allows immediate subdirs linux-x86_64 and linux-aarch64)
        capturer.capture_directory("journal", journal_dir, snapshot.journal_dir, allowed_subdirs={"linux-x86_64", "linux-aarch64"})

        # Desktop tree (root regular files only)
        capturer.capture_directory("desktop", desktop_dir, snapshot.desktop_dir, allowed_subdirs=None)

        # Tmux tree (root regular files only)
        capturer.capture_directory("tmux", tmux_dir, snapshot.tmux_dir, allowed_subdirs=None)

        # Optional explicit bootstrap file
        if bootstrap_file:
            capturer.capture_single_file("bootstrap", bootstrap_file, snapshot.bootstrap_file)

        for directory in sorted(
            (path for path in temp_dir.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ) + [temp_dir]:
            dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

        snapshot.captured_files = capturer.captured_paths
        return snapshot

    except Exception:
        snapshot.cleanup()
        raise
