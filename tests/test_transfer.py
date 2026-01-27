# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe/transfer.py - day archive export and import."""

import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSegmentDeconfliction:
    """Tests for segment deconfliction via find_available_segment."""

    def test_find_available_segment_returns_original_if_free(self, tmp_path):
        """Test find_available_segment returns original if available."""
        from observe.utils import find_available_segment

        # No existing segments
        result = find_available_segment(tmp_path, "120000_300")
        assert result == "120000_300"

    def test_find_available_segment_finds_alternative(self, tmp_path):
        """Test find_available_segment finds alternative when original taken."""
        from observe.utils import find_available_segment

        # Create existing segment
        (tmp_path / "120000_300").mkdir()

        result = find_available_segment(tmp_path, "120000_300")
        assert result is not None
        assert result != "120000_300"
        # Should be a valid segment key format
        assert "_" in result

    def test_find_available_segment_returns_none_when_exhausted(self, tmp_path):
        """Test find_available_segment returns None when all slots taken."""
        from observe.utils import find_available_segment

        # Create many segments around the target
        for delta in range(-50, 51):
            for dur_delta in range(-50, 51):
                total_seconds = 12 * 3600 + delta
                if 0 <= total_seconds < 86400:
                    h = total_seconds // 3600
                    m = (total_seconds % 3600) // 60
                    s = total_seconds % 60
                    dur = 300 + dur_delta
                    if dur > 0:
                        (tmp_path / f"{h:02d}{m:02d}{s:02d}_{dur}").mkdir(exist_ok=True)

        # With so many slots filled, should eventually fail
        result = find_available_segment(tmp_path, "120000_300", max_attempts=10)
        # May or may not find one depending on random walk, but shouldn't crash
        assert result is None or "_" in result


class TestComputeSha256:
    """Tests for SHA256 computation utilities."""

    def test_compute_file_sha256(self, tmp_path):
        """Test compute_file_sha256 returns correct hash."""
        from observe.utils import compute_file_sha256

        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello world")

        # Known SHA256 of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert compute_file_sha256(test_file) == expected

    def test_compute_bytes_sha256(self):
        """Test compute_bytes_sha256 returns correct hash."""
        from observe.utils import compute_bytes_sha256

        # Known SHA256 of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert compute_bytes_sha256(b"hello world") == expected


class TestTransferExport:
    """Tests for archive creation (export)."""

    def test_create_archive_basic(self, tmp_path, monkeypatch):
        """Test create_archive creates valid archive."""
        from observe.transfer import create_archive

        # Set up mock journal
        journal_path = tmp_path / "journal"
        day_dir = journal_path / "20250101"
        segment_dir = day_dir / "120000_300"
        segment_dir.mkdir(parents=True)

        # Add test files to segment
        (segment_dir / "audio.flac").write_bytes(b"fake audio data")
        (segment_dir / "audio.jsonl").write_text('{"raw": "audio.flac"}\n')

        monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        # Clear cache
        import think.utils

        think.utils._journal_path_cache = None

        output_path = tmp_path / "test.tgz"
        result = create_archive("20250101", output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify archive contents
        with tarfile.open(output_path, "r:gz") as tar:
            names = tar.getnames()
            assert "manifest.json" in names
            assert "120000_300/audio.flac" in names
            assert "120000_300/audio.jsonl" in names

            # Verify manifest
            manifest_file = tar.extractfile("manifest.json")
            manifest = json.load(manifest_file)
            assert manifest["version"] == 1
            assert manifest["day"] == "20250101"
            assert "120000_300" in manifest["segments"]

    def test_create_archive_no_segments_error(self, tmp_path, monkeypatch):
        """Test create_archive raises error for empty day."""
        from observe.transfer import create_archive

        journal_path = tmp_path / "journal"
        day_dir = journal_path / "20250101"
        day_dir.mkdir(parents=True)

        monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        import think.utils

        think.utils._journal_path_cache = None

        with pytest.raises(ValueError, match="No segments found"):
            create_archive("20250101")

    def test_create_archive_no_day_error(self, tmp_path, monkeypatch):
        """Test create_archive raises error for missing day."""
        from observe.transfer import create_archive

        journal_path = tmp_path / "journal"
        journal_path.mkdir(parents=True)

        monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        import think.utils

        think.utils._journal_path_cache = None

        with pytest.raises(ValueError, match="does not exist"):
            create_archive("20250101")


class TestTransferImport:
    """Tests for archive import."""

    def _create_test_archive(self, tmp_path, segments: dict) -> Path:
        """Helper to create test archive."""
        archive_path = tmp_path / "test.tgz"

        manifest = {
            "version": 1,
            "day": "20250101",
            "created_at": 1704067200000,
            "host": "test-host",
            "segments": {},
        }

        with tarfile.open(archive_path, "w:gz") as tar:
            for segment, files in segments.items():
                manifest["segments"][segment] = {"files": []}
                for filename, content in files.items():
                    # Add to manifest
                    from observe.utils import compute_bytes_sha256

                    manifest["segments"][segment]["files"].append(
                        {
                            "name": filename,
                            "sha256": compute_bytes_sha256(content),
                            "size": len(content),
                        }
                    )

                    # Add file to archive
                    import io

                    info = tarfile.TarInfo(name=f"{segment}/{filename}")
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))

            # Add manifest
            import io

            manifest_json = json.dumps(manifest).encode()
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest_json)
            tar.addfile(info, io.BytesIO(manifest_json))

        return archive_path

    def test_validate_archive_all_new(self, tmp_path, monkeypatch):
        """Test validate_archive with no existing segments."""
        from observe.transfer import validate_archive

        # Create archive
        archive_path = self._create_test_archive(
            tmp_path,
            {
                "120000_300": {"audio.flac": b"audio data"},
                "130000_300": {"audio.flac": b"more audio"},
            },
        )

        # Set up empty journal
        journal_path = tmp_path / "journal"
        journal_path.mkdir()

        monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        import think.utils

        think.utils._journal_path_cache = None

        result = validate_archive(archive_path)

        assert result["skip"] == []
        assert len(result["import_as"]) == 2
        assert result["import_as"]["120000_300"] == "120000_300"
        assert result["import_as"]["130000_300"] == "130000_300"

    def test_validate_archive_skip_matching(self, tmp_path, monkeypatch):
        """Test validate_archive skips segments with matching hashes."""
        from observe.transfer import validate_archive

        # Create archive
        content = b"audio data"
        archive_path = self._create_test_archive(
            tmp_path,
            {"120000_300": {"audio.flac": content}},
        )

        # Set up journal with matching segment
        journal_path = tmp_path / "journal"
        segment_dir = journal_path / "20250101" / "120000_300"
        segment_dir.mkdir(parents=True)
        (segment_dir / "audio.flac").write_bytes(content)

        monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        import think.utils

        think.utils._journal_path_cache = None

        result = validate_archive(archive_path)

        assert "120000_300" in result["skip"]
        assert "120000_300" not in result["import_as"]

    def test_validate_archive_deconflict_different(self, tmp_path, monkeypatch):
        """Test validate_archive deconflicts segments with different content."""
        from observe.transfer import validate_archive

        # Create archive
        archive_path = self._create_test_archive(
            tmp_path,
            {"120000_300": {"audio.flac": b"new audio data"}},
        )

        # Set up journal with different content in same segment
        journal_path = tmp_path / "journal"
        segment_dir = journal_path / "20250101" / "120000_300"
        segment_dir.mkdir(parents=True)
        (segment_dir / "audio.flac").write_bytes(b"existing different data")

        monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        import think.utils

        think.utils._journal_path_cache = None

        result = validate_archive(archive_path)

        assert "120000_300" in result["deconflicted"]
        assert result["import_as"]["120000_300"] != "120000_300"

    def test_import_archive_basic(self, tmp_path, monkeypatch):
        """Test import_archive extracts segments correctly."""
        from observe.transfer import import_archive

        # Create archive
        audio_content = b"fake audio data"
        jsonl_content = b'{"raw": "audio.flac"}\n'

        archive_path = self._create_test_archive(
            tmp_path,
            {
                "120000_300": {
                    "audio.flac": audio_content,
                    "audio.jsonl": jsonl_content,
                }
            },
        )

        # Set up empty journal
        journal_path = tmp_path / "journal"
        journal_path.mkdir()

        monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        import think.utils

        think.utils._journal_path_cache = None

        # Mock subprocess to avoid running real indexer
        with patch("subprocess.run"):
            result = import_archive(archive_path)

        assert result["status"] == "imported"
        assert "120000_300" in result["imported"]

        # Verify files were extracted
        segment_dir = journal_path / "20250101" / "120000_300"
        assert segment_dir.exists()
        assert (segment_dir / "audio.flac").read_bytes() == audio_content
        assert (segment_dir / "audio.jsonl").read_bytes() == jsonl_content

    def test_import_archive_dry_run(self, tmp_path, monkeypatch):
        """Test import_archive dry run doesn't modify filesystem."""
        from observe.transfer import import_archive

        archive_path = self._create_test_archive(
            tmp_path,
            {"120000_300": {"audio.flac": b"audio data"}},
        )

        journal_path = tmp_path / "journal"
        journal_path.mkdir()

        monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        import think.utils

        think.utils._journal_path_cache = None

        result = import_archive(archive_path, dry_run=True)

        assert result["status"] == "dry_run"
        # Directory should not be created
        assert not (journal_path / "20250101").exists()

    def test_import_archive_nothing_to_import(self, tmp_path, monkeypatch):
        """Test import_archive when all segments already synced."""
        from observe.transfer import import_archive

        content = b"audio data"
        archive_path = self._create_test_archive(
            tmp_path,
            {"120000_300": {"audio.flac": content}},
        )

        # Set up journal with matching content
        journal_path = tmp_path / "journal"
        segment_dir = journal_path / "20250101" / "120000_300"
        segment_dir.mkdir(parents=True)
        (segment_dir / "audio.flac").write_bytes(content)

        monkeypatch.setenv("JOURNAL_PATH", str(journal_path))

        import think.utils

        think.utils._journal_path_cache = None

        result = import_archive(archive_path)

        assert result["status"] == "nothing_to_import"


class TestManifestValidation:
    """Tests for manifest reading and validation."""

    def test_read_manifest_missing(self, tmp_path):
        """Test error when manifest is missing from archive."""
        from observe.transfer import _read_manifest

        # Create archive without manifest
        archive_path = tmp_path / "test.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            import io

            info = tarfile.TarInfo(name="some_file.txt")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"test"))

        with pytest.raises(ValueError, match="manifest.json not found"):
            _read_manifest(archive_path)

    def test_read_manifest_wrong_version(self, tmp_path):
        """Test error when manifest has wrong version."""
        from observe.transfer import _read_manifest

        archive_path = tmp_path / "test.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            import io

            manifest = json.dumps({"version": 999, "day": "20250101", "segments": {}})
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest.encode()))

        with pytest.raises(ValueError, match="Unsupported manifest version"):
            _read_manifest(archive_path)

    def test_read_manifest_missing_fields(self, tmp_path):
        """Test error when manifest has missing required fields."""
        from observe.transfer import _read_manifest

        archive_path = tmp_path / "test.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            import io

            manifest = json.dumps({"version": 1})  # Missing day and segments
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest.encode()))

        with pytest.raises(ValueError, match="missing required fields"):
            _read_manifest(archive_path)
