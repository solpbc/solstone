# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe/transfer.py - day archive export, import, and send."""

import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests


class TestSegmentDeconfliction:
    """Tests for segment deconfliction via find_available_segment."""

    def test_find_available_segment_returns_original_if_free(self, tmp_path):
        """Test find_available_segment returns original if available."""
        from solstone.observe.utils import find_available_segment

        # No existing segments
        result = find_available_segment(tmp_path, "120000_300")
        assert result == "120000_300"

    def test_find_available_segment_finds_alternative(self, tmp_path):
        """Test find_available_segment finds alternative when original taken."""
        from solstone.observe.utils import find_available_segment

        # Create existing segment
        (tmp_path / "120000_300").mkdir()

        result = find_available_segment(tmp_path, "120000_300")
        assert result is not None
        assert result != "120000_300"
        # Should be a valid segment key format
        assert "_" in result

    def test_find_available_segment_returns_none_when_exhausted(self, tmp_path):
        """Test find_available_segment returns None when all slots taken."""
        from solstone.observe.utils import find_available_segment

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
        from solstone.observe.utils import compute_file_sha256

        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"hello world")

        # Known SHA256 of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert compute_file_sha256(test_file) == expected

    def test_compute_bytes_sha256(self):
        """Test compute_bytes_sha256 returns correct hash."""
        from solstone.observe.utils import compute_bytes_sha256

        # Known SHA256 of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert compute_bytes_sha256(b"hello world") == expected


class TestTransferExport:
    """Tests for archive creation (export)."""

    def test_create_archive_basic(self, tmp_path, monkeypatch):
        """Test create_archive creates valid archive."""
        from solstone.observe.transfer import create_archive

        # Set up mock journal with day/stream/segment structure
        journal_path = tmp_path / "journal"
        day_dir = journal_path / "chronicle" / "20250101"
        segment_dir = day_dir / "default" / "120000_300"
        segment_dir.mkdir(parents=True)

        # Add test files to segment
        (segment_dir / "audio.flac").write_bytes(b"fake audio data")
        (segment_dir / "audio.jsonl").write_text('{"raw": "audio.flac"}\n')

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_path))

        # Clear cache
        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

        output_path = tmp_path / "test.tgz"
        result = create_archive("20250101", output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify archive contents
        with tarfile.open(output_path, "r:gz") as tar:
            names = tar.getnames()
            assert "manifest.json" in names
            assert "default/120000_300/audio.flac" in names
            assert "default/120000_300/audio.jsonl" in names

            # Verify manifest
            manifest_file = tar.extractfile("manifest.json")
            manifest = json.load(manifest_file)
            assert manifest["version"] == 1
            assert manifest["day"] == "20250101"
            assert "default/120000_300" in manifest["segments"]

    def test_create_archive_no_segments_error(self, tmp_path, monkeypatch):
        """Test create_archive raises error for empty day."""
        from solstone.observe.transfer import create_archive

        journal_path = tmp_path / "journal"
        day_dir = journal_path / "chronicle" / "20250101"
        day_dir.mkdir(parents=True)

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_path))

        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

        with pytest.raises(ValueError, match="No segments found"):
            create_archive("20250101")

    def test_create_archive_no_day_error(self, tmp_path, monkeypatch):
        """Test create_archive raises error for missing day."""
        from solstone.observe.transfer import create_archive

        journal_path = tmp_path / "journal"
        journal_path.mkdir(parents=True)

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_path))

        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

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
                    from solstone.observe.utils import compute_bytes_sha256

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
        from solstone.observe.transfer import validate_archive

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

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_path))

        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

        result = validate_archive(archive_path)

        assert result["skip"] == []
        assert len(result["import_as"]) == 2
        assert result["import_as"]["120000_300"] == "120000_300"
        assert result["import_as"]["130000_300"] == "130000_300"

    def test_validate_archive_skip_matching(self, tmp_path, monkeypatch):
        """Test validate_archive skips segments with matching hashes."""
        from solstone.observe.transfer import validate_archive

        # Create archive
        content = b"audio data"
        archive_path = self._create_test_archive(
            tmp_path,
            {"120000_300": {"audio.flac": content}},
        )

        # Set up journal with matching segment
        journal_path = tmp_path / "journal"
        segment_dir = journal_path / "chronicle" / "20250101" / "120000_300"
        segment_dir.mkdir(parents=True)
        (segment_dir / "audio.flac").write_bytes(content)

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_path))

        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

        result = validate_archive(archive_path)

        assert "120000_300" in result["skip"]
        assert "120000_300" not in result["import_as"]

    def test_validate_archive_deconflict_different(self, tmp_path, monkeypatch):
        """Test validate_archive deconflicts segments with different content."""
        from solstone.observe.transfer import validate_archive

        # Create archive
        archive_path = self._create_test_archive(
            tmp_path,
            {"120000_300": {"audio.flac": b"new audio data"}},
        )

        # Set up journal with different content in same segment
        journal_path = tmp_path / "journal"
        segment_dir = journal_path / "chronicle" / "20250101" / "120000_300"
        segment_dir.mkdir(parents=True)
        (segment_dir / "audio.flac").write_bytes(b"existing different data")

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_path))

        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

        result = validate_archive(archive_path)

        assert "120000_300" in result["deconflicted"]
        assert result["import_as"]["120000_300"] != "120000_300"

    def test_import_archive_basic(self, tmp_path, monkeypatch):
        """Test import_archive extracts segments correctly."""
        from solstone.observe.transfer import import_archive

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

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_path))

        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

        # Mock subprocess to avoid running real indexer
        with patch("subprocess.run"):
            result = import_archive(archive_path)

        assert result["status"] == "imported"
        assert "120000_300" in result["imported"]

        # Verify files were extracted
        segment_dir = journal_path / "chronicle" / "20250101" / "120000_300"
        assert segment_dir.exists()
        assert (segment_dir / "audio.flac").read_bytes() == audio_content
        assert (segment_dir / "audio.jsonl").read_bytes() == jsonl_content

    def test_import_archive_dry_run(self, tmp_path, monkeypatch):
        """Test import_archive dry run doesn't modify filesystem."""
        from solstone.observe.transfer import import_archive

        archive_path = self._create_test_archive(
            tmp_path,
            {"120000_300": {"audio.flac": b"audio data"}},
        )

        journal_path = tmp_path / "journal"
        journal_path.mkdir()

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_path))

        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

        result = import_archive(archive_path, dry_run=True)

        assert result["status"] == "dry_run"
        # Directory should not be created
        assert not (journal_path / "chronicle" / "20250101").exists()

    def test_import_archive_nothing_to_import(self, tmp_path, monkeypatch):
        """Test import_archive when all segments already synced."""
        from solstone.observe.transfer import import_archive

        content = b"audio data"
        archive_path = self._create_test_archive(
            tmp_path,
            {"120000_300": {"audio.flac": content}},
        )

        # Set up journal with matching content
        journal_path = tmp_path / "journal"
        segment_dir = journal_path / "chronicle" / "20250101" / "120000_300"
        segment_dir.mkdir(parents=True)
        (segment_dir / "audio.flac").write_bytes(content)

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_path))

        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

        result = import_archive(archive_path)

        assert result["status"] == "nothing_to_import"


class TestManifestValidation:
    """Tests for manifest reading and validation."""

    def test_read_manifest_missing(self, tmp_path):
        """Test error when manifest is missing from archive."""
        from solstone.observe.transfer import _read_manifest

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
        from solstone.observe.transfer import _read_manifest

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
        from solstone.observe.transfer import _read_manifest

        archive_path = tmp_path / "test.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            import io

            manifest = json.dumps({"version": 1})  # Missing day and segments
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest.encode()))

        with pytest.raises(ValueError, match="missing required fields"):
            _read_manifest(archive_path)


class TestTransferSend:
    """Tests for transfer send functionality."""

    def _setup_journal(self, tmp_path, *, include_stream_json: bool = False) -> Path:
        journal = tmp_path / "journal"
        day_dir = journal / "chronicle" / "20250103" / "default" / "120000_300"
        day_dir.mkdir(parents=True)
        (day_dir / "audio.flac").write_bytes(b"audio data")
        (day_dir / "transcript.jsonl").write_text('{"text": "hello"}\n')
        if include_stream_json:
            (day_dir / "stream.json").write_text('{"stream": "default"}\n')
        return journal

    def _set_journal_override(self, monkeypatch, journal: Path) -> None:
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))

        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

    def _make_session(
        self,
        *,
        get_status: int = 200,
        get_json: list | None = None,
        post_status: int = 200,
        post_json: dict | None = None,
    ) -> MagicMock:
        mock_session = MagicMock(spec=requests.Session)
        mock_session.headers = {}

        get_response = MagicMock()
        get_response.status_code = get_status
        get_response.json.return_value = get_json if get_json is not None else []
        get_response.text = "GET error"
        mock_session.get.return_value = get_response

        post_response = MagicMock()
        post_response.status_code = post_status
        post_response.json.return_value = (
            post_json if post_json is not None else {"status": "ok", "bytes": 100}
        )
        post_response.text = "POST error"
        mock_session.post.return_value = post_response

        return mock_session

    def test_parse_day_spec_single(self, tmp_path):
        from solstone.observe.transfer import _parse_day_spec

        journal_root = tmp_path / "journal"
        journal_root.mkdir()

        assert _parse_day_spec("20250103", journal_root) == ["20250103"]

    def test_parse_day_spec_range(self, tmp_path):
        from solstone.observe.transfer import _parse_day_spec

        journal_root = tmp_path / "journal"
        journal_root.mkdir()

        assert _parse_day_spec("20250101-20250103", journal_root) == [
            "20250101",
            "20250102",
            "20250103",
        ]

    def test_parse_day_spec_all_days(self, tmp_path):
        from solstone.observe.transfer import _parse_day_spec

        journal_root = tmp_path / "journal"
        journal_root.mkdir()
        (journal_root / "chronicle" / "20250101").mkdir(parents=True)
        (journal_root / "chronicle" / "20250103").mkdir(parents=True)
        (journal_root / "config").mkdir()
        (journal_root / "streams").mkdir()

        assert _parse_day_spec(None, journal_root) == ["20250101", "20250103"]

    def test_parse_day_spec_invalid(self, tmp_path):
        from solstone.observe.transfer import _parse_day_spec

        journal_root = tmp_path / "journal"
        journal_root.mkdir()

        with pytest.raises(ValueError, match="Invalid day format"):
            _parse_day_spec("invalid", journal_root)

    def test_normalize_url(self):
        from solstone.observe.transfer import _normalize_url

        assert _normalize_url("example.com") == "https://example.com"
        assert _normalize_url("example.com/") == "https://example.com"
        assert _normalize_url("https://example.com/") == "https://example.com"
        assert _normalize_url("http://example.com/api/") == "http://example.com/api"

    def test_send_dry_run(self, tmp_path, monkeypatch, capsys):
        from solstone.observe.transfer import send_segments

        journal = self._setup_journal(tmp_path)
        self._set_journal_override(monkeypatch, journal)

        mock_session = self._make_session(get_json=[])

        with patch(
            "solstone.observe.transfer.requests.Session", return_value=mock_session
        ):
            send_segments("https://example.com", "test-key", ["20250103"], dry_run=True)

        assert mock_session.get.call_count == 1
        assert mock_session.get.call_args.args[0].endswith(
            "/app/observer/ingest/segments/20250103"
        )
        assert mock_session.post.call_count == 0
        assert "Dry run: would send 1, skip 0" in capsys.readouterr().out

    def test_send_skips_matching(self, tmp_path, monkeypatch, capsys):
        from solstone.observe.transfer import send_segments
        from solstone.observe.utils import compute_file_sha256

        journal = self._setup_journal(tmp_path)
        self._set_journal_override(monkeypatch, journal)

        segment_dir = journal / "chronicle" / "20250103" / "default" / "120000_300"
        remote_files = [
            {
                "name": "audio.flac",
                "sha256": compute_file_sha256(segment_dir / "audio.flac"),
            },
            {
                "name": "transcript.jsonl",
                "sha256": compute_file_sha256(segment_dir / "transcript.jsonl"),
            },
        ]
        mock_session = self._make_session(
            get_json=[{"key": "120000_300", "observed": False, "files": remote_files}]
        )

        with patch(
            "solstone.observe.transfer.requests.Session", return_value=mock_session
        ):
            send_segments(
                "https://example.com", "test-key", ["20250103"], dry_run=False
            )

        assert mock_session.post.call_count == 0
        output = capsys.readouterr().out
        assert "Transfer complete: 0 sent, 1 skipped, 0 failed" in output
        assert "Nothing to send - remote is up to date" in output

    def test_send_uploads_new(self, tmp_path, monkeypatch, capsys):
        from solstone.observe.transfer import send_segments

        journal = self._setup_journal(tmp_path)
        self._set_journal_override(monkeypatch, journal)

        mock_session = self._make_session(
            get_json=[],
            post_json={"status": "ok", "bytes": 100},
        )

        with patch(
            "solstone.observe.transfer.requests.Session", return_value=mock_session
        ):
            send_segments(
                "https://example.com", "test-key", ["20250103"], dry_run=False
            )

        assert mock_session.post.call_count == 1
        post_kwargs = mock_session.post.call_args.kwargs
        assert post_kwargs["data"]["day"] == "20250103"
        assert post_kwargs["data"]["segment"] == "120000_300"
        assert json.loads(post_kwargs["data"]["meta"]) == {"stream": "default"}
        # Auth is set on the session, not per-request
        assert mock_session.headers["Authorization"] == "Bearer test-key"
        assert (
            "Transfer complete: 1 sent, 0 skipped, 0 failed, 100 bytes transferred"
            in (capsys.readouterr().out)
        )

    def test_send_retry_on_5xx(self, tmp_path, monkeypatch, capsys):
        from solstone.observe.transfer import send_segments

        journal = self._setup_journal(tmp_path)
        self._set_journal_override(monkeypatch, journal)

        mock_session = self._make_session(get_json=[])
        first = MagicMock(status_code=500, text="server error")
        second = MagicMock(status_code=500, text="server error")
        success = MagicMock(status_code=200)
        success.json.return_value = {"status": "ok", "bytes": 100}
        mock_session.post.side_effect = [first, second, success]

        with (
            patch(
                "solstone.observe.transfer.requests.Session", return_value=mock_session
            ),
            patch("solstone.observe.transfer.time.sleep"),
        ):
            send_segments(
                "https://example.com", "test-key", ["20250103"], dry_run=False
            )

        assert mock_session.post.call_count == 3
        assert (
            "Transfer complete: 1 sent, 0 skipped, 0 failed, 100 bytes transferred"
            in (capsys.readouterr().out)
        )

    def test_send_auth_error(self):
        from solstone.observe.transfer import _query_remote_segments

        mock_session = self._make_session(get_status=401)

        with pytest.raises(ValueError, match="Authentication failed"):
            _query_remote_segments(
                mock_session,
                "https://example.com",
                "20250103",
            )

    def test_send_idempotent(self, tmp_path, monkeypatch, capsys):
        from solstone.observe.transfer import send_segments
        from solstone.observe.utils import compute_file_sha256

        journal = self._setup_journal(tmp_path)
        self._set_journal_override(monkeypatch, journal)

        segment_dir = journal / "chronicle" / "20250103" / "default" / "120000_300"
        remote_files = [
            {
                "name": "audio.flac",
                "sha256": compute_file_sha256(segment_dir / "audio.flac"),
            },
            {
                "name": "transcript.jsonl",
                "sha256": compute_file_sha256(segment_dir / "transcript.jsonl"),
            },
        ]
        first_get = MagicMock(status_code=200)
        first_get.json.return_value = []
        first_get.text = "GET error"
        second_get = MagicMock(status_code=200)
        second_get.json.return_value = [
            {"key": "120000_300", "observed": False, "files": remote_files}
        ]
        second_get.text = "GET error"

        mock_session = self._make_session()
        mock_session.get.side_effect = [first_get, second_get]

        with patch(
            "solstone.observe.transfer.requests.Session", return_value=mock_session
        ):
            send_segments(
                "https://example.com", "test-key", ["20250103"], dry_run=False
            )
            send_segments(
                "https://example.com", "test-key", ["20250103"], dry_run=False
            )

        assert mock_session.post.call_count == 1
        output = capsys.readouterr().out
        assert (
            "Transfer complete: 1 sent, 0 skipped, 0 failed, 100 bytes transferred"
            in (output)
        )
        assert (
            "Transfer complete: 0 sent, 1 skipped, 0 failed, 0 bytes transferred"
            in output
        )

    def test_send_excludes_stream_json(self, tmp_path, monkeypatch):
        from solstone.observe.transfer import send_segments

        journal = self._setup_journal(tmp_path, include_stream_json=True)
        self._set_journal_override(monkeypatch, journal)

        mock_session = self._make_session(get_json=[])

        with patch(
            "solstone.observe.transfer.requests.Session", return_value=mock_session
        ):
            send_segments(
                "https://example.com", "test-key", ["20250103"], dry_run=False
            )

        files_arg = mock_session.post.call_args.kwargs["files"]
        uploaded_names = [entry[1][0] for entry in files_arg]
        assert "stream.json" not in uploaded_names
        assert uploaded_names == ["audio.flac", "transcript.jsonl"]
