"""Integration tests for the transcripts indexer."""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from think.indexer import (
    get_index,
    reset_index,
    scan_transcripts,
    search_transcripts,
)


@pytest.mark.integration
def test_transcripts_indexer_scan_and_search():
    """Test scanning and searching transcript files from fixtures."""
    # Use fixtures journal path
    journal_path = Path(__file__).parent.parent.parent / "fixtures" / "journal"
    
    if not journal_path.exists():
        pytest.skip("fixtures/journal not found")
    
    # Create a temporary index directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set environment to use fixtures journal
        old_journal = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = str(journal_path)
        
        try:
            # Set up test index directory
            test_index_dir = Path(tmpdir) / "indexer"
            
            # Monkey-patch the get_index function
            import think.indexer.core
            old_get_index = think.indexer.core.get_index
            
            def test_get_index(*, index, journal=None, day=None):
                # Special handling for transcripts which require day
                if index == "transcripts" and day:
                    db_path = test_index_dir / day / think.indexer.core.DB_NAMES[index]
                else:
                    db_path = test_index_dir / think.indexer.core.DB_NAMES[index]
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(db_path))
                # Ensure schema is created
                for statement in think.indexer.core.SCHEMAS[index]:
                    conn.execute(statement)
                return conn, str(db_path)
            
            think.indexer.core.get_index = test_get_index
            
            # Reset index to ensure clean state
            reset_index(str(journal_path), "transcripts")
            
            # Scan the transcripts
            scan_count = scan_transcripts(str(journal_path))
            
            # We should have scanned transcript files (audio and screen)
            assert scan_count > 0, f"Expected to scan transcript files, got {scan_count}"
            
            # Search for specific content from audio transcripts
            
            # Search for "JWT tokens" (from 123456_audio.md)
            total, results = search_transcripts("JWT tokens")
            assert total > 0, "Should find 'JWT tokens' in audio transcript"
            assert len(results) > 0, "Should have actual results for 'JWT tokens'"
            found_jwt = False
            for result in results:
                if "jwt tokens" in result["text"].lower():
                    found_jwt = True
                    assert result["metadata"]["type"] in ["audio", "screen"]
                    assert "20240101" in result["metadata"]["day"]
                    break
            assert found_jwt, "Should find JWT tokens discussion"
            
            # Search for "authentication endpoints" (from 123456_audio.md)
            total, results = search_transcripts("authentication endpoints")
            assert total > 0, "Should find 'authentication endpoints'"
            
            # Search for "VSCode" (from screen transcript in day 2)
            total, results = search_transcripts("VSCode")
            assert total > 0, "Should find 'VSCode' in screen transcript"
            found_vscode = False
            for result in results:
                if "vscode" in result["text"].lower():
                    found_vscode = True
                    assert result["metadata"]["type"] == "screen"
                    assert "20240102" in result["metadata"]["day"]
                    break
            assert found_vscode, "Should find VSCode in screen transcript"
            
            # Search for "timezone" (from day 2 audio)
            total, results = search_transcripts("timezone")
            assert total > 0, "Should find 'timezone' issue discussion"
            found_timezone = False
            for result in results:
                if "timezone" in result["text"].lower():
                    found_timezone = True
                    assert "20240102" in result["metadata"]["day"]
                    break
            assert found_timezone, "Should find timezone bug discussion"
            
            # Search for "Docker logs" (from screen transcript)
            total, results = search_transcripts("Docker logs")
            assert total > 0, "Should find 'Docker logs' in screen"
            
            # Verify result structure
            if results:
                result = results[0]
                assert "text" in result
                assert "metadata" in result
                assert "score" in result
                assert "day" in result["metadata"]
                assert "type" in result["metadata"]
                assert "path" in result["metadata"]
                assert "time" in result["metadata"]
                assert result["metadata"]["type"] in ["audio", "screen"]
                
        finally:
            # Restore original environment
            if old_journal:
                os.environ["JOURNAL_PATH"] = old_journal
            elif "JOURNAL_PATH" in os.environ:
                del os.environ["JOURNAL_PATH"]
            
            # Restore original function
            think.indexer.core.get_index = old_get_index


@pytest.mark.integration
def test_transcripts_indexer_by_type():
    """Test searching transcripts by type (audio vs screen)."""
    journal_path = Path(__file__).parent.parent.parent / "fixtures" / "journal"
    
    if not journal_path.exists():
        pytest.skip("fixtures/journal not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        old_journal = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = str(journal_path)
        
        try:
            # Set up test index
            test_index_dir = Path(tmpdir) / "indexer"
            
            import think.indexer.core
            old_get_index = think.indexer.core.get_index
            
            def test_get_index(*, index, journal=None, day=None):
                # Special handling for transcripts which require day
                if index == "transcripts" and day:
                    db_path = test_index_dir / day / think.indexer.core.DB_NAMES[index]
                else:
                    db_path = test_index_dir / think.indexer.core.DB_NAMES[index]
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(db_path))
                # Ensure schema is created
                for statement in think.indexer.core.SCHEMAS[index]:
                    conn.execute(statement)
                return conn, str(db_path)
            
            think.indexer.core.get_index = test_get_index
            
            # Reset and scan
            reset_index(str(journal_path), "transcripts")
            scan_transcripts(str(journal_path))
            
            # Search for content that should be in audio
            total, audio_results = search_transcripts("JWT")
            audio_count = sum(1 for r in audio_results if r["metadata"]["type"] == "audio")
            assert audio_count > 0, "Should find JWT in audio transcripts"
            
            # Search for content that should be in screen
            total, screen_results = search_transcripts("IDE")
            screen_count = sum(1 for r in screen_results if r["metadata"]["type"] == "screen")
            assert screen_count > 0, "Should find IDE in screen transcripts"
            
            # Verify both days have transcripts
            total, all_results = search_transcripts("the")  # Common word
            days = set(r["metadata"]["day"] for r in all_results)
            assert "20240101" in days, "Should have transcripts from day 1"
            assert "20240102" in days, "Should have transcripts from day 2"
            
        finally:
            if old_journal:
                os.environ["JOURNAL_PATH"] = old_journal
            elif "JOURNAL_PATH" in os.environ:
                del os.environ["JOURNAL_PATH"]
            
            think.indexer.core.get_index = old_get_index


@pytest.mark.integration
def test_transcripts_indexer_rescan():
    """Test that rescanning transcripts handles updates properly."""
    journal_path = Path(__file__).parent.parent.parent / "fixtures" / "journal"
    
    if not journal_path.exists():
        pytest.skip("fixtures/journal not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        old_journal = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = str(journal_path)
        
        try:
            # Set up test index
            test_index_dir = Path(tmpdir) / "indexer"
            
            import think.indexer.core
            old_get_index = think.indexer.core.get_index
            
            def test_get_index(*, index, journal=None, day=None):
                # Special handling for transcripts which require day
                if index == "transcripts" and day:
                    db_path = test_index_dir / day / think.indexer.core.DB_NAMES[index]
                else:
                    db_path = test_index_dir / think.indexer.core.DB_NAMES[index]
                db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(db_path))
                # Ensure schema is created
                for statement in think.indexer.core.SCHEMAS[index]:
                    conn.execute(statement)
                return conn, str(db_path)
            
            think.indexer.core.get_index = test_get_index
            
            # Initial scan
            reset_index(str(journal_path), "transcripts")
            first_scan = scan_transcripts(str(journal_path))
            assert first_scan > 0
            
            # Search for content
            total1, results1 = search_transcripts("authentication")
            initial_count = total1
            
            # Rescan
            second_scan = scan_transcripts(str(journal_path))
            
            # Results should be consistent
            total2, results2 = search_transcripts("authentication")
            assert total2 == initial_count, "Rescan should not duplicate entries"
            
        finally:
            if old_journal:
                os.environ["JOURNAL_PATH"] = old_journal
            elif "JOURNAL_PATH" in os.environ:
                del os.environ["JOURNAL_PATH"]
            
            think.indexer.core.get_index = old_get_index