"""Indexer package for insights, events, and transcripts.

This module provides backward compatibility by re-exporting all the main
functions from the sub-modules.
"""

# Import from cli
from .cli import main

# Import from core
from .core import (
    DATE_RE,
    DB_NAMES,
    INDEX_DIR,
    SCHEMAS,
    get_index,
    reset_index,
    sanitize_fts_query,
)

# Import from events
from .events import (
    scan_events,
    search_events,
)

# Import from insights
from .insights import (
    find_event_files,
    find_insight_files,
    scan_insights,
    search_insights,
)

# Import from transcripts
from .transcripts import (
    AUDIO_RE,
    SCREEN_RE,
    find_transcript_files,
    scan_transcripts,
    search_transcripts,
)

# All public functions and constants
__all__ = [
    # Core
    "DATE_RE",
    "DB_NAMES",
    "INDEX_DIR",
    "SCHEMAS",
    "get_index",
    "reset_index",
    "sanitize_fts_query",
    # Insights
    "find_event_files",
    "find_insight_files",
    "scan_insights",
    "search_insights",
    # Events
    "scan_events",
    "search_events",
    # Transcripts
    "AUDIO_RE",
    "SCREEN_RE",
    "find_transcript_files",
    "scan_transcripts",
    "search_transcripts",
    # CLI
    "main",
]
