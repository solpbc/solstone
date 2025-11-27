"""Indexer package for insights, events, transcripts, and entities.

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
)

# Import from entities
from .entities import (
    find_entity_files,
    parse_entities,
    scan_entities,
    search_entities,
)

# Import from events
from .events import (
    scan_events,
    search_events,
)

# Import from insights
from .insights import (
    INSIGHT_TYPES,
    INSIGHTS_DIR,
    find_insight_files,
    scan_insights,
    search_insights,
    split_chunks,
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
    # Insights
    "INSIGHT_TYPES",
    "INSIGHTS_DIR",
    "find_insight_files",
    "scan_insights",
    "search_insights",
    "split_chunks",
    # Events
    "scan_events",
    "search_events",
    # Transcripts
    "AUDIO_RE",
    "SCREEN_RE",
    "find_transcript_files",
    "scan_transcripts",
    "search_transcripts",
    # Entities
    "find_entity_files",
    "parse_entities",
    "scan_entities",
    "search_entities",
    # CLI
    "main",
]
