"""Indexer package for summary outputs, events, transcripts, and entities.

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
    find_day_dirs,
    get_index,
    reset_index,
)

# Import from entities
from .entities import (
    ENTITY_ITEM_RE,
    find_entity_files,
    parse_entities,
    parse_entity_line,
    scan_entities,
    search_entities,
)

# Import from events
from .events import (
    scan_events,
    search_events,
)

# Import from summaries
from .summaries import (
    TOPIC_BASENAMES,
    TOPIC_DIR,
    find_summary_files,
    scan_summaries,
    search_summaries,
    split_sentences,
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
    "find_day_dirs",
    "get_index",
    "reset_index",
    # Summaries
    "TOPIC_BASENAMES",
    "TOPIC_DIR",
    "find_summary_files",
    "scan_summaries",
    "search_summaries",
    "split_sentences",
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
    "ENTITY_ITEM_RE",
    "find_entity_files",
    "parse_entities",
    "parse_entity_line",
    "scan_entities",
    "search_entities",
    # CLI
    "main",
]
