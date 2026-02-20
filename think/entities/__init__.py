# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity management with journal-wide identity and facet-scoped relationships.

Entity System Architecture:
- Journal-level entities: entities/<id>/entity.json - canonical identity (name, type, aka)
- Journal-level memory: entities/<id>/ - voiceprints (identity-specific, cross-facet)
- Facet relationships: facets/<facet>/entities/<id>/entity.json - per-facet data
- Detected entities: facets/<facet>/entities/<day>.jsonl - ephemeral daily discoveries
- Facet entity memory: facets/<facet>/entities/<id>/ - observations (facet-specific)

This package is organized into focused modules:
- core: Types, constants, validation, slug generation
- journal: Journal-level entity CRUD
- relationships: Facet relationships and entity memory
- loading: Entity loading functions
- saving: Entity saving functions
- matching: Entity resolution and fuzzy matching
- activity: Activity tracking and detected entities
- observations: Observation CRUD
- formatting: Indexer formatting
"""

# Activity tracking
from think.entities.activity import (
    load_detected_entities_recent,
    parse_knowledge_graph_entities,
    touch_entities_from_activity,
    touch_entity,
)

# Core types and utilities
from think.entities.core import (
    DEFAULT_ACTIVITY_TS,
    ENTITY_TYPES,
    MAX_ENTITY_SLUG_LENGTH,
    EntityDict,
    atomic_write,
    entity_last_active_ts,
    entity_slug,
    get_identity_names,
    is_valid_entity_type,
)

# Formatting (for indexer)
from think.entities.formatting import format_entities

# Journal-level entity management
from think.entities.journal import (
    block_journal_entity,
    delete_journal_entity,
    ensure_journal_entity_memory,
    get_journal_principal,
    get_or_create_journal_entity,
    has_journal_principal,
    journal_entity_memory_path,
    journal_entity_path,
    load_all_journal_entities,
    load_journal_entity,
    save_journal_entity,
    scan_journal_entities,
    unblock_journal_entity,
)

# Entity loading
from think.entities.loading import (
    detected_entities_path,
    load_all_attached_entities,
    load_entities,
    load_entity_names,
    load_recent_entity_names,
    parse_entity_file,
)

# Entity matching and resolution
from think.entities.matching import (
    find_matching_entity,
    resolve_entity,
    validate_aka_uniqueness,
)

# Observations
from think.entities.observations import (
    ObservationNumberError,
    add_observation,
    load_observations,
    observations_file_path,
    save_observations,
)

# Facet relationships and memory
from think.entities.relationships import (
    ensure_entity_memory,
    entity_memory_path,
    facet_relationship_path,
    load_facet_relationship,
    rename_entity_memory,
    save_facet_relationship,
    scan_facet_relationships,
)

# Entity saving
from think.entities.saving import (
    save_detected_entity,
    save_entities,
    update_detected_entity,
    update_entity_description,
)

__all__ = [
    # Core
    "DEFAULT_ACTIVITY_TS",
    "ENTITY_TYPES",
    "MAX_ENTITY_SLUG_LENGTH",
    "EntityDict",
    "atomic_write",
    "entity_last_active_ts",
    "entity_slug",
    "get_identity_names",
    "is_valid_entity_type",
    # Journal
    "block_journal_entity",
    "delete_journal_entity",
    "ensure_journal_entity_memory",
    "get_journal_principal",
    "get_or_create_journal_entity",
    "has_journal_principal",
    "journal_entity_memory_path",
    "journal_entity_path",
    "load_all_journal_entities",
    "load_journal_entity",
    "save_journal_entity",
    "scan_journal_entities",
    "unblock_journal_entity",
    # Relationships
    "ensure_entity_memory",
    "entity_memory_path",
    "facet_relationship_path",
    "load_facet_relationship",
    "rename_entity_memory",
    "save_facet_relationship",
    "scan_facet_relationships",
    # Loading
    "detected_entities_path",
    "load_all_attached_entities",
    "load_entities",
    "load_entity_names",
    "load_recent_entity_names",
    "parse_entity_file",
    # Saving
    "save_detected_entity",
    "save_entities",
    "update_detected_entity",
    "update_entity_description",
    # Matching
    "find_matching_entity",
    "resolve_entity",
    "validate_aka_uniqueness",
    # Activity
    "load_detected_entities_recent",
    "parse_knowledge_graph_entities",
    "touch_entities_from_activity",
    "touch_entity",
    # Observations
    "ObservationNumberError",
    "add_observation",
    "load_observations",
    "observations_file_path",
    "save_observations",
    # Formatting
    "format_entities",
]
