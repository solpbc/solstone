# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Migrate entities from legacy facet-scoped format to journal-wide structure.

This migration:
1. Loads all entities from facets/*/entities.jsonl (legacy format)
2. Fuzzy matches to find common entities across facets
3. Merges matching entities (union akas, use longest name as canonical)
4. Saves journal-level entities to entities/<id>/entity.json
5. Saves facet relationships to facets/<facet>/entities/<id>/entity.json

The legacy entities.jsonl files are NOT modified - they remain in place.

Use --dry-run to preview changes without writing files.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from think.entities import (
    entity_slug,
    parse_entity_file,
    save_facet_relationship,
    save_journal_entity,
)
from think.utils import get_journal, now_ms, setup_cli

logger = logging.getLogger(__name__)

# Fuzzy matching threshold (0-100)
FUZZY_THRESHOLD = 90


@dataclass
class CanonicalEntity:
    """A merged entity representing one logical identity across facets."""

    name: str
    entity_type: str
    aka: set[str] = field(default_factory=set)
    is_principal: bool = False
    source_facets: set[str] = field(default_factory=set)

    # Track which original entities merged into this one
    merged_from: list[tuple[str, str]] = field(default_factory=list)  # (facet, name)

    @property
    def id(self) -> str:
        """Generate ID slug from canonical name."""
        return entity_slug(self.name)

    def to_journal_entity(self) -> dict[str, Any]:
        """Convert to journal entity dict for saving."""
        entity = {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type,
            "created_at": now_ms(),
        }
        if self.aka:
            # Sort for deterministic output
            entity["aka"] = sorted(self.aka)
        if self.is_principal:
            entity["is_principal"] = True
        return entity


@dataclass
class FacetEntity:
    """An entity as it exists in a specific facet."""

    facet: str
    name: str
    entity_type: str
    description: str
    aka: list[str]
    is_principal: bool
    detached: bool
    raw: dict[str, Any]  # All original fields

    # Set during merging - which canonical entity this belongs to
    canonical_id: str | None = None


def load_all_legacy_entities(journal_path: Path) -> list[FacetEntity]:
    """Load all entities from legacy facets/*/entities.jsonl files."""
    facets_dir = journal_path / "facets"
    if not facets_dir.exists():
        return []

    all_entities = []
    skipped_detached = 0

    for facet_path in sorted(facets_dir.iterdir()):
        if not facet_path.is_dir():
            continue

        entities_file = facet_path / "entities.jsonl"
        if not entities_file.exists():
            continue

        facet_name = facet_path.name
        entities = parse_entity_file(str(entities_file))

        for entity in entities:
            # Skip detached entities - they're soft-deleted and shouldn't be migrated
            if entity.get("detached"):
                skipped_detached += 1
                continue

            aka_raw = entity.get("aka", [])
            aka = aka_raw if isinstance(aka_raw, list) else []

            facet_entity = FacetEntity(
                facet=facet_name,
                name=entity.get("name", ""),
                entity_type=entity.get("type", ""),
                description=entity.get("description", ""),
                aka=aka,
                is_principal=entity.get("is_principal", False),
                detached=False,  # Only non-detached entities reach here
                raw=entity,
            )
            all_entities.append(facet_entity)

    return all_entities, skipped_detached


def find_matching_canonical(
    entity: FacetEntity,
    canonicals: list[CanonicalEntity],
    threshold: int = FUZZY_THRESHOLD,
) -> CanonicalEntity | None:
    """Find a canonical entity that matches this facet entity.

    Only matches on NAME - akas are not used for matching (too many false positives).
    If names match, akas get merged into the canonical.
    """
    if not entity.name:
        return None

    entity_name_lower = entity.name.lower()

    for canonical in canonicals:
        # Exact match (case-insensitive)
        if entity_name_lower == canonical.name.lower():
            return canonical

        # Fuzzy match on names only
        try:
            from rapidfuzz import fuzz

            if fuzz.token_sort_ratio(entity.name, canonical.name) >= threshold:
                return canonical
        except ImportError:
            pass

    return None


def merge_into_canonical(canonical: CanonicalEntity, entity: FacetEntity) -> None:
    """Merge a facet entity into an existing canonical entity.

    - Uses the longest name as canonical name
    - Unions all akas
    - Preserves is_principal if either has it
    - Logs the merge
    """
    old_name = canonical.name

    # Use longest name as canonical
    if len(entity.name) > len(canonical.name):
        # Move old canonical name to aka
        canonical.aka.add(canonical.name)
        canonical.name = entity.name
    elif entity.name.lower() != canonical.name.lower():
        # Different name, add as aka
        canonical.aka.add(entity.name)

    # Add all entity akas
    for aka in entity.aka:
        if aka and aka.lower() != canonical.name.lower():
            canonical.aka.add(aka)

    # Remove canonical name from akas if present
    canonical.aka.discard(canonical.name)
    # Case-insensitive removal
    canonical.aka = {a for a in canonical.aka if a.lower() != canonical.name.lower()}

    # Preserve is_principal
    if entity.is_principal:
        canonical.is_principal = True

    # Track source facet
    canonical.source_facets.add(entity.facet)

    # Track merge
    canonical.merged_from.append((entity.facet, entity.name))

    if old_name != canonical.name:
        logger.info(
            f"  Merged '{entity.name}' from {entity.facet} into '{canonical.name}' (name changed from '{old_name}')"
        )
    else:
        logger.info(
            f"  Merged '{entity.name}' from {entity.facet} into '{canonical.name}'"
        )


def create_canonical(entity: FacetEntity) -> CanonicalEntity:
    """Create a new canonical entity from a facet entity."""
    aka_set = set(entity.aka) if entity.aka else set()
    # Remove the name from akas if present
    aka_set.discard(entity.name)
    aka_set = {a for a in aka_set if a.lower() != entity.name.lower()}

    canonical = CanonicalEntity(
        name=entity.name,
        entity_type=entity.entity_type,
        aka=aka_set,
        is_principal=entity.is_principal,
        source_facets={entity.facet},
        merged_from=[(entity.facet, entity.name)],
    )
    return canonical


def build_facet_relationship(entity: FacetEntity) -> dict[str, Any]:
    """Extract relationship-only fields from a facet entity.

    Journal-level fields (id, name, type, aka, is_principal, created_at) are excluded.
    """
    journal_fields = {"id", "name", "type", "aka", "is_principal", "created_at"}

    relationship = {}
    for key, value in entity.raw.items():
        if key not in journal_fields:
            relationship[key] = value

    return relationship


def migrate_entities(
    journal_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the entity migration.

    Returns a summary dict with statistics.
    """
    print(f"Loading entities from {journal_path}/facets/*/entities.jsonl...")

    # Phase 1: Load all legacy entities
    facet_entities, skipped_detached = load_all_legacy_entities(journal_path)
    print(f"  Loaded {len(facet_entities)} entities from legacy files")
    if skipped_detached:
        print(f"  Skipped {skipped_detached} detached entities")

    if not facet_entities:
        print("  No entities to migrate.")
        return {"loaded": 0, "canonicals": 0, "merges": 0, "relationships": 0}

    # Phase 2: Fuzzy merge to build canonical entities
    print("\nBuilding canonical entities with fuzzy matching...")
    canonicals: list[CanonicalEntity] = []
    merge_count = 0

    for entity in facet_entities:
        if not entity.name:
            continue

        # Try to find existing canonical
        match = find_matching_canonical(entity, canonicals)

        if match:
            # Merge into existing
            merge_into_canonical(match, entity)
            entity.canonical_id = match.id
            merge_count += 1
        else:
            # Create new canonical
            canonical = create_canonical(entity)
            canonicals.append(canonical)
            entity.canonical_id = canonical.id
            logger.info(f"  Created canonical '{canonical.name}' from {entity.facet}")

    print(f"  Created {len(canonicals)} canonical entities")
    print(f"  Performed {merge_count} merges")

    # Log type conflicts
    type_conflicts = []
    for canonical in canonicals:
        types_seen = set()
        for facet, name in canonical.merged_from:
            for fe in facet_entities:
                if fe.facet == facet and fe.name == name:
                    types_seen.add(fe.entity_type)
        if len(types_seen) > 1:
            type_conflicts.append((canonical.name, types_seen))

    if type_conflicts:
        print(f"\n  Type conflicts detected ({len(type_conflicts)}):")
        for name, types in type_conflicts:
            print(f"    '{name}': {types}")

    # Phase 3: Save new structure
    if dry_run:
        print("\n[DRY RUN] Would save the following:")
        print(f"  - {len(canonicals)} journal entities to entities/<id>/entity.json")

        rel_count = sum(1 for e in facet_entities if e.name)
        print(
            f"  - {rel_count} facet relationships to facets/<facet>/entities/<id>/entity.json"
        )

        print("\nCanonical entities that would be created:")
        for canonical in sorted(canonicals, key=lambda c: c.name.lower()):
            aka_str = (
                f" (aka: {', '.join(sorted(canonical.aka))})" if canonical.aka else ""
            )
            facets_str = f" [{', '.join(sorted(canonical.source_facets))}]"
            print(
                f"  - {canonical.id}: {canonical.name} [{canonical.entity_type}]{aka_str}{facets_str}"
            )

        return {
            "loaded": len(facet_entities),
            "canonicals": len(canonicals),
            "merges": merge_count,
            "relationships": rel_count,
            "dry_run": True,
        }

    print("\nSaving journal entities...")
    saved_canonicals = 0
    for canonical in canonicals:
        journal_entity = canonical.to_journal_entity()
        save_journal_entity(journal_entity)
        saved_canonicals += 1
        logger.debug(f"  Saved journal entity: {canonical.id}")

    print(f"  Saved {saved_canonicals} journal entities")

    print("\nSaving facet relationships...")
    saved_relationships = 0

    # Build canonical lookup by merged_from
    canonical_lookup: dict[tuple[str, str], CanonicalEntity] = {}
    for canonical in canonicals:
        for facet, name in canonical.merged_from:
            canonical_lookup[(facet, name)] = canonical

    for entity in facet_entities:
        if not entity.name:
            continue

        # Find the canonical this entity belongs to
        canonical = canonical_lookup.get((entity.facet, entity.name))
        if not canonical:
            logger.warning(f"  No canonical found for {entity.facet}/{entity.name}")
            continue

        # Build and save relationship
        relationship = build_facet_relationship(entity)
        save_facet_relationship(entity.facet, canonical.id, relationship)
        saved_relationships += 1
        logger.debug(f"  Saved relationship: {entity.facet}/{canonical.id}")

    print(f"  Saved {saved_relationships} facet relationships")

    return {
        "loaded": len(facet_entities),
        "canonicals": saved_canonicals,
        "merges": merge_count,
        "relationships": saved_relationships,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    args = setup_cli(parser)

    journal_path = Path(get_journal())

    print("=" * 60)
    print("Entity Migration: Legacy to Journal-Wide Structure")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be written]\n")

    results = migrate_entities(journal_path, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"  Entities loaded:       {results['loaded']}")
    print(f"  Canonical entities:    {results['canonicals']}")
    print(f"  Merges performed:      {results['merges']}")
    print(f"  Relationships created: {results['relationships']}")

    if results.get("dry_run"):
        print("\n[DRY RUN] No files were written. Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
