# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for entity management.

Auto-discovered by ``think.call`` and mounted as ``sol call entities ...``.
"""

import re
import shutil
from pathlib import Path

import typer

from think.entities.core import entity_slug, is_valid_entity_type
from think.entities.journal import (
    clear_journal_entity_cache,
    create_journal_entity,
    load_journal_entity,
    save_journal_entity,
)
from think.entities.loading import clear_entity_loading_cache, load_entities
from think.entities.matching import resolve_entity, validate_aka_uniqueness
from think.entities.observations import (
    add_observation,
    load_observations,
    save_observations,
)
from think.entities.relationships import (
    clear_relationship_caches,
    entity_memory_path,
    load_facet_relationship,
    save_facet_relationship,
)
from think.entities.saving import (
    save_detected_entity,
    update_detected_entity,
)
from think.facets import log_call_action
from think.indexer.journal import (
    get_entity_intelligence,
    get_entity_strength,
    search_entities,
)
from think.utils import (
    get_journal,
    now_ms,
    require_solstone,
    resolve_sol_day,
    resolve_sol_facet,
)

app = typer.Typer(help="Entity management.")


@app.callback()
def _require_up() -> None:
    require_solstone()


def _clear_all_caches():
    """Clear all underlying think entity caches."""
    clear_entity_loading_cache()
    clear_relationship_caches()
    clear_journal_entity_cache()


def _resolve_or_exit(facet: str, entity: str) -> dict:
    """Resolve entity or exit with CLI error."""
    resolved, candidates = resolve_entity(facet, entity)
    if resolved:
        return resolved

    blocked_match, _ = resolve_entity(facet, entity, include_blocked=True)
    if blocked_match and blocked_match.get("blocked"):
        name = blocked_match.get("name", entity)
        typer.echo(f"Error: Entity '{name}' is blocked.", err=True)
        raise typer.Exit(1)

    if candidates:
        names = ", ".join(c.get("name", "") for c in candidates[:3])
        typer.echo(
            f"Error: Entity '{entity}' not found. Did you mean: {names}", err=True
        )
        raise typer.Exit(1)

    typer.echo(f"Error: Entity '{entity}' not found in facet '{facet}'.", err=True)
    raise typer.Exit(1)


def _validate_facet_or_exit(facet: str, label: str) -> None:
    """Exit if the facet directory does not exist."""
    facet_path = Path(get_journal()) / "facets" / facet
    if not facet_path.is_dir():
        typer.echo(
            f"Error: Facet '{facet}' ({label}) does not exist.",
            err=True,
        )
        raise typer.Exit(1)


@app.command("list")
def list_entities(
    facet: str | None = typer.Argument(None, help="Facet name (or set SOL_FACET)."),
    day: str | None = typer.Option(
        None, "--day", "-d", help="Day (YYYYMMDD) for detected entities."
    ),
) -> None:
    """List entities for a facet."""
    facet = resolve_sol_facet(facet)
    entities = load_entities(facet, day)

    if not entities:
        typer.echo("No entities found.")
        return

    label = f"detected for {day}" if day else "attached"
    typer.echo(f"{len(entities)} {label} entities:")
    for e in entities:
        typer.echo(f"  - {e.get('name')} ({e.get('type')}): {e.get('description', '')}")


@app.command("move")
def move_entity(
    entity: str = typer.Argument(help="Entity name or partial match."),
    from_facet: str = typer.Option(..., "--from", help="Source facet."),
    to_facet: str = typer.Option(..., "--to", help="Destination facet."),
    merge: bool = typer.Option(
        False,
        "--merge",
        help="Merge if entity already exists in destination.",
    ),
    consent: bool = typer.Option(
        False,
        "--consent",
        help="Assert that explicit user approval was obtained before calling this command (agent audit trail).",
    ),
) -> None:
    """Move an entity from one facet to another."""
    _validate_facet_or_exit(from_facet, "--from")
    _validate_facet_or_exit(to_facet, "--to")

    resolved = _resolve_or_exit(from_facet, entity)
    entity_name = str(resolved.get("name", entity))
    entity_id = entity_slug(entity_name)
    src_dir = entity_memory_path(from_facet, entity_name)
    dst_dir = entity_memory_path(to_facet, entity_name)

    if not src_dir.exists():
        typer.echo("Error: Entity data directory not found in source facet.", err=True)
        raise typer.Exit(1)

    if dst_dir.exists() and not merge:
        typer.echo(
            "Error: Entity already exists in destination facet. Use --merge to merge.",
            err=True,
        )
        raise typer.Exit(1)

    if dst_dir.exists():
        src_relationship = load_facet_relationship(from_facet, entity_id)
        dst_relationship = load_facet_relationship(to_facet, entity_id)
        if src_relationship is not None and dst_relationship is None:
            save_facet_relationship(to_facet, entity_id, src_relationship)

        src_obs = load_observations(from_facet, entity_name)
        dst_obs = load_observations(to_facet, entity_name)

        existing_keys = {(o["content"], o.get("observed_at")) for o in dst_obs}
        merged = list(dst_obs) + [
            o
            for o in src_obs
            if (o["content"], o.get("observed_at")) not in existing_keys
        ]
        save_observations(to_facet, entity_name, merged)

        shutil.rmtree(str(src_dir))
    else:
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_dir), str(dst_dir))

    params: dict[str, object] = {
        "entity": entity_name,
        "moved_from": from_facet,
        "moved_to": to_facet,
    }
    if merge:
        params["merge"] = True
    if consent:
        params["consent"] = True
    log_call_action(facet=from_facet, action="entity_move", params=params)
    typer.echo(f"Moved entity '{entity_name}' from '{from_facet}' to '{to_facet}'.")


@app.command("detect")
def detect_entity(
    type_: str = typer.Argument(metavar="TYPE", help="Entity type."),
    entity: str = typer.Argument(help="Entity name or identifier."),
    description: str = typer.Argument(help="Description."),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name (or set SOL_FACET)."
    ),
    day: str | None = typer.Option(
        None, "--day", "-d", help="Day (YYYYMMDD, or set SOL_DAY)."
    ),
) -> None:
    """Record a detected entity for a day in a facet."""
    facet = resolve_sol_facet(facet)
    day = resolve_sol_day(day)
    if not is_valid_entity_type(type_):
        typer.echo(f"Error: Invalid entity type '{type_}'.", err=True)
        raise typer.Exit(1)

    resolved, _ = resolve_entity(facet, entity)

    if not resolved:
        blocked_match, _ = resolve_entity(facet, entity, include_blocked=True)
        if blocked_match and blocked_match.get("blocked"):
            name = blocked_match.get("name", entity)
            typer.echo(f"Error: Entity '{name}' is blocked.", err=True)
            raise typer.Exit(1)

    name = resolved.get("name", entity) if resolved else entity

    try:
        save_detected_entity(facet, day, type_, name, description)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    log_call_action(
        facet=facet,
        action="entity_detect",
        params={
            "type": type_,
            "entity": entity,
            "name": name,
            "description": description,
        },
        day=day,
    )
    typer.echo(f"Entity '{name}' detected for {day}.")


@app.command("attach")
def attach_entity(
    type_: str = typer.Argument(metavar="TYPE", help="Entity type."),
    entity: str = typer.Argument(help="Entity name."),
    description: str = typer.Argument(help="Description."),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name (or set SOL_FACET)."
    ),
) -> None:
    """Attach an entity permanently to a facet."""
    facet = resolve_sol_facet(facet)
    if not is_valid_entity_type(type_):
        typer.echo(f"Error: Invalid entity type '{type_}'.", err=True)
        raise typer.Exit(1)

    resolved, _ = resolve_entity(
        facet, entity, include_detached=True, include_blocked=True
    )

    if resolved and resolved.get("blocked"):
        name = resolved.get("name", entity)
        typer.echo(f"Error: Entity '{name}' is blocked.", err=True)
        raise typer.Exit(1)

    if resolved and resolved.get("detached"):
        name = resolved.get("name", entity)
        typer.echo(
            f"Error: Entity '{name}' was previously removed by the user.", err=True
        )
        raise typer.Exit(1)

    if resolved:
        typer.echo(f"Entity '{resolved.get('name')}' already attached.")
        return

    name = entity
    now = now_ms()
    entity_id = entity_slug(name)

    # Create journal entity (identity record) if it doesn't exist
    load_journal_entity(entity_id) or create_journal_entity(
        entity_id=entity_id,
        name=name,
        entity_type=type_,
    )

    # Create facet relationship (per-entity file, no load-all needed)
    save_facet_relationship(
        facet,
        entity_id,
        {
            "entity_id": entity_id,
            "description": description,
            "attached_at": now,
            "updated_at": now,
        },
    )

    log_call_action(
        facet=facet,
        action="entity_attach",
        params={
            "type": type_,
            "entity": entity,
            "name": name,
            "description": description,
        },
    )
    typer.echo(f"Entity '{name}' attached.")


@app.command("update")
def update_entity(
    entity: str = typer.Argument(help="Entity name or identifier."),
    description: str = typer.Argument(help="New description."),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name (or set SOL_FACET)."
    ),
    day: str | None = typer.Option(
        None, "--day", "-d", help="Day for detected entities."
    ),
) -> None:
    """Update an entity description."""
    facet = resolve_sol_facet(facet)
    if day is None:
        resolved = _resolve_or_exit(facet, entity)
        resolved_name = resolved.get("name", entity)
        entity_id = resolved.get("id", entity_slug(resolved_name))

        # Load and update only the target entity's relationship file
        relationship = load_facet_relationship(facet, entity_id)
        if relationship is None:
            typer.echo(f"Error: Entity '{resolved_name}' not found.", err=True)
            raise typer.Exit(1)

        relationship["description"] = description
        relationship["updated_at"] = now_ms()
        save_facet_relationship(facet, entity_id, relationship)
        log_call_action(
            facet=facet,
            action="entity_update",
            params={
                "entity": entity,
                "name": resolved_name,
                "description": description,
            },
        )
        typer.echo(f"Entity '{resolved_name}' updated.")
        return

    try:
        update_detected_entity(facet, day, entity, description)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    log_call_action(
        facet=facet,
        action="entity_update",
        params={"entity": entity, "description": description},
        day=day,
    )
    typer.echo(f"Entity '{entity}' updated for {day}.")


@app.command("aka")
def add_aka(
    entity: str = typer.Argument(help="Entity name or identifier."),
    aka_value: str = typer.Argument(metavar="AKA", help="Alias to add."),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name (or set SOL_FACET)."
    ),
) -> None:
    """Add an alias to an attached entity."""
    facet = resolve_sol_facet(facet)
    resolved = _resolve_or_exit(facet, entity)
    resolved_name = resolved.get("name", "")

    base_name = re.sub(r"\s*\([^)]+\)", "", resolved_name).strip()
    first_word = base_name.split()[0] if base_name else None
    if first_word and aka_value.lower() == first_word.lower():
        typer.echo(
            f"Alias '{aka_value}' is the first word of '{resolved_name}' (skipped)."
        )
        return

    aka_list = resolved.get("aka", [])
    if not isinstance(aka_list, list):
        aka_list = []

    if aka_value in aka_list:
        typer.echo(f"Alias '{aka_value}' already exists for '{resolved_name}'.")
        return

    # Validate uniqueness across all entities in facet
    entities = load_entities(
        facet, day=None, include_detached=True, include_blocked=True
    )

    conflict = validate_aka_uniqueness(
        aka_value, entities, exclude_entity_name=resolved_name
    )
    if conflict:
        typer.echo(
            f"Error: Alias '{aka_value}' conflicts with entity '{conflict}'.", err=True
        )
        raise typer.Exit(1)

    entity_id = resolved.get("id", entity_slug(resolved_name))

    # Update journal entity aka (identity-level, not facet-specific)
    journal_entity = load_journal_entity(entity_id)
    if journal_entity:
        existing_aka = set(journal_entity.get("aka", []))
        existing_aka.add(aka_value)
        journal_entity["aka"] = sorted(existing_aka)
        save_journal_entity(journal_entity)

    log_call_action(
        facet=facet,
        action="entity_add_aka",
        params={"entity": entity, "name": resolved_name, "aka": aka_value},
    )
    typer.echo(f"Added alias '{aka_value}' to '{resolved_name}'.")


@app.command("observations")
def list_observations(
    entity: str = typer.Argument(help="Entity name or identifier."),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name (or set SOL_FACET)."
    ),
) -> None:
    """List observations for an attached entity."""
    facet = resolve_sol_facet(facet)
    resolved = _resolve_or_exit(facet, entity)
    resolved_name = resolved.get("name", "")
    obs = load_observations(facet, resolved_name)

    if not obs:
        typer.echo(f"No observations for '{resolved_name}'.")
        return

    typer.echo(f"{len(obs)} observations for '{resolved_name}':")
    for i, o in enumerate(obs, 1):
        typer.echo(f"  {i}. {o.get('content', '')}")


@app.command("observe")
def observe_entity(
    entity: str = typer.Argument(help="Entity name or identifier."),
    content: str = typer.Argument(help="Observation content."),
    facet: str | None = typer.Option(
        None, "--facet", "-f", help="Facet name (or set SOL_FACET)."
    ),
    source_day: str | None = typer.Option(None, "--source-day", help="Day (YYYYMMDD)."),
) -> None:
    """Add an observation to an attached entity."""
    facet = resolve_sol_facet(facet)
    resolved = _resolve_or_exit(facet, entity)
    resolved_name = resolved.get("name", "")

    try:
        add_observation(facet, resolved_name, content, source_day)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    log_call_action(
        facet=facet,
        action="entity_observe",
        params={
            "entity": entity,
            "name": resolved_name,
            "content": content,
        },
    )
    typer.echo(f"Observation added to '{resolved_name}'.")


@app.command("strength")
def entity_strength(
    facet: str | None = typer.Option(None, "--facet", "-f", help="Filter by facet."),
    since: str | None = typer.Option(None, "--since", help="Signals since YYYYMMDD."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results."),
) -> None:
    """Rank entities by relationship strength score."""
    results = get_entity_strength(facet=facet, since=since, limit=limit)
    if not results:
        typer.echo("No entity signals found.")
        return
    for r in results:
        name = r.get("entity_name", "")
        score = r.get("score", 0)
        eid = r.get("entity_id", "")
        label = f"{name} ({eid})" if eid and eid != entity_slug(name) else name
        typer.echo(f"  {score:>8.1f}  {label}")
        typer.echo(
            f"           kg={r['kg_edge_count']} co={r['co_occurrence']} pho={r['photo_count']} rec={r['recency']:.3f} obs={r['observation_depth']} fac={r['facet_breadth']}"
        )


@app.command("search")
def entity_search(
    query: str | None = typer.Option(None, "--query", "-q", help="Text search."),
    type_: str | None = typer.Option(None, "--type", "-t", help="Entity type."),
    facet: str | None = typer.Option(None, "--facet", "-f", help="Filter by facet."),
    since: str | None = typer.Option(None, "--since", help="Signals since YYYYMMDD."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results."),
) -> None:
    """Search entities by text, type, facet, or activity."""
    results = search_entities(
        query=query,
        entity_type=type_,
        facet=facet,
        since=since,
        limit=limit,
    )
    if not results:
        typer.echo("No entities found.")
        return
    typer.echo(f"{len(results)} entities:")
    for e in results:
        facets = ", ".join(e.get("facets", []))
        typer.echo(f"  - {e['name']} ({e['type']}): {e['description']}")
        if facets:
            typer.echo(f"    facets: {facets} | signals: {e.get('signal_count', 0)}")


@app.command("intelligence")
def entity_intel(
    entity: str = typer.Argument(help="Entity name or identifier."),
    facet: str | None = typer.Option(None, "--facet", "-f", help="Filter by facet."),
    brief: bool = typer.Option(
        False, "--brief", "-b", help="Truncate activity and network to 20 items."
    ),
) -> None:
    """Get a full intelligence briefing for an entity."""
    import json as _json

    result = get_entity_intelligence(entity, facet=facet, brief=brief)
    if result is None:
        typer.echo(f"Error: Entity '{entity}' not found.", err=True)
        raise typer.Exit(1)
    typer.echo(_json.dumps(result, indent=2))
