# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for entity management.

Auto-discovered by ``think.call`` and mounted as ``sol call entities ...``.
"""

import re

import typer

from think.entities.core import entity_slug, is_valid_entity_type
from think.entities.journal import (
    get_or_create_journal_entity,
    load_journal_entity,
    save_journal_entity,
)
from think.entities.loading import load_entities
from think.entities.matching import resolve_entity, validate_aka_uniqueness
from think.entities.observations import add_observation, load_observations
from think.entities.relationships import (
    load_facet_relationship,
    save_facet_relationship,
)
from think.entities.saving import (
    save_detected_entity,
    update_detected_entity,
)
from think.facets import log_call_action
from think.utils import now_ms, resolve_sol_day, resolve_sol_facet

app = typer.Typer(help="Entity management.")


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
    get_or_create_journal_entity(
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
