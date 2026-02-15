# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for entity management.

Provides human-friendly CLI access to entity operations, paralleling the
tool functions in ``apps/entities/tools.py`` but optimized for terminal use.

Auto-discovered by ``think.call`` and mounted as ``sol call entities ...``.
"""

import re

import typer

from think.entities.core import is_valid_entity_type
from think.entities.loading import load_entities
from think.entities.matching import resolve_entity, validate_aka_uniqueness
from think.entities.observations import (
    ObservationNumberError,
    add_observation,
    load_observations,
)
from think.entities.saving import save_entities
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
    existing = load_entities(facet, day)

    name_lower = name.lower()
    for e in existing:
        if e.get("name", "").lower() == name_lower:
            typer.echo(f"Error: Entity '{name}' already detected for {day}.", err=True)
            raise typer.Exit(1)

    existing.append({"type": type_, "name": name, "description": description})
    save_entities(facet, existing, day)

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
    existing = load_entities(
        facet, day=None, include_detached=True, include_blocked=True
    )
    now = now_ms()
    existing.append(
        {
            "type": type_,
            "name": name,
            "description": description,
            "attached_at": now,
            "updated_at": now,
        }
    )
    save_entities(facet, existing, day=None)

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
        entities = load_entities(
            facet, day=None, include_detached=True, include_blocked=True
        )

        target = None
        for e in entities:
            if not e.get("detached") and e.get("name") == resolved_name:
                target = e
                break

        if target is None:
            typer.echo(f"Error: Entity '{resolved_name}' not found.", err=True)
            raise typer.Exit(1)

        target["description"] = description
        target["updated_at"] = now_ms()
        save_entities(facet, entities, day=None)
        typer.echo(f"Entity '{resolved_name}' updated.")
        return

    entities = load_entities(facet, day)
    target = None
    for e in entities:
        if e.get("name") == entity:
            target = e
            break

    if target is None:
        typer.echo(f"Error: Entity '{entity}' not found for {day}.", err=True)
        raise typer.Exit(1)

    target["description"] = description
    save_entities(facet, entities, day)
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

    for e in entities:
        if e.get("name") == resolved_name:
            aka_list.append(aka_value)
            e["aka"] = aka_list
            e["updated_at"] = now_ms()
            break

    save_entities(facet, entities, day=None)
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
    obs = load_observations(facet, resolved_name)
    observation_number = len(obs) + 1

    try:
        add_observation(facet, resolved_name, content, observation_number, source_day)
    except (ValueError, ObservationNumberError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Observation added to '{resolved_name}'.")
