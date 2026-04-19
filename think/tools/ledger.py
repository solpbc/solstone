# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import dataclasses
import json as jsonlib

import typer

from think.surfaces import ledger as ledger_surface
from think.utils import require_solstone

app = typer.Typer(help="Ledger: commitments ↔ closures view", no_args_is_help=True)


@app.callback()
def callback() -> None:
    require_solstone()


def _parse_facets_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _echo_json(payload: object) -> None:
    typer.echo(jsonlib.dumps(payload, indent=2, sort_keys=False))


def _render_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        return
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    typer.echo(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    typer.echo("  ".join("-" * width for width in widths))
    for row in rows:
        typer.echo(
            "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        )


def _item_summary(item: ledger_surface.LedgerItem) -> str:
    if item.counterparty:
        return f"{item.owner}: {item.summary} -> {item.counterparty}"
    return f"{item.owner}: {item.summary}"


def _render_items(items: list[ledger_surface.LedgerItem]) -> None:
    if not items:
        typer.echo("No ledger items found.")
        return
    rows = [
        [
            item.id,
            item.state,
            str(item.age_days),
            _item_summary(item),
            item.when or "",
            str(item.opened_at),
            str(item.closed_at or ""),
        ]
        for item in items
    ]
    _render_table(
        ["id", "state", "age_days", "summary", "when", "opened_at", "closed_at"],
        rows,
    )


def _render_decisions(items: list[ledger_surface.Decision]) -> None:
    if not items:
        typer.echo("No decisions found.")
        return
    rows = [
        [item.id, item.day, item.owner, item.action, item.context] for item in items
    ]
    _render_table(["id", "day", "owner", "action", "context"], rows)


@app.command("list")
def list_cmd(
    state: str = typer.Option("open"),
    owner: str | None = typer.Option(None),
    counterparty: str | None = typer.Option(None),
    age_days_gte: int | None = typer.Option(None, "--age-days-gte"),
    closed_since: str | None = typer.Option(None, "--closed-since"),
    top: int | None = typer.Option(None, "--top"),
    sort: str | None = typer.Option(None),
    facets: str | None = typer.Option(None, help="csv"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """List ledger items."""
    if sort is not None and sort not in {
        "age_days_desc",
        "opened_at_desc",
        "closed_at_desc",
    }:
        raise typer.BadParameter(
            "sort must be one of age_days_desc, opened_at_desc, closed_at_desc"
        )
    try:
        items = ledger_surface.list(
            state=state,
            owner=owner,
            counterparty=counterparty,
            age_days_gte=age_days_gte,
            closed_since=closed_since,
            top=top,
            sort=sort,
            facets=_parse_facets_csv(facets),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json:
        _echo_json([dataclasses.asdict(item) for item in items])
        return
    _render_items(items)


@app.command("get")
def get_cmd(item_id: str, json: bool = typer.Option(False, "--json")) -> None:
    """Fetch one ledger item."""
    item = ledger_surface.get(item_id)
    if item is None:
        typer.echo(f"ledger item not found: {item_id}", err=True)
        raise typer.Exit(1)
    if json:
        _echo_json(dataclasses.asdict(item))
        return
    _render_items([item])


@app.command("close")
def close_cmd(
    item_id: str,
    note: str = typer.Option(..., "--note"),
    as_state: str = typer.Option("closed", "--as"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Manually close or drop one ledger item."""
    if as_state not in {"closed", "dropped"}:
        raise typer.BadParameter("as_state must be 'closed' or 'dropped'")
    try:
        item = ledger_surface.close(item_id, note=note, as_state=as_state)
    except KeyError:
        typer.echo(f"ledger item not found: {item_id}", err=True)
        raise typer.Exit(1) from None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json:
        _echo_json(dataclasses.asdict(item))
        return
    _render_items([item])


@app.command("decisions")
def decisions_cmd(
    owner: str | None = typer.Option(None),
    since: str | None = typer.Option(None),
    involving: str | None = typer.Option(None),
    top: int | None = typer.Option(None),
    facets: str | None = typer.Option(None, help="csv"),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """List deduplicated decisions."""
    try:
        items = ledger_surface.decisions(
            owner=owner,
            since=since,
            involving=involving,
            top=top,
            facets=_parse_facets_csv(facets),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json:
        _echo_json([dataclasses.asdict(item) for item in items])
        return
    _render_decisions(items)
