# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for the awareness system.

Auto-discovered by ``think.call`` and mounted as ``sol call awareness ...``.
"""

import json

import typer

app = typer.Typer(help="Awareness system — solstone's self-knowledge.")


@app.command("status")
def status(
    section: str | None = typer.Argument(
        None, help="Section to read (e.g., 'onboarding'). Omit for all."
    ),
) -> None:
    """Show current awareness state."""
    from think.awareness import get_current

    state = get_current()
    if not state:
        typer.echo("No awareness state yet.")
        return

    if section:
        value = state.get(section)
        if value is None:
            typer.echo(f"No '{section}' state.")
            return
        typer.echo(json.dumps(value, indent=2))
    else:
        typer.echo(json.dumps(state, indent=2))


@app.command("onboarding")
def onboarding_cmd(
    path: str | None = typer.Option(
        None, "--path", "-p", help="Onboarding path: 'a' (observe) or 'b' (interview)."
    ),
    skip: bool = typer.Option(False, "--skip", help="Skip onboarding."),
    complete: bool = typer.Option(
        False, "--complete", help="Mark onboarding complete."
    ),
) -> None:
    """Read or update onboarding state."""
    from think.awareness import (
        complete_onboarding,
        get_onboarding,
        skip_onboarding,
        start_onboarding,
    )

    if skip:
        state = skip_onboarding()
        typer.echo(json.dumps(state, indent=2))
        return

    if complete:
        state = complete_onboarding()
        typer.echo(json.dumps(state, indent=2))
        return

    if path:
        if path not in ("a", "b"):
            typer.echo("Error: --path must be 'a' or 'b'", err=True)
            raise typer.Exit(1)
        state = start_onboarding(path)
        typer.echo(json.dumps(state, indent=2))
        return

    # No flags — read current state
    state = get_onboarding()
    if not state:
        typer.echo("No onboarding state yet.")
        return
    typer.echo(json.dumps(state, indent=2))


@app.command("imports")
def imports_cmd(
    record: str | None = typer.Option(
        None, "--record", "-r", help="Record a completed import (source type)."
    ),
    declined: bool = typer.Option(
        False, "--declined", help="Record that user declined import offer."
    ),
    nudge: bool = typer.Option(
        False, "--nudge", help="Record that triage nudged about imports."
    ),
) -> None:
    """Read or update import tracking state."""
    from think.awareness import (
        get_imports,
        record_import,
        record_import_nudge,
        record_import_offer_declined,
    )

    if record:
        state = record_import(record)
        typer.echo(json.dumps(state, indent=2))
        return

    if declined:
        state = record_import_offer_declined()
        typer.echo(json.dumps(state, indent=2))
        return

    if nudge:
        state = record_import_nudge()
        typer.echo(json.dumps(state, indent=2))
        return

    # No flags — read current state
    state = get_imports()
    typer.echo(json.dumps(state, indent=2))


@app.command("log-read")
def log_read_cmd(
    day: str | None = typer.Argument(
        None, help="Day in YYYYMMDD format (defaults to today)."
    ),
    kind: str | None = typer.Option(
        None, "--kind", "-k", help="Filter by entry kind (e.g., 'observation')."
    ),
    limit: int = typer.Option(
        0, "--limit", "-n", help="Max entries to return (0=all)."
    ),
) -> None:
    """Read entries from the daily awareness log."""
    from think.awareness import _today, read_log

    target_day = day or _today()
    entries = read_log(target_day)

    if kind:
        entries = [e for e in entries if e.get("kind") == kind]

    if limit > 0:
        entries = entries[-limit:]

    if not entries:
        typer.echo("No entries found.")
        return

    typer.echo(json.dumps(entries, indent=2))


@app.command("log")
def log_cmd(
    kind: str = typer.Argument(
        help="Entry type: state, observation, nudge, interaction."
    ),
    message: str | None = typer.Argument(None, help="Human-readable message."),
    key: str | None = typer.Option(
        None, "--key", "-k", help="Dotted key for state entries."
    ),
    data: str | None = typer.Option(None, "--data", "-d", help="JSON data payload."),
) -> None:
    """Append an entry to the daily awareness log."""
    from think.awareness import append_log

    parsed_data = None
    if data:
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            typer.echo("Error: --data must be valid JSON", err=True)
            raise typer.Exit(1)

    entry = append_log(kind, key=key, message=message, data=parsed_data)
    typer.echo(json.dumps(entry, indent=2))
