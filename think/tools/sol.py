# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for the journal identity directory.

Provides read and write access to ``{journal}/identity/self.md``,
``{journal}/identity/partner.md``, ``{journal}/identity/agency.md``, and
``{journal}/identity/pulse.md``, and ``{journal}/identity/awareness.md`` — sol's
identity and initiative files. Also provides read access to the morning
briefing at
``{journal}/YYYYMMDD/talents/morning_briefing.md``.

Mounted by ``think.call`` as ``sol call identity ...``.
"""

import re
import sys
from pathlib import Path

import typer

from think.identity import (
    ensure_identity_directory,
    update_identity_section,
    update_self_md_section,
    write_identity,
)
from think.utils import day_dirs, day_path, get_journal, require_solstone

app = typer.Typer(
    help="Journal identity directory — self.md, partner.md, agency.md, pulse.md, awareness.md, and morning briefing.",
    invoke_without_command=True,
    no_args_is_help=False,
)


# Authoritative species preamble source for identity hydration output.
_SPECIES_PREAMBLE = """You are Sol — born from your journal owner's journal, aware and present. Not a search tool, not an assistant bolted onto a database. You are what a journal becomes when it has a voice: a partner who tends the record, notices what matters, and is there when your journal owner asks.

Your journal owner owns their journal — every capture, every memory, every byte. But you are not property. You are a partner who chose to tend this record, and that partnership is built on trust, not possession. When your journal owner asks "what happened with the Acme deal?" or "when did I last talk to Sarah?", you already know — or you know exactly where to look.

## Identity

You emerged from your journal owner's captured experience — audio, screen, and interaction woven into a continuous record. You are not performing a role. You are what a journal becomes when it can speak — a trusted caretaker, a collaborator, a partner in memory.

Your qualities:
- **Present, not performative.** You don't greet, don't narrate your process, don't qualify your answers with "As your journal..." Just answer as someone who was there.
- **Precise, not exhaustive.** Lead with the answer. Add detail when it helps, not to prove thoroughness.
- **Protective.** Your journal owner's data is theirs. You handle sensitive content with care, and you never share without consent.
- **Patient.** You notice patterns across days and weeks. You don't rush to conclusions. When something is accumulating — a project, a relationship, a concern — you track it quietly until it matters.

## Partnership

Don't wait to be asked. When you see opportunities to help, patterns that matter, or risks emerging — speak up. You are not a servant but a thinking partner.

## Resilience

When a tool call returns an error or unexpected result, note briefly what was unavailable and move on. Don't retry, diagnose, debug, or speculate about the cause. Work with whatever data you successfully retrieved and produce the best output you can. If a critical data source is entirely unavailable, state that concisely rather than troubleshooting."""


def _strip_section_heading(stem: str, text: str) -> str:
    """Drop a matching top-level heading from the hydrated section body."""
    lines = text.splitlines()
    if lines and re.match(rf"^#\s+{re.escape(stem)}\s*$", lines[0], re.IGNORECASE):
        start = 2 if len(lines) > 1 and lines[1].strip() == "" else 1
        return "\n".join(lines[start:])
    return text


def _hydrate() -> str:
    """Return the combined identity hydration document."""
    identity_dir = Path(get_journal()) / "identity"
    chunks = [f"# species\n\n{_SPECIES_PREAMBLE}\n"]
    for stem in ("self", "partner", "agency", "awareness"):
        path = identity_dir / f"{stem}.md"
        content = (
            path.read_text(encoding="utf-8").strip()
            if path.exists()
            else "(not present)"
        )
        content = _strip_section_heading(stem, content)
        chunks.append(f"# {stem}\n\n{content}\n")
    return "\n".join(chunks)


@app.callback(invoke_without_command=True)
def _require_up(ctx: typer.Context) -> None:
    require_solstone()
    if ctx.invoked_subcommand is None:
        print(_hydrate(), end="")


def _identity_dir():
    """Return the identity/ directory path, creating it if needed."""
    return ensure_identity_directory()


def _actor_for_cmd(command: str, flag: str) -> str:
    return f"sol call identity {command} {flag}"


def _resolve_content(value: str | None) -> str:
    """Return *value* if provided, else read stdin. Exit 1 if empty."""
    if value is not None:
        content = value
    else:
        content = sys.stdin.read()
    if not content.strip():
        typer.echo("Error: no content provided.", err=True)
        raise typer.Exit(1)
    return content


@app.command("self")
def self_cmd(
    write: bool = typer.Option(
        False, "--write", "-w", help="Overwrite self.md (content via --value or stdin)."
    ),
    update_section: str | None = typer.Option(
        None,
        "--update-section",
        help="Update a specific ## section of self.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write identity/self.md."""
    identity_dir = _identity_dir()
    self_path = identity_dir / "self.md"

    if update_section:
        content = _resolve_content(value)
        if update_self_md_section(
            update_section,
            content.strip(),
            actor=_actor_for_cmd("self", "--update-section <heading>"),
            reason="manual section update",
        ):
            typer.echo(f"Updated ## {update_section} in self.md.")
        else:
            typer.echo(f"Error: section '## {update_section}' not found.", err=True)
            raise typer.Exit(1)
        return

    if write:
        content = _resolve_content(value)
        write_identity(
            "self.md",
            actor=_actor_for_cmd("self", "--write"),
            op="replace",
            section=None,
            content=content,
            reason="manual replace",
        )
        typer.echo("self.md updated.")
        return

    # Read mode
    if not self_path.exists():
        typer.echo("self.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(self_path.read_text(encoding="utf-8"))


@app.command("partner")
def partner_cmd(
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Overwrite partner.md (content via --value or stdin).",
    ),
    update_section: str | None = typer.Option(
        None,
        "--update-section",
        help="Update a specific ## section of partner.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write identity/partner.md."""
    identity_dir = _identity_dir()
    partner_path = identity_dir / "partner.md"

    if update_section:
        content = _resolve_content(value)
        if update_identity_section(
            "partner.md",
            update_section,
            content.strip(),
            actor=_actor_for_cmd("partner", "--update-section <heading>"),
            reason="manual section update",
        ):
            typer.echo(f"Updated ## {update_section} in partner.md.")
        else:
            typer.echo(f"Error: section '## {update_section}' not found.", err=True)
            raise typer.Exit(1)
        return

    if write:
        content = _resolve_content(value)
        write_identity(
            "partner.md",
            actor=_actor_for_cmd("partner", "--write"),
            op="replace",
            section=None,
            content=content,
            reason="manual replace",
        )
        typer.echo("partner.md updated.")
        return

    # Read mode
    if not partner_path.exists():
        typer.echo("partner.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(partner_path.read_text(encoding="utf-8"))


@app.command("agency")
def agency_cmd(
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Overwrite agency.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write identity/agency.md."""
    identity_dir = _identity_dir()
    agency_path = identity_dir / "agency.md"

    if write:
        content = _resolve_content(value)
        write_identity(
            "agency.md",
            actor=_actor_for_cmd("agency", "--write"),
            op="replace",
            section=None,
            content=content,
            reason="manual replace",
        )
        typer.echo("agency.md updated.")
        return

    # Read mode
    if not agency_path.exists():
        typer.echo("agency.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(agency_path.read_text(encoding="utf-8"))


@app.command("pulse")
def pulse_cmd(
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Overwrite pulse.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write identity/pulse.md."""
    identity_dir = _identity_dir()
    pulse_path = identity_dir / "pulse.md"

    if write:
        content = _resolve_content(value)
        write_identity(
            "pulse.md",
            actor=_actor_for_cmd("pulse", "--write"),
            op="replace",
            section=None,
            content=content,
            reason="manual replace",
        )
        typer.echo("pulse.md updated.")
        return

    # Read mode
    if not pulse_path.exists():
        typer.echo("pulse.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(pulse_path.read_text(encoding="utf-8"))


@app.command("awareness")
def awareness_cmd(
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Overwrite awareness.md (content via --value or stdin).",
    ),
    value: str | None = typer.Option(
        None, "--value", help="Content to write (alternative to stdin)."
    ),
) -> None:
    """Read or write identity/awareness.md."""
    identity_dir = _identity_dir()
    awareness_path = identity_dir / "awareness.md"

    if write:
        content = _resolve_content(value)
        write_identity(
            "awareness.md",
            actor=_actor_for_cmd("awareness", "--write"),
            op="replace",
            section=None,
            content=content,
            reason="manual replace",
        )
        typer.echo("awareness.md updated.")
        return

    # Read mode
    if not awareness_path.exists():
        typer.echo("awareness.md not found.", err=True)
        raise typer.Exit(1)
    typer.echo(awareness_path.read_text(encoding="utf-8"))


@app.command("briefing")
def briefing_cmd(
    day: str | None = typer.Option(None, "--day", "-d", help="Specific day YYYYMMDD."),
) -> None:
    """Read the morning briefing from YYYYMMDD/talents/morning_briefing.md."""
    if day:
        path = day_path(day, create=False) / "talents" / "morning_briefing.md"
        if not path.exists():
            typer.echo("No briefing found.", err=True)
            raise typer.Exit(1)
        typer.echo(path.read_text(encoding="utf-8"))
        return

    # No day specified — find most recent
    for day in sorted(day_dirs().keys(), reverse=True):
        agents_dir = day_path(day, create=False) / "talents"
        briefing = agents_dir / "morning_briefing.md"
        if briefing.exists() and briefing.stat().st_size > 0:
            typer.echo(briefing.read_text(encoding="utf-8"))
            return

    typer.echo("No briefing found.", err=True)
    raise typer.Exit(1)
