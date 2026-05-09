# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for chat app tools."""

from __future__ import annotations

import json
from dataclasses import asdict

import typer

from solstone.convey.sol_initiated import start_chat

app = typer.Typer(help="Chat tools.")


@app.callback()
def _callback() -> None:
    """Chat app command group."""


@app.command("start")
def cmd_start(
    summary: str = typer.Option(..., "--summary", help="Short request summary."),
    message: str | None = typer.Option(None, "--message", help="Optional message."),
    category: str = typer.Option(..., "--category", help="Request category."),
    dedupe: str = typer.Option(..., "--dedupe", help="Deduplication key."),
    dedupe_window: str | None = typer.Option(
        None,
        "--dedupe-window",
        help="Deduplication window, e.g. 24h.",
    ),
    since_ts: int = typer.Option(..., "--since-ts", help="Since timestamp in ms."),
    trigger_talent: str = typer.Option(
        ...,
        "--trigger-talent",
        help="Talent that triggered the request.",
    ),
) -> None:
    """Start a sol-initiated chat request."""
    try:
        result = start_chat(
            summary=summary,
            message=message,
            category=category,
            dedupe=dedupe,
            dedupe_window=dedupe_window,
            since_ts=since_ts,
            trigger_talent=trigger_talent,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(json.dumps(asdict(result)))
