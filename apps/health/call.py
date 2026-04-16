# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Health diagnostics — exposes `sol call health <command>` entry points."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

import typer

from think.pipeline_health import summarize_pipeline_day

app = typer.Typer(help="Health diagnostics — sol call health ...")


@app.command("pipeline")
def pipeline(
    day: Optional[str] = typer.Option(
        None, "--day", help="Day to summarize (YYYYMMDD)."
    ),
    yesterday: bool = typer.Option(
        False, "--yesterday", help="Summarize yesterday's pipeline."
    ),
) -> None:
    """Summarize dream pipeline health for one day."""
    if day is not None and yesterday:
        typer.echo("--day and --yesterday are mutually exclusive", err=True)
        raise typer.Exit(1)

    if day is not None:
        target = day
    elif yesterday:
        target = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    else:
        target = datetime.now().strftime("%Y%m%d")

    summary = summarize_pipeline_day(target)
    typer.echo(json.dumps(summary, indent=2, sort_keys=False))
