# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI for delegating work to cogitate agents.

Provides ``sol engage <name>`` for delegating work to cogitate agents.
"""

import sys

import typer

from think.utils import require_solstone

engage_app = typer.Typer(name="engage")


def _engage(
    name: str,
    prompt: str,
    *,
    wait: bool = False,
    facet: str | None = None,
    day: str | None = None,
) -> None:
    config_data = {}
    if facet is not None:
        config_data["facet"] = facet
    if day is not None:
        config_data["day"] = day
    config = config_data or None

    from think.cortex_client import cortex_request

    use_id = cortex_request(prompt=prompt, name=name, config=config)
    if use_id is None:
        typer.echo("Error: failed to send cortex request.", err=True)
        raise typer.Exit(1)

    if not wait:
        typer.echo(use_id)
        return

    from think.cortex_client import read_agent_events, wait_for_agents

    completed, timed_out = wait_for_agents([use_id])
    if use_id in timed_out:
        typer.echo("Error: agent timed out.", err=True)
        raise typer.Exit(1)

    end_state = completed.get(use_id, "error")
    if end_state != "finish":
        typer.echo(f"Error: agent ended with state: {end_state}", err=True)
        raise typer.Exit(1)

    events = read_agent_events(use_id)
    result = ""
    for event in reversed(events):
        if event.get("event") == "finish":
            result = event.get("result", "")
            break

    typer.echo(result)


@engage_app.command()
def engage(
    name: str = typer.Argument(help="Agent name to delegate to (e.g. coder)."),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Block until the agent completes and print its result.",
    ),
    facet: str | None = typer.Option(
        None, "--facet", help="Facet context for the agent."
    ),
    day: str | None = typer.Option(
        None, "--day", help="Day context for the agent (e.g. 20260404)."
    ),
) -> None:
    """Delegate work to a cogitate agent.

    Reads a prompt from stdin, sends it to cortex as an agent request.
    By default, prints the use_id and exits immediately (fire-and-forget).

    Example::

        echo 'Fix the matching bug' | sol engage coder
        echo 'Fix the matching bug' | sol engage coder --wait
    """
    prompt = sys.stdin.read()
    if not prompt.strip():
        typer.echo("Error: no prompt provided on stdin.", err=True)
        raise typer.Exit(1)

    _engage(name, prompt.strip(), wait=wait, facet=facet, day=day)


def main() -> None:
    """Entry point for ``sol engage``."""
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        engage_app()
        return
    require_solstone()
    engage_app()
