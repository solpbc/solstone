# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for the agent identity system.

Auto-discovered by ``think.call`` and mounted as ``sol call agent ...``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(help="Agent identity — name and status.")


def _get_agent_config() -> dict:
    """Read agent config from journal config."""
    from think.utils import get_config

    return get_config().get(
        "agent",
        {
            "name": "sol",
            "name_status": "default",
            "named_date": None,
            "proposal_count": 0,
        },
    )


def _update_agent_config(updates: dict) -> dict:
    """Update agent config in journal.json and return the full agent block."""
    from think.utils import get_config, get_journal

    config = get_config()
    agent = config.get(
        "agent",
        {
            "name": "sol",
            "name_status": "default",
            "named_date": None,
            "proposal_count": 0,
        },
    )
    agent.update(updates)
    config["agent"] = agent

    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return agent


@app.command("name")
def name() -> None:
    """Show the current agent name and status."""
    agent = _get_agent_config()
    typer.echo(json.dumps(agent, indent=2))


@app.command("set-name")
def set_name(
    name: str = typer.Argument(..., help="New agent name."),
    status: str = typer.Option(
        "chosen",
        "--status",
        "-s",
        help="Name status (chosen, self-named, deferred, default).",
    ),
) -> None:
    """Set the agent name."""
    agent = _update_agent_config(
        {
            "name": name,
            "name_status": status,
            "named_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
    )
    typer.echo(json.dumps(agent, indent=2))


@app.command("reset")
def reset() -> None:
    """Reset the agent name to default."""
    agent = _update_agent_config(
        {
            "name": "sol",
            "name_status": "default",
            "named_date": None,
        }
    )
    typer.echo(json.dumps(agent, indent=2))


@app.command("thickness")
def thickness() -> None:
    """Show journal thickness signals for naming readiness."""
    from think.awareness import compute_thickness

    typer.echo(json.dumps(compute_thickness(), indent=2))
