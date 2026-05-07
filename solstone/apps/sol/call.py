# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for the agent identity system.

Auto-discovered by ``think.call`` and mounted as ``sol call sol ...``.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import typer

from solstone.think.utils import get_project_root, require_solstone

app = typer.Typer(help="Agent identity — name and status.")


@app.callback()
def _require_up() -> None:
    require_solstone()


def _get_agent_config() -> dict:
    """Read agent config from journal config."""
    from solstone.think.utils import get_config

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
    from solstone.think.utils import get_config, get_journal

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
    os.chmod(config_path, 0o600)

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
    # Update identity/self.md with new name
    from solstone.think.identity import update_self_md_opening, update_self_md_section

    named_date = agent.get("named_date", "")
    update_self_md_opening(
        f"I am {name}. this is a new journal — we're just getting started.",
        actor="sol call sol set-name",
        reason="agent name updated",
    )
    if named_date:
        update_self_md_section(
            "my name",
            f"{name} (named {named_date})",
            actor="sol call sol set-name",
            reason="agent name updated",
        )
    else:
        update_self_md_section(
            "my name",
            name,
            actor="sol call sol set-name",
            reason="agent name updated",
        )
    project_root = Path(get_project_root())
    subprocess.run(
        ["make", "skills"], cwd=project_root, check=False, capture_output=True
    )


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
    project_root = Path(get_project_root())
    subprocess.run(
        ["make", "skills"], cwd=project_root, check=False, capture_output=True
    )


@app.command("thickness")
def thickness() -> None:
    """Show journal thickness signals for naming readiness."""
    from solstone.think.awareness import compute_thickness

    typer.echo(json.dumps(compute_thickness(), indent=2))


@app.command("set-owner")
def set_owner(
    name: str = typer.Argument(..., help="Owner name."),
    bio: str = typer.Option(None, "--bio", "-b", help="Short owner bio."),
) -> None:
    """Set the journal owner's name (and optional bio)."""
    from solstone.think.identity import update_self_md_section
    from solstone.think.utils import get_config, get_journal

    config = get_config()
    identity = config.get("identity", {})
    identity["name"] = name
    if bio is not None:
        identity["bio"] = bio
    config["identity"] = identity

    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    os.chmod(config_path, 0o600)

    # Update identity/self.md
    owner_content = name
    if bio:
        owner_content += f"\n{bio}"
    update_self_md_section(
        "who I'm here for",
        owner_content,
        actor="sol call sol set-owner",
        reason="owner identity updated",
    )

    typer.echo(json.dumps({"name": name, "bio": bio or ""}, indent=2))
    project_root = Path(get_project_root())
    subprocess.run(
        ["make", "skills"], cwd=project_root, check=False, capture_output=True
    )


@app.command("sol-init")
def sol_init() -> None:
    """Initialize the identity directory with self.md and agency.md."""
    from solstone.think.identity import ensure_identity_directory

    identity_dir = ensure_identity_directory()
    typer.echo(
        json.dumps({"identity_dir": str(identity_dir), "status": "ok"}, indent=2)
    )
