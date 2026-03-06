# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for chat navigation and thread management.

Auto-discovered by ``think.call`` and mounted as ``sol call chat ...``.
"""

import typer

from think.callosum import callosum_send

app = typer.Typer(help="Chat tools.")


@app.command("navigate")
def navigate(
    path: str | None = typer.Argument(None, help="URL path to navigate to."),
    facet: str | None = typer.Option(None, "--facet", "-f", help="Facet to switch to."),
) -> None:
    """Navigate the browser to a path and/or switch facet."""
    if not path and not facet:
        typer.echo("Error: provide a path and/or --facet", err=True)
        raise typer.Exit(1)

    fields: dict = {}
    if path is not None:
        fields["path"] = path
    if facet is not None:
        fields["facet"] = facet

    callosum_send("navigate", "request", **fields)

    parts = []
    if path:
        parts.append(path)
    if facet:
        parts.append(f"[{facet}]")
    typer.echo(f"Navigate: {' '.join(parts)}")


@app.command("redirect")
def redirect(
    message: str = typer.Argument(..., help="User message to redirect to chat."),
    app_name: str = typer.Option("", "--app", help="Source app name."),
    path: str = typer.Option("", "--path", help="Source URL path."),
    facet: str = typer.Option("", "--facet", "-f", help="Active facet."),
    muse: str = typer.Option("default", "--muse", "-m", help="Muse agent to use."),
) -> None:
    """Create a chat thread with the specified muse and navigate the browser to it."""
    context_lines = []
    if app_name:
        context_lines.append(f"Current app: {app_name}")
    if path:
        context_lines.append(f"Current path: {path}")
    if facet:
        context_lines.append(f"Current facet: {facet}")
    full_prompt = (
        "\n".join(context_lines) + "\n\n" + message if context_lines else message
    )

    from think.models import resolve_provider

    provider, _model = resolve_provider("muse.system.default", "cogitate")

    from apps.chat.routes import generate_chat_title

    title = generate_chat_title(message)

    import convey.state
    from apps.utils import get_app_storage_path
    from convey.utils import save_json
    from think.utils import get_journal, now_ms

    convey.state.journal_root = str(get_journal())

    from think.cortex_client import cortex_request

    config: dict = {}
    if facet:
        config["facet"] = facet
    agent_id = cortex_request(
        prompt=full_prompt,
        name=muse,
        provider=provider,
        config=config,
    )
    if agent_id is None:
        typer.echo("Error: failed to create chat thread", err=True)
        raise typer.Exit(1)

    chat_id = agent_id
    ts = now_ms()
    chat_record = {
        "ts": ts,
        "facet": facet or None,
        "provider": provider,
        "muse": muse,
        "title": title,
        "agent_ids": [agent_id],
    }
    chats_dir = get_app_storage_path("chat", "chats")
    save_json(chats_dir / f"{chat_id}.json", chat_record)

    callosum_send("navigate", "request", path=f"/app/chat#{chat_id}")
    typer.echo(f"Redirected to chat: {chat_id}")


@app.command("history")
def history(
    muse: str | None = typer.Option(None, "--muse", "-m", help="Filter by muse name."),
    facet: str | None = typer.Option(None, "--facet", "-f", help="Filter by facet."),
    day: str | None = typer.Option(None, "--day", "-d", help="Filter by day YYYYMMDD."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results."),
) -> None:
    """List chat threads with optional filters."""
    from datetime import datetime

    import convey.state
    from apps.chat.routes import _load_all_chats
    from think.utils import get_journal

    convey.state.journal_root = str(get_journal())

    chats, _unread = _load_all_chats()

    # Filter
    if muse:
        chats = [c for c in chats if c.get("muse") == muse]
    if facet:
        chats = [c for c in chats if c.get("facet") == facet]
    if day:
        chats = [
            c
            for c in chats
            if datetime.fromtimestamp(c["ts"] / 1000).strftime("%Y%m%d") == day
        ]

    # Sort by ts descending
    chats.sort(key=lambda c: c.get("ts", 0), reverse=True)
    chats = chats[:limit]

    if not chats:
        typer.echo("No chats found.")
        return

    for c in chats:
        chat_day = datetime.fromtimestamp(c["ts"] / 1000).strftime("%Y%m%d")
        parts = [
            c.get("chat_id", ""),
            c.get("title", "(untitled)"),
            chat_day,
        ]
        if c.get("facet"):
            parts.append(c["facet"])
        if c.get("muse"):
            parts.append(c["muse"])
        typer.echo("  ".join(parts))


@app.command("read")
def read(
    chat_id: str = typer.Argument(..., help="Chat ID to read."),
    summary: bool = typer.Option(
        False, "--summary", "-s", help="Show only prompts and results."
    ),
    max_bytes: int = typer.Option(16384, "--max", help="Max output bytes."),
) -> None:
    """Display conversation events for a chat thread."""
    import json

    import convey.state
    from apps.chat.routes import _load_chat
    from think.cortex_client import read_agent_events
    from think.utils import get_journal, truncated_echo

    convey.state.journal_root = str(get_journal())

    chat = _load_chat(chat_id)
    if not chat:
        typer.echo(f"Chat not found: {chat_id}", err=True)
        raise typer.Exit(1)

    agent_ids = chat.get("agent_ids", [])
    all_events: list[dict] = []
    for agent_id in agent_ids:
        try:
            events = read_agent_events(agent_id)
            all_events.extend(events)
        except FileNotFoundError:
            pass

    if summary:
        all_events = [e for e in all_events if e.get("event") in ("request", "finish")]

    lines: list[str] = []
    for e in all_events:
        event_type = e.get("event", "unknown")
        if event_type == "request":
            lines.append(f"[request] {e.get('prompt', '')}")
        elif event_type == "start":
            name = e.get("name", "")
            provider = e.get("provider", "")
            model = e.get("model", "")
            lines.append(f"[start] {name} ({provider}/{model})")
        elif event_type == "thinking":
            lines.append(f"[thinking] {e.get('content', '')}")
        elif event_type == "tool_start":
            tool = e.get("tool", "")
            args = json.dumps(e.get("args", {}))
            lines.append(f"[tool_start] {tool}({args})")
        elif event_type == "tool_end":
            lines.append(f"[tool_end] {e.get('result', '')}")
        elif event_type == "finish":
            lines.append(f"[finish] {e.get('result', '')}")

    if not lines:
        typer.echo("No events found.")
        return

    truncated_echo("\n".join(lines), max_bytes)
