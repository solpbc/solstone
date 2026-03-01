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
) -> None:
    """Create a chat thread with the default muse and navigate the browser to it."""
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

    from think.cortex_client import cortex_request

    config: dict = {}
    if facet:
        config["facet"] = facet
    agent_id = cortex_request(
        prompt=full_prompt,
        name="default",
        provider=provider,
        config=config,
    )
    if agent_id is None:
        typer.echo("Error: failed to create chat thread", err=True)
        raise typer.Exit(1)

    import convey.state
    from apps.utils import get_app_storage_path
    from convey.utils import save_json
    from think.utils import get_journal, now_ms

    convey.state.journal_root = str(get_journal())

    chat_id = agent_id
    ts = now_ms()
    chat_record = {
        "ts": ts,
        "facet": facet or None,
        "provider": provider,
        "title": title,
        "agent_ids": [agent_id],
    }
    chats_dir = get_app_storage_path("chat", "chats")
    save_json(chats_dir / f"{chat_id}.json", chat_record)

    callosum_send("navigate", "request", path=f"/app/chat#{chat_id}")
    typer.echo(f"Redirected to chat: {chat_id}")
