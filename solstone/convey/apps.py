# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""App plugin system context processors and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from flask import Flask, g, request, url_for

from solstone.apps import AppRegistry


def _get_facets_data() -> list[dict]:
    """Get active facets data for templates."""
    from solstone.think.facets import get_facets

    from .config import apply_facet_order, load_convey_config

    all_facets = get_facets()
    facets_list = []

    for name, data in all_facets.items():
        if data.get("muted", False):
            continue

        facets_list.append(
            {
                "name": name,
                "title": data.get("title", name),
                "color": data.get("color", ""),
                "emoji": data.get("emoji", ""),
            }
        )

    # Apply custom ordering from config
    config = load_convey_config()
    return apply_facet_order(facets_list, config)


def _get_selected_facet() -> str | None:
    """Get selected facet from cookie, syncing with config.

    Cookie takes precedence - if it differs from config, update config.
    If no cookie exists, use config value as default.
    Validates against active (non-muted) facets; stale values are cleared.
    """
    from .config import get_selected_facet, set_selected_facet

    cookie_facet = request.cookies.get("selectedFacet")
    config_facet = get_selected_facet()

    # Empty string cookie -> treat as no selection, expire it
    if cookie_facet == "":
        set_selected_facet(None)
        g.clear_facet_cookie = True
        return None

    # Resolve: cookie takes precedence
    facet = cookie_facet if cookie_facet is not None else config_facet

    # Validate against active (non-muted) facets
    if facet:
        active_names = {f["name"] for f in _get_facets_data()}
        if facet not in active_names:
            set_selected_facet(None)
            g.clear_facet_cookie = True
            return None

    # Sync: cookie takes precedence, update config if different
    if cookie_facet is not None and cookie_facet != config_facet:
        set_selected_facet(cookie_facet)

    return facet


@dataclass
class AttentionItem:
    """A system attention item for the chat bar and triage context."""

    placeholder_text: str
    context_lines: list[str]


def _resolve_attention(awareness_current: dict) -> AttentionItem | None:
    """Check attention sources P0-P3, return highest priority or None."""
    # P0: Cortex errors
    try:
        import json
        from datetime import datetime
        from pathlib import Path

        from solstone.think.utils import get_journal

        journal = Path(get_journal())
        today = datetime.now().strftime("%Y%m%d")
        day_index = journal / "talents" / f"{today}.jsonl"
        if day_index.exists():
            errors: dict[str, float] = {}
            successes: dict[str, float] = {}
            for line in day_index.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    name = entry.get("name", "")
                    ts = entry.get("ts", 0)
                    if entry.get("status") == "error":
                        if ts > errors.get(name, 0):
                            errors[name] = ts
                    elif entry.get("status") == "completed":
                        if ts > successes.get(name, 0):
                            successes[name] = ts
                except (json.JSONDecodeError, TypeError):
                    continue
            unresolved = [
                name
                for name, err_ts in errors.items()
                if successes.get(name, 0) <= err_ts
            ]
            if unresolved:
                count = len(unresolved)
                names = ", ".join(sorted(unresolved)[:3])
                suffix = f" (+{count - 3} more)" if count > 3 else ""
                placeholder = (
                    f"{count} agent error{'s' if count != 1 else ''} today"
                    " — ask what happened"
                )
                context = [
                    f"System health: {count} unresolved agent error(s) today: "
                    f"{names}{suffix}. If user asks what needs attention, "
                    "summarize which agents failed."
                ]
                return AttentionItem(
                    placeholder_text=placeholder,
                    context_lines=context,
                )
    except Exception:
        pass

    # P1: Recent import completion
    imports = awareness_current.get("imports", {})
    last_completed = imports.get("last_completed")
    last_summary = imports.get("last_result_summary")
    if last_completed and last_summary:
        try:
            from datetime import datetime, timedelta

            completed_dt = datetime.fromisoformat(last_completed)
            if datetime.now() - completed_dt < timedelta(hours=1):
                placeholder = f"Import complete: {last_summary} — ask me about it"
                if len(placeholder) > 90:
                    placeholder = "New import complete — ask me what arrived"
                context = [
                    f"System health: import recently completed — {last_summary}. "
                    "If user asks what needs attention, mention the new import."
                ]
                return AttentionItem(
                    placeholder_text=placeholder,
                    context_lines=context,
                )
        except Exception:
            pass

    # P2: Daily analysis highlights
    journal_state = awareness_current.get("journal", {})
    if journal_state.get("first_daily_ready"):
        try:
            from datetime import datetime
            from pathlib import Path

            from solstone.think.utils import get_journal

            journal = Path(get_journal())
            today = datetime.now().strftime("%Y%m%d")
            agents_dir = journal / today / "talents"
            if agents_dir.is_dir():
                outputs = sorted(p.stem for p in agents_dir.glob("*.md"))
                if outputs:
                    count = len(outputs)
                    placeholder = (
                        f"{count} analysis report{'s' if count != 1 else ''} ready"
                        " — ask about your day"
                    )
                    context = [
                        f"System health: {count} daily analysis report(s) "
                        f"available today: {', '.join(outputs)}. User can ask "
                        "about any of these topics."
                    ]
                    return AttentionItem(
                        placeholder_text=placeholder,
                        context_lines=context,
                    )
        except Exception:
            pass

    # P3: Owner voiceprint candidate ready for confirmation
    voiceprint = awareness_current.get("voiceprint", {})
    if voiceprint.get("status") == "candidate":
        cluster_size = voiceprint.get("cluster_size", 0)
        placeholder = "Voice pattern detected — confirm in Speakers"
        context = [
            f"System detected owner voice pattern from {cluster_size} voice samples. "
            "Direct user to the Speakers app (/app/speakers) to confirm their voiceprint."
        ]
        return AttentionItem(placeholder_text=placeholder, context_lines=context)

    return None


def _resolve_placeholder(awareness_current: dict, day_count: int) -> str:
    """Resolve fallback chat bar placeholder text based on journal state."""
    imports = awareness_current.get("imports", {})
    if not imports.get("has_imported") and day_count < 3:
        return "Bring in past conversations, calendar, or notes to give me context..."
    if awareness_current.get("journal", {}).get("first_daily_ready"):
        if day_count < 2:
            return "Your first daily analysis is ready — ask me what I found..."
        if day_count >= 7:
            return "Ask me about your day, search your journal, or explore insights..."
        return "Your daily analysis is ready — ask about today or anything in your journal..."
    return "observing — your first daily analysis will be ready soon..."


def register_app_context(app: Flask, registry: AppRegistry) -> None:
    """Register app system context processors and template filters."""
    from .utils import DATE_RE, format_date_short

    # Register Jinja2 filters
    app.jinja_env.filters["format_date_short"] = format_date_short

    @app.context_processor
    def inject_app_context() -> dict:
        """Inject app registry and facets context for new app system."""
        from solstone.convey import copy as convey_copy

        from .config import apply_app_order, load_convey_config, reporting_enabled

        # Parse URL path: /app/{app_name}/{day}/...
        path_parts = request.path.split("/")

        # Auto-extract app name from URL for /app/{app_name}/... routes
        current_app_name = None
        if (
            len(path_parts) > 2
            and path_parts[1] == "app"
            and path_parts[2] in registry.apps
        ):
            current_app_name = path_parts[2]

        # Auto-extract day from URL for apps with date_nav enabled
        # Pattern: /app/{app_name}/{YYYYMMDD} or /app/{app_name}/{YYYYMMDD}/*
        day = None
        if (
            current_app_name
            and registry.apps[current_app_name].date_nav_enabled()
            and len(path_parts) > 3
            and DATE_RE.fullmatch(path_parts[3])
        ):
            day = path_parts[3]

        facets = _get_facets_data()
        selected_facet = _get_selected_facet()

        # Build apps dict for menu-bar
        apps_dict = {}
        for app_instance in registry.apps.values():
            apps_dict[app_instance.name] = {
                "icon": app_instance.icon,
                "label": app_instance.label,
            }

        # Apply custom ordering from config
        config = load_convey_config()
        apps_dict = apply_app_order(apps_dict, config)

        # Override sol label if agent has a chosen name
        if "sol" in apps_dict:
            try:
                from solstone.think.utils import get_config as _get_journal_config

                journal_config = _get_journal_config()
                agent_block = journal_config.get("agent", {})
                if agent_block.get("name_status") in ("chosen", "self-named"):
                    agent_name = agent_block.get("name", "").strip()
                    if agent_name:
                        apps_dict["sol"]["label"] = agent_name
            except Exception:
                pass  # Keep default label on any error

        # Get starred apps list
        starred_apps = config.get("apps", {}).get("starred", [])

        # Chat bar placeholder based on journal state
        chat_bar_placeholder = "Send a message..."
        chat_bar_attention = None
        try:
            from solstone.think.awareness import get_current
            from solstone.think.utils import day_dirs

            awareness_current = get_current()
            day_count = len(day_dirs())
            attention = _resolve_attention(awareness_current)
            if attention:
                chat_bar_attention = {"placeholder_text": attention.placeholder_text}
            chat_bar_placeholder = _resolve_placeholder(awareness_current, day_count)
        except Exception:
            pass  # Default placeholder on any error

        today = date.today().strftime("%Y%m%d")
        from solstone.convey.chat_stream import read_chat_events
        from solstone.convey.sol_initiated.state import (
            latest_unresolved_sol_chat_request,
        )

        unresolved_request = latest_unresolved_sol_chat_request(read_chat_events(today))
        chat_bar_sol_request = (
            {**unresolved_request, "day": today} if unresolved_request else None
        )

        return {
            "app_registry": registry,
            "app": current_app_name,
            "apps": apps_dict,
            "facets": facets,
            "selected_facet": selected_facet,
            "starred_apps": starred_apps,
            "day": day,
            "chat_bar_placeholder": chat_bar_placeholder,
            "chat_bar_attention": chat_bar_attention,
            "chat_bar_sol_request": chat_bar_sol_request,
            "convey_settings": {"reporting_enabled": reporting_enabled()},
            "CONVEY_COPY": {
                name.removeprefix("CONVEY_"): getattr(convey_copy, name)
                for name in convey_copy.__all__
                if name.startswith("CONVEY_")
            },
        }

    @app.context_processor
    def inject_vendor_helper() -> dict:
        """Provide convenient vendor library helper for templates."""

        def vendor_lib(library_name: str, file: str | None = None) -> str:
            """Generate URL for vendor library.

            Args:
                library_name: Name of vendor library (e.g., 'marked')
                file: Optional specific file, defaults to {library}.min.js

            Returns:
                URL to the vendor library file

            Example:
                {{ vendor_lib('marked') }}
                → /static/vendor/marked/marked.min.js
            """
            if file is None:
                file = f"{library_name}.min.js"
            return url_for("root.static", filename=f"vendor/{library_name}/{file}")

        return {"vendor_lib": vendor_lib}

    @app.after_request
    def clear_stale_facet_cookie(response):
        if getattr(g, "clear_facet_cookie", False):
            response.delete_cookie("selectedFacet", path="/", samesite="Lax")
        return response
