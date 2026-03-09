# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""App plugin system context processors and helpers."""

from __future__ import annotations

from flask import Flask, request, url_for

from apps import AppRegistry


def _get_facets_data(include_muted: bool = False) -> list[dict]:
    """Get facets data for templates.

    Args:
        include_muted: If True, include facets marked as disabled

    Returns:
        List of facet dicts with name, title, color, emoji, and muted status
    """
    from think.facets import get_facets

    from .config import apply_facet_order, load_convey_config

    all_facets = get_facets()
    facets_list = []

    for name, data in all_facets.items():
        is_muted = data.get("muted", False)

        # Skip muted facets unless explicitly requested
        if is_muted and not include_muted:
            continue

        facets_list.append(
            {
                "name": name,
                "title": data.get("title", name),
                "color": data.get("color", ""),
                "emoji": data.get("emoji", ""),
                "muted": is_muted,  # Include muted status for styling
            }
        )

    # Apply custom ordering from config
    config = load_convey_config()
    return apply_facet_order(facets_list, config)


def _get_selected_facet() -> str | None:
    """Get selected facet from cookie, syncing with config.

    Cookie takes precedence - if it differs from config, update config.
    If no cookie exists, use config value as default.
    """
    from .config import get_selected_facet, set_selected_facet

    cookie_facet = request.cookies.get("selectedFacet")
    config_facet = get_selected_facet()

    # Sync: cookie takes precedence, update config if different
    if cookie_facet is not None and cookie_facet != config_facet:
        set_selected_facet(cookie_facet)
        return cookie_facet

    # No cookie: use config default
    return cookie_facet if cookie_facet is not None else config_facet


def _resolve_placeholder(
    onboarding_status: str, awareness_current: dict, day_count: int
) -> str:
    """Resolve chat bar placeholder text based on journal state."""
    if onboarding_status == "observing":
        return "I'm learning how you work — ask me what I've noticed..."
    if onboarding_status == "ready":
        return "I have suggestions for organizing your journal — let's review"
    if onboarding_status == "interviewing":
        return "Tell me about your work..."
    if onboarding_status in ("complete", "skipped"):
        if awareness_current.get("journal", {}).get("first_daily_ready"):
            if day_count < 2:
                return "Your first daily analysis is ready — ask me what I found..."
            if day_count >= 7:
                return (
                    "Ask me about your day, search your journal, or explore insights..."
                )
            return "Your daily analysis is ready — ask about today or anything in your journal..."
        return "Capture is running — your first daily analysis will be ready soon..."
    return "Send a message..."


def register_app_context(app: Flask, registry: AppRegistry) -> None:
    """Register app system context processors and template filters."""
    from .utils import DATE_RE, format_date_short

    # Register Jinja2 filters
    app.jinja_env.filters["format_date_short"] = format_date_short

    @app.context_processor
    def inject_app_context() -> dict:
        """Inject app registry and facets context for new app system."""
        from .config import apply_app_order, load_convey_config

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

        # Determine if current app wants muted facets shown
        include_muted = False
        if current_app_name:
            include_muted = registry.apps[current_app_name].show_muted_facets()

        facets = _get_facets_data(include_muted=include_muted)
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

        # Get starred apps list
        starred_apps = config.get("apps", {}).get("starred", [])

        # Chat bar placeholder based on onboarding state
        chat_bar_placeholder = "Send a message..."
        try:
            from think.awareness import get_current, get_onboarding
            from think.utils import day_dirs

            onboarding = get_onboarding()
            onboarding_status = onboarding.get("status", "")
            awareness_current = get_current()
            day_count = len(day_dirs())
            chat_bar_placeholder = _resolve_placeholder(
                onboarding_status, awareness_current, day_count
            )
        except Exception:
            pass  # Default placeholder on any error

        return {
            "app_registry": registry,
            "app": current_app_name,
            "apps": apps_dict,
            "facets": facets,
            "selected_facet": selected_facet,
            "starred_apps": starred_apps,
            "day": day,
            "chat_bar_placeholder": chat_bar_placeholder,
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
            return url_for("static", filename=f"vendor/{library_name}/{file}")

        return {"vendor_lib": vendor_lib}
