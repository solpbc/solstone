"""App plugin system context processors and helpers."""

from __future__ import annotations

from flask import Flask, request, url_for

from apps import AppRegistry


def _get_facets_data() -> list[dict]:
    """Get facets data for templates."""
    from think.facets import get_facets

    from .config import apply_facet_order, load_convey_config

    all_facets = get_facets()
    active_facets = []

    for name, data in all_facets.items():
        if not data.get("disabled", False):
            active_facets.append(
                {
                    "name": name,
                    "title": data.get("title", name),
                    "color": data.get("color", ""),
                    "emoji": data.get("emoji", ""),
                }
            )

    # Apply custom ordering from config
    config = load_convey_config()
    return apply_facet_order(active_facets, config)


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


def register_app_context(app: Flask, registry: AppRegistry) -> None:
    """Register app system context processors."""

    @app.context_processor
    def inject_app_context() -> dict:
        """Inject app registry and facets context for new app system."""
        from .config import apply_app_order, load_convey_config

        facets = _get_facets_data()
        selected_facet = _get_selected_facet()

        # Build apps dict for menu-bar (includes submenu items)
        apps_dict = {}
        for app_instance in registry.apps.values():
            submenu = app_instance.get_submenu_items(facets, selected_facet)
            apps_dict[app_instance.name] = {
                "icon": app_instance.icon,
                "label": app_instance.label,
                "submenu": submenu if submenu else None,
            }

        # Apply custom ordering from config
        config = load_convey_config()
        apps_dict = apply_app_order(apps_dict, config)

        # Get starred apps list
        starred_apps = config.get("apps", {}).get("starred", [])

        return {
            "app_registry": registry,
            "apps": apps_dict,
            "facets": facets,
            "selected_facet": selected_facet,
            "starred_apps": starred_apps,
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
