"""Inbox app hooks - submenu with badge."""

from __future__ import annotations

from think import messages


def get_submenu_items(facets, selected_facet):
    """Show active/archived tabs with badge.

    Args:
        facets: List of active facet dicts with name, title, color, emoji
        selected_facet: Currently selected facet name, or None

    Returns:
        List of dicts with keys: label, path, count (optional)
    """
    try:
        unread_count = messages.get_unread_count()
    except Exception:
        unread_count = 0

    items = [
        {"label": "Active", "path": "/app/inbox", "count": unread_count},
        {"label": "Archived", "path": "/app/inbox?status=archived"},
    ]

    return items
