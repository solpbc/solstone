"""Tests for navigation badge helpers in convey package."""

from __future__ import annotations

from typing import Any

import pytest

import convey.__init__ as convey_app


def test_count_pending_todos_today_counts_incomplete(monkeypatch):
    """Unfinished todos should increment the nav badge count."""

    def fake_get_domains_with_todos(day: str):
        return ["personal", "work"]

    def fake_get_todos(day: str, domain: str):
        # Return different todos for each domain
        if domain == "personal":
            return [
                {"completed": False, "cancelled": False},
                {"completed": True, "cancelled": False},
            ]
        elif domain == "work":
            return [
                {"completed": False, "cancelled": True},
            ]
        return []

    monkeypatch.setattr(
        convey_app.todo_store, "get_domains_with_todos", fake_get_domains_with_todos
    )
    monkeypatch.setattr(convey_app.todo_store, "get_todos", fake_get_todos)

    count = convey_app._count_pending_todos_today()

    # Should count only 1 incomplete, non-cancelled todo from personal domain
    assert count == 1


def test_resolve_nav_badges_filters_zero(monkeypatch):
    """Providers returning zero should be omitted."""

    def first_provider():
        return 5

    def second_provider():
        return 0

    monkeypatch.setattr(
        convey_app,
        "NAV_BADGE_PROVIDERS",
        {"one": first_provider, "two": second_provider},
    )

    badges = convey_app._resolve_nav_badges()

    assert badges == {"one": 5}
