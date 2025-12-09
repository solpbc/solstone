import json


def test_get_month_event_counts(tmp_path, monkeypatch):
    """Test get_month_event_counts scans event files correctly."""
    from think.events import get_month_event_counts

    journal = tmp_path

    # Create events for two facets in January 2024
    work_events = journal / "facets" / "work" / "events"
    work_events.mkdir(parents=True)
    personal_events = journal / "facets" / "personal" / "events"
    personal_events.mkdir(parents=True)

    # Work facet: 2 events on Jan 1, 1 event on Jan 5
    (work_events / "20240101.jsonl").write_text(
        json.dumps({"title": "Meeting 1", "start": "09:00:00"})
        + "\n"
        + json.dumps({"title": "Meeting 2", "start": "14:00:00"})
        + "\n"
    )
    (work_events / "20240105.jsonl").write_text(
        json.dumps({"title": "Code review", "start": "10:00:00"}) + "\n"
    )

    # Personal facet: 1 event on Jan 1
    (personal_events / "20240101.jsonl").write_text(
        json.dumps({"title": "Gym session", "start": "18:00:00"}) + "\n"
    )

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    result = get_month_event_counts("202401")

    assert "20240101" in result
    assert result["20240101"]["work"] == 2
    assert result["20240101"]["personal"] == 1
    assert "20240105" in result
    assert result["20240105"]["work"] == 1
    assert "personal" not in result["20240105"]


def test_get_month_event_counts_future_dates(tmp_path, monkeypatch):
    """Test that future dates without day directories are included."""
    from think.events import get_month_event_counts

    journal = tmp_path

    # Create events for a future month (no day directories exist)
    work_events = journal / "facets" / "work" / "events"
    work_events.mkdir(parents=True)

    # Future events (anticipated)
    (work_events / "20251215.jsonl").write_text(
        json.dumps(
            {"title": "Holiday party", "start": "18:00:00", "occurred": False}
        )
        + "\n"
    )
    (work_events / "20251220.jsonl").write_text(
        json.dumps({"title": "Year review", "start": "10:00:00", "occurred": False})
        + "\n"
        + json.dumps({"title": "Team lunch", "start": "12:00:00", "occurred": False})
        + "\n"
    )

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    result = get_month_event_counts("202512")

    assert "20251215" in result
    assert result["20251215"]["work"] == 1
    assert "20251220" in result
    assert result["20251220"]["work"] == 2


def test_get_month_event_counts_skips_entries_without_title(tmp_path, monkeypatch):
    """Test that entries without title are not counted."""
    from think.events import get_month_event_counts

    journal = tmp_path

    work_events = journal / "facets" / "work" / "events"
    work_events.mkdir(parents=True)

    # Mix of valid and invalid entries
    (work_events / "20240101.jsonl").write_text(
        json.dumps({"title": "Valid event", "start": "09:00:00"})
        + "\n"
        + json.dumps({"start": "10:00:00"})  # No title
        + "\n"
        + json.dumps({"title": "", "start": "11:00:00"})  # Empty title
        + "\n"
        + json.dumps({"title": "Another valid", "start": "14:00:00"})
        + "\n"
    )

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    result = get_month_event_counts("202401")

    assert result["20240101"]["work"] == 2  # Only 2 valid events


def test_get_month_event_counts_empty_month(tmp_path, monkeypatch):
    """Test that empty month returns empty dict."""
    from think.events import get_month_event_counts

    journal = tmp_path

    # Create facets directory but no events for requested month
    work_events = journal / "facets" / "work" / "events"
    work_events.mkdir(parents=True)

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    result = get_month_event_counts("202402")

    assert result == {}


def test_get_month_event_counts_no_journal_path(monkeypatch):
    """Test that missing JOURNAL_PATH returns empty dict."""
    from think.events import get_month_event_counts

    monkeypatch.delenv("JOURNAL_PATH", raising=False)

    result = get_month_event_counts("202401")

    assert result == {}
