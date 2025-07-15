import importlib


def test_calendar_days_api(tmp_path):
    review = importlib.import_module("dream")
    (tmp_path / "20240101").mkdir()
    (tmp_path / "20240102").mkdir()
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/calendar/api/days"):
        resp = review.calendar_days()
    assert resp.json == ["20240101", "20240102"]
