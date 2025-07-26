import importlib
import json
import os


def test_search_summaries_api(tmp_path):
    indexer = importlib.import_module("think.indexer")
    review = importlib.import_module("dream")
    search_view = importlib.import_module("dream.views.search")

    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)
    day = journal / "20240107" / "topics"
    day.mkdir(parents=True)
    (day / "files.md").write_text("This is a test sentence.\n")

    indexer.scan_summaries(str(journal))
    review.journal_root = str(journal)

    with review.app.test_request_context("/search/api/summaries?q=test"):
        resp = search_view.search_summaries_api()

    assert resp.json["total"] == 1
    assert resp.json["results"][0]["topic"] == "files"


def test_search_summaries_api_filters(tmp_path):
    indexer = importlib.import_module("think.indexer")
    review = importlib.import_module("dream")
    search_view = importlib.import_module("dream.views.search")

    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    day1 = journal / "20240107" / "topics"
    day1.mkdir(parents=True)
    (day1 / "files.md").write_text("Sentence one.\n")

    day2 = journal / "20240108" / "topics"
    day2.mkdir(parents=True)
    (day2 / "other.md").write_text("Sentence two.\n")

    indexer.scan_summaries(str(journal))
    review.journal_root = str(journal)

    with review.app.test_request_context(
        "/search/api/summaries?q=Sentence&day=20240107"
    ):
        resp = search_view.search_summaries_api()
    assert resp.json["total"] == 1
    assert resp.json["results"][0]["day"] == "20240107"

    with review.app.test_request_context(
        "/search/api/summaries?q=Sentence&topic=files"
    ):
        resp = search_view.search_summaries_api()
    assert resp.json["total"] == 1
    assert resp.json["results"][0]["topic"] == "files"


def test_search_events_api(tmp_path):
    indexer = importlib.import_module("think.indexer")
    review = importlib.import_module("dream")
    search_view = importlib.import_module("dream.views.search")

    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    day = journal / "20240101"
    day.mkdir()
    data = {
        "day": "20240101",
        "occurrences": [
            {
                "type": "meeting",
                "source": "topics/meetings.md",
                "start": "09:00:00",
                "end": "09:30:00",
                "title": "Standup",
                "summary": "Daily sync",
            }
        ],
    }
    topics_dir = day / "topics"
    topics_dir.mkdir()
    (topics_dir / "meetings.json").write_text(json.dumps(data))

    indexer.scan_events(str(journal))
    review.journal_root = str(journal)

    with review.app.test_request_context("/search/api/events?q=Standup"):
        resp = search_view.search_events_api()

    assert resp.json["total"] == 1
    assert resp.json["results"][0]["topic"] == "meetings"


def test_search_transcripts_api(tmp_path):
    indexer = importlib.import_module("think.indexer")
    review = importlib.import_module("dream")
    search_view = importlib.import_module("dream.views.search")

    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    day_dir = journal / "20240103"
    day_dir.mkdir()
    (day_dir / "123000_audio.json").write_text(
        json.dumps(
            [
                {
                    "start": "00:00:01",
                    "source": "mic",
                    "speaker": 1,
                    "text": "hello raw",
                },
                {"topics": ["t"], "setting": "personal"},
            ]
        )
    )

    indexer.scan_transcripts(str(journal))
    review.journal_root = str(journal)

    with review.app.test_request_context(
        "/search/api/transcripts?q=hello&day=20240103&topic=ignoreme"
    ):
        resp = search_view.search_transcripts_api()

    assert resp.json["total"] == 1
    assert resp.json["results"][0]["time"] == "123000"
    assert resp.json["results"][0]["type"] == "audio"
