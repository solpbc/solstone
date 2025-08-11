import importlib


def test_agents_list(monkeypatch, tmp_path):
    from unittest.mock import Mock

    review = importlib.import_module("dream")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    review.journal_root = str(tmp_path)

    # Mock cortex client to return expected data
    mock_client = Mock()
    mock_client.list_agents.return_value = {
        "agents": [
            {
                "id": "100000",
                "started_at": 100000,  # milliseconds
                "status": "running",
                "pid": 12345,
                "metadata": {"prompt": "hi", "persona": "p", "model": "m"},
            }
        ],
        "pagination": {"limit": 10, "offset": 0, "total": 1},
    }

    # Mock the cortex client getter
    monkeypatch.setattr(
        "dream.cortex_client.get_global_cortex_client", lambda: mock_client
    )
    monkeypatch.setattr("time.time", lambda: 160)

    with review.app.test_request_context("/agents/api/list"):
        resp = review.agents_list()

    expected_response = {
        "agents": [
            {
                "id": "100000",
                "start": 100.0,
                "since": "1 minute ago",
                "model": "m",
                "persona": "p",
                "prompt": "hi",
                "status": "running",
                "pid": 12345,
            }
        ],
        "pagination": {"limit": 10, "offset": 0, "total": 1},
    }
    assert resp.json == expected_response
