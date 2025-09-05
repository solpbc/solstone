"""Test supervisor scheduling functionality."""

from unittest.mock import AsyncMock, Mock, patch

from think.supervisor import spawn_scheduled_agents


@patch("think.supervisor.CortexClient")
@patch("think.supervisor.get_personas")
def test_spawn_scheduled_agents(mock_get_personas, mock_cortex_client_class):
    """Test that scheduled agents are spawned correctly via Cortex."""
    # Mock personas with one scheduled and one not
    mock_get_personas.return_value = {
        "todo": {
            "title": "TODO Task Manager",
            "config": {"schedule": "daily", "backend": "openai", "model": "gpt-4"},
        },
        "default": {
            "title": "Default Assistant",
            "config": {},  # No schedule
        },
        "another_daily": {
            "title": "Another Daily Task",
            "config": {"schedule": "daily"},  # No model specified
        },
    }

    # Mock the CortexClient instance
    mock_client = AsyncMock()
    mock_cortex_client_class.return_value.__aenter__.return_value = mock_client
    mock_cortex_client_class.return_value.__aexit__.return_value = None

    # Mock spawn to return agent IDs
    mock_client.spawn.side_effect = ["agent_123", "agent_456"]

    # Call the function
    spawn_scheduled_agents("/test/journal")

    # Should have created CortexClient with correct journal path
    mock_cortex_client_class.assert_called_once_with(journal_path="/test/journal")

    # Should spawn 2 agents (todo and another_daily)
    assert mock_client.spawn.call_count == 2

    # Check first spawn call (todo)
    first_call = mock_client.spawn.call_args_list[0]
    assert first_call[1]["persona"] == "todo"
    assert first_call[1]["backend"] == "openai"
    assert first_call[1]["config"] == {"model": "gpt-4"}
    assert "Running daily scheduled task for todo" in first_call[1]["prompt"]

    # Check second spawn call (another_daily)
    second_call = mock_client.spawn.call_args_list[1]
    assert second_call[1]["persona"] == "another_daily"
    assert second_call[1]["backend"] == "openai"  # default
    assert second_call[1]["config"] == {}  # no model specified
    assert "Running daily scheduled task for another_daily" in second_call[1]["prompt"]


@patch("think.supervisor.spawn_scheduled_agents")
@patch("think.supervisor.run_process_day")
def test_supervisor_runs_scheduled_after_process_day(
    mock_run_process_day, mock_spawn_scheduled
):
    """Test that scheduled agents run only after successful process_day."""
    from think.supervisor import supervise

    # Test successful process_day
    mock_run_process_day.return_value = True

    with patch("think.supervisor.datetime") as mock_datetime:
        with patch("think.supervisor.time.sleep") as mock_sleep:
            with patch("think.supervisor.check_health") as mock_check_health:
                # Mock dates to trigger daily processing
                mock_now = Mock()
                mock_now.date.side_effect = [
                    Mock(name="day1"),
                    Mock(name="day2"),  # Different day triggers process_day
                    Mock(name="day2"),  # Same day after processing
                ]
                mock_datetime.now.return_value = mock_now
                mock_check_health.return_value = []  # No stale processes

                # Use side effect to break loop after first iteration
                mock_sleep.side_effect = KeyboardInterrupt

                try:
                    supervise("/test/journal", daily=True)
                except KeyboardInterrupt:
                    pass

                # Should have called process_day and spawn_scheduled_agents
                mock_run_process_day.assert_called_once()
                mock_spawn_scheduled.assert_called_once_with("/test/journal")


@patch("think.supervisor.spawn_scheduled_agents")
@patch("think.supervisor.run_process_day")
def test_supervisor_skips_scheduled_on_process_day_failure(
    mock_run_process_day, mock_spawn_scheduled
):
    """Test that scheduled agents don't run if process_day fails."""
    from think.supervisor import supervise

    # Test failed process_day
    mock_run_process_day.return_value = False

    with patch("think.supervisor.datetime") as mock_datetime:
        with patch("think.supervisor.time.sleep") as mock_sleep:
            with patch("think.supervisor.check_health") as mock_check_health:
                # Mock dates to trigger daily processing
                mock_now = Mock()
                mock_now.date.side_effect = [
                    Mock(name="day1"),
                    Mock(name="day2"),  # Different day triggers process_day
                    Mock(name="day2"),  # Same day after processing
                ]
                mock_datetime.now.return_value = mock_now
                mock_check_health.return_value = []  # No stale processes

                # Use side effect to break loop after first iteration
                mock_sleep.side_effect = KeyboardInterrupt

                try:
                    supervise("/test/journal", daily=True)
                except KeyboardInterrupt:
                    pass

                # Should have called process_day but NOT spawn_scheduled_agents
                mock_run_process_day.assert_called_once()
                mock_spawn_scheduled.assert_not_called()
