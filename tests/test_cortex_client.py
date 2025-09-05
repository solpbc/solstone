"""Tests for the Cortex file-based client."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from think.cortex_client import CortexClient, run_agent, run_agent_with_events


@pytest.fixture
def temp_journal(tmp_path):
    """Create a temporary journal directory."""
    journal_path = tmp_path / "journal"
    journal_path.mkdir()
    agents_dir = journal_path / "agents"
    agents_dir.mkdir()
    return journal_path


@pytest.fixture
def mock_time():
    """Mock time.time() for consistent agent IDs."""
    with patch("think.cortex_client.time.time") as mock:
        mock.return_value = 1234567890.123
        yield mock


class TestCortexClient:
    """Test CortexClient class."""

    def test_init_with_journal_path(self, temp_journal):
        """Test initialization with explicit journal path."""
        client = CortexClient(journal_path=str(temp_journal))
        assert client.journal_path == temp_journal
        assert client.agents_dir == temp_journal / "agents"

    def test_init_with_env_var(self, temp_journal, monkeypatch):
        """Test initialization with JOURNAL_PATH env var."""
        monkeypatch.setenv("JOURNAL_PATH", str(temp_journal))
        client = CortexClient()
        assert client.journal_path == temp_journal
        assert client.agents_dir == temp_journal / "agents"

    def test_init_creates_agents_dir(self, tmp_path):
        """Test that initialization creates the agents directory."""
        journal_path = tmp_path / "new_journal"
        client = CortexClient(journal_path=str(journal_path))
        assert client.agents_dir.exists()

    @pytest.mark.asyncio
    async def test_context_manager(self, temp_journal):
        """Test async context manager."""
        async with CortexClient(journal_path=str(temp_journal)) as client:
            assert isinstance(client, CortexClient)
            assert client.agents_dir.exists()

    @pytest.mark.asyncio
    async def test_spawn_creates_request_file(self, temp_journal, mock_time):
        """Test that spawn creates and activates request file."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = await client.spawn(
            prompt="Test prompt",
            persona="tester",
            backend="openai",
            config={"model": "gpt-4"},
            handoff={"persona": "reviewer"},
            handoff_from="previous_agent"
        )
        
        assert agent_id == "1234567890123"
        
        # Check that active file exists
        active_file = client.agents_dir / f"{agent_id}_active.jsonl"
        assert active_file.exists()
        
        # Check request content
        with open(active_file, "r") as f:
            request = json.loads(f.readline())
            
        assert request["event"] == "request"
        assert request["ts"] == 1234567890123
        assert request["prompt"] == "Test prompt"
        assert request["persona"] == "tester"
        assert request["backend"] == "openai"
        assert request["config"] == {"model": "gpt-4"}
        assert request["handoff"] == {"persona": "reviewer"}
        assert request["handoff_from"] == "previous_agent"

    @pytest.mark.asyncio
    async def test_spawn_minimal_params(self, temp_journal, mock_time):
        """Test spawn with minimal parameters."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = await client.spawn("Simple prompt")
        
        active_file = client.agents_dir / f"{agent_id}_active.jsonl"
        with open(active_file, "r") as f:
            request = json.loads(f.readline())
            
        assert request["prompt"] == "Simple prompt"
        assert request["persona"] == "default"
        assert request["backend"] == "openai"
        assert request["config"] == {}
        assert "handoff" not in request
        assert "handoff_from" not in request

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, temp_journal):
        """Test listing agents when none exist."""
        client = CortexClient(journal_path=str(temp_journal))
        
        result = await client.list_agents()
        
        assert result["agents"] == []
        assert result["pagination"]["total"] == 0
        assert result["pagination"]["has_more"] is False

    @pytest.mark.asyncio
    async def test_list_agents_with_files(self, temp_journal):
        """Test listing agents with various file states."""
        client = CortexClient(journal_path=str(temp_journal))
        
        # Create some agent files
        agents_data = [
            ("1234567890001", "active", {"event": "request", "ts": 1234567890001, "prompt": "Test 1", "persona": "default", "backend": "openai"}),
            ("1234567890002", "completed", {"event": "request", "ts": 1234567890002, "prompt": "Test 2", "persona": "tester", "backend": "google"}),
            ("1234567890003", "completed", {"event": "request", "ts": 1234567890003, "prompt": "Test 3", "persona": "reviewer", "backend": "anthropic"}),
        ]
        
        for agent_id, status, request in agents_data:
            if status == "active":
                file_path = client.agents_dir / f"{agent_id}_active.jsonl"
            else:
                file_path = client.agents_dir / f"{agent_id}.jsonl"
                
            with open(file_path, "w") as f:
                f.write(json.dumps(request) + "\n")
                
        result = await client.list_agents(limit=2, offset=0)
        
        assert len(result["agents"]) == 2
        assert result["pagination"]["total"] == 3
        assert result["pagination"]["has_more"] is True
        
        # Check first agent
        agent = result["agents"][0]
        assert agent["status"] in ["running", "completed"]
        assert agent["persona"] in ["default", "tester", "reviewer"]
        assert agent["backend"] in ["openai", "google", "anthropic"]

    @pytest.mark.asyncio
    async def test_list_agents_exclude_active(self, temp_journal):
        """Test listing agents excluding active ones."""
        client = CortexClient(journal_path=str(temp_journal))
        
        # Create active and completed files
        active_file = client.agents_dir / "1234567890001_active.jsonl"
        completed_file = client.agents_dir / "1234567890002.jsonl"
        
        for file_path in [active_file, completed_file]:
            with open(file_path, "w") as f:
                f.write(json.dumps({"event": "request", "ts": 1234567890000, "prompt": "Test", "persona": "default", "backend": "openai"}) + "\n")
                
        result = await client.list_agents(include_active=False)
        
        # Should only include completed file
        assert len(result["agents"]) == 1
        assert result["agents"][0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_read_events_from_active_file(self, temp_journal):
        """Test reading events from an active agent file."""
        client = CortexClient(journal_path=str(temp_journal))
        
        # Create active file with events
        agent_id = "1234567890001"
        active_file = client.agents_dir / f"{agent_id}_active.jsonl"
        
        events = [
            {"event": "request", "ts": 1234567890001, "prompt": "Test"},
            {"event": "start", "ts": 1234567890002, "persona": "default"},
            {"event": "tool_start", "ts": 1234567890003, "tool": "search"},
            {"event": "tool_end", "ts": 1234567890004, "tool": "search", "result": "data"},
            {"event": "finish", "ts": 1234567890005, "result": "Done"},
        ]
        
        with open(active_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
                
        # Read events
        received_events = []
        await client.read_events(
            agent_id, 
            lambda e: received_events.append(e),
            follow=False
        )
        
        assert len(received_events) == 5
        assert received_events[0]["event"] == "request"
        assert received_events[-1]["event"] == "finish"

    @pytest.mark.asyncio
    async def test_read_events_from_completed_file(self, temp_journal):
        """Test reading events from a completed agent file."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = "1234567890001"
        completed_file = client.agents_dir / f"{agent_id}.jsonl"
        
        events = [
            {"event": "request", "ts": 1234567890001},
            {"event": "finish", "ts": 1234567890002, "result": "Completed"},
        ]
        
        with open(completed_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
                
        received_events = []
        await client.read_events(
            agent_id,
            lambda e: received_events.append(e),
            follow=False
        )
        
        assert len(received_events) == 2
        assert received_events[1]["result"] == "Completed"

    @pytest.mark.asyncio
    async def test_read_events_file_not_found(self, temp_journal):
        """Test reading events from non-existent agent."""
        client = CortexClient(journal_path=str(temp_journal))
        
        with pytest.raises(FileNotFoundError, match="Agent file not found"):
            await client.read_events("nonexistent", lambda e: None)

    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self, temp_journal):
        """Test waiting for agent completion."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = "1234567890001"
        active_file = client.agents_dir / f"{agent_id}_active.jsonl"
        
        # Write initial events
        with open(active_file, "w") as f:
            f.write(json.dumps({"event": "request", "ts": 1234567890001}) + "\n")
            f.write(json.dumps({"event": "start", "ts": 1234567890002}) + "\n")
            
        # Simulate agent completing after a delay
        async def complete_agent():
            await asyncio.sleep(0.1)
            with open(active_file, "a") as f:
                f.write(json.dumps({"event": "finish", "ts": 1234567890003, "result": "Task completed"}) + "\n")
                
        # Start completion in background
        asyncio.create_task(complete_agent())
        
        # Wait for completion
        result = await client.wait_for_completion(agent_id, timeout=1)
        assert result == "Task completed"

    @pytest.mark.asyncio
    async def test_wait_for_completion_error(self, temp_journal):
        """Test wait_for_completion with agent error."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = "1234567890001"
        active_file = client.agents_dir / f"{agent_id}_active.jsonl"
        
        with open(active_file, "w") as f:
            f.write(json.dumps({"event": "request", "ts": 1234567890001}) + "\n")
            f.write(json.dumps({"event": "error", "ts": 1234567890002, "error": "Agent failed"}) + "\n")
            
        with pytest.raises(RuntimeError, match="Agent error: Agent failed"):
            await client.wait_for_completion(agent_id)

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self, temp_journal):
        """Test wait_for_completion timeout."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = "1234567890001"
        active_file = client.agents_dir / f"{agent_id}_active.jsonl"
        
        # Create file with no finish event
        with open(active_file, "w") as f:
            f.write(json.dumps({"event": "request", "ts": 1234567890001}) + "\n")
            f.write(json.dumps({"event": "start", "ts": 1234567890002}) + "\n")
            
        with pytest.raises(TimeoutError, match="timed out after 0.1 seconds"):
            await client.wait_for_completion(agent_id, timeout=0.1)

    @pytest.mark.asyncio
    async def test_get_agent_status_running(self, temp_journal):
        """Test getting status of running agent."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = "1234567890001"
        active_file = client.agents_dir / f"{agent_id}_active.jsonl"
        active_file.touch()
        
        status = await client.get_agent_status(agent_id)
        assert status == "running"

    @pytest.mark.asyncio
    async def test_get_agent_status_completed(self, temp_journal):
        """Test getting status of completed agent."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = "1234567890001"
        completed_file = client.agents_dir / f"{agent_id}.jsonl"
        
        with open(completed_file, "w") as f:
            f.write(json.dumps({"event": "finish", "ts": 1234567890001}) + "\n")
            
        status = await client.get_agent_status(agent_id)
        assert status == "completed"

    @pytest.mark.asyncio
    async def test_get_agent_status_failed(self, temp_journal):
        """Test getting status of failed agent."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = "1234567890001"
        completed_file = client.agents_dir / f"{agent_id}.jsonl"
        
        with open(completed_file, "w") as f:
            f.write(json.dumps({"event": "error", "ts": 1234567890001}) + "\n")
            
        status = await client.get_agent_status(agent_id)
        assert status == "failed"

    @pytest.mark.asyncio
    async def test_get_agent_status_not_found(self, temp_journal):
        """Test getting status of non-existent agent."""
        client = CortexClient(journal_path=str(temp_journal))
        
        status = await client.get_agent_status("nonexistent")
        assert status == "not_found"

    @pytest.mark.asyncio
    async def test_get_agent_events(self, temp_journal):
        """Test getting all events for an agent."""
        client = CortexClient(journal_path=str(temp_journal))
        
        agent_id = "1234567890001"
        completed_file = client.agents_dir / f"{agent_id}.jsonl"
        
        events = [
            {"event": "request", "ts": 1234567890001},
            {"event": "start", "ts": 1234567890002},
            {"event": "finish", "ts": 1234567890003},
        ]
        
        with open(completed_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
                
        result = await client.get_agent_events(agent_id)
        
        assert len(result) == 3
        assert result[0]["event"] == "request"
        assert result[2]["event"] == "finish"

    @pytest.mark.asyncio
    async def test_get_agent_events_not_found(self, temp_journal):
        """Test getting events for non-existent agent."""
        client = CortexClient(journal_path=str(temp_journal))
        
        with pytest.raises(FileNotFoundError, match="Agent file not found"):
            await client.get_agent_events("nonexistent")


class TestHelperFunctions:
    """Test helper functions."""

    @pytest.mark.asyncio
    async def test_run_agent(self, temp_journal, mock_time):
        """Test run_agent helper."""
        # Create a mock agent file that completes
        agent_id = "1234567890123"
        active_file = temp_journal / "agents" / f"{agent_id}_active.jsonl"
        
        async def simulate_agent():
            await asyncio.sleep(0.05)
            with open(active_file, "a") as f:
                f.write(json.dumps({"event": "finish", "ts": 1234567890124, "result": "Test result"}) + "\n")
                
        # Start simulation in background
        asyncio.create_task(simulate_agent())
        
        result = await run_agent(
            "Test prompt",
            journal_path=str(temp_journal),
            persona="tester",
            backend="openai"
        )
        
        assert result == "Test result"

    @pytest.mark.asyncio
    async def test_run_agent_with_events(self, temp_journal, mock_time):
        """Test run_agent_with_events helper."""
        events_received = []
        
        def on_event(event):
            events_received.append(event)
            
        # Create a mock agent file with events
        agent_id = "1234567890123"
        active_file = temp_journal / "agents" / f"{agent_id}_active.jsonl"
        
        async def simulate_agent():
            await asyncio.sleep(0.05)
            with open(active_file, "a") as f:
                f.write(json.dumps({"event": "tool_start", "ts": 1234567890124, "tool": "search"}) + "\n")
                f.write(json.dumps({"event": "tool_end", "ts": 1234567890125, "tool": "search"}) + "\n")
                f.write(json.dumps({"event": "finish", "ts": 1234567890126, "result": "Done"}) + "\n")
                
        # Start simulation in background
        asyncio.create_task(simulate_agent())
        
        result = await run_agent_with_events(
            "Test prompt",
            on_event,
            journal_path=str(temp_journal)
        )
        
        assert result == "Done"
        assert len(events_received) == 4  # request + tool_start + tool_end + finish
        assert events_received[0]["event"] == "request"
        assert events_received[1]["event"] == "tool_start"
        assert events_received[2]["event"] == "tool_end"
        assert events_received[3]["event"] == "finish"
        # Check that agent_id was added to events
        assert all("agent_id" in e for e in events_received)

    @pytest.mark.asyncio
    async def test_run_agent_with_async_event_handler(self, temp_journal, mock_time):
        """Test run_agent_with_events with async event handler."""
        events_received = []
        
        async def on_event(event):
            await asyncio.sleep(0.01)  # Simulate async work
            events_received.append(event)
            
        agent_id = "1234567890123"
        active_file = temp_journal / "agents" / f"{agent_id}_active.jsonl"
        
        async def simulate_agent():
            await asyncio.sleep(0.05)
            with open(active_file, "a") as f:
                f.write(json.dumps({"event": "finish", "ts": 1234567890124, "result": "Done"}) + "\n")
                
        asyncio.create_task(simulate_agent())
        
        result = await run_agent_with_events(
            "Test prompt",
            on_event,
            journal_path=str(temp_journal)
        )
        
        assert result == "Done"
        assert len(events_received) == 2  # request + finish