"""Tests for the Cortex WebSocket client."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from think.cortex_client import CortexClient, run_agent, run_agent_with_events


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    ws.close = AsyncMock()

    # Make ws async iterable by default (empty)
    async def default_aiter():
        return
        yield  # Make it a generator

    ws.__aiter__ = lambda self: default_aiter()
    return ws


@pytest.fixture
def mock_connect(mock_websocket):
    """Mock websockets.connect."""
    with patch(
        "think.cortex_client.websockets.connect",
        new=AsyncMock(return_value=mock_websocket),
    ):
        yield mock_websocket


@pytest.fixture
def temp_journal(tmp_path):
    """Create a temporary journal directory with cortex.uri."""
    journal_path = tmp_path / "journal"
    journal_path.mkdir()
    agents_dir = journal_path / "agents"
    agents_dir.mkdir()

    # Write custom cortex URI
    uri_file = agents_dir / "cortex.uri"
    uri_file.write_text("ws://test-cortex:9999/ws/cortex")

    return journal_path


class TestCortexClient:
    """Test CortexClient class."""

    def test_uri_discovery_from_file(self, temp_journal, monkeypatch):
        """Test URI discovery from cortex.uri file."""
        monkeypatch.setenv("JOURNAL_PATH", str(temp_journal))

        client = CortexClient()
        assert client.uri == "ws://test-cortex:9999/ws/cortex"

    def test_uri_discovery_fallback(self, monkeypatch):
        """Test URI fallback when no journal path."""
        monkeypatch.delenv("JOURNAL_PATH", raising=False)

        client = CortexClient()
        assert client.uri == "ws://127.0.0.1:2468/ws/cortex"

    def test_uri_explicit(self):
        """Test explicit URI overrides discovery."""
        client = CortexClient(uri="ws://custom:1234/ws")
        assert client.uri == "ws://custom:1234/ws"

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_connect):
        """Test successful connection."""
        client = CortexClient()

        await client.connect()

        assert client.ws == mock_connect
        mock_connect.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mock_connect):
        """Test connect when already connected."""
        client = CortexClient()

        await client.connect()
        await client.connect()  # Second call should be no-op

        # Should only connect once
        with patch("think.cortex_client.websockets.connect") as mock:
            await client.connect()
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_connect):
        """Test disconnection."""
        client = CortexClient()

        await client.connect()
        await client.disconnect()

        mock_connect.close.assert_called_once()
        assert client.ws is None

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_connect):
        """Test async context manager."""
        async with CortexClient() as client:
            assert client.ws == mock_connect

        mock_connect.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_success(self, mock_connect):
        """Test successful agent spawn."""
        mock_connect.recv.return_value = json.dumps(
            {"type": "agent_spawned", "agent_id": "test_agent_123"}
        )

        client = CortexClient()
        await client.connect()

        agent_id = await client.spawn(
            prompt="Test prompt",
            persona="tester",
            backend="openai",
            config={"model": "gpt-4"},
            handoff={"persona": "reviewer"},
        )

        assert agent_id == "test_agent_123"

        # Check request sent
        mock_connect.send.assert_called_once()
        sent_data = json.loads(mock_connect.send.call_args[0][0])
        assert sent_data["action"] == "spawn"
        assert sent_data["prompt"] == "Test prompt"
        assert sent_data["persona"] == "tester"
        assert sent_data["backend"] == "openai"
        assert sent_data["config"] == {"model": "gpt-4"}
        assert sent_data["handoff"] == {"persona": "reviewer"}

    @pytest.mark.asyncio
    async def test_spawn_error(self, mock_connect):
        """Test spawn error handling."""
        mock_connect.recv.return_value = json.dumps(
            {"type": "error", "message": "Spawn failed: invalid persona"}
        )

        client = CortexClient()
        await client.connect()

        with pytest.raises(RuntimeError, match="Spawn failed: invalid persona"):
            await client.spawn("Test prompt")

    @pytest.mark.asyncio
    async def test_spawn_not_connected(self):
        """Test spawn when not connected."""
        client = CortexClient()

        with pytest.raises(RuntimeError, match="Not connected to Cortex"):
            await client.spawn("Test prompt")

    @pytest.mark.asyncio
    async def test_list_agents(self, mock_connect):
        """Test listing agents."""
        mock_connect.recv.return_value = json.dumps(
            {
                "type": "agent_list",
                "agents": [
                    {"id": "agent1", "status": "running"},
                    {"id": "agent2", "status": "running"},
                ],
                "pagination": {"limit": 10, "offset": 0, "total": 2, "has_more": False},
            }
        )

        client = CortexClient()
        await client.connect()

        result = await client.list_agents(limit=10, offset=0)

        assert result["type"] == "agent_list"
        assert len(result["agents"]) == 2
        assert result["pagination"]["total"] == 2

        # Check request
        mock_connect.send.assert_called_once()
        sent_data = json.loads(mock_connect.send.call_args[0][0])
        assert sent_data["action"] == "list"
        assert sent_data["limit"] == 10
        assert sent_data["offset"] == 0

    @pytest.mark.asyncio
    async def test_attach_and_events(self, mock_connect):
        """Test attaching to agent and receiving events."""
        events_received = []

        async def on_event(event):
            events_received.append(event)

        # Mock message stream
        messages = [
            json.dumps({"type": "attached", "agent_id": "test_agent"}),
            json.dumps(
                {
                    "type": "agent_event",
                    "event": {"event": "tool_start", "tool": "search"},
                }
            ),
            json.dumps(
                {
                    "type": "agent_event",
                    "event": {"event": "tool_end", "tool": "search", "result": "data"},
                }
            ),
            json.dumps(
                {"type": "agent_event", "event": {"event": "finish", "result": "Done"}}
            ),
            json.dumps({"type": "agent_finished", "agent_id": "test_agent"}),
        ]

        # Create async generator
        async def async_messages():
            for msg in messages:
                yield msg

        mock_connect.__aiter__ = lambda self: async_messages()

        client = CortexClient()
        await client.connect()

        await client.attach("test_agent", on_event)

        # Check events received
        assert len(events_received) == 3
        assert events_received[0]["event"] == "tool_start"
        assert events_received[1]["event"] == "tool_end"
        assert events_received[2]["event"] == "finish"

    @pytest.mark.asyncio
    async def test_detach(self, mock_connect):
        """Test detaching from agent."""
        mock_connect.recv.return_value = json.dumps({"type": "detached"})

        client = CortexClient()
        await client.connect()

        await client.detach()

        mock_connect.send.assert_called_once()
        sent_data = json.loads(mock_connect.send.call_args[0][0])
        assert sent_data["action"] == "detach"

    @pytest.mark.asyncio
    async def test_wait_for_completion(self, mock_connect):
        """Test waiting for agent completion."""

        # Mock attach behavior
        async def mock_attach(agent_id, on_event):
            await on_event({"event": "tool_start", "tool": "search"})
            await on_event({"event": "tool_end", "tool": "search"})
            await on_event({"event": "finish", "result": "Task completed"})

        client = CortexClient()
        await client.connect()
        client.attach = mock_attach

        result = await client.wait_for_completion("test_agent", timeout=5)

        assert result == "Task completed"

    @pytest.mark.asyncio
    async def test_wait_for_completion_error(self, mock_connect):
        """Test wait_for_completion with agent error."""

        async def mock_attach(agent_id, on_event):
            await on_event({"event": "error", "error": "Agent failed"})

        client = CortexClient()
        await client.connect()
        client.attach = mock_attach

        with pytest.raises(RuntimeError, match="Agent error: Agent failed"):
            await client.wait_for_completion("test_agent")

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self, mock_connect):
        """Test wait_for_completion timeout."""

        async def mock_attach(agent_id, on_event):
            # Never send finish event
            await asyncio.sleep(10)

        client = CortexClient()
        await client.connect()
        client.attach = mock_attach

        with pytest.raises(TimeoutError, match="timed out after 0.1 seconds"):
            await client.wait_for_completion("test_agent", timeout=0.1)


class TestHelperFunctions:
    """Test helper functions."""

    @pytest.mark.asyncio
    async def test_run_agent(self, mock_connect):
        """Test run_agent helper."""
        # Mock spawn and wait_for_completion
        with patch("think.cortex_client.CortexClient") as MockClient:
            client = AsyncMock()
            MockClient.return_value.__aenter__.return_value = client
            MockClient.return_value.__aexit__.return_value = None

            client.spawn.return_value = "agent_123"
            client.wait_for_completion.return_value = "Result text"

            result = await run_agent("Test prompt", persona="tester", backend="openai")

            assert result == "Result text"
            client.spawn.assert_called_once_with(
                "Test prompt", persona="tester", backend="openai"
            )
            client.wait_for_completion.assert_called_once_with("agent_123")

    @pytest.mark.asyncio
    async def test_run_agent_with_events(self, mock_connect):
        """Test run_agent_with_events helper."""
        events_received = []

        def on_event(event):
            events_received.append(event)

        with patch("think.cortex_client.CortexClient") as MockClient:
            client = AsyncMock()
            MockClient.return_value.__aenter__.return_value = client
            MockClient.return_value.__aexit__.return_value = None

            client.spawn.return_value = "agent_123"

            # Mock attach to send events
            async def mock_attach(agent_id, handler):
                await handler({"event": "tool_start", "tool": "search"})
                await handler({"event": "finish", "result": "Done"})

            client.attach = mock_attach

            result = await run_agent_with_events(
                "Test prompt", on_event, persona="tester"
            )

            assert result == "Done"
            assert len(events_received) == 2
            assert events_received[0]["event"] == "tool_start"
            assert events_received[1]["event"] == "finish"
