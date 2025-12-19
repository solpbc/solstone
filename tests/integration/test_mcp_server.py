"""Integration tests for MCP server with full protocol testing."""

import asyncio
import json
import os
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_tool_registration(integration_journal_path):
    """Test that MCP server registers all tools correctly via direct API.

    This is a fast smoke test that verifies all tools and resources are
    properly registered after the modular refactoring. It doesn't test
    the MCP protocol itself, just the registration.
    """
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Import after setting JOURNAL_PATH
    from muse.mcp import TOOL_PACKS, mcp

    # Get all registered tools
    tools = await mcp.get_tools()

    # Verify all expected tools are registered
    expected_tools = set()
    for pack_tools in TOOL_PACKS.values():
        expected_tools.update(pack_tools)

    registered_tool_names = set(tools.keys())

    # Assert all tools are present
    assert expected_tools.issubset(
        registered_tool_names
    ), f"Missing tools: {expected_tools - registered_tool_names}"

    # Verify specific tool metadata
    assert "todo_list" in tools
    assert tools["todo_list"].name == "todo_list"
    # Parameters is a JSON schema dict
    tool_params = tools["todo_list"].parameters
    if isinstance(tool_params, dict):
        # Check for JSON schema structure
        if "properties" in tool_params:
            assert "day" in tool_params["properties"]
            assert "facet" in tool_params["properties"]
        else:
            assert "day" in tool_params
    else:
        # It's a list of parameter objects
        param_names = [p.name if hasattr(p, "name") else p for p in tool_params]
        assert "day" in param_names

    assert "search_journal" in tools
    assert tools["search_journal"].name == "search_journal"

    assert "entity_list" in tools
    assert tools["entity_list"].name == "entity_list"

    # Get all registered resources
    resources = await mcp.get_resources()

    # Resources may be empty if using templates, check resource templates instead
    if len(resources) == 0:
        # Try getting resource templates
        try:
            templates = await mcp.get_resource_templates()
            template_uris = [t.uriTemplate for t in templates]
            assert any("journal://insight" in uri for uri in template_uris)
            assert any("journal://transcripts" in uri for uri in template_uris)
        except AttributeError:
            # Older FastMCP version, skip resource check
            pass
    else:
        # Verify key resources are present
        resource_uris = [r.uri for r in resources]
        assert any("journal://insight" in uri for uri in resource_uris)
        assert any("journal://transcripts" in uri for uri in resource_uris)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_stdio_e2e(integration_journal_path):
    """Test MCP server end-to-end via stdio transport with actual tool call.

    This test validates the complete MCP stack:
    1. Server starts via stdio transport
    2. Client can connect and initialize
    3. Tools are discoverable via MCP protocol
    4. Tools can be invoked and return correct results
    5. Resources are accessible
    6. Error handling works properly
    """
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Create test todo for the test
    facet = "test-facet"
    day = "20991231"  # Future date to avoid validation issues

    facets_dir = integration_journal_path / "facets" / facet
    facets_dir.mkdir(parents=True, exist_ok=True)

    # Create facet.json
    facet_json = facets_dir / "facet.json"
    facet_json.write_text(
        json.dumps({"title": "Test Facet", "description": "Integration test"}),
        encoding="utf-8",
    )

    # Create todos directory with a test todo (JSONL format)
    todos_dir = facets_dir / "todos"
    todos_dir.mkdir(exist_ok=True)
    todo_file = todos_dir / f"{day}.jsonl"
    todo_file.write_text(
        json.dumps({"text": "Integration test todo"}) + "\n", encoding="utf-8"
    )

    # Configure server parameters - use muse-mcp-tools entry point
    server_params = StdioServerParameters(
        command="muse-mcp-tools",
        args=["--transport", "stdio"],
        env={**os.environ, "JOURNAL_PATH": str(integration_journal_path)},
    )

    # Connect to server via stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # Test 1: List all tools
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]

            # Verify key tools are present from each module
            assert "todo_list" in tool_names, "todo tool missing"
            assert "todo_add" in tool_names, "todo tool missing"
            assert "search_journal" in tool_names, "search tool missing"
            assert "get_events" in tool_names, "events tool missing"
            assert "entity_list" in tool_names, "entity tool missing"
            assert "entity_attach" in tool_names, "entity tool missing"
            assert "get_facet" in tool_names, "facet tool missing"
            assert "facet_news" in tool_names, "facet tool missing"
            assert "send_message" in tool_names, "messaging tool missing"
            assert "get_resource" in tool_names, "messaging tool missing"

            # Verify we have the expected number of tools (15 core tools after unification)
            assert (
                len(tool_names) >= 15
            ), f"Expected at least 15 tools, got {len(tool_names)}"

            # Test 2: Call a tool (todo_list)
            result = await session.call_tool(
                "todo_list", arguments={"day": day, "facet": facet}
            )

            # Verify result structure
            assert not result.isError, "Tool call returned error"
            assert len(result.content) > 0, "Tool call returned no content"

            # Parse the result
            result_text = result.content[0].text
            result_data = json.loads(result_text)

            # Verify the todo was returned
            assert result_data["day"] == day
            assert result_data["facet"] == facet
            assert "Integration test todo" in result_data["markdown"]
            assert "1:" in result_data["markdown"], "Expected numbered output"

            # Test 3: List resources (may be empty if using templates)
            resources_result = await session.list_resources()
            # Resources might be empty - the server uses resource templates
            # which are dynamically resolved. This is expected behavior.

            # Test 4: Call another tool to verify different module
            # Call search_journal to test search module
            search_result = await session.call_tool(
                "search_journal",
                arguments={"query": "test", "limit": 1},
            )

            assert not search_result.isError
            search_data = json.loads(search_result.content[0].text)
            assert "total" in search_data
            assert "results" in search_data
            assert "limit" in search_data

            # Test 5: Error handling - call tool with missing required argument
            # FastMCP validates arguments and returns error result (not exception)
            error_result = await session.call_tool(
                "todo_list",
                arguments={"day": day},  # Missing 'facet' argument
            )
            # Server logs the error but returns a result with isError=True
            assert error_result.isError, "Expected error result for missing argument"

            # Test 6: Successful tool call with nonexistent facet
            # Note: todo_list creates empty todos file if it doesn't exist
            # This is by design, not an error
            result_nonexistent = await session.call_tool(
                "todo_list",
                arguments={"day": "20991231", "facet": "nonexistent-facet"},
            )

            # Should succeed and return empty todos
            assert not result_nonexistent.isError
            data_nonexistent = json.loads(result_nonexistent.content[0].text)
            assert "markdown" in data_nonexistent
            assert "0:" in data_nonexistent["markdown"]  # Empty todos


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_tool_packs_coverage(integration_journal_path):
    """Verify all tool packs have their tools registered.

    This test ensures that the TOOL_PACKS dictionary in muse/mcp.py
    accurately reflects what's actually registered, which is important
    for agents that use tool packs.
    """
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    from muse.mcp import TOOL_PACKS, mcp

    tools = await mcp.get_tools()
    registered_tool_names = set(tools.keys())

    # Check each tool pack
    for pack_name, pack_tools in TOOL_PACKS.items():
        for tool_name in pack_tools:
            assert (
                tool_name in registered_tool_names
            ), f"Tool '{tool_name}' from pack '{pack_name}' not registered"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_multiple_tool_calls(integration_journal_path):
    """Test making multiple sequential tool calls to verify state handling.

    This ensures the server can handle multiple requests without issues.
    """
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Setup test data
    facet = "multi-test"
    day = "20991231"

    facets_dir = integration_journal_path / "facets" / facet
    facets_dir.mkdir(parents=True, exist_ok=True)

    facet_json = facets_dir / "facet.json"
    facet_json.write_text(
        json.dumps({"title": "Multi Test", "description": "Multiple calls test"}),
        encoding="utf-8",
    )

    todos_dir = facets_dir / "todos"
    todos_dir.mkdir(exist_ok=True)

    server_params = StdioServerParameters(
        command="muse-mcp-tools",
        args=["--transport", "stdio"],
        env={**os.environ, "JOURNAL_PATH": str(integration_journal_path)},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Call 1: List empty todos
            result1 = await session.call_tool(
                "todo_list", arguments={"day": day, "facet": facet}
            )
            data1 = json.loads(result1.content[0].text)
            # Empty todos returns "0: (no todos)", not an error
            assert "markdown" in data1

            # Call 2: Search journal
            result2 = await session.call_tool(
                "search_journal", arguments={"query": "test", "limit": 5}
            )
            data2 = json.loads(result2.content[0].text)
            assert "results" in data2

            # Call 3: Get facet info
            result3 = await session.call_tool("get_facet", arguments={"facet": facet})
            data3 = json.loads(result3.content[0].text)
            assert "facet" in data3
            assert data3["facet"] == facet

            # All three calls should succeed
            assert not result1.isError
            assert not result2.isError
            assert not result3.isError


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_get_resource_tool(integration_journal_path):
    """Test the get_resource tool can fetch journal resources.

    This verifies that the Context injection works correctly and
    resources can be fetched via the tool wrapper.
    """
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Setup test data - create a todo resource to fetch
    facet = "resource-test"
    day = "20991231"

    facets_dir = integration_journal_path / "facets" / facet
    facets_dir.mkdir(parents=True, exist_ok=True)

    facet_json = facets_dir / "facet.json"
    facet_json.write_text(
        json.dumps({"title": "Resource Test", "description": "Test get_resource"}),
        encoding="utf-8",
    )

    todos_dir = facets_dir / "todos"
    todos_dir.mkdir(exist_ok=True)
    todo_file = todos_dir / f"{day}.jsonl"
    # JSONL format: one JSON object per line
    todo_file.write_text(
        json.dumps({"text": "Test resource fetch"})
        + "\n"
        + json.dumps({"text": "Already done", "completed": True})
        + "\n",
        encoding="utf-8",
    )

    server_params = StdioServerParameters(
        command="muse-mcp-tools",
        args=["--transport", "stdio"],
        env={**os.environ, "JOURNAL_PATH": str(integration_journal_path)},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Call get_resource to fetch the todo resource
            result = await session.call_tool(
                "get_resource",
                arguments={"uri": f"journal://todo/{facet}/{day}"},
            )

            # Verify success
            assert not result.isError, f"get_resource failed: {result.content}"
            assert len(result.content) > 0

            # The result should contain the todo content
            result_text = result.content[0].text
            assert "Test resource fetch" in result_text
            assert "Already done" in result_text
