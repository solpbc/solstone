# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration tests for MCP server with full protocol testing."""

import json
import os

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_tool_registration(integration_journal_path):
    """Test that MCP server registers all tools correctly via direct API.

    This is a fast smoke test that verifies all tools are properly registered
    after the modular refactoring. It doesn't test the MCP protocol itself,
    just the registration.
    """
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    # Import after setting JOURNAL_PATH
    from think.mcp import TOOL_PACKS, mcp

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

    assert "entity_list" in tools
    assert tools["entity_list"].name == "entity_list"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_stdio_e2e(integration_journal_path):
    """Test MCP server end-to-end via stdio transport with actual tool call.

    This test validates the complete MCP stack:
    1. Server starts via stdio transport
    2. Client can connect and initialize
    3. Tools are discoverable via MCP protocol
    4. Tools can be invoked and return correct results
    5. Error handling works properly
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

    # Configure server parameters - use sol mcp entry point
    server_params = StdioServerParameters(
        command="sol",
        args=["mcp", "--transport", "stdio"],
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
            assert "entity_list" in tool_names, "entity tool missing"
            assert "entity_attach" in tool_names, "entity tool missing"
            assert "send_message" in tool_names, "messaging tool missing"

            # Verify we have the expected number of tools
            assert (
                len(tool_names) >= 10
            ), f"Expected at least 10 tools, got {len(tool_names)}"

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

            # Test 3: Error handling - call tool with missing required argument
            # FastMCP validates arguments and returns error result (not exception)
            error_result = await session.call_tool(
                "todo_list",
                arguments={"day": day},  # Missing 'facet' argument
            )
            # Server logs the error but returns a result with isError=True
            assert error_result.isError, "Expected error result for missing argument"

            # Test 4: Successful tool call with nonexistent facet
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

    This test ensures that the TOOL_PACKS dictionary in think/mcp.py
    accurately reflects what's actually registered, which is important
    for agents that use tool packs.
    """
    os.environ["JOURNAL_PATH"] = str(integration_journal_path)

    from think.mcp import TOOL_PACKS, mcp

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
        command="sol",
        args=["mcp", "--transport", "stdio"],
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

            # Call 2: Add a todo
            result2 = await session.call_tool(
                "todo_add",
                arguments={
                    "day": day,
                    "facet": facet,
                    "line_number": 1,
                    "text": "second call item",
                },
            )
            data2 = json.loads(result2.content[0].text)
            assert "markdown" in data2

            # Both calls should succeed
            assert not result1.isError
            assert not result2.isError
