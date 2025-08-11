#!/usr/bin/env python3
"""Tests for HTTP MCP integration in utils.py."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from think.utils import create_mcp_client


class TestCreateMCPClientHTTP:
    """Test HTTP MCP client creation and URI auto-discovery."""

    def test_with_uri_file(self):
        """Test client uses URI file when available."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create agents/mcp.uri file
            agents_dir = Path(tmp_dir) / "agents"
            agents_dir.mkdir()
            uri_file = agents_dir / "mcp.uri"
            test_uri = "http://127.0.0.1:6270/mcp/"
            uri_file.write_text(test_uri)

            with patch.dict(os.environ, {"JOURNAL_PATH": tmp_dir}):
                with patch("fastmcp.Client") as mock_client:
                    result = create_mcp_client()

                    # Should create client with discovered URI
                    mock_client.assert_called_once_with(test_uri)
                    assert result == mock_client.return_value

    def test_uri_file_whitespace_handling(self):
        """Test URI file content is properly stripped of whitespace."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create agents/mcp.uri file with whitespace
            agents_dir = Path(tmp_dir) / "agents"
            agents_dir.mkdir()
            uri_file = agents_dir / "mcp.uri"
            test_uri = "http://127.0.0.1:6270/mcp/"
            uri_file.write_text(f"  {test_uri}  \n")

            with patch.dict(os.environ, {"JOURNAL_PATH": tmp_dir}):
                with patch("fastmcp.Client") as mock_client:
                    result = create_mcp_client()

                    # Should strip whitespace from URI
                    mock_client.assert_called_once_with(
                        test_uri
                    )  # stripped, no whitespace
                    assert result == mock_client.return_value

    def test_no_journal_path(self):
        """Test error when JOURNAL_PATH is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="JOURNAL_PATH not set"):
                create_mcp_client()

    def test_uri_file_not_exists(self):
        """Test error when URI file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(os.environ, {"JOURNAL_PATH": tmp_dir}):
                with pytest.raises(RuntimeError, match="MCP server URI file not found"):
                    create_mcp_client()

    def test_uri_file_read_error(self):
        """Test error when URI file can't be read."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create agents directory
            agents_dir = Path(tmp_dir) / "agents"
            agents_dir.mkdir()
            # Create a directory instead of file to cause read error
            (agents_dir / "mcp.uri").mkdir()

            with patch.dict(os.environ, {"JOURNAL_PATH": tmp_dir}):
                with pytest.raises(RuntimeError, match="Failed to read MCP server URI"):
                    create_mcp_client()

    def test_uri_file_empty(self):
        """Test error when URI file is empty."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create agents/mcp.uri file but leave it empty
            agents_dir = Path(tmp_dir) / "agents"
            agents_dir.mkdir()
            uri_file = agents_dir / "mcp.uri"
            uri_file.write_text("")

            with patch.dict(os.environ, {"JOURNAL_PATH": tmp_dir}):
                with pytest.raises(RuntimeError, match="MCP server URI file is empty"):
                    create_mcp_client()

    def test_uri_file_whitespace_only(self):
        """Test error when URI file contains only whitespace."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create agents/mcp.uri file with only whitespace
            agents_dir = Path(tmp_dir) / "agents"
            agents_dir.mkdir()
            uri_file = agents_dir / "mcp.uri"
            uri_file.write_text("   \n  \t  ")

            with patch.dict(os.environ, {"JOURNAL_PATH": tmp_dir}):
                with pytest.raises(RuntimeError, match="MCP server URI file is empty"):
                    create_mcp_client()
