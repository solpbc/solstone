import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest


def test_agent_instructions_default():
    utils = importlib.import_module("think.utils")
    system, extra, meta = utils.agent_instructions()
    assert system.startswith("You are Sunstone")
    assert "Current Date and Time" in extra
    assert meta.get("title") == "Journal Chat"


def test_agent_instructions_with_domains(monkeypatch):
    """Test that agent instructions include domains with correct hashtag format."""
    utils = importlib.import_module("think.utils")
    
    # Use the fixtures journal which has test domains
    fixtures_path = Path(__file__).parent.parent / "fixtures" / "journal"
    monkeypatch.setenv("JOURNAL_PATH", str(fixtures_path))
    
    system, extra, meta = utils.agent_instructions("default")
    
    # Domains are added to the extra context, not system instructions
    # Check that domains section exists
    assert "## Domains" in extra
    
    # Check for specific domain with new hashtag format
    # Format should be: Domain: 🧪 Test Domain (#test-domain) - Description
    assert "(#test-domain)" in extra
    assert "🧪 Test Domain (#test-domain)" in extra
    assert "A test domain for validating matter functionality" in extra
    
    # Check other domains are included with hashtag format
    assert "(#full-featured)" in extra
    assert "🚀 Full Featured Domain (#full-featured)" in extra
    assert "(#minimal-domain)" in extra
    assert "Minimal Domain (#minimal-domain)" in extra


def test_agent_instructions_domain_format(monkeypatch):
    """Test the exact format of domain entries in agent instructions."""
    utils = importlib.import_module("think.utils")
    
    fixtures_path = Path(__file__).parent.parent / "fixtures" / "journal"
    monkeypatch.setenv("JOURNAL_PATH", str(fixtures_path))
    
    system, extra, meta = utils.agent_instructions("default")
    
    # Extract just the domains section from extra context
    lines = extra.split('\n')
    domain_lines = []
    in_domains = False
    
    for line in lines:
        if '## Domains' in line:
            in_domains = True
        elif in_domains and line.startswith('## '):
            break
        elif in_domains and line.strip().startswith('* Domain:'):
            domain_lines.append(line.strip())
    
    # Should have at least one domain
    assert len(domain_lines) > 0
    
    # Check format of each domain line
    for line in domain_lines:
        assert line.startswith("* Domain:")
        # Should contain hashtag format
        assert "(#" in line and ")" in line
        
    # Check specific expected formats
    expected_patterns = [
        "* Domain: 🧪 Test Domain (#test-domain) - A test domain for validating matter functionality",
        "* Domain: 🚀 Full Featured Domain (#full-featured) - A domain for testing all features",
        "* Domain: Minimal Domain (#minimal-domain)"  # No description
    ]
    
    for pattern in expected_patterns:
        assert any(pattern in line for line in domain_lines), f"Expected pattern not found: {pattern}"


def test_agent_instructions_no_journal(monkeypatch):
    """Test agent instructions when no journal path is set."""
    utils = importlib.import_module("think.utils")
    
    # Clear JOURNAL_PATH
    monkeypatch.setenv("JOURNAL_PATH", "")
    
    system, extra, meta = utils.agent_instructions("default")
    
    # Should not have domains section without journal (check extra context)
    assert "## Domains" not in extra
    
    # Should still have basic functionality
    assert system.startswith("You are Sunstone")
    assert "Current Date and Time" in extra


def test_agent_instructions_empty_domains(monkeypatch, tmp_path):
    """Test agent instructions when journal has no domains."""
    utils = importlib.import_module("think.utils")
    
    # Create empty journal structure
    empty_journal = tmp_path / "empty_journal"
    empty_journal.mkdir()
    (empty_journal / "domains").mkdir()
    
    monkeypatch.setenv("JOURNAL_PATH", str(empty_journal))
    
    system, extra, meta = utils.agent_instructions("default")
    
    # Should not add domains section if no domains exist (check extra context)
    assert "## Domains" not in extra
