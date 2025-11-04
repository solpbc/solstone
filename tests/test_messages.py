"""Tests for inbox message management functionality."""

import json
import time
from pathlib import Path

import pytest

from think.messages import (
    archive_message,
    delete_message,
    get_message,
    get_unread_count,
    list_messages,
    mark_read,
    send_message,
    unarchive_message,
)


@pytest.fixture
def temp_journal(tmp_path, monkeypatch):
    """Create a temporary journal directory for testing."""
    journal_path = tmp_path / "journal"
    journal_path.mkdir()
    monkeypatch.setenv("JOURNAL_PATH", str(journal_path))
    return journal_path


def test_send_message(temp_journal):
    """Test sending a new message to the inbox."""
    message_id = send_message(
        body="Test message content",
        from_type="agent",
        from_id="test_agent",
        context={"facet": "test_facet", "day": "20250117"},
    )

    assert message_id.startswith("msg_")

    # Check the message file was created
    message_path = temp_journal / "inbox" / "active" / f"{message_id}.json"
    assert message_path.exists()

    # Verify message content
    with open(message_path, "r") as f:
        message = json.load(f)

    assert message["id"] == message_id
    assert message["body"] == "Test message content"
    assert message["from"]["type"] == "agent"
    assert message["from"]["id"] == "test_agent"
    assert message["status"] == "unread"
    assert message["context"]["facet"] == "test_facet"
    assert message["context"]["day"] == "20250117"

    # Check activity log
    log_path = temp_journal / "inbox" / "activity_log.jsonl"
    assert log_path.exists()

    with open(log_path, "r") as f:
        log_entry = json.loads(f.readline())

    assert log_entry["action"] == "received"
    assert log_entry["message_id"] == message_id
    assert log_entry["from_type"] == "agent"
    assert log_entry["from_id"] == "test_agent"


def test_send_message_without_context(temp_journal):
    """Test sending a message without context."""
    message_id = send_message(
        body="Simple message", from_type="system", from_id="system"
    )

    message = get_message(message_id)
    assert message is not None
    assert "context" not in message


def test_get_message(temp_journal):
    """Test retrieving a message by ID."""
    message_id = send_message("Test message", from_id="test")

    message = get_message(message_id)
    assert message is not None
    assert message["id"] == message_id
    assert message["body"] == "Test message"

    # Test non-existent message
    assert get_message("msg_nonexistent") is None


def test_mark_read(temp_journal):
    """Test marking a message as read."""
    message_id = send_message("Unread message", from_id="test")

    # Verify initial status
    message = get_message(message_id)
    assert message["status"] == "unread"

    # Mark as read
    assert mark_read(message_id) is True

    # Verify status changed
    message = get_message(message_id)
    assert message["status"] == "read"

    # Test marking non-existent message
    assert mark_read("msg_nonexistent") is False

    # Check activity log
    log_path = temp_journal / "inbox" / "activity_log.jsonl"
    with open(log_path, "r") as f:
        lines = f.readlines()

    log_entry = json.loads(lines[-1])
    assert log_entry["action"] == "read"
    assert log_entry["message_id"] == message_id


def test_archive_message(temp_journal):
    """Test archiving a message."""
    message_id = send_message("Message to archive", from_id="test")

    # Archive the message
    assert archive_message(message_id) is True

    # Verify moved to archived folder
    active_path = temp_journal / "inbox" / "active" / f"{message_id}.json"
    archived_path = temp_journal / "inbox" / "archived" / f"{message_id}.json"

    assert not active_path.exists()
    assert archived_path.exists()

    # Verify status changed
    message = get_message(message_id)
    assert message["status"] == "archived"

    # Test archiving non-existent message
    assert archive_message("msg_nonexistent") is False

    # Check activity log
    log_path = temp_journal / "inbox" / "activity_log.jsonl"
    with open(log_path, "r") as f:
        lines = f.readlines()

    log_entry = json.loads(lines[-1])
    assert log_entry["action"] == "archived"
    assert log_entry["message_id"] == message_id


def test_unarchive_message(temp_journal):
    """Test unarchiving a message."""
    message_id = send_message("Message to unarchive", from_id="test")

    # Archive first
    archive_message(message_id)

    # Unarchive the message
    assert unarchive_message(message_id) is True

    # Verify moved back to active folder
    active_path = temp_journal / "inbox" / "active" / f"{message_id}.json"
    archived_path = temp_journal / "inbox" / "archived" / f"{message_id}.json"

    assert active_path.exists()
    assert not archived_path.exists()

    # Verify status changed to read (not unread)
    message = get_message(message_id)
    assert message["status"] == "read"

    # Test unarchiving non-existent message
    assert unarchive_message("msg_nonexistent") is False


def test_list_messages(temp_journal):
    """Test listing messages."""
    # Create multiple messages
    msg_ids = []
    for i in range(3):
        time.sleep(0.001)  # Ensure different timestamps
        msg_id = send_message(f"Message {i}", from_id=f"sender_{i}")
        msg_ids.append(msg_id)

    # Mark one as read
    mark_read(msg_ids[1])

    # Archive one
    archive_message(msg_ids[2])

    # List active messages
    active_messages = list_messages("active")
    assert len(active_messages) == 2
    # Should be sorted newest first
    assert active_messages[0]["id"] == msg_ids[1]
    assert active_messages[1]["id"] == msg_ids[0]

    # List archived messages
    archived_messages = list_messages("archived")
    assert len(archived_messages) == 1
    assert archived_messages[0]["id"] == msg_ids[2]


def test_get_unread_count(temp_journal):
    """Test getting unread message count."""
    # Initially no messages
    assert get_unread_count() == 0

    # Add unread messages
    msg1 = send_message("Unread 1", from_id="test")
    msg2 = send_message("Unread 2", from_id="test")
    msg3 = send_message("Unread 3", from_id="test")

    assert get_unread_count() == 3

    # Mark one as read
    mark_read(msg1)
    assert get_unread_count() == 2

    # Archive one (still unread)
    archive_message(msg2)
    assert get_unread_count() == 1  # Archived messages not counted

    # Mark last one as read
    mark_read(msg3)
    assert get_unread_count() == 0


def test_delete_message(temp_journal):
    """Test permanently deleting a message."""
    # Create and delete from active
    msg1 = send_message("To delete", from_id="test")
    assert delete_message(msg1) is True
    assert get_message(msg1) is None

    # Create, archive, and delete from archived
    msg2 = send_message("To delete archived", from_id="test")
    archive_message(msg2)
    assert delete_message(msg2) is True
    assert get_message(msg2) is None

    # Test deleting non-existent message
    assert delete_message("msg_nonexistent") is False

    # Check activity log
    log_path = temp_journal / "inbox" / "activity_log.jsonl"
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Should have two delete entries
    delete_entries = [
        json.loads(line) for line in lines if json.loads(line)["action"] == "deleted"
    ]
    assert len(delete_entries) == 2


def test_fixtures_loading(monkeypatch):
    """Test that fixture messages load correctly."""
    fixtures_path = Path(__file__).parent.parent / "fixtures" / "journal"
    monkeypatch.setenv("JOURNAL_PATH", str(fixtures_path))

    # Check active messages
    active_messages = list_messages("active")
    assert len(active_messages) == 3

    # Check specific fixture message
    research_msg = get_message("msg_1736000000000")
    assert research_msg is not None
    assert research_msg["from"]["id"] == "research_agent"
    assert research_msg["status"] == "unread"
    assert "ML papers" in research_msg["body"]

    # Check archived messages
    archived_messages = list_messages("archived")
    assert len(archived_messages) == 1

    # Check archived fixture message
    archived_msg = get_message("msg_1735990000000")
    assert archived_msg is not None
    assert archived_msg["from"]["type"] == "facet"
    assert archived_msg["status"] == "archived"

    # Check unread count
    assert get_unread_count() == 2  # Two unread in fixtures


def test_message_timestamp_ordering(temp_journal):
    """Test that messages are ordered by timestamp correctly."""
    # Create messages with specific timestamps
    msg_ids = []
    base_time = int(time.time() * 1000)

    for i in range(5):
        timestamp = base_time + (i * 1000)
        msg_id = f"msg_{timestamp}"

        # Manually create message to control timestamp
        message = {
            "id": msg_id,
            "timestamp": timestamp,
            "from": {"type": "system", "id": "test"},
            "body": f"Message {i}",
            "status": "unread",
        }

        message_path = temp_journal / "inbox" / "active" / f"{msg_id}.json"
        message_path.parent.mkdir(parents=True, exist_ok=True)

        with open(message_path, "w") as f:
            json.dump(message, f, indent=2)

        msg_ids.append(msg_id)

    # List messages and verify ordering (newest first)
    messages = list_messages("active")
    assert len(messages) == 5

    for i, msg in enumerate(messages):
        expected_idx = 4 - i  # Reverse order
        assert msg["body"] == f"Message {expected_idx}"
