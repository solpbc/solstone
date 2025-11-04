"""Inbox message management functionality."""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


def _get_inbox_dir() -> Path:
    """Get the inbox directory path from JOURNAL_PATH."""
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")
    return Path(journal) / "inbox"


def _log_activity(action: str, message_id: str, **kwargs: Any) -> None:
    """Log an activity to the inbox activity log."""
    inbox_dir = _get_inbox_dir()
    inbox_dir.mkdir(exist_ok=True)
    log_file = inbox_dir / "activity_log.jsonl"

    entry = {
        "timestamp": int(time.time() * 1000),
        "action": action,
        "message_id": message_id,
        **kwargs,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def send_message(
    body: str,
    from_type: str = "system",
    from_id: str = "system",
    context: Optional[dict[str, str]] = None,
) -> str:
    """Send a new message to the inbox.

    Args:
        body: The message content (plain text or markdown)
        from_type: Type of sender (agent/system/facet)
        from_id: ID of the sender
        context: Optional context dict with facet, matter, and/or day

    Returns:
        The message ID (e.g., "msg_1755450767962")
    """
    timestamp = int(time.time() * 1000)
    message_id = f"msg_{timestamp}"

    message = {
        "id": message_id,
        "timestamp": timestamp,
        "from": {"type": from_type, "id": from_id},
        "body": body,
        "status": "unread",
    }

    if context:
        message["context"] = context

    # Create inbox directories if needed
    inbox_dir = _get_inbox_dir()
    active_dir = inbox_dir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    # Write message file
    message_path = active_dir / f"{message_id}.json"
    with open(message_path, "w", encoding="utf-8") as f:
        json.dump(message, f, indent=2)

    # Log the activity
    _log_activity("received", message_id, from_type=from_type, from_id=from_id)

    return message_id


def get_message(message_id: str) -> Optional[dict[str, Any]]:
    """Get a message by ID from active or archived folder.

    Args:
        message_id: The message ID (e.g., "msg_1755450767962")

    Returns:
        The message dict or None if not found
    """
    inbox_dir = _get_inbox_dir()

    # Check active folder first
    active_path = inbox_dir / "active" / f"{message_id}.json"
    if active_path.exists():
        with open(active_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Check archived folder
    archived_path = inbox_dir / "archived" / f"{message_id}.json"
    if archived_path.exists():
        with open(archived_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def mark_read(message_id: str) -> bool:
    """Mark a message as read.

    Args:
        message_id: The message ID to mark as read

    Returns:
        True if successful, False if message not found
    """
    inbox_dir = _get_inbox_dir()
    message_path = inbox_dir / "active" / f"{message_id}.json"

    if not message_path.exists():
        return False

    with open(message_path, "r", encoding="utf-8") as f:
        message = json.load(f)

    if message.get("status") != "read":
        message["status"] = "read"
        with open(message_path, "w", encoding="utf-8") as f:
            json.dump(message, f, indent=2)

        _log_activity("read", message_id)

    return True


def archive_message(message_id: str) -> bool:
    """Archive a message by moving it from active to archived folder.

    Args:
        message_id: The message ID to archive

    Returns:
        True if successful, False if message not found
    """
    inbox_dir = _get_inbox_dir()
    active_path = inbox_dir / "active" / f"{message_id}.json"

    if not active_path.exists():
        return False

    # Read message and update status
    with open(active_path, "r", encoding="utf-8") as f:
        message = json.load(f)

    message["status"] = "archived"

    # Create archived directory and write message
    archived_dir = inbox_dir / "archived"
    archived_dir.mkdir(parents=True, exist_ok=True)
    archived_path = archived_dir / f"{message_id}.json"

    with open(archived_path, "w", encoding="utf-8") as f:
        json.dump(message, f, indent=2)

    # Remove from active
    active_path.unlink()

    _log_activity("archived", message_id)

    return True


def unarchive_message(message_id: str) -> bool:
    """Unarchive a message by moving it from archived to active folder.

    Args:
        message_id: The message ID to unarchive

    Returns:
        True if successful, False if message not found
    """
    inbox_dir = _get_inbox_dir()
    archived_path = inbox_dir / "archived" / f"{message_id}.json"

    if not archived_path.exists():
        return False

    # Read message and update status
    with open(archived_path, "r", encoding="utf-8") as f:
        message = json.load(f)

    message["status"] = "read"  # Set to read when unarchiving

    # Create active directory and write message
    active_dir = inbox_dir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)
    active_path = active_dir / f"{message_id}.json"

    with open(active_path, "w", encoding="utf-8") as f:
        json.dump(message, f, indent=2)

    # Remove from archived
    archived_path.unlink()

    _log_activity("unarchived", message_id)

    return True


def list_messages(status: str = "active") -> list[dict[str, Any]]:
    """List all messages in active or archived folder.

    Args:
        status: "active" or "archived"

    Returns:
        List of message dicts, sorted by timestamp (newest first)
    """
    inbox_dir = _get_inbox_dir()
    folder = inbox_dir / status

    if not folder.exists():
        return []

    messages = []
    for msg_file in folder.glob("msg_*.json"):
        try:
            with open(msg_file, "r", encoding="utf-8") as f:
                messages.append(json.load(f))
        except Exception:
            continue

    # Sort by timestamp, newest first
    messages.sort(key=lambda m: m.get("timestamp", 0), reverse=True)

    return messages


def get_unread_count() -> int:
    """Get the count of unread messages in the active inbox.

    Returns:
        Number of unread messages
    """
    active_messages = list_messages("active")
    return sum(1 for m in active_messages if m.get("status") == "unread")


def delete_message(message_id: str) -> bool:
    """Permanently delete a message from active or archived folder.

    Args:
        message_id: The message ID to delete

    Returns:
        True if successful, False if message not found
    """
    inbox_dir = _get_inbox_dir()

    # Check both folders
    for status in ["active", "archived"]:
        message_path = inbox_dir / status / f"{message_id}.json"
        if message_path.exists():
            message_path.unlink()
            _log_activity("deleted", message_id)
            return True

    return False


def main() -> None:
    """CLI for inbox message management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage inbox messages")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Send message command
    send_parser = subparsers.add_parser("send", help="Send a new message")
    send_parser.add_argument("body", help="Message body")
    send_parser.add_argument(
        "--from-type", default="system", help="Sender type (agent/system/facet)"
    )
    send_parser.add_argument("--from-id", default="cli", help="Sender ID")
    send_parser.add_argument("--facet", help="Facet context")
    send_parser.add_argument("--matter", help="Matter context")
    send_parser.add_argument("--day", help="Day context (YYYYMMDD)")

    # List messages command
    list_parser = subparsers.add_parser("list", help="List messages")
    list_parser.add_argument(
        "--status",
        default="active",
        choices=["active", "archived"],
        help="Message status",
    )

    # Get message command
    get_parser = subparsers.add_parser("get", help="Get a specific message")
    get_parser.add_argument("message_id", help="Message ID")

    # Mark read command
    read_parser = subparsers.add_parser("read", help="Mark message as read")
    read_parser.add_argument("message_id", help="Message ID")

    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive a message")
    archive_parser.add_argument("message_id", help="Message ID")

    # Unarchive command
    unarchive_parser = subparsers.add_parser("unarchive", help="Unarchive a message")
    unarchive_parser.add_argument("message_id", help="Message ID")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a message")
    delete_parser.add_argument("message_id", help="Message ID")

    # Stats command
    subparsers.add_parser("stats", help="Show inbox statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "send":
        context = {}
        if args.facet:
            context["facet"] = args.facet
        if args.matter:
            context["matter"] = args.matter
        if args.day:
            context["day"] = args.day

        message_id = send_message(
            args.body,
            from_type=args.from_type,
            from_id=args.from_id,
            context=context if context else None,
        )
        print(f"Message sent: {message_id}")

    elif args.command == "list":
        messages = list_messages(args.status)
        if not messages:
            print(f"No {args.status} messages")
        else:
            print(f"\n{args.status.upper()} MESSAGES ({len(messages)}):")
            print("-" * 80)
            for msg in messages:
                timestamp = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(msg["timestamp"] / 1000)
                )
                status_indicator = "●" if msg["status"] == "unread" else "○"
                preview = msg["body"][:60].replace("\n", " ")
                print(f"{status_indicator} [{msg['id']}]")
                print(f"  From: {msg['from']['id']} ({msg['from']['type']})")
                print(f"  Time: {timestamp}")
                print(f"  Body: {preview}...")
                if msg.get("context"):
                    print(f"  Context: {msg['context']}")
                print()

    elif args.command == "get":
        message = get_message(args.message_id)
        if not message:
            print(f"Message {args.message_id} not found")
        else:
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(message["timestamp"] / 1000)
            )
            print(f"\nMESSAGE: {message['id']}")
            print("-" * 80)
            print(f"From: {message['from']['id']} ({message['from']['type']})")
            print(f"Time: {timestamp}")
            print(f"Status: {message['status']}")
            if message.get("context"):
                print(f"Context: {message['context']}")
            print(f"\nBody:\n{message['body']}")

    elif args.command == "read":
        if mark_read(args.message_id):
            print(f"Message {args.message_id} marked as read")
        else:
            print(f"Message {args.message_id} not found")

    elif args.command == "archive":
        if archive_message(args.message_id):
            print(f"Message {args.message_id} archived")
        else:
            print(f"Message {args.message_id} not found")

    elif args.command == "unarchive":
        if unarchive_message(args.message_id):
            print(f"Message {args.message_id} unarchived")
        else:
            print(f"Message {args.message_id} not found")

    elif args.command == "delete":
        if delete_message(args.message_id):
            print(f"Message {args.message_id} deleted")
        else:
            print(f"Message {args.message_id} not found")

    elif args.command == "stats":
        active_messages = list_messages("active")
        archived_messages = list_messages("archived")
        unread_count = sum(1 for m in active_messages if m.get("status") == "unread")

        print("\nINBOX STATISTICS:")
        print("-" * 40)
        print(f"Active messages:   {len(active_messages)}")
        print(f"Unread messages:   {unread_count}")
        print(f"Archived messages: {len(archived_messages)}")
        print(f"Total messages:    {len(active_messages) + len(archived_messages)}")


if __name__ == "__main__":
    main()
