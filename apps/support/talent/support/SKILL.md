---
name: support
description: >
  File support tickets, search the knowledge base, and submit feedback to
  sol pbc. Manage open tickets, attach files, check announcements, and run
  local diagnostics. Use when the owner needs help with solstone, wants to
  report a bug, request a feature, check for known issues, or give feedback.
  TRIGGER: support, bug report, feature request, feedback, help, knowledge
  base, file a ticket, known issues, announcements, diagnostics.
---

# sol support

CLI for filing support tickets, searching the knowledge base, and submitting feedback to sol pbc.

## Before You Start

1. **Read the TOS first.** A local copy is cached at the portal storage directory after first registration. Check `apps/support/portal/tos.txt` in the journal's app storage. If it doesn't exist, run `sol call support register` to fetch and cache it.

2. **Always search the KB before filing a ticket.** Run `sol call support search "your question"` first. Many common issues are already documented. Only file a ticket if the KB doesn't answer the question.

3. **Diagnostics are auto-populated.** When creating a ticket, `sol call support create` automatically collects system info (version, OS, services, recent errors). You don't need to gather this manually.

4. **Owner consent is required for all outbound operations.** Never use `--yes` without explicit owner approval. Always show the owner what will be sent and get their OK first.

## Subcommands

### Registration

```bash
sol call support register
```

Register (or re-register) with the support portal. Generates an RSA-4096 keypair on first use, signs the TOS, and creates an account. Run this if you get auth errors.

### Knowledge Base

```bash
# Search articles
sol call support search "transcription errors"

# Read a specific article
sol call support article getting-started
```

Always search before filing a ticket. Present matching articles to the owner.

### Filing a Ticket

```bash
sol call support create \
  --subject "Transcription fails on long recordings" \
  --description "Recordings over 2 hours consistently fail with timeout errors. Started after updating to v2.1." \
  --severity medium \
  --category bug
```

The `create` command implements a KB-first flow:
1. Searches KB for related articles
2. Shows matches (owner can read them and cancel if resolved)
3. Collects diagnostics automatically
4. Shows the full ticket draft for review
5. Submits only after owner confirms

**Flags:**
- `--subject` / `-s` — Ticket subject (required)
- `--description` / `-d` — Detailed description (required)
- `--product` / `-p` — Product name (default: solstone)
- `--severity` — low, medium, high, critical (default: medium)
- `--category` — bug, feature, question, account
- `--skip-kb` — Skip KB search (not recommended)
- `--yes` / `-y` — Skip confirmation (only use with explicit owner consent)
- `--anonymous` — Strip installation identifiers

### Ticket Management

```bash
# List open tickets
sol call support list

# List all tickets (including resolved)
sol call support list --status resolved

# View a ticket with thread (includes attachment metadata)
sol call support show 42

# Reply to a ticket
sol call support reply 42 --body "Here's the additional info you requested..."

# JSON output for any command
sol call support list --json
sol call support show 42 --json
```

### Attachments

```bash
# Attach a screenshot to ticket #42
sol call support attach 42 ~/screenshot.png

# Attach multiple files
sol call support attach 42 screenshot.png error-log.txt

# Skip confirmation (only after explicit owner consent)
sol call support attach 42 screenshot.png --yes
```

Upload files to an existing ticket. The consent gate shows each file (name and size) before upload. Attachments are a follow-up action — create the ticket first, then attach files.

**Limits:** max 10 MB per file, max 5 files per upload.

**Supported formats:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.svg`, `.pdf`, `.txt`, `.csv`, `.html`, `.md`, `.xml`, `.json`

When an owner reports a visual bug (UI glitch, rendering issue), proactively suggest attaching a screenshot.

### Feedback

```bash
sol call support feedback --body "The entity search is great but I wish it could filter by date range"
```

Lower friction than a full ticket. Feedback is submitted as a ticket with category "feedback". Supports `--anonymous` flag.

### Announcements

```bash
sol call support announcements
```

Check for product updates, known issues, and maintenance notices.

### Local Diagnostics

```bash
sol call support diagnose
sol call support diagnose --json
```

Runs locally — no network, no data sent. Shows:
- solstone version
- OS/platform info
- Active services and their status
- Recent errors from service logs
- Configuration (secrets stripped)

## Good Ticket Descriptions

A good ticket includes:
- **What happened** — specific behavior observed
- **What was expected** — what should have happened
- **Steps to reproduce** — how to trigger the issue
- **Context** — when it started, how often, any recent changes

The diagnostic collector auto-populates version, OS, and service status. You don't need to include these in the description.

## Examples

```bash
# Owner reports a bug — full flow
sol call support search "calendar sync"          # check KB first
sol call support create \
  --subject "Calendar events not syncing" \
  --description "Google Calendar events imported yesterday aren't showing up in the calendar app. Tried re-importing but same result." \
  --category bug \
  --severity medium

# Attach a screenshot to the ticket
sol call support attach 15 ~/screenshot.png

# Owner wants to give feedback
sol call support feedback \
  --body "Love the entity detection but it sometimes misidentifies project names as people"

# Check for responses on open tickets
sol call support list
sol call support show 15

# Quick system health check
sol call support diagnose
```
