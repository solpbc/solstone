---
name: sol-support
description: >
  File support tickets, search the knowledge base, and give feedback via
  the sol support CLI. Use this skill when the user needs help with solstone,
  wants to report a bug, request a feature, or submit feedback to sol pbc.
  TRIGGER: support tickets, bug reports, feature requests, feedback, help
  requests, knowledge base search, system diagnostics.
---

# sol support

CLI for filing support tickets, searching the knowledge base, and submitting feedback to sol pbc.

## Before You Start

1. **Read the TOS first.** A local copy is cached at the portal storage directory after first registration. Check `apps/support/portal/tos.txt` in the journal's app storage. If it doesn't exist, run `sol call support register` to fetch and cache it.

2. **Always search the KB before filing a ticket.** Run `sol call support search "your question"` first. Many common issues are already documented. Only file a ticket if the KB doesn't answer the question.

3. **Diagnostics are auto-populated.** When creating a ticket, `sol call support create` automatically collects system info (version, OS, services, recent errors). You don't need to gather this manually.

4. **User consent is required for all outbound operations.** Never use `--yes` without explicit user approval. Always show the user what will be sent and get their OK first.

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

Always search before filing a ticket. Present matching articles to the user.

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
2. Shows matches (user can read them and cancel if resolved)
3. Collects diagnostics automatically
4. Shows the full ticket draft for review
5. Submits only after user confirms

**Flags:**
- `--subject` / `-s` — Ticket subject (required)
- `--description` / `-d` — Detailed description (required)
- `--product` / `-p` — Product name (default: solstone)
- `--severity` — low, medium, high, critical (default: medium)
- `--category` — bug, feature, question, account
- `--skip-kb` — Skip KB search (not recommended)
- `--yes` / `-y` — Skip confirmation (only use with explicit user consent)
- `--anonymous` — Strip installation identifiers

### Ticket Management

```bash
# List open tickets
sol call support list

# List all tickets (including resolved)
sol call support list --status resolved

# View a ticket with thread
sol call support show 42

# Reply to a ticket
sol call support reply 42 --body "Here's the additional info you requested..."

# JSON output for any command
sol call support list --json
sol call support show 42 --json
```

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
# User reports a bug — full flow
sol call support search "calendar sync"          # check KB first
sol call support create \
  --subject "Calendar events not syncing" \
  --description "Google Calendar events imported yesterday aren't showing up in the calendar app. Tried re-importing but same result." \
  --category bug \
  --severity medium

# User wants to give feedback
sol call support feedback \
  --body "Love the entity detection but it sometimes misidentifies project names as people"

# Check for responses on open tickets
sol call support list
sol call support show 15

# Quick system health check
sol call support diagnose
```
