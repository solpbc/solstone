# Backlog

Tactical work items prioritized for implementation.

---

## Apps

- [ ] Consolidate insights app functionality into agents app
- [ ] Add tabs or navigation mode to entities app all-facet view (reduce vertical scrolling)
- [ ] Audit apps for #fragment deep linking and improve coverage

## Agents

- [ ] Update supervisor/dream interaction to use dynamic daily schedule from daily schedule agent output
- [ ] Create segment agent for voiceprint detection and updating via hooks
- [ ] Refactor think/agents.py to use run_agent for all generation
- [ ] Evaluate moving cortex.py logic into agents.py for better separation of concerns
- [ ] Pass 'day' context through to tools in daily agents for correct storage location
- [ ] Surface named hook outputs in agents app and sol muse CLI
- [ ] Make daily schedule agents idempotent with state tracking (show existing vs new segments)
- [ ] Add activities attach/update MCP tools for facet curation (like entity tools)

## Integrations

- [ ] Add OpenRouter provider support with observe integration for multimodal models
- [ ] Automated Fireflies importer

## Indexer

## Infrastructure

- [ ] Create system/user service install for sol supervisor (remove terminal dependency)
- [ ] Health monitor and diagnostics agent (explore Claude Code SDK)

## Indexer

- [ ] Performance tune SQLite usage
- [ ] Investigate potential double-indexing of insights and occurrences
- [ ] Refactor entity detection to be per-entity

## Testing

- [ ] Move fixtures/ into tests/
- [ ] Enable clean fixture-based service startup (Convey on dynamic port) for integration/dev testing with screenshots
