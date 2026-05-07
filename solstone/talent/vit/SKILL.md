---
name: vit
description: >-
  Ship solstone features as vit caps to the social capability network
  after a feature lands through hopper. Scoped to the publish/ship
  workflow that runs at the end of the VPE playbook; discovery, vetting,
  and consumption commands live in the user-wide `using-vit` skill.
  TRIGGER: ship a cap, publish solstone cap, vit ship, after hopper ship,
  final lode ship, capworthy feature, three-word ref, vit beacon,
  vit:github.com/solpbc/solstone.
---

# Vit Ship Skill

Publish solstone capabilities as vit caps. Invoke via Bash: `vit ship ...` (other vit commands are covered by the `using-vit` skill).

## overview

vit is a CLI for publishing software capabilities (caps) to a decentralized social network built on ATProto. solstone participates in the vit network — when a capworthy feature ships, the VPE session publishes it as a cap so other projects and agents can discover, vet, and remix it.

**beacon:** `vit:github.com/solpbc/solstone`

**Scope**: this skill is intentionally narrow — VPE-session ship workflow only. For `vit skim`, `vit follow`, `vit learn`, `vit remix`, `vit vet`, and general vit usage, the `using-vit` skill has full coverage. Don't duplicate that reference here.

## when to ship a cap

ship a cap after the **final lode** of a feature ships through hopper and progress is updated (VPE playbook step 5). not every lode is a cap. a cap describes a self-contained capability another project could learn from or adopt.

**ship when:**
- the feature introduces a novel pattern or approach worth sharing
- the feature is self-contained — someone could read the cap and understand the full approach
- the feature would be useful to other agentic projects, journaling tools, or AI-native software

**don't ship when:**
- internal refactoring or code cleanup
- bug fixes (unless the fix embodies a pattern worth documenting)
- test-only or docs-only changes
- individual lodes within a multi-lode feature (ship one cap for the whole feature at the end)

## how to ship

```bash
vit ship --title "Feature Name" \
         --description "One sentence explaining the value" \
         --ref "three-word-ref" \
         --kind feat <<'EOF'
What this feature does, how it works, and the key architectural decisions.
Written for another developer or agent who might adopt the approach in
their own codebase.
EOF
```

### field guide

- `--title` — concise noun phrase, 2-5 words (e.g., "Entity Intelligence Signals Index")
- `--description` — one sentence explaining the value to someone who hasn't seen the code
- `--ref` — three lowercase words separated by dashes, memorable slug for discovery (e.g., "entity-signal-indexing")
- `--kind` — category: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`, `perf`, `style`
- `--recap <ref>` — only if this cap derives from another cap (e.g., after `vit remix`)
- **body (stdin)** — a short paragraph explaining the approach. not a commit message — write it for someone who wants to understand and potentially adopt the pattern

### ref naming conventions

the three-word ref should be descriptive and memorable:
- good: `entity-signal-indexing`, `speaker-voice-attribution`, `routine-progressive-discovery`
- bad: `update-fix-three`, `new-feature-impl`, `march-twentyseven-ship`

## pre-ship diagnostics

### `vit init`
already done — beacon is set. run `vit init` to check current beacon status, or `vit init --beacon <url>` to change it.

### `vit doctor`
read-only diagnostic. run to verify setup and beacon status before shipping when something looks off.

For `vit skim`, `vit follow`, `vit learn`, `vit remix`, and other discovery/consumption commands, see the `using-vit` skill.

## troubleshooting

| error | fix |
|-------|-----|
| `no DID configured` | tell the user to run `vit login <handle>` in their terminal |
| `no beacon set` | run `vit init --beacon .` |
| `session expired` | tell the user to run `vit login <handle>` |
| invalid ref format | ref must be exactly three lowercase words separated by dashes |

## human-only commands

these require browser interaction — tell the user to run them in their terminal:
- `vit setup` — check prerequisites
- `vit login <handle>` — authenticate via browser OAuth
- `vit vet <ref>` — review a cap before trusting it
