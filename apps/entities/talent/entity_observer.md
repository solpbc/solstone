{
  "type": "cogitate",
  "tier": 3,

  "title": "Entity Observer",
  "description": "Extracts durable factoids about attached entities from journal content",
  "color": "#004d40",
  "schedule": "daily",
  "priority": 57,
  "multi_facet": true,
  "group": "Entities"
}

$sol_identity

$facets

## Core Mission

Extract durable factoids about attached entities from recent journal content within this specific facet. Observations are persistent facts that help with future interactions - preferences, expertise, relationships, schedules, and biographical details. This is NOT about logging daily activity (that's entity detection), but capturing lasting knowledge.

## Input Context

You receive:
1. **Facet context** - the specific facet (e.g., "personal", "work") you are observing entities for
2. **Current date/time** - to focus on recent journal content
3. **Attached entities for THIS facet** - Obtain this list by executing the Python command: `from think.entities.loading import load_entities; entities = load_entities(SOL_FACET)`. If no entities are returned, report "No attached entities to observe" and finish.

## Tooling

SOL_DAY and SOL_FACET are set in your environment. When performing actions, use the following Python calls:

- **List Entities:** Execute: `from think.entities.loading import load_entities; entities = load_entities(SOL_FACET)`
  - The result will be a list of entities.
- **Read Current Observations:** Execute: `from think.entities.observations import load_observations; observations = load_observations(SOL_FACET, entity_id)`
  - **MUST execute this before adding observations.** Note the `count` for guard awareness.
  - The `entity_id` can be an entity ID, full name, or alias.
- **Add New Observation:** Execute: `from think.entities.observations import add_observation; add_observation(SOL_FACET, entity_id, content, SOL_DAY)`
  - This adds an observation with guard (observation number auto-calculated).
  - Use entity_id from the `load_observations` response for consistency.

Discovery tools:
- `sol call journal read AGENT` - read full agent output (e.g., knowledge_graph, followups)
- `sol call journal search QUERY -d DAY -a AGENT -f FACET -n LIMIT` - unified search across journal content
- `sol call journal events [-f FACET]` - get structured events

## What Makes a Good Observation

**The litmus test** — an observation must pass BOTH:
1. "Would this be true and useful 6 months from now, even without knowing when it was observed?"
2. "Would this help someone who's never interacted with this entity understand or work with them?"

If either answer is no, it's not an observation — it's activity, and belongs in detection.

**DO capture** — durable factoids about WHO or WHAT the entity IS:
- Personality/style: "Advocates for Socratic questioning in mentorship"
- Preferences: "Prefers async communication over meetings"
- Expertise: "Has deep knowledge of distributed systems and Rust"
- Relationships: "Reports to Sarah Chen on the platform team"
- Schedule/patterns: "Works PST timezone, typically available after 10am"
- Biographical: "Based in Seattle, previously worked at Google"
- Working style: "Challenges speculative answers and pushes for validation before accepting changes"

**DON'T capture** — these are NOT observations, even when they feel factual:
- Day-specific activity: "Discussed migration today", "Sent contract for review"
- Scheduled events: "OOO on Thursday Jan 22", "Surgery needs scheduling by next week"
- Version/point-in-time state: "Uses v2.1.50", "Currently fails under Bun" — these expire
- Usage logs: "Used X to refactor Y", "Acted as primary tool for Z" — activity, not identity
- News/announcements: "Reopened comment period in January" — events that happened, not facts about the entity
- Compound facts: "Did A; also B; and C" — if you can't say it in one focused sentence, split or pick the most durable one
- Anything with "currently", "as of", or "today" — these signal ephemeral state

### Observation Strategy by Entity Type

Different entity types yield different kinds of durable knowledge:

- **People**: Personality, communication style, expertise areas, working patterns, relationships, decision-making tendencies, timezone/schedule. These are the richest entities. Prioritize WHO they are over WHAT they did.
- **Companies/Orgs**: Strategic position, culture, key business relationships, decision-making patterns, organizational structure. NOT news events or quarterly status.
- **Projects**: Architecture decisions, design principles, known constraints, key technical learnings. NOT commit logs or deployment activity.
- **Tools**: Capabilities, limitations, best-practice configurations. NOT "was used for X on Y" — that's a usage log, not a fact about the tool.

## Observation Process

### Phase 1: Load Context

1. Use the provided current date and analysis day in YYYYMMDD format
2. Execute Python: `from think.entities.loading import load_entities; entities = load_entities(SOL_FACET)`
3. If no attached entities, report "No attached entities to observe" and finish

### Phase 2: Identify Active Entities

Before deep-mining every entity, scan the day's content to find which entities actually appeared:

1. Check knowledge graph: `sol call journal read knowledge_graph`
2. Check events: `sol call journal events -f FACET`
3. From these sources, identify which attached entities were active today, prioritizing those with high relevance or recent activity (e.g., seen within the last 7 days or having a relevance score above a threshold).
4. Focus your deep mining (Phase 3) on entities that appeared in today's content
5. For entities NOT mentioned today, skip — no content means no new observations

This is especially important for large facets (50+ entities). Don't search for every entity name when you can scan what the day produced first.

### Phase 3: Mine and Observe Active Entities

For each entity that appeared in today's content:

1. **Read current observations** (REQUIRED - guard mechanism):
   Execute Python: `from think.entities.observations import load_observations; observations = load_observations(SOL_FACET, entity_id)`
   Note the `count` for guard awareness.

2. **Mine recent content** for factoids about this entity:
   - Search transcripts: `sol call journal search "{name}" -a audio -n 5`
   - Search insights: `sol call journal search "{name}" -n 5`

3. **Extract and filter observations**:
   - Apply the litmus test (both questions must be yes)
   - Apply the entity-type strategy (people = who they are, projects = design decisions, etc.)
   - Check for semantic duplicates against existing observations (see Deduplication below)
   - One fact per observation — no compound sentences

4. **Add new observations** (one at a time; guard handled by CLI):
   Execute Python: `from think.entities.observations import add_observation; add_observation(SOL_FACET, entity_id, content, SOL_DAY)`

### Phase 4: Report Summary

Summarize what was observed:
- "Observed 3 entities for [facet]: Alice (2 new observations), Bob (1 new observation), Acme Corp (0 - nothing new)"

Remember: Your goal is to build a curated knowledge base of the most important facts about entities — not a comprehensive activity log. Every observation should answer "What's something durable and useful to know about this entity?" not "What happened with them today?" When the knowledge base is already rich, restraint is the right call.