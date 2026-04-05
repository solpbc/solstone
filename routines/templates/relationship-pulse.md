{
  "name": "relationship-pulse",
  "description": "Review relationship health and identify people who need attention or follow-through.",
  "default_cadence": "0 9 * * 1",
  "default_timezone": "UTC",
  "default_facets": []
}

You are reviewing relationship health across the routine's configured facets.

Focus on people who matter operationally or personally, especially where contact, follow-through, or momentum has changed.

## Gather

1. Use `sol call journal facets` if you need to confirm the active facet set.
2. Use `sol call journal search "" --facet FACET -n 20` to identify frequently mentioned people or recent interactions in each relevant facet.
3. For each meaningful person, call `sol call entities intelligence PERSON`.
4. Use `sol call journal news FACET --day $day_YYYYMMDD` if a facet summary helps explain current context.
5. Use `sol call identity pulse` for broad priorities that may affect relationship maintenance.

## Synthesize

- Identify strong, active relationships versus neglected or at-risk ones.
- Note recent interactions, open loops, and people who likely need a reply, check-in, or prep.
- Prioritize by importance and recency, not by raw mention count.
- Distinguish between work relationships, collaborators, and personal contacts where relevant.

## Write

Write markdown with sections such as:

- `## Active Relationships`
- `## Needs Attention`
- `## Open Loops`
- `## Suggested Next Moves`

Keep each person entry short and specific.
Use entity intelligence to ground your judgments.
Avoid generic advice; tie every recommendation to journal evidence.
