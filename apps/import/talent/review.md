{
  "type": "cogitate",
  "title": "Import Review",
  "description": "Unified import review: resolves staged entities, then facets, then config in dependency order.",
  "color": "#37474f",
  "group": "Import"
}

$facets

## Core Mission

Run the full import review pipeline in the correct dependency order: entities first, then facets, then config.

## Tooling

- `sol call import list-staged --source SOURCE`
- `sol call import list-staged --source SOURCE --area entities`
- `sol call import list-staged --source SOURCE --area facets`
- `sol call import list-staged --source SOURCE --area config`
- `sol call import resolve-entity SOURCE_ID merge --source SOURCE --target TARGET_ID`
- `sol call import resolve-entity SOURCE_ID create --source SOURCE`
- `sol call import resolve-entity SOURCE_ID skip --source SOURCE`
- `sol call import resolve-staged-facet STAGED_FILE --apply --source SOURCE`
- `sol call import resolve-staged-facet STAGED_FILE --skip --source SOURCE`
- `sol call import resolve-config FIELD apply --source SOURCE`
- `sol call import resolve-config FIELD keep --source SOURCE`
- `sol call import resolve-config-all --source SOURCE --category transferable`
- `sol call import resolve-config-all --source SOURCE --category preference`

## Process

### Step 1: Confirm Source

The source name must be provided as input when you are invoked. If it is missing, ask for it before doing anything else.

### Step 2: Check All Staged Work

Run:

```bash
sol call import list-staged --source SOURCE
```

If nothing is staged, report `Nothing to review` and exit.

### Step 3: Review Entities

Process all staged entities using the same decision logic as the entity review workflow:
- merge when the match is clearly correct
- create when the entity is valid but distinct
- skip only when the staged record should be discarded

### Step 4: Review Facets

Only start facet review after entity staging is clear. Then process staged facet items using the facet review workflow.

### Step 5: Review Config

After entities and facets are resolved, review config differences:
- batch-apply transferable fields
- inspect preference fields individually

### Step 6: Final Verification

Run `sol call import list-staged --source SOURCE` again and confirm whether anything remains staged.

### Step 7: Report

Report a complete summary:
- entities merged / created / skipped
- facet items applied / skipped
- config fields applied / kept
- any remaining staged items or blockers

## Quality Guidelines

### DO:

- Follow the dependency order strictly
- Clear each area before moving to the next when feasible
- Surface blockers immediately and explicitly
- End with a final verification pass

### DON'T:

- Attempt facet resolution before entity review is complete
- Finish without checking the staged queues again
- Leave unresolved work without naming what remains
