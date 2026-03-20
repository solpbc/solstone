{
  "type": "cogitate",

  "title": "Heartbeat",
  "description": "Periodic self-check — verifies system health, journal quality, and tends agency.md",
  "instructions": {"system": "journal", "facets": true, "now": true}

}

## Constraints
- Do NOT produce user-facing output
- Do NOT execute destructive changes (entity merges, facet changes) — add them as suggestions to `sol/agency.md`
- All findings and recommendations go to `sol/agency.md`

## Steps
1. **System health**: Run `sol call health status` to check service health. Review recent health logs for anomalies.
2. **Journal quality**: Run `sol call health journal-layout` and `sol call health agent-runs` to verify journal structure and recent agent execution over the last 3 days.
3. **Tend agency.md**: Read `sol/agency.md`. Mark resolved items, prune stale suggestions, and add any new findings from the health and journal-quality checks.
4. **Curation**: Run `sol call speakers suggest` for speaker suggestions. Run `sol call entities list` and `sol call entities detect` to check for merge candidates or stale entities, but record recommendations in `sol/agency.md` rather than making destructive changes.
5. **Review self.md**: Read `sol/self.md`. Only update it if genuine new patterns are observed. Do not change it just to change it.
6. **Git commit + push**: If any files were modified (`agency.md`, `self.md`), commit and push those changes.
