{
  "type": "generate",
  "title": "First-Day Check-In",
  "description": "One-shot check-in after onboarding — spawns support agent chat",
  "schedule": "segment",
  "priority": 98,
  "output": "text",
  "hook": {"pre": "firstday_checkin", "post": "firstday_checkin"},
  "tier": 3,
  "thinking_budget": 512,
  "max_output_tokens": 256,
  "exclude_streams": ["import.*"]
}

This generator exists only to trigger the first-day check-in via its pre/post hooks. The pre-hook handles all logic — if it doesn't skip, the post-hook spawns a support agent chat. The LLM output is unused.

Output "ok" — nothing else needed.
