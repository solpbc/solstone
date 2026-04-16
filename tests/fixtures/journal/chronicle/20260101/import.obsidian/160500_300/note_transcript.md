## Betaworks Call Prep

> Source: vault/meetings/betaworks-call-prep.md

Meeting with David Park from [[Betaworks]] on Thursday. Need to cover:

### Key points to make
- Solstone's approach to trust isn't just verification — it's about creating *composable trust signals* that applications can build on
- The protocol layer is intentionally minimal; the value accrues at the edges where communities define their own trust schemas
- We're not competing with existing identity providers — we're the connective tissue between them

### Questions for David
1. How is Betaworks thinking about AI agent identity? Their portfolio has several agent-forward companies — are any of them running into the "who is this agent acting for" problem?
2. What's the appetite for open protocol investment vs. platform plays right now? Last time we talked he seemed bullish on protocols but the fund dynamics may have shifted.
3. Camp interest — would a trust/identity track at [[Betaworks Camp]] make sense for summer cohort? Could be a good forcing function for the spec.

### Prep to-do
- [ ] Pull the latest [[trust architecture]] diagram
- [ ] Have the 2-minute demo of credential issuance flow ready
- [ ] Review David's recent tweets for any signal on what he's focused on

#meetings #betaworks #partnerships

---

## Stream Processing Patterns

> Source: vault/projects/solstone/stream-processing-patterns.md

Documenting the patterns we keep reaching for. These are emerging from actual implementation, not theoretical.

### Pattern 1: Windowed Aggregation
Group events by time window (we're using 5-minute windows) and produce a summary segment. This is what the journal does — raw events flow in, segments crystallize out. Borrowed from [[Kafka]] Streams' windowed aggregation, but we're not using Kafka. The window is the unit of work.

### Pattern 2: Event Sourcing with Projections
Following [[event sourcing]] principles: the append-only event log is the source of truth. Everything else is a projection — a read-optimized view derived from replaying events. This means we can always rebuild any view from the raw stream. The cost is storage; the benefit is auditability and the ability to ask new questions of old data.

### Pattern 3: Stream Joining
This is where it gets interesting for the [[trust architecture]]. When a credential issuance event arrives on one stream and a verification request arrives on another, we need to join them by DID. Temporal joins with a grace period handle the case where events arrive out of order. The join window should match the credential validity window — no point joining against an expired credential.

### Pattern 4: Backpressure as Signal
When a downstream consumer can't keep up, that's not just a scaling problem — it's information. A trust verification service that falls behind is telling us something about the health of the network. We should surface backpressure metrics as first-class trust signals.

Still evolving. The connection between stream processing and [[trust architecture]] is tighter than I initially thought — trust *is* a stream problem.

#engineering #streams #architecture #solstone
