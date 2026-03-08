## Trust Architecture for Decentralized Identity

> Source: vault/projects/solstone/trust-architecture.md

Sketching out the trust layer for Solstone. The core idea is that identity verification shouldn't depend on a single authority — it needs to be distributed and composable.

Key principles:
- **Self-sovereign roots**: Every entity holds its own [[DID spec|decentralized identifier]]. No central registry owns the namespace.
- **Layered attestation**: Trust is built from [[verifiable credentials]] issued by multiple parties. A single credential means little; a *pattern* of credentials from independent issuers creates real confidence.
- **Contextual disclosure**: Holders reveal only what's needed for a given interaction. The architecture must support selective disclosure natively, not as an afterthought.

### Open questions

1. How do we handle revocation without introducing a centralized status list? The [[DID spec]] mentions status registries but leaves implementation open.
2. Credential refresh cadence — stale credentials erode trust, but aggressive refresh creates friction. Need to find the right decay curve.
3. Relationship to the [[stream processing]] layer: trust signals are themselves events. Can we model credential issuance and verification as streams?

#trust #identity #solstone #architecture

---

## Book Rec from Nadia

> Source: vault/reading/governing-the-commons.md

Nadia recommended *Governing the Commons* by Elinor Ostrom over coffee today. She said it completely changed how she thinks about shared resource management — not tragedy-of-the-commons inevitability, but actual documented cases of communities self-organizing durable governance.

Relevant to what we're building with Solstone: Ostrom's design principles for long-enduring commons institutions might map onto protocol governance. Especially the idea of graduated sanctions and conflict resolution mechanisms built into the system itself rather than imposed from outside.

Adding to the reading queue. Should pair well with the [[trust architecture]] thinking.

#reading #governance #commons
