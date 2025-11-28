# Quick ADR Generator Skill

## Purpose
Rapidly create Architectural Decision Records (ADRs) for significant technical decisions in the Physical AI Textbook project, ensuring decisions are documented with rationale, alternatives, and consequences.

## When to Use
- During `/sp.plan` when making architectural choices
- When selecting between multiple implementation approaches
- When choosing external services or libraries
- When defining data models or API contracts
- When establishing deployment strategies
- When selecting authentication mechanisms (Better-Auth vs alternatives)
- When designing RAG architecture (embedding model, chunking strategy)

## Three-Part ADR Significance Test

An architectural decision is significant if ALL three are true:
1. **Impact**: Has long-term consequences (framework, data model, API, security, platform)
2. **Alternatives**: Multiple viable options were considered
3. **Scope**: Cross-cutting and influences system design

## Input Parameters

- **title**: Brief decision title (3-7 words, e.g., "Use Qdrant for Vector Storage")
- **context**: Why this decision is needed (problem statement, constraints)
- **options**: List of alternatives considered (2-4 options)
- **decision**: Which option was chosen
- **rationale**: Why this option was selected (trade-off analysis)
- **consequences**: Positive and negative outcomes expected
- **related_features**: Optional - links to related specs/tasks

## ADR Template Structure

```markdown
# ADR-NNNN: [Title]

**Status**: Accepted | Proposed | Deprecated | Superseded
**Date**: [YYYY-MM-DD]
**Context**: [Feature/Module]
**Related ADRs**: [Links to related ADRs if any]

## Context

[Describe the problem, constraint, or architectural challenge that requires a decision. Include relevant background.]

## Decision

[State the decision clearly in one sentence.]

We will [decision statement].

## Alternatives Considered

### Option 1: [Name]
**Description**: [What this option entails]

**Pros**:
- [Advantage 1]
- [Advantage 2]

**Cons**:
- [Disadvantage 1]
- [Disadvantage 2]

**Complexity**: Low | Medium | High

### Option 2: [Name]
[Same structure]

### Option 3: [Name]
[Same structure]

## Rationale

[Explain why the chosen option is best given the context, constraints, and trade-offs. Reference constitution principles if applicable.]

Key factors:
1. [Factor 1 and why it matters]
2. [Factor 2 and why it matters]
3. [Factor 3 and why it matters]

## Consequences

### Positive
- [Expected benefit 1]
- [Expected benefit 2]
- [Expected benefit 3]

### Negative
- [Trade-off or cost 1]
- [Trade-off or cost 2]

### Neutral
- [Changes that are neither clearly positive nor negative]

## Implementation Notes

- [Specific guidance for implementation]
- [Key technical details]
- [Migration steps if replacing existing solution]

## Success Metrics

- [How to measure if this decision was correct]
- [KPIs or validation criteria]

## Review Date

[Optional: Date to revisit this decision if circumstances change]

---

**Related Specifications**: [Links to specs/plans/tasks]
**Related PRs**: [Links when implemented]
```

## Automatic ADR Numbering

The skill will:
1. Scan `history/adr/` directory
2. Find highest existing ADR number
3. Increment by 1
4. Format as 4 digits (e.g., 0001, 0042, 0123)

## File Naming Convention

`history/adr/NNNN-kebab-case-title.md`

Examples:
- `history/adr/0001-use-qdrant-for-vector-storage.md`
- `history/adr/0002-fastapi-for-rag-backend.md`
- `history/adr/0003-better-auth-for-authentication.md`

## Common ADRs for This Project

### High-Priority ADRs to Create

1. **RAG Architecture**
   - Title: "RAG Architecture with OpenAI and Qdrant"
   - Options: OpenAI embeddings vs open-source, Qdrant vs Pinecone vs Weaviate
   - Decision: OpenAI text-embedding-3-small + Qdrant Cloud Free Tier

2. **Authentication System**
   - Title: "Better-Auth for User Authentication"
   - Options: Better-Auth vs NextAuth vs custom JWT vs Supabase Auth
   - Decision: Better-Auth for flexible profile questions

3. **Content Personalization Strategy**
   - Title: "User Profile-Based Content Adaptation"
   - Options: Client-side JS rendering vs server-side generation vs static variants
   - Decision: [To be determined during planning]

4. **Translation Approach**
   - Title: "Urdu Translation Implementation"
   - Options: Google Translate API vs OpenAI translation vs human translation vs i18n files
   - Decision: [To be determined during planning]

5. **Docusaurus Deployment**
   - Title: "GitHub Pages vs Vercel for Deployment"
   - Options: GitHub Pages (free, GitHub Actions) vs Vercel (faster, preview deployments)
   - Decision: [To be determined during planning]

6. **RAG Chunking Strategy**
   - Title: "Semantic Chunking for Textbook Content"
   - Options: Fixed-size chunks vs paragraph-based vs heading-based vs hybrid
   - Decision: [To be determined during planning]

## Output Format

```markdown
✅ ADR Created Successfully

**ADR Number**: 0003
**File Path**: history/adr/0003-better-auth-for-authentication.md
**Title**: Better-Auth for User Authentication
**Status**: Accepted
**Date**: 2025-11-28

## Summary
Decision to use Better-Auth for authentication system with custom profile questions
for software/hardware background capture, enabling content personalization.

## Next Steps
1. Reference this ADR in authentication feature specification
2. Link from implementation plan
3. Update constitution if this establishes new patterns
```

## Usage Example

```bash
# Via Skill tool
skill: "adr-quick"
  title: "Use Qdrant for Vector Storage"
  context: "Need vector database for RAG system with free tier for hackathon"
  options: ["Qdrant Cloud Free", "Pinecone Free Tier", "Weaviate Cloud"]
  decision: "Qdrant Cloud Free Tier"
  rationale: "1GB free storage sufficient for textbook content, easy Python SDK, proven performance"
  consequences_positive: ["No cost for hackathon", "Fast vector search", "Simple API"]
  consequences_negative: ["Vendor lock-in", "Free tier limits at scale"]
```

## Integration with Constitution

Per Constitution Principle VII:
> "Architectural decisions (RAG architecture, auth approach, translation strategy) MUST be documented as ADRs"

This skill ensures compliance by making ADR creation fast and consistent.

## Quality Checks

Before finalizing ADR, ensure:
- [ ] At least 2 alternatives considered
- [ ] Clear rationale with trade-off analysis
- [ ] Both positive and negative consequences listed
- [ ] Implementation notes provide actionable guidance
- [ ] Success metrics defined
- [ ] Related to constitution principles where applicable

## Validation

Run constitution-check skill on generated ADR to ensure quality:
```bash
skill: "constitution-check"
  artifact_type: "architecture"
  artifact_path: "history/adr/NNNN-title.md"
```

## Success Metrics

- ADR creation time reduced from 20 minutes to 3 minutes
- 100% of significant decisions documented
- Clear decision trail for hackathon judges
- Reusable patterns for future features
