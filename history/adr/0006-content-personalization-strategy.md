# ADR-0006: Content Personalization Strategy

- **Status:** Accepted
- **Date:** 2025-11-28
- **Feature:** 002-textbook-docusaurus-setup
- **Context:** Bonus feature (Phase 6) requires personalized content based on user background (software skills 1-5, hardware skills 1-5). Must display personalized variant within <1s (SC-007) after clicking "Personalize" button. 32-40 chapters need 3 variants each (beginner/intermediate/advanced). Critical decision: pre-computed variants at build time vs on-demand LLM generation, variant storage mechanism, and selection logic.

## Decision

**Use the following personalization approach:**

- **Variant Generation**: Pre-computed at build time (not on-demand)
  - Generate 3 markdown files per chapter during Phase 2 content creation
  - File naming: `chapter1-introduction.md` (default), `chapter1-introduction-beginner.md`, `chapter1-introduction-advanced.md`
  - Total files: 32 chapters × 3 variants = 96 markdown files (~500KB compressed)

- **Variant Selection Logic** (Assumption #14):
  - Input: User profile (software_background + hardware_background, each 1-5 scale)
  - Sum: `total_score = software_background + hardware_background` (range 2-10)
  - Mapping:
    - Beginner: total_score ≤ 3 (e.g., 1+1, 1+2, 2+1)
    - Intermediate: 4 ≤ total_score ≤ 7 (e.g., 2+3, 3+3, 4+3)
    - Advanced: total_score ≥ 8 (e.g., 4+4, 4+5, 5+5)

- **Variant Delivery**:
  - Client-side selection: React component reads user profile from localStorage → selects variant → requests correct markdown file
  - No API call required for personalization (variant selection is pure computation)
  - Cache in browser localStorage: `chapter-{id}-{variant}` → <0.5s subsequent loads

- **Variant Differences**:
  - Beginner: More definitions, step-by-step explanations, simpler examples, basic exercises
  - Intermediate: Assumes foundational knowledge, moderate complexity examples, mixed exercises
  - Advanced: Minimal definitions, complex examples, theory-heavy, challenging exercises

## Consequences

### Positive

- **Instant Display**: No LLM API call → variant selection <100ms → easily meets SC-007 <1s requirement
- **No API Costs**: Pre-computed variants → zero OpenAI charges for personalization → stays within $5 budget
- **Predictable Quality**: Manually reviewed variants → consistent quality vs LLM hallucinations
- **Offline-First**: Cached variants work without network → better UX for slow connections
- **Simple Implementation**: File-based storage → no database complexity, no cache invalidation logic
- **SEO Benefit**: All variants indexed by Google → improves search rankings for different skill levels
- **Easy Debugging**: Can inspect variant markdown files directly → easier to fix content issues

### Negative

- **Storage Overhead**: 96 files vs 32 → 3x repository size (mitigated: markdown compresses well, ~500KB total)
- **Build Time**: Generating 96 chapters takes 3x longer → ~6-9 hours vs 2-3 hours (mitigated: parallelizable with chapter-generator subagent)
- **Content Drift**: Must update 3 files when fixing bugs → 3x maintenance burden (mitigated: shared content sections using MDX imports)
- **Coarse Granularity**: Only 3 variants for 9 possible scores → users with score=4 and score=7 both get "intermediate" (acceptable: finer gradations not worth complexity)
- **No Dynamic Adjustment**: Cannot personalize based on reading behavior (e.g., user struggling with intermediate) → fixed at signup (acceptable: bonus feature, not core requirement)
- **Limited Scalability**: Adding 4th variant (expert) requires regenerating 32 more files → linear scaling (acceptable: hackathon scope fixed at 3 variants)

## Alternatives Considered

### Alternative 1: On-Demand LLM Generation (GPT-4o-mini)

**Stack**:
- Store only default variant in repository (32 files)
- On "Personalize" click → Send chapter content + user profile to OpenAI API → Generate personalized variant on-the-fly
- Cache generated variant in Neon Postgres or Redis

**Pros**:
- **Lower Storage**: Only 32 default files → minimal repository size
- **Infinite Personalization**: Can generate unique content for each user → hyper-personalized experience
- **Easy Updates**: Fix bugs in 1 default file → all personalizations auto-update
- **Dynamic Adaptation**: Can adjust based on user's reading history, quiz scores, time spent

**Cons**:
- **Latency Risk**: LLM generation takes 2-5s → violates SC-007 <1s requirement (500ms cache miss + 2000ms generation + 500ms rendering = 3s)
- **API Costs**: 32 chapters × 2000 words × $0.60/1M tokens output ≈ $0.04 per user → 100 users = $4 → exhausts $5 budget quickly
- **Quality Unpredictability**: LLM may hallucinate facts, miss technical nuances → risk broken code examples
- **Caching Complexity**: Need Redis or Postgres caching layer → additional service dependency
- **First-Use Penalty**: Every user's first personalization takes 2-5s → poor UX vs instant pre-computed variants

**Why Rejected**: SC-007 requires <1s display → on-demand generation's 2-5s latency is unacceptable. API costs threaten $5 budget (100 users × $0.04 = $4 just for personalization). Pre-computed approach guarantees instant display and zero runtime costs. Quality control is critical for educational content → manual review of pre-computed variants safer than LLM generation.

### Alternative 2: Hybrid Approach (Pre-computed Beginner/Advanced Only)

**Stack**:
- Pre-compute beginner and advanced variants (64 files)
- Intermediate = default variant (32 files)
- Total: 96 files, but intermediate is just symlink or duplicate of default

**Pros**:
- **Reduced Effort**: Only generate 2 custom variants → 33% less writing work
- **Faster Builds**: Generate 64 custom files vs 96 → shorter build time
- **Same Performance**: Still instant display, no API calls

**Cons**:
- **Inconsistent UX**: Beginner/advanced users get customized content, intermediate users get generic → unfair experience
- **Wasted Opportunity**: Intermediate is largest user segment (scores 4-7, 5 out of 9 combinations) → neglecting majority
- **Complexity Without Benefit**: Still need personalization logic, but worse user experience → worst of both worlds
- **Confusing Mental Model**: Why do some variants customize and others don't? Hard to explain to users

**Why Rejected**: Optimizing for wrong metric (build time vs user experience). Intermediate users likely largest segment → should get best experience, not worst. Hackathon scoring awards quality → better to invest effort in 3 excellent variants than save time with 2. Complexity savings minimal (variant selection logic same for 2 or 3 variants).

### Alternative 3: Client-Side LLM (Transformers.js + Llama 3.1 8B)

**Stack**:
- Bundle Llama 3.1 8B quantized model (4GB) in browser via Transformers.js
- Run personalization entirely client-side using WebGPU or WASM
- No API calls, no server-side processing

**Pros**:
- **Zero API Costs**: All processing local → unlimited personalizations
- **Maximum Privacy**: User data never leaves browser → no server storage
- **Offline Support**: Works without internet after initial load → best offline UX

**Cons**:
- **Massive Download**: 4GB model download → violates SC-001 <3s page load (4GB at 10Mbps = 53 minutes)
- **Browser Compatibility**: WebGPU only in Chrome 113+, Safari 18+ → excludes many users
- **Slow Inference**: CPU inference takes 30-60s per chapter → violates SC-007 <1s (even WebGPU takes 5-10s)
- **Memory Pressure**: 4GB model + browser memory → tab crashes on low-RAM devices (mitigated: but terrible UX)
- **Limited Model Quality**: Quantized 8B model weaker than GPT-4o-mini → lower personalization quality
- **Experimental Tech**: Transformers.js still beta → high risk of bugs during hackathon

**Why Rejected**: 4GB download completely defeats page load performance goals. Even with perfect caching, initial load takes minutes → no user will wait. Inference latency (5-60s) orders of magnitude worse than SC-007 requirement. Cutting-edge tech (WebGPU, Transformers.js) too risky for hackathon deadline → high probability of last-minute compatibility bugs. Pre-computed approach achieves same goal (zero API costs) with 1000x better performance.

## References

- Feature Spec: [specs/002-textbook-docusaurus-setup/spec.md](../../specs/002-textbook-docusaurus-setup/spec.md) (FR-035 through FR-037, Assumption #14)
- Implementation Plan: [specs/002-textbook-docusaurus-setup/plan.md](../../specs/002-textbook-docusaurus-setup/plan.md#phase-6-content-personalization-bonus-days-13-14)
- Related ADRs: ADR-0005 (Authentication - provides user background scores), ADR-0002 (Frontend Stack - Docusaurus serves variant files)
- Success Criteria: SC-007 (<1s personalization display)
- Assumption #14 Details: [spec.md line 260-265](../../specs/002-textbook-docusaurus-setup/spec.md#L260-L265)
