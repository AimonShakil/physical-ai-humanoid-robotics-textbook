# ADR-0007: Translation Architecture

- **Status:** Accepted
- **Date:** 2025-11-28
- **Feature:** 002-textbook-docusaurus-setup
- **Context:** Bonus feature (Phase 6) requires Urdu translation for textbook chapters while preserving code blocks in English (FR-040 through FR-046). Must display translated content within <3s on first request and <0.5s for cached content (SC-008). 32-40 chapters × ~2000 words each need translation. Critical decision: translation API provider, caching strategy (client vs server vs hybrid), and code block preservation mechanism.

## Decision

**Use the following translation architecture:**

- **Translation API**: OpenAI Translation (not Google Translate or DeepL)
  - Model: GPT-4o-mini with translation prompt
  - Preserves markdown formatting natively
  - Reuses same API key as RAG chatbot (no additional provider)

- **Caching Strategy**: Two-tier caching (browser localStorage + server-side Neon Postgres)
  - **Tier 1 (Client)**: Browser localStorage with cache key `translation-{chapter_id}-urdu`
    - Stores translated markdown (2-3KB per chapter)
    - Instant retrieval (<50ms) for repeat views
    - Survives page reloads, expires only on manual clear

  - **Tier 2 (Server)**: Neon Postgres `translations` table
    - Schema: `{chapter_id, language, translated_content, created_at}`
    - Serves as fallback when localStorage cleared
    - Shared across users (one translation per chapter per language)
    - Reduces API calls: first user pays 3s, subsequent users get <0.5s

- **Code Block Preservation**:
  - Translation prompt: "Translate to Urdu but keep all content within triple backticks (```) unchanged"
  - Regex validation: Extract code blocks before translation → Translate text → Re-insert original code blocks
  - Test validation: All code blocks must pass `code_validator.match(original, translated)` assertion

- **Translation Flow**:
  1. User clicks "Translate to Urdu" → Check localStorage for cached translation
  2. Cache miss → Call backend POST /api/translate with chapter_id
  3. Backend checks Neon Postgres for existing translation
  4. DB miss → Call OpenAI API with translation prompt → Store in Postgres → Return to frontend
  5. Frontend stores in localStorage + renders Urdu content
  6. Subsequent requests: localStorage returns <50ms (meets SC-008 <0.5s)

## Consequences

### Positive

- **Fast Cached Access**: localStorage <50ms → easily meets SC-008 <0.5s requirement
- **Shared Translations**: Server cache benefits all users → first user pays 3s, others get <0.5s
- **Cost Efficiency**: 32 chapters × $0.60/1M tokens × 2000 words ≈ $0.05 total → negligible cost, one-time per chapter
- **API Consolidation**: Reuses OpenAI API key → no additional provider, billing, or credential management
- **Formatting Preservation**: OpenAI natively understands markdown → headings, lists, emphasis preserved automatically
- **Offline Access**: localStorage survives offline → users can re-read translated content without network
- **Simple Invalidation**: Clear localStorage → re-fetch from server → easy debugging and content updates

### Negative

- **localStorage Size Limit**: 5-10MB cap → 32 chapters × 3KB = 96KB used (acceptable, but limits future expansion)
- **Cache Inconsistency**: localStorage per-browser → user switching devices loses cache (acceptable: server cache still works)
- **First-User Penalty**: First user waits 3s for translation → poor UX for that user (mitigated: admin can pre-translate)
- **OpenAI Dependency**: If OpenAI API down, no translations work → no fallback to Google Translate (acceptable: RAG also depends on OpenAI)
- **Code Block Risk**: Regex extraction might fail on malformed markdown → breaks code preservation (mitigated: validation in tests)
- **No Incremental Updates**: Translating updated chapter requires re-translating entire content → wasteful for small edits (acceptable: hackathon scope, infrequent updates)

## Alternatives Considered

### Alternative 1: Google Translate API + Client-Only Caching

**Stack**:
- Google Cloud Translation API (v3)
- localStorage caching only (no server-side cache)
- Custom code block masking (replace code with placeholders before translation)

**Pros**:
- **Higher Quality**: Google Translate optimized for Urdu → potentially better translation quality than GPT-4o-mini
- **Lower Cost**: Google Translate $20/1M characters vs OpenAI $0.60/1M tokens → ~50% cheaper for text-heavy content
- **Simpler Architecture**: No server-side cache → one less database table, simpler backend

**Cons**:
- **No Shared Cache**: Every user pays 3s for first translation → no benefit from previous translations
- **Additional Provider**: Need Google Cloud account, API key, billing setup → more complexity vs reusing OpenAI
- **Code Preservation Complexity**: Google Translate doesn't understand markdown → must manually mask/unmask code blocks → higher bug risk
- **API Key Management**: Two API keys (OpenAI + Google) → more secrets to manage in Fly.io environment
- **localStorage-Only Risk**: If user clears cache, 3s penalty every time → no server fallback

**Why Rejected**: Lack of shared server cache hurts UX for hackathon demo (multiple judges testing → each waits 3s). OpenAI API reuse simplifies architecture (one provider, one key, one billing dashboard). Code block preservation simpler with GPT-4o-mini (understands markdown natively vs Google's placeholder workaround). Cost savings ($0.02 vs $0.05) negligible for hackathon scope.

### Alternative 2: Pre-computed Translations (Build-Time)

**Stack**:
- Generate Urdu markdown files at build time (32 chapters × 2 languages = 64 files)
- Store translations alongside English files: `chapter1-introduction.md`, `chapter1-introduction-ur.md`
- No API calls at runtime, instant delivery via Docusaurus routing

**Pros**:
- **Zero Runtime Cost**: No OpenAI API calls during demo → $0 translation cost
- **Instant Delivery**: No 3s first-request penalty → <0.5s for all users, all requests
- **No Caching Needed**: Files served by GitHub Pages CDN → automatic caching, no localStorage complexity
- **Same Pattern as Personalization**: Matches ADR-0006 pre-computed variants → consistent architecture

**Cons**:
- **Build Time Explosion**: 32 chapters × 3 variants × 2 languages = 192 files → 6x repository size
- **Translation Quality Control**: Must manually review 32 Urdu translations for accuracy → high time investment (8-16 hours)
- **Update Complexity**: Fixing bug in English requires re-translating and reviewing Urdu → 2x maintenance burden
- **Urdu Expertise Required**: Need Urdu speaker to validate translations → team lacks Urdu fluency, can't validate quality
- **Storage Overhead**: 192 files × 2KB = 384KB markdown → acceptable size but messy repository structure

**Why Rejected**: Team lacks Urdu fluency → cannot validate translation quality. Pre-computing unvalidated translations risks embarrassing errors during demo (judges may speak Urdu). API-based translation allows iterative testing (translate → test → refine prompt). Build time explosion (192 files) makes repository harder to navigate. Bonus feature (lowest priority) doesn't justify 2x maintenance burden. Runtime cost ($0.05) negligible compared to quality risk.

### Alternative 3: Client-Side Translation (Transformers.js + NLLB-200)

**Stack**:
- Meta's NLLB-200 distilled model (600MB) for in-browser translation
- Transformers.js for WebAssembly inference
- No API calls, fully offline translation

**Pros**:
- **Zero API Cost**: No OpenAI or Google charges → unlimited translations
- **Privacy**: Translation happens locally → no chapter content sent to external APIs
- **Offline-First**: Works without internet → best offline UX

**Cons**:
- **Massive Download**: 600MB model → violates SC-001 <3s page load (600MB at 10Mbps = 8 minutes)
- **Slow Inference**: NLLB-200 takes 10-20s per chapter on CPU → violates SC-008 <3s requirement
- **Browser Compatibility**: WebAssembly SIMD required → excludes older browsers (Safari <16.4, Firefox <89)
- **Quality**: NLLB-200 distilled weaker than GPT-4o-mini → lower translation quality, more errors
- **Code Block Preservation**: NLLB-200 doesn't understand markdown → need custom masking like Google Translate

**Why Rejected**: 600MB download defeats page load goals. 10-20s inference violates SC-008 <3s requirement. Experimental tech (Transformers.js in-browser inference) too risky for hackathon deadline. Translation quality critical for educational content → NLLB-200 distilled model not as reliable as GPT-4o-mini. Two-tier caching achieves same goal (zero API cost for cached translations) with 1000x better performance.

## References

- Feature Spec: [specs/002-textbook-docusaurus-setup/spec.md](../../specs/002-textbook-docusaurus-setup/spec.md) (FR-040 through FR-046)
- Implementation Plan: [specs/002-textbook-docusaurus-setup/plan.md](../../specs/002-textbook-docusaurus-setup/plan.md#phase-6-content-personalization-bonus-days-13-14)
- Related ADRs: ADR-0004 (RAG Architecture - reuses OpenAI API key), ADR-0003 (Backend Hosting - Neon Postgres for translation cache), ADR-0005 (Authentication - requires signin for translation)
- Success Criteria: SC-008 (<3s first request, <0.5s cached)
- OpenAI Translation Best Practices: https://platform.openai.com/docs/guides/translation
