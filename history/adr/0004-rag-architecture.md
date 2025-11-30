# ADR-0004: RAG Architecture

- **Status:** Accepted
- **Date:** 2025-11-28
- **Feature:** 002-textbook-docusaurus-setup
- **Context:** Need RAG system for chatbot answering questions about textbook content. Must respond <2s (SC-003), achieve 95% citation accuracy (SC-004), handle 50-query test set (beginner/intermediate/advanced), support text-selection queries, and stay within OpenAI free tier ($5 credit) during development. Critical decision: LLM provider + embedding model + vector DB + citation linking mechanism.

## Decision

**Use the following integrated RAG stack:**

- **LLM (Chat)**: OpenAI GPT-4o-mini (not GPT-4o)
  - Cost: $0.15/$0.60 per 1M tokens (60% cheaper than GPT-4o)
  - Latency: ~500-800ms response time (vs 1-1.5s for GPT-4o)

- **Embedding Model**: OpenAI text-embedding-3-small (not text-embedding-3-large)
  - Cost: $0.02 per 1M tokens (5x cheaper than large variant)
  - Dimensions: 1536 (vs 3072 for large) → faster vector search

- **Vector Database**: Qdrant Cloud Free Tier
  - Storage: 1GB (sufficient for 200-400 chunks × 1536 dims)
  - Performance: <500ms for top-5 similarity search (Constitution Principle VI)

- **Relational Database**: Neon Serverless Postgres Free Tier
  - Purpose: Chunk metadata (chapter_id, section_heading, content_type), chat history, user sessions
  - Storage: 0.5GB (sufficient for metadata + 1K chat messages)

- **Chunking Strategy** (per Assumption #13 from spec clarifications):
  - Semantic chunking: Respect H2/H3 heading boundaries
  - Max 1000 tokens per chunk (except code blocks kept whole)
  - 100-token overlap between adjacent chunks (preserve context)
  - Priority: Full section → H3 sub-section → Paragraph → Character split (last resort)

- **Citation Linking**: URL anchors to Docusaurus heading IDs
  - Format: `/docs/module1-ros2/chapter1-introduction#what-is-ros-2`
  - Docusaurus auto-generates heading IDs from H2/H3 text
  - Browser native anchor navigation + smooth scroll

## Consequences

### Positive

- **Cost-Effective**: GPT-4o-mini + text-embedding-3-small = 60-80% cost savings vs GPT-4o + text-embedding-3-large → stays within $5 budget
- **Meets Performance Target**: 500ms (embedding) + 800ms (LLM) + 500ms (vector search) + 200ms (overhead) = ~2s total → meets SC-003
- **High Citation Accuracy**: Semantic chunking with H2/H3 boundaries → chunks map cleanly to heading IDs → reliable citation generation (targets SC-004 95%)
- **Free Tier Hosting**: Qdrant Cloud + Neon Serverless both have generous free tiers → zero ongoing costs during development
- **Overlap Prevents Context Loss**: 100-token overlap ensures questions spanning chunk boundaries still retrieve relevant context
- **Code Block Preservation**: Keeping code >1000 tokens whole prevents breaking syntax examples → maintains code validation (SC-009)
- **Browser-Native Citations**: URL anchors work without JavaScript → shareable links, works with screen readers (SC-005 Accessibility)

### Negative

- **OpenAI Vendor Lock-in**: Switching to Anthropic Claude or local LLMs requires rewriting embedding pipeline (1536-dim vectors incompatible)
- **Quality vs GPT-4o**: GPT-4o-mini has slightly lower reasoning quality → may impact advanced query responses (mitigation: test with 50-query set early)
- **Free Tier Limits**: Qdrant 1GB → max ~600 chunks (18-20 chapters if 30 chunks/chapter) → may need paid tier if expanding beyond hackathon scope
- **Citation Fragility**: If Docusaurus heading ID generation changes (e.g., slug collisions), citations break → need heading ID validation in tests
- **Chunking Complexity**: Semantic chunking requires custom Markdown parsing logic → cannot use simple character-split libraries
- **No Hybrid Search**: Using vector-only (no keyword/BM25 hybrid) → may miss exact term matches (e.g., "ROS 2" vs "ROS2")

## Alternatives Considered

### Alternative 1: GPT-4o + text-embedding-3-large + Pinecone

**Stack**:
- OpenAI GPT-4o (highest quality LLM)
- text-embedding-3-large (3072 dimensions, best accuracy)
- Pinecone (managed vector DB)
- PostgreSQL (metadata)

**Pros**:
- **Highest Quality**: GPT-4o best-in-class reasoning → better advanced query responses
- **Best Embeddings**: 3072-dim embeddings → slightly better retrieval accuracy (97-98% vs 95% for small)
- **Mature Platform**: Pinecone has better docs/support than Qdrant

**Cons**:
- **Cost**: GPT-4o $2.50/$10 per 1M tokens + text-embedding-3-large $0.13/1M → 5-10x more expensive → exceeds $5 budget quickly
- **Latency**: GPT-4o takes 1-1.5s to respond → harder to meet SC-003 <2s total (including vector search + overhead)
- **Pinecone Free Tier**: Only 1 index, 1GB storage → same limits as Qdrant but requires credit card upfront

**Why Rejected**: Cost exceeds hackathon budget. GPT-4o-mini quality sufficient for textbook Q&A (tested in spec analysis). Latency risk - GPT-4o slower response time threatens SC-003 compliance.

### Alternative 2: Anthropic Claude 3.5 Sonnet + Voyage AI embeddings + Weaviate

**Stack**:
- Anthropic Claude 3.5 Sonnet (competitive with GPT-4o quality)
- Voyage AI embeddings (optimized for retrieval tasks)
- Weaviate (open-source vector DB with GraphQL)
- PostgreSQL

**Pros**:
- **Non-OpenAI**: Reduces vendor lock-in, Claude has better long-context handling (200K tokens)
- **Retrieval-Optimized**: Voyage embeddings designed specifically for RAG → potentially better than OpenAI
- **Open Source**: Weaviate can be self-hosted → no vendor lock-in

**Cons**:
- **Multiple Vendors**: OpenAI (already using for translation bonus) + Anthropic + Voyage → 3 API keys, 3 billing systems
- **Voyage Cost**: $0.10 per 1M tokens (5x more expensive than text-embedding-3-small) → budget risk
- **Weaviate Complexity**: GraphQL query language → steeper learning curve vs Qdrant's simple similarity search API
- **Less Familiar**: Team has OpenAI experience, Claude integration requires learning new SDK

**Why Rejected**: Multi-vendor complexity not justified for hackathon. OpenAI already needed for translation (bonus feature Phase 6) → reuse same API key. Voyage embeddings too expensive. Weaviate GraphQL overhead slows Phase 3 development.

### Alternative 3: Local Llama 3.1 8B + BGE embeddings + ChromaDB

**Stack**:
- Llama 3.1 8B (open-source LLM, run locally or Replicate)
- BGE-large-en embeddings (open-source, 1024 dims)
- ChromaDB (local vector DB)
- SQLite (metadata)

**Pros**:
- **Zero API Costs**: No OpenAI charges, unlimited queries
- **Privacy**: All data stays local, no external API calls
- **Open Source**: No vendor lock-in, can modify models

**Cons**:
- **Hosting Complexity**: Need GPU server for Llama 3.1 (8GB VRAM minimum) → Fly.io GPU instances expensive, defeats cost savings
- **Latency**: Local LLM inference 2-4s on CPU, 500-800ms on GPU → similar to GPT-4o-mini but requires GPU
- **Quality Gap**: Llama 3.1 8B weaker than GPT-4o-mini for reasoning → may fail advanced queries in 50-query test set
- **Embedding Quality**: BGE-large-en not as good as OpenAI embeddings → lower retrieval accuracy
- **ChromaDB Scaling**: SQLite-based, not suitable for production → would need migration later

**Why Rejected**: Hosting complexity (GPU setup, VRAM management) consumes Phase 3 timeline. Quality risk - Llama 3.1 8B not tested for textbook Q&A. Defeats hackathon goal of rapid deployment (Qdrant Cloud + Neon provision in 5 minutes vs days setting up GPU server).

## References

- Feature Spec: [specs/002-textbook-docusaurus-setup/spec.md](../../specs/002-textbook-docusaurus-setup/spec.md) (FR-018 through FR-027, Assumption #13)
- Implementation Plan: [specs/002-textbook-docusaurus-setup/plan.md](../../specs/002-textbook-docusaurus-setup/plan.md#phase-3-rag-backend-development-days-6-8)
- Related ADRs: ADR-0003 (Backend Hosting - FastAPI endpoints), ADR-0007 (Translation - reuses OpenAI API key)
- Success Criteria: SC-003 (<2s response), SC-004 (95% citation accuracy), Constitution Principle VI (<500ms vector search)
- Chunking Strategy Detail: [plan.md lines 701-708](../../specs/002-textbook-docusaurus-setup/plan.md#L701-L708)
- 50-Query Test Set: [plan.md lines 58-61](../../specs/002-textbook-docusaurus-setup/plan.md#L58-L61)
