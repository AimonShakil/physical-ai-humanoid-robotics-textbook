---
id: 0003
title: Textbook Implementation Planning
stage: plan
date: 2025-11-28
surface: agent
model: claude-sonnet-4-5-20250929
feature: 002-textbook-docusaurus-setup
branch: 002-textbook-docusaurus-setup
user: hackathon-participant
command: /sp.plan
labels: ["plan", "architecture", "design", "phases", "constitution-check", "data-models", "api-contracts"]
links:
  spec: specs/002-textbook-docusaurus-setup/spec.md
  ticket: null
  adr: history/adr/ (10 ADRs to be created in Phase 0)
  pr: null
files:
  - specs/002-textbook-docusaurus-setup/plan.md
tests:
  - N/A (planning stage)
---

## Prompt

(Automatic execution via `/sp.plan` command following Spec-Kit Plus workflow)

Create comprehensive technical implementation plan for Physical AI & Humanoid Robotics Textbook based on specification (specs/002-textbook-docusaurus-setup/spec.md) and clarifications (Assumptions #13-17).

## Response snapshot

Created complete implementation plan (plan.md, 1497 lines) covering all required sections:

**1. Technical Context** (Fully Specified):
- Languages: TypeScript 5.x + React 18.x (frontend), Python 3.11+ (backend)
- Stack: Docusaurus 3.x, FastAPI, Qdrant Cloud, Neon Postgres, Better-Auth
- Performance goals: <3s page load, <2s RAG response, <500ms vector search
- Constraints: 32-40 chapters by Nov 30, 95% citation accuracy, 100% code validation
- Scale: 96-120 markdown files, 200-400 vector chunks, <100 concurrent users

**2. Constitution Check** (✅ ALL 8 PRINCIPLES COMPLIANT):
- Principle I (Educational-First): Chapter structure enforced (Learning Objectives, Core Concepts, Hands-On Lab, Exercises, Summary)
- Principle II (Modular Architecture): 4 independent modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA)
- Principle III (RAG Integration): Chatbot as first-class feature with text-selection queries and citations
- Principle IV (Code Validation): 100% validation via code-validator subagent
- Principle V (Accessibility): Responsive design, personalization, Urdu translation
- Principle VI (Performance): Lighthouse >90, <3s load, <2s chat, rate limiting
- Principle VII (Spec-Driven): All workflows followed, 4 subagents + 2 skills created, 10 ADRs planned
- Principle VIII (Git Hygiene): Conventional commits, <500 lines per commit, Claude co-author

**3. Project Structure** (Detailed Monorepo):
- Frontend: docs/ (Docusaurus with 19 React components: Chatbot, Auth, Personalization, Translation)
- Backend: backend/ (FastAPI with 20 Python modules: RAG, chunking, auth, translation services)
- Scripts: chapter-generator, code-validator, rag-indexer, test-generator subagent invocation
- Tests: Playwright E2E (20+ scenarios), pytest backend (4 test suites)
- CI/CD: 4 GitHub Actions workflows (frontend deploy, backend deploy, tests, Lighthouse)

**4. Implementation Phases** (7 Phases, Days 1-18):
- Phase 1 (Days 1-2): Foundation & Infrastructure - Docusaurus + FastAPI scaffold, CI/CD
- Phase 2 (Days 3-5): Content Architecture - Module 1 (8-10 chapters), personalization variants, code validation
- Phase 3 (Days 6-8): RAG Backend - Qdrant + Neon setup, semantic chunking, citation generation
- Phase 4 (Days 9-10): Chatbot UI - ChatbotButton, text selection, citation navigation, E2E tests
- Phase 5 (Days 11-12): Authentication (BONUS) - Better-Auth, HTTP-only cookies, session management
- Phase 6 (Days 13-15): Personalization & Translation (BONUS) - Variant selection, Urdu API, caching
- Phase 7 (Days 16-18): Testing & Delivery - Lighthouse audits, demo video, final deployment by 5:00 PM Nov 30

**5. Data Models** (6 Entities Fully Specified):
- Module: 4 modules, static metadata in Docusaurus
- Chapter: 32-40 chapters, 3 variants each, markdown storage
- User: Neon Postgres, bcrypt password hash, background scores (1-5)
- Session: JWT with 7-day expiry, auto-refresh at 5 days, HTTP-only cookies
- ContentChunk: Qdrant + Neon, 1536-dim embeddings, semantic chunking (H2/H3, 100-token overlap)
- ChatMessage: Conversation context, citations array, response time tracking (<2s)

**6. API Contracts** (3 OpenAPI Specifications):
- contracts/chat-api.yaml: POST /api/chat (RAG query → answer + citations), POST /api/embed (admin indexing)
- contracts/auth-api.yaml: POST /api/signup, /api/signin, /api/signout, GET /api/profile
- contracts/translation-api.yaml: POST /api/translate (Urdu translation with code preservation)

**7. Phase 0 Research Tasks** (10 ADRs to Create):
- ADR-001: Monorepo vs Multi-Repo (decided: monorepo)
- ADR-002: Docusaurus 2.x vs 3.x Version
- ADR-003: Backend Hosting Platform (recommend: Fly.io for no cold starts)
- ADR-004: RAG Citation Linking (recommend: URL anchors to heading IDs)
- ADR-005: OpenAI Model Selection (recommend: GPT-4o-mini + text-embedding-3-small)
- ADR-006: Authentication Session Management (validate Assumption #16: HTTP-only cookies)
- ADR-007: Personalization Implementation (validate Assumption #14: pre-computed variants)
- ADR-008: Translation Caching Strategy (recommend: two-tier localStorage + Neon)
- ADR-009: E2E Testing Framework (recommend: Playwright)
- ADR-010: Deployment Timeline Strategy (recommend: continuous from Day 1)

**8. Risk Mitigation** (5 Major Risks):
- Risk 1 (High/High): Content volume too ambitious → Prioritize Module 1 MVP, use chapter-generator subagent
- Risk 2 (Medium/High): RAG citation accuracy <95% → Iterate chunking early, test with 50-query set
- Risk 3 (Medium/Medium): API costs exceed budget → Context7 MCP caching, $5 soft cap
- Risk 4 (Low/Critical): Last-minute deployment failures → Deploy early/often, backup platforms ready
- Risk 5 (Medium/High): Bonus features break base → Feature flags, test in isolation, prioritize base (Phases 1-4)

**9. Validation Plan**:
- Unit tests: pytest for RAG, chunking, auth, translation services
- E2E tests: Playwright (20+ scenarios generated by test-generator subagent)
- Performance tests: Lighthouse CI (4 pages: homepage + 3 chapters)
- Manual tests: Demo video rehearsal, cross-browser, mobile responsiveness
- Automation: 9/12 SCs fully/partially automated (75% automated validation)

## Outcome

- ✅ Impact: Production-ready implementation plan with complete technical architecture, 7 time-boxed phases, 6 fully specified data models, 3 OpenAPI contracts, 10 architectural decisions queued for research, 5 risk mitigation strategies, and comprehensive validation plan. Ready for Phase 0 research.md creation.
- 🧪 Tests: N/A (planning stage; testing strategy defined in plan)
- 📁 Files: plan.md (1497 lines, ~12-15 pages), embedded API contracts (chat-api.yaml, auth-api.yaml, translation-api.yaml structures)
- 🔁 Next prompts: 1) Create research.md (Phase 0: resolve 10 ADRs), 2) Create data-model.md (Phase 1: entity schemas), 3) Create quickstart.md (Phase 1: setup guide), 4) Run /sp.tasks (Phase 2: detailed task breakdown)
- 🧠 Reflection: Plan successfully balances comprehensiveness with actionability. Constitution Check confirms 100% compliance across all 8 principles. Monorepo structure (ADR-001) simplifies development. Pre-computed personalization (Assumption #14) validates as only approach meeting SC-007 (<1s). Semantic chunking strategy (Assumption #13) detailed with overlap mechanism. Risk mitigation prioritizes base features (Phases 1-4, 150 points) before bonuses (Phases 5-6, +150 points). Critical path identified: Foundation → Content → RAG Backend → Chatbot UI. Deployment timeline (continuous from Day 1 per ADR-010) de-risks final submission. Phase 0 research will validate remaining architecture decisions before implementation.

## Evaluation notes (flywheel)

- Failure modes observed: None - plan creation followed template structure, all required sections filled
- Graders run and results (PASS/FAIL): PASS - Constitution Check shows ✅ ALL 8 PRINCIPLES COMPLIANT
- Prompt variant (if applicable): N/A (standard /sp.plan workflow)
- Next experiment (smallest change to try): During research.md creation (Phase 0), validate Fly.io vs Railway.app for backend hosting by testing cold start latency with sample FastAPI deployment (affects SC-003 <2s response time requirement)
