# Implementation Plan: Physical AI & Humanoid Robotics Textbook with Docusaurus

**Branch**: `002-textbook-docusaurus-setup` | **Date**: 2025-11-28 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-textbook-docusaurus-setup/spec.md`

## Summary

Create an interactive educational textbook for teaching Physical AI and Humanoid Robotics using Docusaurus static site generator, integrated with a RAG-powered chatbot for contextual Q&A. The textbook covers 4 modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA) with 32-40 chapters total. Base requirements include readable content and functional RAG chatbot; bonus features include user authentication, content personalization, and Urdu translation.

**Technical Approach**: Monorepo with Docusaurus frontend (React-based static site) and FastAPI backend (RAG API). Content chunking and vector indexing via rag-indexer subagent, embeddings stored in Qdrant Cloud, metadata in Neon Postgres. Pre-computed personalization variants (3 per chapter) for <1s display. HTTP-only cookie authentication with Better-Auth. Two-tier translation caching (localStorage + server). Continuous deployment via GitHub Actions to GitHub Pages (frontend) and Fly.io (backend).

**Hackathon Context**: Deadline Nov 30, 2025 6:00 PM. Scoring: 100 base points + 200 bonus (50 for subagents, 50 for auth, 50 for personalization, 50 for translation).

## Technical Context

**Language/Version**:
- Frontend: TypeScript 5.x + React 18.x (Docusaurus 3.x requirement)
- Backend: Python 3.11+ (FastAPI async support)
- Node.js: 18+ (Docusaurus build toolchain)

**Primary Dependencies**:
- Frontend: Docusaurus 3.x, React 18, @docusaurus/preset-classic, prism-react-renderer (syntax highlighting), mermaid (diagrams)
- Backend: FastAPI 0.104+, uvicorn (ASGI server), openai (embeddings + chat), qdrant-client (vector DB), psycopg2-binary (Postgres), better-auth (authentication), bcrypt (password hashing)
- Testing: Playwright (E2E), pytest (backend unit tests), Lighthouse CI (performance)

**Storage**:
- Vector Database: Qdrant Cloud Free Tier (1GB, text-embedding-3-small vectors at 1536 dimensions)
- Relational Database: Neon Serverless Postgres Free Tier (User, ContentChunk, ChatMessage, Session tables)
- Static Assets: GitHub Pages CDN (Docusaurus build output)
- Caching: Browser localStorage (personalization variants, translations), Server-side Neon Postgres (translation cache)

**Testing**:
- E2E: Playwright tests for navigation, chatbot, auth, personalization, translation (generated via test-generator subagent)
- Backend: pytest with pytest-asyncio for RAG service, chunking service, auth service
- Performance: Lighthouse CI for SC-005 validation (Performance >90, Accessibility >95)
- Code Validation: code-validator subagent for all Python/Bash/XML examples in chapters

**Target Platform**:
- Frontend: Modern browsers (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+) with ES2020+ support
- Backend: Linux x86_64 (Fly.io VMs running Ubuntu 22.04)
- Development: Ubuntu 22.04 / macOS 13+ / Windows 11 with WSL2

**Project Type**: Web application (Docusaurus static site + FastAPI REST API backend)

**Performance Goals**:
- Page load time: <3s for any chapter (SC-001)
- RAG response time: <2s from query to answer with citations (SC-003)
- Vector search latency: <500ms for top-5 chunk retrieval (Constitution Principle VI)
- Personalization display: <1s for variant selection (SC-007)
- Translation (first request): <3s including API call (SC-008)
- Translation (cached): <0.5s from cache (SC-008)
- Lighthouse scores: Performance >90, Accessibility >95, Best Practices >90, SEO >90 (SC-005)

**Constraints**:
- Content volume: 32-40 chapters across 4 modules by Nov 30 (SC-002)
- RAG citation accuracy: 95% of 50-query test set returns valid citations (SC-004)
- Code validation: 100% of code examples must pass code-validator subagent (SC-009)
- Git history quality: 100% conventional commits, <500 lines per commit, Claude co-author (SC-012)
- API costs: Must stay within OpenAI free tier during development ($5 credit), monitor via usage dashboard
- Deployment deadline: Nov 30, 5:00 PM (1 hour before submission, SC-010)
- Demo video: <90 seconds demonstrating base features + 2 bonus features (SC-011)

**Scale/Scope**:
- Content: 32-40 chapters × 2000-3000 words = ~64K-120K words total
- Personalization variants: 32-40 chapters × 3 variants = 96-120 markdown files
- Vector chunks: ~200-400 chunks (10-12 chunks per chapter × 32-40 chapters)
- Expected users (hackathon): <100 concurrent, <1000 total during judging period
- Test coverage: Minimum 20 E2E scenarios (5 navigation, 5 chatbot, 3 auth, 3 personalization, 3 translation, 1 smoke test)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Educational-First Content Design ✅ COMPLIANT

- Content structured as progression: Module 1 (ROS 2 foundation) → Module 2 (Simulation) → Module 3 (Advanced perception) → Module 4 (VLA integration)
- Each chapter includes: Learning Objectives (3-6), Introduction, Core Concepts, Hands-On Lab (5-10 steps), Exercises (beginner/intermediate/advanced), Summary, Further Reading (per Assumption #15)
- Code examples validated via code-validator subagent, runnable and tested (FR-013, SC-009)
- Visual aids: Minimum 2 diagrams per chapter using Mermaid or static images (FR-016, Assumption #15)
- Personalization adjusts depth based on user background (FR-036, bonus feature)

**Validation**: Chapter template in data-model.md will enforce structure. chapter-generator subagent configured to follow Assumption #15 requirements.

---

### Principle II: Modular Content Architecture ✅ COMPLIANT

- 4 independent modules: ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA (FR-002)
- Each module has 8-10 chapters, completable independently (FR-003)
- Docusaurus sidebar structure enforces module boundaries (sidebars.js configuration)
- Cross-module references use explicit links with context (e.g., "As introduced in Module 1, Chapter 3: ROS 2 Nodes...")
- Content versioning via Git tags (v1.0.0-module1, v1.0.0-module2, etc.)

**Validation**: Docusaurus sidebar configuration in quickstart.md will show explicit module structure. No circular dependencies in content flow.

---

### Principle III: Interactive Learning Through RAG ✅ COMPLIANT

- RAG chatbot accessible from all pages via floating button (FR-018, FR-019)
- Text-selection queries: "Ask about this" context menu on selected text (FR-025)
- Citations include chapter title, section heading, clickable URL with anchor (FR-024, SC-004)
- Multi-turn conversation context maintained via conversationId (FR-026)
- Handles beginner ("What is ROS 2?") and advanced queries ("How does VSLAM differ from traditional SLAM?") (FR-027)
- Non-intrusive UI: Floating button bottom-right, chat panel slides in from side

**Validation**: API contract in contracts/chat-api.yaml will define citation format. E2E tests will validate text selection and multi-turn context.

---

### Principle IV: Code-First Technical Validation ✅ COMPLIANT

- All code examples validated via code-validator subagent before deployment (SC-009)
- Validation includes: Python AST parsing, import checks, ROS 2 API compliance (Humble/Iron), no hardcoded secrets (Assumption #15)
- Bash scripts validated with shellcheck (dangerous command detection)
- URDF/XML validated for structure (code-validator subagent capability)
- Expected outputs included in Hands-On Lab sections (FR-014)

**Validation**: CI/CD pipeline includes code validation step. code-validator subagent outputs validation report. 100% pass rate required for deployment (SC-009).

---

### Principle V: Accessibility and Personalization ✅ COMPLIANT

- Responsive design: Docusaurus default theme supports mobile/tablet/desktop (FR-009)
- No assumptions about robotics background: Jargon defined on first use (FR-017)
- User profile questions: Software background (1-5), Hardware background (1-5) captured at signup (FR-029)
- Personalization logic: Sum of backgrounds → variant selection (beginner/intermediate/advanced) (Assumption #14)
- Urdu translation with code preservation (FR-040 through FR-046, Assumption #17)
- Accessibility: Docusaurus meets WCAG 2.1 AA by default, Lighthouse Accessibility score >95 (SC-005)

**Validation**: Lighthouse Accessibility audit in CI/CD. Manual testing with screen readers (VoiceOver, NVDA). Translation validation ensures code blocks remain untranslated.

---

### Principle VI: Deployment and Performance Standards ✅ COMPLIANT

- Docusaurus build completes without errors (FR-047, tested in CI/CD)
- GitHub Actions automated deployment to GitHub Pages + Fly.io (FR-048)
- Page load <3s (SC-001, Lighthouse Performance >90)
- RAG response <2s (SC-003, monitored via responseTime field in API)
- Vector search <500ms (Qdrant Cloud performance, monitored via backend logs)
- Authentication: HTTP-only cookies with secure, httpOnly, sameSite=strict flags (Assumption #16)
- Rate limiting: 10 requests/minute per IP for /api/chat, 5 requests/minute for /api/translate

**Validation**: Lighthouse CI enforces SC-005 thresholds. Backend logs track response times. Load testing with 50 concurrent users during Phase 7.

---

### Principle VII: Spec-Driven Development with Claude Code ✅ COMPLIANT

- Specification created via /sp.specify (completed)
- Clarifications added via /sp.clarify (completed, Assumptions #13-17)
- This plan created via /sp.plan (in progress)
- Tasks will be generated via /sp.tasks (Phase 2, after plan approval)
- Subagents created for reusable workflows:
  - chapter-generator: Constitution-compliant chapter creation
  - code-validator: Example validation (Python, Bash, XML)
  - rag-indexer: Semantic chunking and vector indexing
  - test-generator: Playwright E2E test generation
- Skills created:
  - constitution-check: Validate artifacts against all principles
  - adr-quick: Rapid ADR generation for architectural decisions
- ADRs will be created for: Monorepo structure, Backend hosting (Fly.io), RAG citation linking, OpenAI model selection, Authentication session management, Personalization implementation, Translation caching, E2E framework, Deployment strategy (10 total, see research.md)

**Validation**: PHRs created for all significant work. ADRs linked in plan.md and tasks.md. Subagent/skill usage tracked in commit messages.

---

### Principle VIII: Version Control and Collaboration Hygiene ✅ COMPLIANT

- Conventional commits enforced: `<type>(<scope>): <subject>` format (SC-012)
- Branch naming: `002-textbook-docusaurus-setup` (current feature branch)
- Atomic commits: <500 lines per commit (SC-012, monitored via git log)
- Claude Code co-author footer on all AI-assisted commits (SC-012)
- Secrets management: `.env.example` with placeholders, `.env` in `.gitignore`, environment variables in CI/CD
- README includes: Setup instructions, deployment steps, architecture overview, demo video link
- Dependencies pinned: package.json uses exact versions (no ^/~), requirements.txt uses ==

**Validation**: Pre-commit hooks validate commit message format. CI/CD fails on commits >500 lines. Git log audit before submission validates SC-012.

---

**GATE RESULT**: ✅ ALL PRINCIPLES COMPLIANT - Proceed to Phase 0 Research

---

## Project Structure

### Documentation (this feature)

```text
specs/002-textbook-docusaurus-setup/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file (in progress)
├── research.md          # Phase 0: Architecture decisions and unknowns (to be created)
├── data-model.md        # Phase 1: Entity schemas and chapter structure (to be created)
├── quickstart.md        # Phase 1: Development setup and deployment guide (to be created)
├── contracts/           # Phase 1: API contracts (to be created)
│   ├── chat-api.yaml    # OpenAPI spec for /api/chat, /api/embed
│   ├── auth-api.yaml    # OpenAPI spec for /api/signup, /api/signin, /api/signout
│   └── translation-api.yaml  # OpenAPI spec for /api/translate
├── tasks.md             # Phase 2: Task breakdown (created by /sp.tasks, not this command)
└── checklists/
    ├── requirements.md      # Spec quality checklist (completed)
    ├── smart-analysis.md    # SMART criteria analysis (completed)
    └── smart-fixes-summary.md  # SMART fixes documentation (completed)
```

### Source Code (repository root)

```text
# Monorepo structure (ADR-001: Monorepo vs Multi-Repo)

docs/                                    # Docusaurus frontend
├── docusaurus.config.js                 # Site config, plugins, theme, navbar/footer
├── sidebars.js                          # Module/chapter navigation structure
├── src/
│   ├── components/
│   │   ├── Chatbot/
│   │   │   ├── ChatbotButton.tsx        # Floating button (bottom-right, z-index 1000)
│   │   │   ├── ChatPanel.tsx            # Chat UI panel (slides in from right)
│   │   │   ├── MessageList.tsx          # Conversation history with scroll
│   │   │   ├── MessageBubble.tsx        # Single message (user/assistant styling)
│   │   │   ├── CitationLink.tsx         # Clickable citation with chapter/section
│   │   │   ├── TextSelectionMenu.tsx    # "Ask about this" context menu
│   │   │   └── useChatAPI.ts            # Custom hook for /api/chat calls
│   │   ├── Auth/
│   │   │   ├── SignUpForm.tsx           # Email, password, background sliders (1-5)
│   │   │   ├── SignInForm.tsx           # Email, password, remember me
│   │   │   ├── ProfileIndicator.tsx     # User email + sign out button
│   │   │   ├── AuthGuard.tsx            # Protect personalization/translation features
│   │   │   └── useAuth.ts               # Auth context: user, session, signIn/signOut
│   │   ├── Personalization/
│   │   │   ├── PersonalizeButton.tsx    # Variant switcher (beginner/intermediate/advanced)
│   │   │   └── usePersonalization.ts    # Variant selection logic based on user profile
│   │   ├── Translation/
│   │   │   ├── TranslateButton.tsx      # Language toggle (English ↔ Urdu)
│   │   │   └── useTranslation.ts        # Translation API hook with caching
│   │   ├── Content/
│   │   │   ├── CodeBlock.tsx            # Syntax highlighting (Prism) + copy button
│   │   │   ├── MermaidDiagram.tsx       # Mermaid renderer with error handling
│   │   │   └── ExerciseBlock.tsx        # Exercise display (beginner/intermediate/advanced badges)
│   │   └── Layout/
│   │       └── DocItemWrapper.tsx       # Swizzled Docusaurus component (inject chatbot/personalize/translate buttons)
│   ├── css/
│   │   ├── custom.css                   # Theme customization (colors, fonts, spacing)
│   │   └── chatbot.module.css           # Scoped styles for chatbot components
│   ├── pages/
│   │   ├── index.tsx                    # Homepage (hero, module cards, demo video embed)
│   │   └── signup.tsx                   # Signup page (redirects to textbook after signup)
│   └── theme/                           # Docusaurus theme overrides (navbar, footer)
├── docs/                                # Content (markdown chapters)
│   ├── intro.md                         # Welcome page (course overview, prerequisites)
│   ├── module1-ros2/
│   │   ├── _category_.json              # Module metadata (label, position)
│   │   ├── chapter1-introduction.md     # Default variant
│   │   ├── chapter1-introduction-beginner.md
│   │   ├── chapter1-introduction-advanced.md
│   │   ├── chapter2-nodes-topics.md
│   │   ├── chapter2-nodes-topics-beginner.md
│   │   ├── chapter2-nodes-topics-advanced.md
│   │   └── ... (8-10 chapters × 3 variants = 24-30 files)
│   ├── module2-gazebo-unity/
│   │   └── ... (8-10 chapters × 3 variants)
│   ├── module3-nvidia-isaac/
│   │   └── ... (8-10 chapters × 3 variants)
│   └── module4-vla/
│       └── ... (8-10 chapters × 3 variants)
├── static/
│   ├── img/                             # Diagrams, screenshots, logos
│   └── demo-video.mp4                   # Hackathon demo video (<90s)
├── package.json                         # Dependencies (Docusaurus, React, TypeScript)
├── tsconfig.json                        # TypeScript config
└── babel.config.js                      # Babel config for React JSX

backend/                                 # FastAPI backend
├── app/
│   ├── main.py                          # FastAPI app, CORS, startup events
│   ├── config.py                        # Environment variables, settings (Pydantic BaseSettings)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py                      # POST /api/chat (RAG query)
│   │   ├── embed.py                     # POST /api/embed (admin: index content)
│   │   ├── translate.py                 # POST /api/translate (Urdu translation)
│   │   └── auth.py                      # POST /api/signup, /api/signin, /api/signout, GET /api/profile
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag_service.py               # RAG logic: retrieve top-5 chunks + generate answer
│   │   ├── chunking_service.py          # Semantic chunking (Assumption #13: H2/H3, 100-token overlap)
│   │   ├── embedding_service.py         # OpenAI text-embedding-3-small wrapper
│   │   ├── translation_service.py       # OpenAI translation + code preservation
│   │   └── auth_service.py              # Better-Auth integration + session management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py                      # User SQLAlchemy model (email, password_hash, backgrounds)
│   │   ├── chunk.py                     # ContentChunk model (chapter_id, content, qdrant_point_id)
│   │   ├── message.py                   # ChatMessage model (conversation_id, role, content, citations)
│   │   └── session.py                   # Session model (user_id, token, expires_at)
│   ├── db/
│   │   ├── __init__.py
│   │   ├── qdrant_client.py             # Qdrant Cloud connection + collection management
│   │   └── postgres_client.py           # Neon Postgres connection + SQLAlchemy engine
│   └── utils/
│       ├── __init__.py
│       ├── citation_formatter.py        # Map chunks to URLs with anchors
│       ├── security.py                  # JWT generation/validation, CSRF tokens
│       └── validators.py                # Input validation (email, password strength)
├── scripts/
│   ├── index_content.py                 # Invoke rag-indexer subagent for all chapters
│   ├── validate_code.py                 # Invoke code-validator subagent for all chapters
│   └── seed_database.py                 # Create Qdrant collection, Postgres tables
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Pytest fixtures (test client, mock DB)
│   ├── test_rag_service.py              # Unit tests for RAG retrieval + generation
│   ├── test_chunking_service.py         # Unit tests for semantic chunking
│   ├── test_auth_service.py             # Unit tests for signup/signin/session
│   └── test_translation_service.py      # Unit tests for Urdu translation + code preservation
├── Dockerfile                           # Multi-stage build (Python 3.11-slim)
├── requirements.txt                     # Python dependencies (FastAPI, openai, qdrant-client, etc.)
└── .env.example                         # Environment variable template

scripts/                                 # Automation scripts (repository root)
├── generate-chapter.sh                  # Invoke chapter-generator subagent
├── validate-code.sh                     # Invoke code-validator subagent
├── index-content.sh                     # Invoke rag-indexer subagent
├── generate-tests.sh                    # Invoke test-generator subagent
└── check-success-criteria.sh            # Validate all 12 SCs (SC-001 through SC-012)

tests/e2e/                               # Playwright E2E tests (repository root)
├── playwright.config.ts                 # Playwright config (browsers, base URL, timeout)
├── tests/
│   ├── navigation.spec.ts               # Test sidebar, search, prev/next chapter
│   ├── chatbot.spec.ts                  # Test query, citation click, text selection
│   ├── auth.spec.ts                     # Test signup, signin, session persistence
│   ├── personalization.spec.ts          # Test variant selection, content changes
│   ├── translation.spec.ts              # Test Urdu display, code preservation
│   └── smoke.spec.ts                    # Test homepage, modules, chatbot opens
└── fixtures/
    └── test-queries.json                # 50-query test set for SC-004 validation

.github/
└── workflows/
    ├── deploy-frontend.yml              # Build Docusaurus, deploy to GitHub Pages
    ├── deploy-backend.yml               # Build Docker image, deploy to Fly.io
    ├── test.yml                         # Run Playwright E2E + pytest backend tests
    └── lighthouse.yml                   # Run Lighthouse CI for SC-005 validation

.specify/                                # Spec-Kit Plus artifacts
├── memory/
│   └── constitution.md                  # Project constitution v1.0.0
├── templates/
│   └── phr-template.prompt.md           # Prompt History Record template
├── scripts/
│   └── bash/
│       ├── setup-plan.sh
│       ├── create-phr.sh
│       └── update-agent-context.sh
└── context/
    └── claude.md                        # Claude Code context file (updated by update-agent-context.sh)

.claude/                                 # Claude Code subagents and skills
├── subagents/
│   ├── chapter-generator.md             # Generate constitution-compliant chapters
│   ├── code-validator.md                # Validate Python/Bash/XML code examples
│   ├── rag-indexer.md                   # Semantic chunking + vector indexing
│   └── test-generator.md                # Generate Playwright E2E tests
└── skills/
    ├── constitution-check.md            # Validate artifacts against constitution
    └── adr-quick.md                     # Rapid ADR generation

history/
├── prompts/
│   ├── constitution/
│   │   └── 0001-physical-ai-textbook-constitution.constitution.prompt.md
│   ├── 002-textbook-docusaurus-setup/
│   │   ├── 0001-textbook-docusaurus-specification.spec.prompt.md
│   │   └── 0002-specification-clarification-review.spec.prompt.md
│   └── general/
│       └── 0001-constitution-git-commit.general.prompt.md
└── adr/                                 # Architectural Decision Records (to be created in Phase 0)
    ├── 0001-monorepo-vs-multi-repo.md
    ├── 0002-docusaurus-theme-strategy.md
    ├── 0003-backend-hosting-platform.md
    ├── 0004-rag-citation-linking.md
    ├── 0005-openai-model-selection.md
    ├── 0006-authentication-session-management.md
    ├── 0007-personalization-implementation.md
    ├── 0008-translation-caching-strategy.md
    ├── 0009-e2e-testing-framework.md
    └── 0010-deployment-timeline-strategy.md

.gitignore                               # Ignore .env, node_modules/, __pycache__/, etc.
.env.example                             # Environment variables template
README.md                                # Project overview, setup, deployment, demo link
package.json                             # Root package.json (npm workspaces for docs/)
```

**Structure Decision**: **Monorepo with separate frontend (docs/) and backend (backend/) folders** (ADR-001). This structure simplifies development (single repo clone), keeps API contracts co-located with implementations, and enables single CI/CD pipeline for both deployments. Docusaurus content (docs/docs/) is separate from React components (docs/src/) per Docusaurus conventions. E2E tests at repository root test the integrated system (frontend + backend).

## Complexity Tracking

> **No violations - This section is empty per template instructions**

All Constitution principles are compliant (see Constitution Check section). No complexity violations require justification.

---

## Phase 0: Research & Architectural Decisions

**Objective**: Resolve all unknowns from Technical Context, make architectural decisions, document ADRs.

### Research Tasks

The following unknowns must be researched and resolved before Phase 1 design:

1. **Docusaurus 2.x vs 3.x Version Selection**
   - Research stability, features, Lighthouse performance
   - Decision factors: Build speed, React 18 support, plugin ecosystem
   - Output: ADR-002 (Docusaurus Theme Strategy) - recommend version

2. **Backend Hosting Platform Selection**
   - Research Railway.app, Fly.io, Vercel Serverless, AWS EC2
   - Decision factors: Cold start latency (SC-003 <2s), free tier limits, deployment complexity
   - Output: ADR-003 (Backend Hosting Platform) - recommend Fly.io for no cold starts

3. **OpenAI Model Selection for RAG**
   - Research GPT-4o vs GPT-4o-mini vs GPT-3.5-turbo (chat)
   - Research text-embedding-3-small vs text-embedding-3-large (embeddings)
   - Decision factors: Cost, latency, quality (SC-003, SC-004)
   - Output: ADR-005 (OpenAI Model Selection) - recommend GPT-4o-mini + text-embedding-3-small

4. **Better-Auth Session Management Best Practices**
   - Research HTTP-only cookies vs localStorage vs sessionStorage
   - Research JWT generation/validation, CSRF protection, token refresh
   - Decision factors: Security (XSS protection), SC-006 <1min signup
   - Output: ADR-006 (Authentication Session Management) - validate Assumption #16

5. **Citation Linking Mechanism**
   - Research Docusaurus URL anchor generation (heading IDs)
   - Research scroll behavior, browser compatibility, shareable URLs
   - Decision factors: Reliability (SC-004 95% citation accuracy)
   - Output: ADR-004 (RAG Citation Linking) - recommend URL anchors to heading IDs

6. **Content Personalization Variant Storage**
   - Research build-time pre-computation vs on-demand LLM generation
   - Research cache invalidation, storage overhead (96-120 files)
   - Decision factors: SC-007 <1s display, build time, API costs
   - Output: ADR-007 (Personalization Implementation) - validate Assumption #14 (pre-computed)

7. **Translation Caching Strategy**
   - Research client-only (localStorage) vs server-only (Neon/Redis) vs two-tier
   - Research cache hit rates, storage costs, API call reduction
   - Decision factors: SC-008 <3s first request, <0.5s cached
   - Output: ADR-008 (Translation Caching Strategy) - recommend two-tier caching

8. **E2E Testing Framework Selection**
   - Research Playwright vs Cypress vs Selenium
   - Research test execution speed, multi-browser support, CI/CD integration
   - Decision factors: test-generator subagent compatibility, hackathon timeline
   - Output: ADR-009 (E2E Testing Framework) - recommend Playwright

9. **Deployment Timeline Strategy**
   - Research continuous deployment vs milestone-based vs end-only
   - Research GitHub Actions caching, preview deployments, rollback
   - Decision factors: SC-010 (deploy by 5:00 PM Nov 30), risk mitigation
   - Output: ADR-010 (Deployment Timeline Strategy) - recommend continuous from Day 1

10. **Monorepo vs Multi-Repo Structure**
    - Research monorepo tools (npm workspaces, Turborepo, Nx)
    - Research deployment complexity (separate frontend/backend deploys)
    - Decision factors: Development velocity, CI/CD simplicity
    - Output: ADR-001 (Monorepo vs Multi-Repo) - already decided (monorepo), document rationale

### Research Deliverable

Create `research.md` with the following structure for each decision:

```markdown
## [Decision Title]

**Context**: [Why this decision is needed, what problem it solves]

**Options Considered**:
1. Option A: [Description, pros, cons]
2. Option B: [Description, pros, cons]
3. Option C: [Description, pros, cons]

**Decision**: [Chosen option]

**Rationale**: [Why this option was chosen, how it meets constraints/goals]

**Consequences**: [Trade-offs accepted, future implications]

**References**: [Links to docs, benchmarks, best practices]
```

All 10 decisions will be documented in `research.md` and corresponding ADRs will be created in `history/adr/`.

---

## Phase 1: Data Models & API Contracts

**Objective**: Design entity schemas, database models, API contracts, and development quickstart guide.

**Prerequisites**: Phase 0 research.md complete with all decisions made.

### Task 1.1: Extract Entities and Create data-model.md

From the specification (spec.md), extract the following entities and document their schemas:

#### Entity: Module

**Purpose**: Represents a major learning unit in the textbook.

**Attributes**:
- `module_id` (string, primary key): Unique identifier (e.g., "module1-ros2")
- `module_number` (integer): Display order (1-4)
- `title` (string): Module name (e.g., "ROS 2 Fundamentals")
- `description` (string): Overview of module content
- `prerequisites` (array of module_id): Required prior modules (e.g., [] for Module 1, ["module1-ros2"] for Module 2)
- `chapter_count` (integer): Number of chapters in module (8-10)
- `estimated_duration` (string): Total reading time (e.g., "8 hours")

**Relationships**:
- Has many Chapters (one-to-many)

**Validation Rules**:
- `module_number` must be 1-4
- `chapter_count` must be 8-10 (per FR-003)
- `prerequisites` must reference existing modules

**Storage**: Not stored in database (static metadata in Docusaurus sidebars.js and _category_.json files)

---

#### Entity: Chapter

**Purpose**: Represents a single learning unit within a module.

**Attributes**:
- `chapter_id` (string, primary key): Unique identifier (e.g., "module1-chapter1-introduction")
- `chapter_number` (integer): Display order within module (1-10)
- `title` (string): Chapter name (e.g., "Introduction to ROS 2")
- `module_id` (string, foreign key): Parent module
- `content_default` (markdown string): Default variant content
- `content_beginner` (markdown string, optional): Beginner variant content
- `content_advanced` (markdown string, optional): Advanced variant content
- `learning_objectives` (array of strings): 3-6 objectives per Assumption #15
- `estimated_reading_time` (integer): Minutes (e.g., 15)
- `exercise_count` (integer): Always 3 (beginner, intermediate, advanced per Assumption #15)
- `code_block_count` (integer): Minimum 3 per Assumption #15
- `diagram_count` (integer): Minimum 2 per Assumption #15

**Relationships**:
- Belongs to Module (many-to-one)
- Has many ContentChunks (one-to-many, for RAG indexing)

**Validation Rules** (per Assumption #15):
- `chapter_number` must be 1-10
- `learning_objectives` must have 3-6 items
- `estimated_reading_time` must be 10-20 minutes
- `content_default` must be 2000-3000 words (excluding code blocks)
- `exercise_count` must be exactly 3
- `code_block_count` must be ≥3
- `diagram_count` must be ≥2

**Storage**: Content stored as markdown files in `docs/docs/module{N}-{name}/`, metadata extracted during Docusaurus build.

**State Transitions**: N/A (static content, no workflow)

---

#### Entity: User

**Purpose**: Represents a signed-in student (bonus feature, FR-028).

**Database Table**: `users` (Neon Postgres)

**Attributes**:
- `id` (UUID, primary key): Auto-generated
- `email` (string, unique, indexed): User email (validated per FR-030)
- `password_hash` (string): bcrypt hash (10 rounds, per Assumption #16)
- `software_background` (integer): Scale 1-5 (FR-029)
- `hardware_background` (integer): Scale 1-5 (FR-029)
- `created_at` (timestamp): Account creation time
- `updated_at` (timestamp): Last profile update

**Relationships**:
- Has many Sessions (one-to-many)
- Has many ChatMessages (one-to-many, optional: anonymous users have NULL user_id)

**Validation Rules**:
- `email` must match regex: `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
- `password` (pre-hash) must be ≥8 characters (FR-030)
- `software_background` must be 1-5
- `hardware_background` must be 1-5

**Storage**: Neon Postgres `users` table

**Indexes**:
- Unique index on `email` (fast lookup for signin)
- Index on `created_at` (analytics, if needed)

**Security**:
- Password NEVER stored in plaintext
- Password hash uses bcrypt with salt
- Email not exposed in API responses (only in authenticated /api/profile)

---

#### Entity: Session

**Purpose**: Represents an authenticated user session (bonus feature, Assumption #16).

**Database Table**: `sessions` (Neon Postgres)

**Attributes**:
- `id` (UUID, primary key): Auto-generated
- `user_id` (UUID, foreign key): References `users.id`
- `token` (string, unique, indexed): JWT token (signed with HS256)
- `expires_at` (timestamp): 7 days from creation (per Assumption #16)
- `created_at` (timestamp): Session creation time
- `last_accessed_at` (timestamp): Last API call with this token (for auto-refresh logic)

**Relationships**:
- Belongs to User (many-to-one)

**Validation Rules**:
- `token` must be valid JWT (signature verified)
- `expires_at` must be in future (expired sessions invalid)
- Auto-refresh if `last_accessed_at` > 5 days old (per Assumption #16)

**Storage**: Neon Postgres `sessions` table

**Indexes**:
- Unique index on `token` (fast lookup for validation)
- Index on `user_id` (find all sessions for user)
- Index on `expires_at` (cleanup expired sessions via cron job)

**State Transitions**:
1. **Created**: User signs in → session created with 7-day expiry
2. **Refreshed**: API call with token age >5 days → `expires_at` extended, `last_accessed_at` updated
3. **Expired**: Current time > `expires_at` → session invalid, user must sign in again
4. **Invalidated**: User signs out → session deleted

---

#### Entity: ContentChunk

**Purpose**: Represents a semantic section of textbook content indexed for RAG (FR-021, Assumption #13).

**Database Table**: `content_chunks` (Neon Postgres for metadata)
**Vector Storage**: Qdrant Cloud `textbook_chunks` collection (for embeddings)

**Attributes**:
- `id` (UUID, primary key): Auto-generated
- `chapter_id` (string, indexed): References Chapter (e.g., "module1-chapter1-introduction")
- `module_name` (string): Module identifier (e.g., "ros2")
- `chapter_title` (string): Chapter name (e.g., "Introduction to ROS 2")
- `section_heading` (string): H2/H3 heading (e.g., "What is ROS 2?")
- `content` (text): Chunk text (max 1000 tokens per Assumption #13)
- `content_type` (enum): `text` | `code` | `exercise`
- `heading_level` (integer): 2 (H2) or 3 (H3)
- `token_count` (integer): Actual token count (for validation)
- `qdrant_point_id` (string, unique): Qdrant vector ID (UUID)
- `created_at` (timestamp): Indexing time

**Relationships**:
- Belongs to Chapter (many-to-one, via `chapter_id` string reference)

**Validation Rules** (per Assumption #13):
- `token_count` must be ≤1000 (except code blocks, which can exceed)
- `content_type` = `code` → chunk can exceed 1000 tokens (kept whole)
- `heading_level` must be 2 or 3
- `qdrant_point_id` must exist in Qdrant collection

**Storage**:
- Metadata: Neon Postgres `content_chunks` table
- Vector: Qdrant Cloud `textbook_chunks` collection (1536-dimension embedding from text-embedding-3-small)

**Qdrant Payload** (stored with vector):
```json
{
  "chunk_id": "uuid",
  "chapter_id": "module1-chapter1-introduction",
  "module_name": "ros2",
  "chapter_title": "Introduction to ROS 2",
  "section_heading": "What is ROS 2?",
  "content_type": "text",
  "token_count": 380
}
```

**Indexes**:
- Index on `chapter_id` (find all chunks for chapter)
- Index on `module_name` (find all chunks for module)
- Index on `qdrant_point_id` (map Qdrant results to Postgres metadata)

**Chunking Strategy** (per Assumption #13):
1. Priority 1: Full section if <1000 tokens (H2/H3 heading + content)
2. Priority 2: Sub-section at H3 level if H2 section too large
3. Priority 3: Paragraph-level split if H3 still too large
4. Priority 4: Character split as last resort (avoid if possible)
5. Special case: Code blocks >1000 tokens kept whole, context added to metadata

**Overlap Strategy**: 100-token overlap between adjacent chunks (last 100 tokens of chunk N = first 100 tokens of chunk N+1) to preserve context across boundaries.

---

**Chunking Example (Concrete Implementation)**:

Given this sample chapter structure:

```markdown
# Chapter 1: Introduction to ROS 2

## What is ROS 2? (350 tokens)
ROS 2 (Robot Operating System 2) is the next generation of ROS, a flexible
framework for writing robot software... [300 more words of explanation]

## Installing ROS 2 (850 tokens total)
This section guides you through installing ROS 2 on Ubuntu 22.04...
[200 words of introduction]

### Prerequisites (280 tokens)
Before installing ROS 2, ensure your system meets these requirements:
- Ubuntu 22.04 LTS
- 4GB RAM minimum
[150 more words]

### Installation Steps (420 tokens)
Follow these steps to install ROS 2 Humble:

```bash
sudo apt update
sudo apt install ros-humble-desktop
# ... 30 more lines of installation commands
```

[100 words of post-installation verification]

## Creating Your First Node (650 tokens)
Let's create a simple publisher node... [500 words + code example]
```

**Expected Chunking Output**:

**Chunk 1**: `module1-chapter1-intro-what-is-ros2`
- **Section**: H2 "What is ROS 2?"
- **Content**: Full section (350 tokens, under 1000 limit)
- **Content Type**: `text`
- **Token Count**: 350
- **Rationale**: Priority 1 - Full section fits under 1000 tokens

**Chunk 2**: `module1-chapter1-intro-installing-ros2-intro`
- **Section**: H2 "Installing ROS 2" (intro only)
- **Content**: Introduction paragraph before H3 (200 tokens) + 100-token overlap from Chunk 3
- **Content Type**: `text`
- **Token Count**: 300 (200 content + 100 overlap)
- **Rationale**: Priority 2 - H2 section too large (850 tokens), split at H3 boundary

**Chunk 3**: `module1-chapter1-intro-installing-prerequisites`
- **Section**: H3 "Prerequisites"
- **Content**: 100-token overlap from Chunk 2 + Full H3 content (280 tokens)
- **Content Type**: `text`
- **Token Count**: 380 (100 overlap + 280 content)
- **Rationale**: Priority 2 - H3 sub-section under 1000 tokens

**Chunk 4**: `module1-chapter1-intro-installing-steps`
- **Section**: H3 "Installation Steps"
- **Content**: Code block (150 tokens) + surrounding text (270 tokens) + 100-token overlap
- **Content Type**: `code` (contains substantial code block)
- **Token Count**: 520 (100 overlap + 270 text + 150 code)
- **Rationale**: Special case - Code block kept whole even though total H3 is 420 tokens

**Chunk 5**: `module1-chapter1-intro-first-node`
- **Section**: H2 "Creating Your First Node"
- **Content**: Full section (650 tokens) with 100-token overlap from Chunk 4
- **Content Type**: `text`
- **Token Count**: 750 (100 overlap + 650 content)
- **Rationale**: Priority 1 - Full section under 1000 tokens

**Total Chunks**: 5 chunks from 1 chapter (1850 tokens of content)

**Overlap Visualization**:
```
Chunk 2: [200 tokens] + [100 overlap with Chunk 3]
Chunk 3: [100 overlap from Chunk 2] + [280 tokens] + [overlap with Chunk 4]
Chunk 4: [overlap from Chunk 3] + [420 tokens]
Chunk 5: [overlap from Chunk 4] + [650 tokens]
```

**Edge Cases Handled**:
1. **Code blocks >1000 tokens**: Keep whole, add surrounding context to metadata
2. **H2 section >1000 tokens**: Split at H3 boundaries
3. **H3 section >1000 tokens**: Split at paragraph boundaries (Priority 3)
4. **No natural boundaries**: Character split with sentence awareness (Priority 4, last resort)

**Validation Checks**:
- ✓ All chunks ≤1000 tokens (except code-only chunks)
- ✓ 100-token overlap between adjacent chunks
- ✓ Section headings preserved in metadata
- ✓ Code blocks not split mid-syntax

---

#### Entity: ChatMessage

**Purpose**: Represents a single message in a RAG chatbot conversation (FR-026).

**Database Table**: `chat_messages` (Neon Postgres)

**Attributes**:
- `id` (UUID, primary key): Auto-generated
- `conversation_id` (UUID, indexed): Groups messages in multi-turn conversation
- `user_id` (UUID, foreign key, nullable): References `users.id` (NULL for anonymous users)
- `role` (enum): `user` | `assistant`
- `content` (text): Message text (user query or assistant response)
- `citations` (JSON, nullable): Array of citation objects (only for `assistant` messages)
- `response_time_ms` (integer, nullable): API response time for `assistant` messages (for SC-003 validation)
- `created_at` (timestamp): Message timestamp

**Relationships**:
- Belongs to User (many-to-one, optional: anonymous users have NULL user_id)
- Belongs to Conversation (many-to-one, via `conversation_id`)

**Validation Rules**:
- `role` must be `user` or `assistant`
- `content` must not be empty
- `citations` required for `assistant` messages (can be empty array if no sources)
- `response_time_ms` must be <2000 for `assistant` messages (SC-003 compliance)

**Citation Object Schema**:
```json
{
  "chapterTitle": "Introduction to ROS 2",
  "sectionHeading": "What is ROS 2?",
  "url": "/docs/module1-ros2/chapter1-introduction#what-is-ros-2"
}
```

**Storage**: Neon Postgres `chat_messages` table

**Indexes**:
- Index on `conversation_id` (fetch all messages for conversation)
- Index on `user_id` (fetch user's chat history, if needed)
- Index on `created_at` (chronological ordering)

**State Transitions**: N/A (immutable, append-only log)

---

### Task 1.2: Generate API Contracts

Create OpenAPI 3.0 specifications for all API endpoints in `contracts/` directory:

#### contracts/chat-api.yaml

```yaml
openapi: 3.0.0
info:
  title: RAG Chat API
  version: 1.0.0
  description: FastAPI endpoints for RAG chatbot queries and content indexing

servers:
  - url: http://localhost:8000
    description: Local development
  - url: https://textbook-api.fly.dev
    description: Production (Fly.io)

paths:
  /api/chat:
    post:
      summary: Send RAG query and receive answer with citations
      operationId: sendChatQuery
      tags:
        - Chat
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - query
              properties:
                query:
                  type: string
                  description: User question (max 500 characters)
                  example: "What is a ROS 2 node?"
                conversationId:
                  type: string
                  format: uuid
                  description: Optional conversation ID for multi-turn context
                context:
                  type: string
                  description: Optional selected text for text-selection queries
                  example: "A ROS 2 node is a process that performs computation..."
      responses:
        '200':
          description: Successful RAG response
          content:
            application/json:
              schema:
                type: object
                required:
                  - answer
                  - citations
                  - conversationId
                  - responseTime
                properties:
                  answer:
                    type: string
                    description: LLM-generated answer
                    example: "A ROS 2 node is a process that performs computation. It communicates with other nodes via topics, services, or actions."
                  citations:
                    type: array
                    description: Source citations (minimum 1 for SC-004)
                    items:
                      type: object
                      required:
                        - chapterTitle
                        - sectionHeading
                        - url
                      properties:
                        chapterTitle:
                          type: string
                          example: "Introduction to ROS 2"
                        sectionHeading:
                          type: string
                          example: "What is ROS 2?"
                        url:
                          type: string
                          format: uri
                          example: "/docs/module1-ros2/chapter1-introduction#what-is-ros-2"
                  conversationId:
                    type: string
                    format: uuid
                    description: Conversation ID for follow-up queries
                  responseTime:
                    type: integer
                    description: API response time in milliseconds (must be <2000 per SC-003)
                    example: 1450
        '400':
          description: Invalid request (empty query, query too long)
        '500':
          description: Server error (OpenAI API failure, Qdrant unavailable)
      security:
        - cookieAuth: []  # Optional: works for both authenticated and anonymous users

  /api/embed:
    post:
      summary: Index chapter content into vector database (admin only)
      operationId: embedContent
      tags:
        - Indexing
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - chapterId
                - content
              properties:
                chapterId:
                  type: string
                  description: Chapter identifier
                  example: "module1-chapter1-introduction"
                content:
                  type: string
                  description: Full chapter markdown content
      responses:
        '200':
          description: Content successfully indexed
          content:
            application/json:
              schema:
                type: object
                properties:
                  chunksCreated:
                    type: integer
                    description: Number of chunks created
                    example: 12
                  vectorsIndexed:
                    type: integer
                    description: Number of vectors added to Qdrant
                    example: 12
        '400':
          description: Invalid request
        '500':
          description: Indexing failed
      security:
        - apiKeyAuth: []  # Admin only (not exposed publicly)

components:
  securitySchemes:
    cookieAuth:
      type: apiKey
      in: cookie
      name: session_token
    apiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
```

#### contracts/auth-api.yaml

```yaml
openapi: 3.0.0
info:
  title: Authentication API
  version: 1.0.0
  description: Better-Auth powered signup/signin endpoints

paths:
  /api/signup:
    post:
      summary: Create new user account
      operationId: signup
      tags:
        - Auth
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - email
                - password
                - softwareBackground
                - hardwareBackground
              properties:
                email:
                  type: string
                  format: email
                  example: "student@example.com"
                password:
                  type: string
                  format: password
                  minLength: 8
                  example: "SecurePass123"
                softwareBackground:
                  type: integer
                  minimum: 1
                  maximum: 5
                  description: Self-rated software experience (1=beginner, 5=expert)
                  example: 3
                hardwareBackground:
                  type: integer
                  minimum: 1
                  maximum: 5
                  description: Self-rated hardware/robotics experience (1=beginner, 5=expert)
                  example: 2
      responses:
        '201':
          description: Account created, user auto-signed in
          headers:
            Set-Cookie:
              schema:
                type: string
                example: "session_token=<jwt>; HttpOnly; Secure; SameSite=Strict; Max-Age=604800"
          content:
            application/json:
              schema:
                type: object
                properties:
                  userId:
                    type: string
                    format: uuid
                  email:
                    type: string
                  softwareBackground:
                    type: integer
                  hardwareBackground:
                    type: integer
        '400':
          description: Invalid input (weak password, invalid email)
        '409':
          description: Email already exists

  /api/signin:
    post:
      summary: Sign in existing user
      operationId: signin
      tags:
        - Auth
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - email
                - password
              properties:
                email:
                  type: string
                  format: email
                password:
                  type: string
                  format: password
      responses:
        '200':
          description: Successfully signed in
          headers:
            Set-Cookie:
              schema:
                type: string
                example: "session_token=<jwt>; HttpOnly; Secure; SameSite=Strict; Max-Age=604800"
          content:
            application/json:
              schema:
                type: object
                properties:
                  userId:
                    type: string
                    format: uuid
                  email:
                    type: string
                  softwareBackground:
                    type: integer
                  hardwareBackground:
                    type: integer
        '401':
          description: Invalid credentials

  /api/signout:
    post:
      summary: Sign out current user
      operationId: signout
      tags:
        - Auth
      responses:
        '200':
          description: Successfully signed out
          headers:
            Set-Cookie:
              schema:
                type: string
                example: "session_token=; Max-Age=0"
      security:
        - cookieAuth: []

  /api/profile:
    get:
      summary: Get current user profile
      operationId: getProfile
      tags:
        - Auth
      responses:
        '200':
          description: User profile
          content:
            application/json:
              schema:
                type: object
                properties:
                  userId:
                    type: string
                    format: uuid
                  email:
                    type: string
                  softwareBackground:
                    type: integer
                  hardwareBackground:
                    type: integer
        '401':
          description: Not authenticated
      security:
        - cookieAuth: []
```

#### contracts/translation-api.yaml

```yaml
openapi: 3.0.0
info:
  title: Translation API
  version: 1.0.0
  description: Urdu translation endpoint with code preservation

paths:
  /api/translate:
    post:
      summary: Translate chapter content to Urdu
      operationId: translateContent
      tags:
        - Translation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - chapterId
                - content
                - targetLang
              properties:
                chapterId:
                  type: string
                  description: Chapter identifier for caching
                  example: "module1-chapter1-introduction"
                content:
                  type: string
                  description: Markdown content to translate
                targetLang:
                  type: string
                  enum: [urdu]
                  description: Target language (only Urdu supported for bonus feature)
      responses:
        '200':
          description: Translated content
          content:
            application/json:
              schema:
                type: object
                properties:
                  translatedContent:
                    type: string
                    description: Markdown with text translated, code blocks preserved
                  cached:
                    type: boolean
                    description: Whether result was served from cache
        '400':
          description: Invalid request
        '500':
          description: Translation API failed
      security:
        - cookieAuth: []  # Requires authentication (bonus feature)
```

---

### Task 1.3: Create quickstart.md

**Purpose**: Provide step-by-step development setup and deployment guide.

**Contents**:
1. Prerequisites (Node.js 18+, Python 3.11+, Docker, Git)
2. Repository clone and initial setup
3. Environment variables configuration (.env.example → .env)
4. Docusaurus local development (npm install, npm start)
5. FastAPI local development (pip install, uvicorn)
6. Database setup (Qdrant collection, Neon Postgres tables)
7. Content generation workflow (chapter-generator subagent)
8. RAG indexing workflow (rag-indexer subagent)
9. Running tests (Playwright E2E, pytest backend)
10. Deployment (GitHub Actions to GitHub Pages + Fly.io)

(Full quickstart.md content will be generated in Phase 1 execution)

---

### Task 1.4: Update Agent Context

After completing data-model.md and contracts/, update Claude Code context:

```bash
.specify/scripts/bash/update-agent-context.sh claude
```

This script will:
1. Detect Claude Code as the active agent
2. Read `.specify/context/claude.md`
3. Add new technologies from this plan (Docusaurus 3.x, FastAPI, Qdrant, Neon Postgres, Better-Auth)
4. Preserve manual additions between `<!-- MANUAL ADDITIONS -->` markers
5. Write updated context back to `.specify/context/claude.md`

---

## Phase 2: Task Breakdown (Not Part of /sp.plan)

**Note**: Phase 2 (task generation) is performed by the `/sp.tasks` command, NOT by `/sp.plan`. This plan provides the foundation for task creation but does not generate tasks.md.

**Expected /sp.tasks Output**:
- Detailed task breakdown for 7 implementation phases (Foundation, Content, RAG Backend, Chatbot UI, Auth, Personalization/Translation, Testing)
- Each task with: Description, Acceptance Criteria, Dependencies, Estimated Effort
- Test cases for each task (unit tests, E2E tests, validation checks)
- Mapping to Success Criteria (SC-001 through SC-012)

---

## Implementation Phases (High-Level Overview)

This section provides a roadmap for implementation. Detailed tasks will be generated by `/sp.tasks`.

### Phase 1: Foundation & Infrastructure (Days 1-2)

**Objective**: Establish project scaffold and deployment pipeline.

**Key Deliverables**:
- Docusaurus 3.x initialized with TypeScript, custom theme configured
- FastAPI backend with health check endpoint
- Monorepo structure: docs/, backend/, scripts/, tests/e2e/
- GitHub Actions CI/CD for frontend (GitHub Pages) and backend (Fly.io)
- Docker Compose for local Qdrant + Postgres (development only)
- Environment variables configured (.env.example created)

**Success Criteria Validated**: SC-010 (partial - site deployed)

---

### Phase 2: Content Architecture & MVP Chapters (Days 3-5)

**Objective**: Create content structure and generate Module 1 (ROS 2) chapters.

**Key Deliverables**:
- Docusaurus sidebar configuration for 4 modules (sidebars.js)
- Custom React components: CodeBlock, MermaidDiagram, ExerciseBlock
- Module 1 content: 8-10 chapters generated via chapter-generator subagent
- All code examples validated via code-validator subagent (100% pass rate)
- Personalization variants: 3 per chapter (beginner/intermediate/advanced)

**Success Criteria Validated**: SC-002 (partial - 25% content), SC-009 (100% code validation)

---

### Phase 3: RAG Backend Development (Days 6-8)

**Objective**: Build working RAG system with vector search and citation generation.

**Key Deliverables**:
- Qdrant Cloud collection created (`textbook_chunks`, 1536 dimensions)
- Neon Postgres schema created (ContentChunk, ChatMessage, User, Session tables)
- Semantic chunking service (per Assumption #13: H2/H3 boundaries, 100-token overlap)
- RAG service: Query → Vector search (top 5) → LLM generation → Citations
- FastAPI endpoints: POST /api/chat, POST /api/embed (admin)
- Module 1 content indexed (~80-120 chunks)

**RAG Prompt Template for Citation Extraction**:

The following prompt template ensures GPT-4o-mini generates answers with proper citations:

**System Prompt**:
```
You are an expert ROS 2 tutor helping students learn robotics concepts. Your role is to:

1. Answer questions accurately using ONLY the provided context from the textbook
2. Cite sources for EVERY piece of information using the format [^N]
3. Be concise but thorough (aim for 3-5 sentences)
4. If the context doesn't contain enough information, say "I don't have enough information in the textbook to answer that completely"

Citation Format:
- Use [^1], [^2], etc. inline where you reference information
- At the end of your response, list all citations in this exact format:
  [^1]: Chapter Title - Section Heading
  [^2]: Chapter Title - Section Heading

Example Response:
"ROS 2 uses DDS for communication[^1]. To create a publisher, use rclpy.create_publisher()[^2].

Citations:
[^1]: Introduction to ROS 2 - What is ROS 2?
[^2]: ROS 2 Nodes and Topics - Creating a Publisher"
```

**User Prompt Template**:
```python
def format_rag_prompt(query: str, chunks: List[ContentChunk]) -> str:
    context_blocks = []
    for idx, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[{idx}] Chapter: {chunk.chapter_title} | Section: {chunk.section_heading}\n"
            f"Content: {chunk.content}\n"
        )

    context = "\n".join(context_blocks)

    return f"""Context from textbook:
{context}

Question: {query}

Instructions: Answer using ONLY the context above. Cite sources using [^N] format and list all citations at the end."""
```

**Expected LLM Output Format**:
```
ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software[^1].
It uses DDS (Data Distribution Service) for inter-process communication[^2]. To install
ROS 2 Humble on Ubuntu 22.04, you need to add the ROS 2 apt repository and install the
desktop package[^3].

Citations:
[^1]: Introduction to ROS 2 - What is ROS 2?
[^2]: Introduction to ROS 2 - DDS Communication
[^3]: Installing ROS 2 - Ubuntu 22.04 Installation
```

**Citation Extraction Logic** (backend/app/services/rag_service.py):
```python
import re
from typing import List, Dict

def extract_citations(llm_response: str, chunks: List[ContentChunk]) -> List[Dict]:
    """
    Extract citations from LLM response and map to chapter URLs.

    Returns: [
        {
            "chapterTitle": "Introduction to ROS 2",
            "sectionHeading": "What is ROS 2?",
            "url": "/docs/module1-ros2/chapter1-introduction#what-is-ros-2"
        }
    ]
    """
    citations = []

    # Extract citation section (after "Citations:")
    citation_match = re.search(r'Citations:\s*\n((?:\[\^\d+\]:.*\n?)+)', llm_response)
    if not citation_match:
        return citations

    citation_text = citation_match.group(1)

    # Parse each citation line: [^1]: Chapter Title - Section Heading
    citation_pattern = r'\[\^(\d+)\]:\s*(.+?)\s*-\s*(.+?)(?:\n|$)'
    for match in re.finditer(citation_pattern, citation_text):
        citation_num = int(match.group(1))
        chapter_title = match.group(2).strip()
        section_heading = match.group(3).strip()

        # Map to chunk URL (find matching chunk)
        chunk = chunks[citation_num - 1] if citation_num <= len(chunks) else None
        if chunk:
            url = f"/docs/{chunk.module_name}/chapter{chunk.chapter_id}#{slugify(section_heading)}"
            citations.append({
                "chapterTitle": chapter_title,
                "sectionHeading": section_heading,
                "url": url
            })

    return citations

def slugify(text: str) -> str:
    """Convert heading to Docusaurus anchor ID (lowercase, hyphens)"""
    return re.sub(r'[^\w\s-]', '', text).strip().lower().replace(' ', '-')
```

**Validation Checks**:
- ✓ Every citation [^N] in answer text has corresponding entry in Citations section
- ✓ Citation numbers sequential (1, 2, 3, not 1, 3, 5)
- ✓ All cited chunks exist in top-5 retrieved chunks
- ✓ URLs resolve to valid chapter sections (404 check during testing)

**Error Handling**:
- If LLM output missing "Citations:" section → Return answer with empty citations array, log warning
- If citation [^N] references non-existent chunk → Skip that citation, log error
- If LLM hallucinates section heading not in chunks → Use chunk.section_heading from matching chapter instead

**Success Criteria Validated**: SC-003 (partial - backend <2s), SC-004 (partial - citations)

---

### Phase 4: Chatbot Frontend Integration (Days 9-10)

**Objective**: Integrate RAG backend with Docusaurus UI for interactive chatbot.

**Key Deliverables**:
- Chatbot UI components: ChatbotButton, ChatPanel, MessageList, CitationLink
- Text selection feature: "Ask about this" context menu
- Citation navigation: Clickable links with smooth scroll to section
- Conversation context: sessionStorage for multi-turn queries
- E2E tests: Chatbot query, citation click, text selection

**Success Criteria Validated**: SC-001 (page load <3s), SC-003 (full <2s), SC-004 (95% citations)

---

### Phase 5: Authentication System (Days 11-12) ⭐ BONUS

**Objective**: Implement Better-Auth for user accounts with profile questions.

**Key Deliverables**:
- Better-Auth integration with HTTP-only cookies (per Assumption #16)
- User schema: Email, password_hash, software_background, hardware_background
- API endpoints: POST /api/signup, POST /api/signin, POST /api/signout, GET /api/profile
- Frontend components: SignUpForm, SignInForm, ProfileIndicator
- Session management: JWT with 7-day expiry, auto-refresh at 5 days

**Success Criteria Validated**: SC-006 (signup <1 minute)

---

### Phase 6: Personalization & Translation (Days 13-15) ⭐ BONUS

**Objective**: Implement content personalization and Urdu translation.

**Key Deliverables**:
- Personalization: Client-side variant selection (beginner/intermediate/advanced)
- PersonalizeButton component with localStorage caching
- Translation: OpenAI API integration with code preservation
- TranslateButton component with two-tier caching (localStorage + Neon Postgres)
- E2E tests: Variant selection, content changes, Urdu display, code preservation

**Success Criteria Validated**: SC-007 (<1s personalization), SC-008 (<3s translation, <0.5s cached)

---

### Phase 7: Testing, Optimization & Delivery (Days 16-18)

**Objective**: Finalize all features, optimize performance, create demo video.

**Key Deliverables**:
- E2E test suite: 20+ scenarios (navigation, chatbot, auth, personalization, translation)
- Lighthouse audits: Performance >90, Accessibility >95, Best Practices >90, SEO >90
- Code validation: 100% pass rate via code-validator subagent
- Demo video: <90 seconds, shows base features + 2 bonus features
- Final deployment: Nov 30, 5:00 PM (1 hour before deadline)
- Git history audit: Conventional commits, <500 lines per commit, Claude co-author

**Lighthouse Optimization Playbook**:

Run Lighthouse audits on: Homepage, Module 1 Chapter 1, Module 2 Chapter 1, Module 3 Chapter 1

**Target Scores**: Performance >90, Accessibility >95, Best Practices >90, SEO >90

---

### **If Performance Score <90**

**Diagnosis**: Run Lighthouse audit with detailed performance breakdown

**Common Issues & Fixes**:

1. **Large JavaScript Bundles** (usually biggest issue)
   - **Symptom**: "Reduce JavaScript execution time" warning, bundle >500KB
   - **Tool**: `npx webpack-bundle-analyzer` or `npm run analyze` (if configured)
   - **Fixes**:
     - Code splitting: Lazy load chatbot component
       ```tsx
       const ChatPanel = React.lazy(() => import('./components/ChatPanel'));
       ```
     - Remove unused dependencies: Run `npx depcheck` to find unused packages
     - Tree-shaking: Ensure `sideEffects: false` in package.json
   - **Target**: Total JavaScript <300KB gzipped

2. **Unoptimized Images**
   - **Symptom**: "Properly size images" or "Serve images in modern formats"
   - **Tool**: `sharp` or `imagemin`
   - **Fixes**:
     - Convert PNG to WebP:
       ```bash
       npx @squoosh/cli --webp auto static/img/*.png
       ```
     - Resize large images: Max 1920px width for hero images, 800px for diagrams
     - Use `loading="lazy"` for images below fold
   - **Target**: All images <100KB, WebP format

3. **Render-Blocking Resources**
   - **Symptom**: "Eliminate render-blocking resources" for CSS/fonts
   - **Fixes**:
     - Inline critical CSS (Docusaurus does this automatically for most themes)
     - Font optimization: Preload fonts or use `font-display: swap`
       ```css
       @font-face {
         font-family: 'CustomFont';
         font-display: swap;
       }
       ```
   - **Target**: First Contentful Paint (FCP) <1.5s

4. **Slow Server Response Time**
   - **Symptom**: "Reduce server response time (TTFB)" >600ms
   - **Fixes**:
     - Enable GitHub Pages CDN caching (automatic for static files)
     - Minimize HTML file size (Docusaurus minifies automatically)
     - Check if backend API calls block initial render (they shouldn't for static pages)
   - **Target**: Time to First Byte (TTFB) <300ms

5. **Unused CSS**
   - **Symptom**: "Remove unused CSS" >50KB unused
   - **Fixes**:
     - Purge unused Tailwind classes (if using Tailwind)
     - Remove unnecessary Docusaurus theme overrides
     - Use PurgeCSS in production build
   - **Target**: CSS <50KB gzipped

**Performance Budget**:
```json
{
  "resourceSizes": [
    { "resourceType": "script", "budget": 300 },
    { "resourceType": "stylesheet", "budget": 50 },
    { "resourceType": "image", "budget": 500 },
    { "resourceType": "total", "budget": 1000 }
  ],
  "resourceCounts": [
    { "resourceType": "script", "budget": 10 },
    { "resourceType": "stylesheet", "budget": 5 }
  ]
}
```

---

### **If Accessibility Score <95**

**Common Issues & Fixes**:

1. **Missing ARIA Labels**
   - **Symptom**: "Elements must have sufficient color contrast" or "Interactive elements must have labels"
   - **Fixes**:
     - Add `aria-label` to icon buttons:
       ```tsx
       <button aria-label="Open chatbot">
         <ChatIcon />
       </button>
       ```
     - Add `alt` text to all images
     - Use semantic HTML (`<nav>`, `<main>`, `<article>`)
   - **Tool**: Run `npx @axe-core/cli` for detailed accessibility audit

2. **Color Contrast Issues**
   - **Symptom**: "Background and foreground colors do not have sufficient contrast ratio"
   - **Fixes**:
     - Ensure contrast ratio ≥4.5:1 for normal text, ≥3:1 for large text
     - Use contrast checker: https://webaim.org/resources/contrastchecker/
     - Common fix: Darken text color or lighten background
   - **Tool**: Browser DevTools > Inspect > Accessibility tab

3. **Keyboard Navigation**
   - **Symptom**: "Links do not have a discernible name" or focus indicators missing
   - **Fixes**:
     - Add visible focus styles:
       ```css
       button:focus { outline: 2px solid blue; }
       ```
     - Ensure all interactive elements reachable via Tab key
     - Test: Navigate entire page using only keyboard
   - **Target**: All features accessible without mouse

4. **Form Labels**
   - **Symptom**: "Form elements do not have associated labels"
   - **Fixes**:
     - Associate labels with inputs:
       ```tsx
       <label htmlFor="email">Email</label>
       <input id="email" type="email" />
       ```
     - Add `aria-describedby` for error messages

**Accessibility Testing**:
- Automated: `npx @axe-core/cli http://localhost:3000`
- Manual: Test with screen reader (NVDA on Windows, VoiceOver on macOS)
- Keyboard: Navigate using Tab, Enter, Escape keys only

---

### **If Best Practices Score <90**

**Common Issues & Fixes**:

1. **HTTPS Issues**
   - **Symptom**: "Does not use HTTPS" (should not occur on GitHub Pages)
   - **Fix**: Ensure all external resources (CDN, fonts) use HTTPS

2. **Console Errors**
   - **Symptom**: "Browser errors were logged to the console"
   - **Fix**: Check browser console, fix all errors and warnings
   - Common: React key warnings, failed API calls during build

3. **Deprecated APIs**
   - **Symptom**: "Uses deprecated APIs"
   - **Fix**: Update dependencies to latest versions
   - Run: `npm audit` and `npm outdated`

---

### **If SEO Score <90**

**Common Issues & Fixes**:

1. **Missing Meta Tags**
   - **Symptom**: "Document does not have a meta description"
   - **Fix**: Add to `docusaurus.config.js`:
     ```js
     metadata: [
       { name: 'description', content: 'Learn Physical AI and Humanoid Robotics with ROS 2' }
     ]
     ```

2. **Missing Heading Hierarchy**
   - **Symptom**: "Heading elements are not in a sequentially-descending order"
   - **Fix**: Ensure H1 → H2 → H3 order (no skipping levels)

3. **Links Missing Text**
   - **Symptom**: "Links do not have descriptive text"
   - **Fix**: Avoid "click here", use descriptive text like "Read ROS 2 installation guide"

---

### **Optimization Workflow**

```bash
# 1. Run initial Lighthouse audit
npx @lhci/cli autorun --config lighthouserc.json

# 2. If Performance <90:
npm run analyze  # Check bundle size
npx depcheck     # Find unused deps
npx @squoosh/cli --webp auto static/img/*.png  # Optimize images

# 3. If Accessibility <95:
npx @axe-core/cli http://localhost:3000

# 4. Re-run Lighthouse (run 3 times, take median score)
npx @lhci/cli autorun --config lighthouserc.json
```

**Lighthouse CI Config** (`.github/workflows/lighthouse.yml`):
```yaml
- name: Run Lighthouse CI
  run: |
    npm install -g @lhci/cli
    lhci autorun
  env:
    LHCI_GITHUB_APP_TOKEN: ${{ secrets.LHCI_GITHUB_APP_TOKEN }}
```

**lighthouserc.json**:
```json
{
  "ci": {
    "collect": {
      "url": [
        "http://localhost:3000",
        "http://localhost:3000/docs/module1-ros2/chapter1-introduction"
      ],
      "numberOfRuns": 3
    },
    "assert": {
      "assertions": {
        "categories:performance": ["error", { "minScore": 0.9 }],
        "categories:accessibility": ["error", { "minScore": 0.95 }],
        "categories:best-practices": ["error", { "minScore": 0.9 }],
        "categories:seo": ["error", { "minScore": 0.9 }]
      }
    }
  }
}
```

**Success Criteria Validated**: SC-005 (Lighthouse), SC-010 (deployed), SC-011 (demo), SC-012 (git)

---

## Risk Mitigation Strategies

### Risk 1: Content Volume (32-40 chapters) Too Ambitious

**Likelihood**: High | **Impact**: High (blocks base requirements)

**Mitigation**:
- Prioritize Module 1 (ROS 2, 8-10 chapters) as MVP - base requirement satisfied
- Use chapter-generator subagent for acceleration (automated content creation)
- Manual review process: 30 min per chapter for quality assurance
- Defer Modules 2-4 to "nice to have" if timeline pressured

**Contingency**:
- If time runs out by Day 15, submit with 1 complete module + placeholder outlines for others
- Judges evaluate quality over quantity - better to have 10 excellent chapters than 40 mediocre ones

---

### Risk 2: RAG Quality Doesn't Meet 95% Citation Accuracy (SC-004)

**Likelihood**: Medium | **Impact**: High (base requirement failure)

**Mitigation**:
- Iterate on chunking strategy early (Days 6-7): Test with 10 queries, measure accuracy
- Use semantic chunking with rich metadata (chapter title, section heading, content type)
- Test with diverse query types: Beginner ("What is X?"), intermediate ("How do I Y?"), advanced ("Compare X vs Y")
- Monitor Qdrant search quality: Log top-5 results for manual inspection

**Contingency**:
- If 95% proves unachievable, reduce target to 90% and document reason in submission notes
- Focus on citation format correctness (URL resolves, section exists) over semantic perfection

---

### Risk 3: API Costs Exceed Budget During Development

**Likelihood**: Medium | **Impact**: Medium (blocks features if overspend)

**Mitigation**:
- Use Context7 MCP for semantic caching (reduces duplicate OpenAI calls)
- Set OpenAI usage limits: $5 soft cap, monitor daily via dashboard
- Optimize prompts: Minimize token count in RAG context window (top 5 chunks, not top 10)
- Pre-translate common queries during testing (avoid repeated translation API calls)

**Contingency**:
- If nearing limit, pause translation feature development (lowest priority bonus)
- Use free tier alternatives: Google Translate API has higher free tier than OpenAI translation

---

### Risk 4: Deployment Issues on Nov 30 (Last-Minute Failures)

**Likelihood**: Low | **Impact**: Critical (submission failure)

**Mitigation**:
- Deploy early and often: Phase 1 (Day 2), Phase 4 (Day 10), Phase 6 (Day 15)
- Smoke tests in CI/CD: Homepage loads, chatbot responds, 1 chapter per module accessible
- Backup deployment platform: If GitHub Pages fails, use Vercel (5-minute switch)
- Backend backup: If Fly.io fails, use Railway.app (similar deployment)

**Contingency**:
- Final deployment by 5:00 PM Nov 30 (1 hour buffer before 6:00 PM deadline per SC-010)
- If production fails, submit staging URL with note explaining issue

---

### Risk 5: Bonus Features Break Base Features

**Likelihood**: Medium | **Impact**: High (lose base points chasing bonus)

**Mitigation**:
- Feature flags: Disable auth/personalization/translation if bugs arise (environment variable toggle)
- Test in isolation: Separate E2E test suites for base vs bonus features
- Prioritize ruthlessly: Base features (Phases 1-4) complete before starting bonuses (Phases 5-6)

**Contingency**:
- If bonus feature breaks chatbot on Day 17, disable bonus and submit base only
- 150 points (base 100 + subagents 50) is competitive - bonuses are icing, not cake

---

## Validation Plan

### Testing Strategy

**Unit Tests** (Backend, pytest):
- `test_rag_service.py`: Vector search, LLM generation, citation formatting
- `test_chunking_service.py`: Semantic chunking, overlap, code block handling
- `test_auth_service.py`: Signup, signin, session validation, JWT generation
- `test_translation_service.py`: Urdu translation, code preservation, caching

**E2E Tests** (Playwright, generated by test-generator subagent):
- Navigation: Sidebar navigation, search, prev/next chapter (5 scenarios)
- Chatbot: Send query, verify response, click citation, text selection (5 scenarios)
- Auth: Signup flow, signin flow, session persistence, signout (3 scenarios)
- Personalization: Variant selection, content changes, cache hit (3 scenarios)
- Translation: Urdu display, code preservation, cache hit (3 scenarios)
- Smoke: Homepage, modules, chatbot opens (1 scenario)

**Performance Tests** (Lighthouse CI):
- Run on: Homepage, Module 1 Chapter 1, Module 2 Chapter 1, Module 3 Chapter 1
- Validate: Performance >90, Accessibility >95, Best Practices >90, SEO >90
- Fail build if any score below threshold (SC-005 compliance)

**Manual Tests**:
- Demo video rehearsal (ensure <90 seconds, all features shown)
- Cross-browser testing (Chrome, Firefox, Safari)
- Mobile responsiveness (iPhone 12, iPad, Android tablet)

---

### Success Criteria Validation Checklist

| SC | Description | Validation Method | Automated? |
|----|-------------|-------------------|------------|
| SC-001 | Page load <3s | Lighthouse Performance score, Chrome DevTools Network tab | ✅ Yes (Lighthouse CI) |
| SC-002 | 32-40 complete chapters | Chapter count script, manual review of structure | ⚠️ Partial (script counts, human reviews) |
| SC-003 | Chatbot <2s response | API responseTime field, E2E test assertion | ✅ Yes (E2E test) |
| SC-004 | 95% citation accuracy | 50-query test set, citation validation script | ✅ Yes (test set in tests/e2e/fixtures/) |
| SC-005 | Lighthouse scores >thresholds | Lighthouse CI | ✅ Yes |
| SC-006 | Signup <1 minute | E2E test with timer | ✅ Yes (E2E test) |
| SC-007 | Personalization <1s | E2E test with performance.now() | ✅ Yes (E2E test) |
| SC-008 | Translation <3s (first), <0.5s (cached) | E2E test with timer | ✅ Yes (E2E test) |
| SC-009 | 100% code validation | code-validator subagent output, CI/CD gate | ✅ Yes (CI/CD) |
| SC-010 | Deployed by 5:00 PM Nov 30 | Manual verification (URL accessible, smoke tests pass) | ⚠️ Partial (smoke tests automated) |
| SC-011 | Demo video <90s | Manual video creation and upload, manual duration check | ❌ No (manual) |
| SC-012 | Git history quality | Git log analysis script (conventional commits, <500 lines, co-author) | ✅ Yes (script in scripts/check-success-criteria.sh) |

**Overall Automation**: 9/12 SCs fully or partially automated (75% automated validation)

---

### Performance Benchmarking Approach

**Baseline Metrics** (establish during Phase 1):
- Empty Docusaurus site: Lighthouse Performance score (target: >95)
- FastAPI health check: Response time (target: <50ms)

**Phase Benchmarks**:
- Phase 2 (Content): Lighthouse Performance with 10 chapters (target: >92, allows 2-point degradation)
- Phase 3 (RAG Backend): /api/chat response time with 100 chunks (target: <1.5s, 500ms buffer)
- Phase 4 (Chatbot UI): End-to-end query time including UI render (target: <2.5s total)
- Phase 5 (Auth): Signup flow time (target: <45s, 15s buffer)
- Phase 6 (Personalization): Variant switch time (target: <0.8s, 200ms buffer)
- Phase 6 (Translation): Translation time (first: <2.5s, cached: <0.3s, buffers)

**Regression Detection**: If any metric degrades >10% between phases, investigate and optimize before proceeding.

---

## Post-Plan Actions

After this plan is approved:

1. **Create research.md** (Phase 0):
   - Research and document all 10 architectural decisions
   - Create ADRs in history/adr/ for each decision
   - Resolve all unknowns from Technical Context

2. **Create data-model.md** (Phase 1):
   - Document entity schemas (Module, Chapter, User, Session, ContentChunk, ChatMessage)
   - Include validation rules, relationships, storage details

3. **Create API contracts** (Phase 1):
   - Generate contracts/chat-api.yaml (OpenAPI spec for RAG endpoints)
   - Generate contracts/auth-api.yaml (OpenAPI spec for auth endpoints)
   - Generate contracts/translation-api.yaml (OpenAPI spec for translation endpoint)

4. **Create quickstart.md** (Phase 1):
   - Step-by-step setup guide for local development
   - Deployment instructions for GitHub Pages + Fly.io

5. **Update agent context** (Phase 1):
   - Run .specify/scripts/bash/update-agent-context.sh claude
   - Verify new technologies added to .specify/context/claude.md

6. **Run /sp.tasks** (Phase 2):
   - Generate detailed task breakdown in tasks.md
   - Include test cases, acceptance criteria, dependencies for each task

7. **Begin implementation** (Phases 1-7):
   - Execute tasks in dependency order (Foundation → Content → RAG → Chatbot → Auth → Personalization/Translation → Testing)
   - Track progress via tasks.md
   - Create PHRs for significant work

---

**Plan Status**: ✅ COMPLETE - Ready for Phase 0 (Research)

**Next Command**: Proceed with research.md creation to resolve architectural decisions, then /sp.tasks for task generation.
