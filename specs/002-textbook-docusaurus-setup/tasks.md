# Task Breakdown: Physical AI & Humanoid Robotics Textbook

**Feature**: 002-textbook-docusaurus-setup
**Generated**: 2025-11-29
**Source**: [spec.md](./spec.md) | [plan.md](./plan.md)

---

## Atomic Task Definition

**Task Size**: 15-30 minutes each
**Acceptance Criteria**: Single, testable condition per task
**Format**: `- [ ] T### [Story] Description → AC: <condition> (Spec: REF)`

**Story Tags**:
- **[SETUP]**: Foundation infrastructure
- **[US1]**: User Story 1 - Read Textbook Content (P1 MVP)
- **[US2]**: User Story 2 - RAG Chatbot (P2 Base)
- **[US3]**: User Story 3 - Authentication (P3 Bonus)
- **[US4]**: User Story 4 - Personalization (P3 Bonus)
- **[US5]**: User Story 5 - Translation (P3 Bonus)
- **[TEST]**: Testing & Quality Assurance
- **[DEPLOY]**: Deployment & Delivery

---

## Phase 0: Foundation Setup (Days 1-2)

**Objective**: Establish project infrastructure
**Checkpoint**: T014 - Infrastructure ready for content development

### Repository Initialization (30 min)

- [ ] T001 [SETUP] Create monorepo directory structure
  - **AC**: Directories exist: `docs/`, `backend/`, `tests/e2e/`, `scripts/`, `.github/workflows/`
  - **Spec**: Plan.md Project Structure section
  - **Test**: `ls -la` shows all directories

- [ ] T002 [SETUP] Create .gitignore file
  - **AC**: .gitignore contains: `node_modules/`, `__pycache__/`, `.env`, `build/`, `dist/`, `.DS_Store`
  - **Spec**: Plan.md line 393
  - **Test**: Verify patterns match common exclusions

- [ ] T003 [SETUP] Create .env.example template
  - **AC**: File contains placeholders: `OPENAI_API_KEY=`, `QDRANT_URL=`, `QDRANT_API_KEY=`, `NEON_DATABASE_URL=`, `JWT_SECRET=`
  - **Spec**: Plan.md line 394
  - **Test**: All 5 required env vars present

- [ ] T004 [SETUP] Create root README.md skeleton
  - **AC**: README has sections: Project Overview, Features, Tech Stack, Setup, Deployment, License
  - **Spec**: SC-010, Plan.md lines 395
  - **Test**: All 6 sections present with placeholders

### Docusaurus Frontend Setup (60 min)

- [ ] T005 [SETUP] Initialize Docusaurus 3.x project
  - **AC**: `npx create-docusaurus@latest docs classic --typescript` completes successfully
  - **Spec**: FR-001, Plan.md line 16
  - **Test**: `docs/` directory contains `docusaurus.config.js`, `package.json`, `src/`
  - **Depends**: T001

- [ ] T006 [SETUP] Configure Docusaurus site metadata
  - **AC**: `docusaurus.config.js` has `title: "Physical AI & Humanoid Robotics"`, `url`, `baseUrl`
  - **Spec**: FR-001
  - **Test**: Metadata fields populated in config
  - **Depends**: T005

- [ ] T007 [SETUP] Install Docusaurus dependencies
  - **AC**: `package.json` includes `@docusaurus/preset-classic`, `prism-react-renderer`, `mermaid@^10`
  - **Spec**: Plan.md line 21
  - **Test**: `npm install` in `docs/` completes without errors
  - **Depends**: T006

- [ ] T008 [SETUP] Verify Docusaurus local dev server
  - **AC**: `npm start` in `docs/` launches site at `http://localhost:3000` showing default homepage
  - **Spec**: SC-001
  - **Test**: Navigate to localhost:3000, see "Dinosaur" placeholder
  - **Depends**: T007

### FastAPI Backend Setup (45 min)

- [ ] T009 [SETUP] Create backend directory structure
  - **AC**: Directories exist: `backend/app/`, `backend/app/api/`, `backend/app/services/`, `backend/app/models/`, `backend/app/db/`, `backend/scripts/`, `backend/tests/`
  - **Spec**: Plan.md lines 276-318
  - **Test**: `ls -la backend/app/` shows subdirectories
  - **Depends**: T001

- [ ] T010 [SETUP] Create FastAPI main.py with health endpoint
  - **AC**: `backend/app/main.py` contains FastAPI app with `GET /health` returning `{"status": "ok"}`
  - **Spec**: Plan.md line 278
  - **Test**: Code contains `@app.get("/health")` decorator
  - **Depends**: T009

- [ ] T011 [SETUP] Add CORS middleware to FastAPI app
  - **AC**: `main.py` includes `CORSMiddleware` with `allow_origins=["http://localhost:3000"]`
  - **Spec**: Plan.md line 278
  - **Test**: CORS configured for local Docusaurus dev server
  - **Depends**: T010

- [ ] T012 [SETUP] Create backend requirements.txt
  - **AC**: File contains: `fastapi==0.104.1`, `uvicorn[standard]==0.24.0`, `python-dotenv==1.0.0`
  - **Spec**: Plan.md line 22
  - **Test**: All 3 base dependencies listed with versions
  - **Depends**: T009

- [ ] T013 [SETUP] Verify FastAPI local dev server
  - **AC**: `uvicorn app.main:app --reload` launches API at `http://localhost:8000`, `curl localhost:8000/health` returns 200
  - **Spec**: Plan.md line 278
  - **Test**: Health endpoint responds successfully
  - **Depends**: T010, T011, T012

### Database Setup (60 min)

- [ ] T014 [SETUP] Create Neon Serverless Postgres project
  - **AC**: Neon project created, connection string obtained and added to `.env` as `NEON_DATABASE_URL`
  - **Spec**: Plan.md line 27, FR-022
  - **Test**: Connection string format: `postgres://user:pass@host/db`
  - **Depends**: T003

- [ ] T015 [SETUP] Create backend/app/db/postgres_client.py
  - **AC**: File contains SQLAlchemy engine creation using `NEON_DATABASE_URL` from env
  - **Spec**: Plan.md line 301
  - **Test**: Import succeeds, engine object created
  - **Depends**: T012, T014

- [ ] T016 [SETUP] Test Neon Postgres connection
  - **AC**: `backend/scripts/test_db_connection.py` connects to Neon, executes `SELECT 1`, prints success
  - **Spec**: Plan.md line 301
  - **Test**: Script runs without errors, outputs "Connected to Neon"
  - **Depends**: T015

- [ ] T017 [SETUP] Create Qdrant Cloud free tier account
  - **AC**: Qdrant Cloud account created, cluster provisioned, API key and URL obtained
  - **Spec**: FR-021, Plan.md line 26
  - **Test**: Can access Qdrant Cloud dashboard
  - **Depends**: T003

- [ ] T018 [SETUP] Add Qdrant credentials to .env
  - **AC**: `.env` contains `QDRANT_URL=<cluster-url>` and `QDRANT_API_KEY=<api-key>`
  - **Spec**: Plan.md line 26
  - **Test**: Both variables populated in .env
  - **Depends**: T017

- [ ] T019 [SETUP] Create backend/app/db/qdrant_client.py
  - **AC**: File contains Qdrant client initialization using credentials from env
  - **Spec**: Plan.md line 301
  - **Test**: Import succeeds, client object created
  - **Depends**: T012, T018

- [ ] T020 [SETUP] Test Qdrant Cloud connection
  - **AC**: `backend/scripts/test_qdrant_connection.py` connects to Qdrant, lists collections, prints success
  - **Spec**: Plan.md line 301
  - **Test**: Script runs without errors, outputs "Connected to Qdrant"
  - **Depends**: T019

### CI/CD Pipeline Setup (90 min)

- [ ] T021 [SETUP] Create GitHub repository
  - **AC**: Repository created on GitHub, local repo initialized with `git init`, first commit made
  - **Spec**: SC-012
  - **Test**: `git remote -v` shows GitHub URL
  - **Depends**: T004

- [ ] T022 [SETUP] Create GitHub Actions workflow for frontend deploy
  - **AC**: `.github/workflows/deploy-frontend.yml` exists with jobs: install, build, deploy to GitHub Pages
  - **Spec**: FR-048, Plan.md lines 344-345
  - **Test**: File contains `name: Deploy Frontend`, `runs-on: ubuntu-latest`
  - **Depends**: T021

- [ ] T023 [SETUP] Configure GitHub Pages settings
  - **AC**: GitHub Pages enabled for repository, source set to `gh-pages` branch
  - **Spec**: FR-010, SC-010
  - **Test**: Pages URL shown in repository settings
  - **Depends**: T022

- [ ] T024 [SETUP] Create backend Dockerfile
  - **AC**: `backend/Dockerfile` exists with multi-stage build: `FROM python:3.11-slim`, `COPY requirements.txt`, `RUN pip install`, `COPY app/`
  - **Spec**: Plan.md line 319
  - **Test**: Dockerfile contains all 4 stages
  - **Depends**: T012

- [ ] T025 [SETUP] Create GitHub Actions workflow for backend deploy
  - **AC**: `.github/workflows/deploy-backend.yml` exists with jobs: build Docker image, push to Fly.io
  - **Spec**: Plan.md line 345
  - **Test**: File contains `docker build`, `flyctl deploy`
  - **Depends**: T024

- [ ] T026 [SETUP] Setup Fly.io account and app
  - **AC**: Fly.io account created, app provisioned, `FLY_API_TOKEN` added to GitHub secrets
  - **Spec**: Plan.md line 39 (ADR-003)
  - **Test**: `fly status` shows app running
  - **Depends**: T025

**🔍 CHECKPOINT 1**: Infrastructure Complete (T001-T026)
**Validation**:
- ✅ Docusaurus dev server runs at localhost:3000
- ✅ FastAPI dev server runs at localhost:8000/health
- ✅ Neon Postgres connection succeeds
- ✅ Qdrant Cloud connection succeeds
- ✅ GitHub repo initialized with CI/CD workflows

---

## Phase 1: User Story 1 - Read Textbook Content (Days 3-5)

**User Story**: Students access the online textbook to learn Physical AI and Humanoid Robotics
**Success Criteria**: SC-001 (page load <3s), SC-002 (32-40 chapters), SC-009 (100% code validation)
**Checkpoint**: T055 - Module 1 content complete and validated

### Docusaurus Content Architecture (60 min)

- [ ] T027 [US1] Create docs/docs/intro.md welcome page
  - **AC**: File contains: Course overview, prerequisites (ROS 2 basics), learning path (4 modules)
  - **Spec**: FR-011, Plan.md line 253
  - **Test**: File exists with 3 sections (200+ words)
  - **Depends**: T008

- [ ] T028 [US1] Create module1-ros2/_category_.json
  - **AC**: File contains `{"label": "Module 1: ROS 2 Fundamentals", "position": 1}`
  - **Spec**: FR-002, FR-004
  - **Test**: JSON valid, label and position set
  - **Depends**: T027

- [ ] T029 [US1] Create module2-gazebo-unity/_category_.json
  - **AC**: File contains `{"label": "Module 2: Simulation", "position": 2}`
  - **Spec**: FR-002, FR-004
  - **Test**: JSON valid, label and position set
  - **Depends**: T027

- [ ] T030 [US1] Create module3-nvidia-isaac/_category_.json
  - **AC**: File contains `{"label": "Module 3: NVIDIA Isaac", "position": 3}`
  - **Spec**: FR-002, FR-004
  - **Test**: JSON valid, label and position set
  - **Depends**: T027

- [ ] T031 [US1] Create module4-vla/_category_.json
  - **AC**: File contains `{"label": "Module 4: Vision-Language-Action", "position": 4}`
  - **Spec**: FR-002, FR-004
  - **Test**: JSON valid, label and position set
  - **Depends**: T027

- [ ] T032 [US1] Configure docs/sidebars.js for 4 modules
  - **AC**: sidebars.js exports sidebar with 4 module categories, each containing chapter placeholders
  - **Spec**: FR-004, Plan.md line 271
  - **Test**: Sidebar renders in dev server showing 4 modules
  - **Depends**: T028, T029, T030, T031

### Custom React Components (120 min)

- [ ] T033 [US1] Create CodeBlock component skeleton
  - **AC**: `docs/src/components/Content/CodeBlock.tsx` exists with TypeScript interface for props: `{code: string, language: string}`
  - **Spec**: FR-007, Plan.md line 240
  - **Test**: Component imports without errors
  - **Depends**: T008

- [ ] T034 [US1] Add Prism syntax highlighting to CodeBlock
  - **AC**: CodeBlock uses `prism-react-renderer` to highlight code based on `language` prop
  - **Spec**: FR-007
  - **Test**: Render `<CodeBlock code="print('test')" language="python" />`, see highlighted syntax
  - **Depends**: T033

- [ ] T035 [US1] Add copy button to CodeBlock
  - **AC**: CodeBlock has copy-to-clipboard button, clicking copies code to clipboard
  - **Spec**: FR-007
  - **Test**: Click button, paste into editor, code matches original
  - **Depends**: T034

- [ ] T036 [US1] Create MermaidDiagram component skeleton
  - **AC**: `docs/src/components/Content/MermaidDiagram.tsx` exists with props: `{chart: string}`
  - **Spec**: FR-008, Plan.md line 241
  - **Test**: Component imports without errors
  - **Depends**: T008

- [ ] T037 [US1] Integrate Mermaid library into component
  - **AC**: MermaidDiagram uses `mermaid` library to render chart from `chart` prop
  - **Spec**: FR-008
  - **Test**: Render `<MermaidDiagram chart="graph TD; A-->B" />`, see flowchart
  - **Depends**: T036

- [ ] T038 [US1] Add error handling to MermaidDiagram
  - **AC**: Invalid Mermaid syntax displays error message instead of crashing
  - **Spec**: FR-008
  - **Test**: Render invalid syntax, see "Invalid diagram syntax" message
  - **Depends**: T037

- [ ] T039 [US1] Create ExerciseBlock component
  - **AC**: `docs/src/components/Content/ExerciseBlock.tsx` exists with props: `{difficulty: 'beginner'|'intermediate'|'advanced', content: string}`
  - **Spec**: FR-015, Plan.md line 242
  - **Test**: Component renders with colored badge (green/yellow/red) based on difficulty
  - **Depends**: T008

### Module 1 Content Generation (480 min = 8 hours)

**Note**: Each chapter generation is 60 min (use chapter-generator subagent)

- [ ] T040 [US1] Generate Module 1 Chapter 1: Introduction to ROS 2 - skeleton
  - **AC**: `docs/docs/module1-ros2/chapter1-introduction.md` created with frontmatter: `title`, `sidebar_position: 1`
  - **Spec**: FR-003, FR-011
  - **Test**: File exists with valid frontmatter
  - **Depends**: T032

- [ ] T041 [US1] Add Learning Objectives to Chapter 1
  - **AC**: Chapter 1 contains section "## Learning Objectives" with 3-6 bullet points
  - **Spec**: FR-012, Assumption #15
  - **Test**: Section has 3-6 objectives (e.g., "Understand ROS 2 architecture")
  - **Depends**: T040

- [ ] T042 [US1] Add Introduction section to Chapter 1
  - **AC**: Chapter 1 contains "## Introduction" with 200-300 words explaining ROS 2 context
  - **Spec**: FR-011
  - **Test**: Section exists with 200+ words
  - **Depends**: T041

- [ ] T043 [US1] Add Core Concepts section to Chapter 1
  - **AC**: Chapter 1 contains "## Core Concepts" with 800-1000 words covering: Nodes, Topics, DDS
  - **Spec**: FR-011, FR-017
  - **Test**: Section covers 3 core concepts with definitions
  - **Depends**: T042

- [ ] T044 [US1] Add first code example to Chapter 1
  - **AC**: Core Concepts section includes Python code block (20-30 lines) for "Hello World" ROS 2 node with comments
  - **Spec**: FR-013, Assumption #15 (min 3 code blocks)
  - **Test**: Code block exists with `rclpy.init()`, comments on key lines
  - **Depends**: T043

- [ ] T045 [US1] Add first diagram to Chapter 1
  - **AC**: Core Concepts section includes Mermaid diagram showing ROS 2 node communication
  - **Spec**: FR-016, Assumption #15 (min 2 diagrams)
  - **Test**: Diagram renders showing nodes, topics, messages
  - **Depends**: T044

- [ ] T046 [US1] Add Hands-On Lab section to Chapter 1
  - **AC**: Chapter 1 contains "## Hands-On Lab" with 5-10 step-by-step instructions for creating a publisher node
  - **Spec**: FR-014, Assumption #15
  - **Test**: Lab has 5+ numbered steps with expected outputs
  - **Depends**: T045

- [ ] T047 [US1] Add second code example to Chapter 1 (Lab code)
  - **AC**: Hands-On Lab includes complete Python publisher code (30-40 lines) with comments
  - **Spec**: FR-013, FR-014
  - **Test**: Code is complete, runnable, includes expected output
  - **Depends**: T046

- [ ] T048 [US1] Add Exercises section to Chapter 1
  - **AC**: Chapter 1 contains "## Exercises" with 3 exercises: 1 beginner, 1 intermediate, 1 advanced
  - **Spec**: FR-015, Assumption #15
  - **Test**: Each exercise has difficulty label, clear task, success criteria
  - **Depends**: T047

- [ ] T049 [US1] Add Summary section to Chapter 1
  - **AC**: Chapter 1 contains "## Summary" with 100-150 word recap of key points
  - **Spec**: FR-011
  - **Test**: Summary exists with 100+ words
  - **Depends**: T048

- [ ] T050 [US1] Add Further Reading section to Chapter 1
  - **AC**: Chapter 1 contains "## Further Reading" with 3-5 external links (ROS 2 docs, tutorials)
  - **Spec**: FR-011
  - **Test**: Section has 3+ valid URLs
  - **Depends**: T049

- [ ] T051 [US1] Add second diagram to Chapter 1
  - **AC**: Summary section includes Mermaid diagram showing ROS 2 architecture overview
  - **Spec**: FR-016, Assumption #15 (min 2 diagrams)
  - **Test**: Diagram exists and renders
  - **Depends**: T050

- [ ] T052 [US1] Add third code example to Chapter 1
  - **AC**: Exercises section includes example solution for beginner exercise (15-20 lines)
  - **Spec**: Assumption #15 (min 3 code blocks)
  - **Test**: Code block exists with working solution
  - **Depends**: T051

**Repeat pattern T040-T052 for Chapters 2-8** (consolidated for brevity)

- [ ] T053 [US1] Generate Module 1 Chapter 2: ROS 2 Nodes and Topics (complete structure)
  - **AC**: Chapter 2 follows same structure as Chapter 1 (Learning Objectives → Further Reading), 2000-3000 words
  - **Spec**: FR-003, FR-011, Assumption #15
  - **Test**: All 8 sections present, 2000+ words, 3+ code blocks, 2+ diagrams
  - **Depends**: T052

- [ ] T054 [US1] Generate Module 1 Chapter 3: Publishers and Subscribers (complete structure)
  - **AC**: Chapter 3 complete with all sections, 2000-3000 words
  - **Spec**: Same as T053
  - **Depends**: T053

- [ ] T055 [US1] Generate Module 1 Chapter 4: Services and Actions (complete structure)
  - **AC**: Chapter 4 complete with all sections, 2000-3000 words
  - **Spec**: Same as T053
  - **Depends**: T054

- [ ] T056 [US1] Generate Module 1 Chapter 5: Parameters and Launch Files (complete structure)
  - **AC**: Chapter 5 complete with all sections, 2000-3000 words
  - **Spec**: Same as T053
  - **Depends**: T055

- [ ] T057 [US1] Generate Module 1 Chapter 6: tf2 and Coordinate Frames (complete structure)
  - **AC**: Chapter 6 complete with all sections, 2000-3000 words
  - **Spec**: Same as T053
  - **Depends**: T056

- [ ] T058 [US1] Generate Module 1 Chapter 7: ROS 2 Bags and Logging (complete structure)
  - **AC**: Chapter 7 complete with all sections, 2000-3000 words
  - **Spec**: Same as T053
  - **Depends**: T057

- [ ] T059 [US1] Generate Module 1 Chapter 8: Simulation with Gazebo (complete structure)
  - **AC**: Chapter 8 complete with all sections, 2000-3000 words
  - **Spec**: Same as T053
  - **Depends**: T058

### Code Validation (60 min)

- [ ] T060 [US1] Create code validation script
  - **AC**: `backend/scripts/validate_code.py` extracts all code blocks from Module 1 chapters, validates syntax
  - **Spec**: SC-009, Plan.md line 310
  - **Test**: Script runs, finds all code blocks
  - **Depends**: T059

- [ ] T061 [US1] Validate Python code examples
  - **AC**: Script uses `ast.parse()` to validate Python syntax, reports errors if any
  - **Spec**: SC-009
  - **Test**: All Python code blocks parse without errors
  - **Depends**: T060

- [ ] T062 [US1] Validate Bash code examples
  - **AC**: Script uses `shellcheck` to validate Bash scripts, reports errors if any
  - **Spec**: SC-009
  - **Test**: All Bash code blocks pass shellcheck
  - **Depends**: T060

- [ ] T063 [US1] Validate XML/YAML code examples
  - **AC**: Script validates XML/YAML syntax, checks ROS 2 launch file structure
  - **Spec**: SC-009
  - **Test**: All XML/YAML code blocks parse without errors
  - **Depends**: T060

- [ ] T064 [US1] Achieve 100% code validation pass rate
  - **AC**: `validate_code.py` reports 100% pass rate for all Module 1 code examples
  - **Spec**: SC-009
  - **Test**: Script output: "All code examples validated successfully"
  - **Depends**: T061, T062, T063

### Navigation Features (45 min)

- [ ] T065 [US1] Swizzle Docusaurus DocItem Footer component
  - **AC**: `docs/src/theme/DocItem/Footer/index.tsx` created via `npm run swizzle`
  - **Spec**: FR-006, Plan.md line 244
  - **Test**: Component file exists, default Docusaurus footer renders
  - **Depends**: T032

- [ ] T066 [US1] Add Previous Chapter button to footer
  - **AC**: Footer component includes "← Previous Chapter" button using Docusaurus metadata
  - **Spec**: FR-006
  - **Test**: Button appears on Chapter 2+, links to previous chapter
  - **Depends**: T065

- [ ] T067 [US1] Add Next Chapter button to footer
  - **AC**: Footer component includes "Next Chapter →" button using Docusaurus metadata
  - **Spec**: FR-006
  - **Test**: Button appears on Chapter 1-7, links to next chapter
  - **Depends**: T066

- [ ] T068 [US1] Configure Docusaurus search plugin
  - **AC**: `docusaurus.config.js` includes `@docusaurus/plugin-content-docs` with search enabled
  - **Spec**: FR-005, Plan.md line 274
  - **Test**: Search bar appears in navbar
  - **Depends**: T032

- [ ] T069 [US1] Test search functionality
  - **AC**: Search for "ROS 2 node", results include Chapter 1 and Chapter 2
  - **Spec**: FR-005
  - **Test**: Search returns relevant results from multiple chapters
  - **Depends**: T068

**🔍 CHECKPOINT 2**: Module 1 Content Complete (T027-T069)
**Validation**:
- ✅ 8 chapters in Module 1, each 2000-3000 words
- ✅ All chapters have 3+ code blocks, 2+ diagrams, 3 exercises
- ✅ 100% code validation pass rate (SC-009)
- ✅ Navigation (prev/next, search) works

---

## Phase 2: User Story 2 - RAG Chatbot (Days 6-10)

**User Story**: Students ask questions via AI chatbot with citations
**Success Criteria**: SC-003 (<2s response), SC-004 (95% citation accuracy)
**Checkpoint**: T127 - Chatbot functional with 95% citation accuracy

### Database Schema (90 min)

- [ ] T070 [US2] Install SQLAlchemy and psycopg2
  - **AC**: `backend/requirements.txt` includes `sqlalchemy==2.0.23`, `psycopg2-binary==2.9.9`
  - **Spec**: Plan.md line 22
  - **Test**: `pip install -r requirements.txt` succeeds
  - **Depends**: T016

- [ ] T071 [US2] Create User model
  - **AC**: `backend/app/models/user.py` defines User table: `id` (UUID), `email` (unique), `password_hash`, `software_background` (1-5), `hardware_background` (1-5), `created_at`, `updated_at`
  - **Spec**: FR-028, FR-029, Plan.md lines 576-600
  - **Test**: Model imports without errors, has 7 columns
  - **Depends**: T070

- [ ] T072 [US2] Create ContentChunk model
  - **AC**: `backend/app/models/chunk.py` defines ContentChunk table: `id` (UUID), `chapter_id`, `module_name`, `chapter_title`, `section_heading`, `content` (text), `content_type` (enum), `heading_level`, `token_count`, `qdrant_point_id`, `created_at`
  - **Spec**: FR-021, Plan.md lines 650-700
  - **Test**: Model has 11 columns, content_type enum defined
  - **Depends**: T070

- [ ] T073 [US2] Create ChatMessage model
  - **AC**: `backend/app/models/message.py` defines ChatMessage table: `id` (UUID), `conversation_id` (UUID), `user_id` (nullable FK), `role` (enum: user/assistant), `content` (text), `citations` (JSON), `response_time_ms` (int), `created_at`
  - **Spec**: FR-026, Plan.md lines 810-850
  - **Test**: Model has 8 columns, role enum, citations JSON type
  - **Depends**: T070

- [ ] T074 [US2] Create Session model
  - **AC**: `backend/app/models/session.py` defines Session table: `id` (UUID), `user_id` (FK), `token` (unique), `expires_at`, `created_at`, `last_accessed_at`
  - **Spec**: Assumption #16, Plan.md lines 613-646
  - **Test**: Model has 6 columns, user_id foreign key
  - **Depends**: T071

- [ ] T075 [US2] Install Alembic for migrations
  - **AC**: `backend/requirements.txt` includes `alembic==1.13.0`, `alembic init` executed
  - **Spec**: Plan.md line 23
  - **Test**: `backend/alembic/` directory exists with env.py
  - **Depends**: T070

- [ ] T076 [US2] Create initial migration for all tables
  - **AC**: `alembic revision --autogenerate -m "Initial schema"` creates migration with User, ContentChunk, ChatMessage, Session tables
  - **Spec**: Plan.md line 23
  - **Test**: Migration file contains CREATE TABLE for 4 tables
  - **Depends**: T071, T072, T073, T074, T075

- [ ] T077 [US2] Apply migration to Neon database
  - **AC**: `alembic upgrade head` executes successfully against Neon
  - **Spec**: FR-022
  - **Test**: Query Neon: `\dt` shows 4 tables
  - **Depends**: T076

### Qdrant Collection Setup (30 min)

- [ ] T078 [US2] Install qdrant-client
  - **AC**: `backend/requirements.txt` includes `qdrant-client==1.7.0`
  - **Spec**: Plan.md line 22
  - **Test**: `pip install qdrant-client` succeeds
  - **Depends**: T020

- [ ] T079 [US2] Create Qdrant collection creation script
  - **AC**: `backend/scripts/create_qdrant_collection.py` creates `textbook_chunks` collection with 1536 dimensions, Cosine distance
  - **Spec**: FR-021, Plan.md line 655
  - **Test**: Script runs, creates collection
  - **Depends**: T078

- [ ] T080 [US2] Execute collection creation
  - **AC**: Run script, verify collection exists in Qdrant Cloud dashboard
  - **Spec**: FR-021
  - **Test**: `client.get_collection('textbook_chunks')` succeeds
  - **Depends**: T079

### Semantic Chunking Implementation (120 min)

- [ ] T081 [US2] Install tiktoken for token counting
  - **AC**: `backend/requirements.txt` includes `tiktoken==0.5.2`
  - **Spec**: FR-021 (max 1000 tokens)
  - **Test**: `import tiktoken` succeeds
  - **Depends**: T070

- [ ] T082 [US2] Create ChunkingService class skeleton
  - **AC**: `backend/app/services/chunking_service.py` defines `ChunkingService` class with method `chunk_chapter(markdown: str, chapter_id: str) -> List[ContentChunk]`
  - **Spec**: FR-021, Plan.md line 289
  - **Test**: Class imports without errors
  - **Depends**: T072, T081

- [ ] T083 [US2] Implement markdown parsing for H2/H3 headings
  - **AC**: ChunkingService extracts all H2/H3 sections using regex `r'^#{2,3}\s+(.+)$'`
  - **Spec**: Assumption #13, Plan.md line 702
  - **Test**: Parse sample chapter, returns list of sections with headings
  - **Depends**: T082

- [ ] T084 [US2] Implement token counting for sections
  - **AC**: Each section's token count calculated using `tiktoken.encoding_for_model('gpt-4o-mini')`
  - **Spec**: FR-021 (max 1000 tokens)
  - **Test**: Count tokens for "Introduction to ROS 2" section, output integer
  - **Depends**: T083

- [ ] T085 [US2] Implement Priority 1 chunking (full section <1000 tokens)
  - **AC**: If section ≤1000 tokens, create single chunk with full content
  - **Spec**: Assumption #13 Priority 1, Plan.md line 702
  - **Test**: Section with 350 tokens → 1 chunk
  - **Depends**: T084

- [ ] T086 [US2] Implement Priority 2 chunking (split at H3 boundaries)
  - **AC**: If H2 section >1000 tokens, split at H3 boundaries
  - **Spec**: Assumption #13 Priority 2, Plan.md line 703
  - **Test**: H2 section with 850 tokens, 2 H3 subsections → 2 chunks
  - **Depends**: T085

- [ ] T087 [US2] Implement 100-token overlap between chunks
  - **AC**: Adjacent chunks share 100 tokens (last 100 of chunk N = first 100 of chunk N+1)
  - **Spec**: Assumption #13, Plan.md line 708
  - **Test**: Verify overlap in chunked output
  - **Depends**: T086

- [ ] T088 [US2] Implement code block preservation (special case)
  - **AC**: Code blocks >1000 tokens kept whole, not split
  - **Spec**: Assumption #13 special case, Plan.md line 706
  - **Test**: Chapter with 1200-token code block → code block intact in single chunk
  - **Depends**: T087

- [ ] T089 [US2] Test chunking with sample chapter from plan.md
  - **AC**: Chunk sample chapter from Plan.md lines 716-746, output matches expected 5 chunks
  - **Spec**: Plan.md Chunking Example
  - **Test**: Compare output to Plan.md lines 750-785
  - **Depends**: T088

### Embedding Service (60 min)

- [ ] T090 [US2] Install OpenAI Python SDK
  - **AC**: `backend/requirements.txt` includes `openai==1.6.0`
  - **Spec**: Plan.md line 22
  - **Test**: `import openai` succeeds
  - **Depends**: T070

- [ ] T091 [US2] Add OPENAI_API_KEY to .env
  - **AC**: `.env` contains `OPENAI_API_KEY=<your-key>`, key valid
  - **Spec**: Plan.md line 3
  - **Test**: Key authenticates with OpenAI API
  - **Depends**: T003

- [ ] T092 [US2] Create EmbeddingService class
  - **AC**: `backend/app/services/embedding_service.py` defines `EmbeddingService` with method `embed(text: str) -> List[float]`
  - **Spec**: FR-021, Plan.md line 290
  - **Test**: Class imports without errors
  - **Depends**: T090

- [ ] T093 [US2] Implement text-embedding-3-small API call
  - **AC**: `embed()` calls `openai.embeddings.create(model='text-embedding-3-small', input=text)`, returns 1536-dim vector
  - **Spec**: Assumption #7, Plan.md line 429
  - **Test**: Embed "test text", verify output length = 1536
  - **Depends**: T091, T092

- [ ] T094 [US2] Add rate limiting to embedding calls
  - **AC**: EmbeddingService limits to 20 requests/minute using `time.sleep()` if needed
  - **Spec**: Plan.md line 290
  - **Test**: Call embed() 25 times rapidly, verify delays after 20th call
  - **Depends**: T093

- [ ] T095 [US2] Add error handling for API failures
  - **AC**: `embed()` catches `openai.OpenAIError`, retries 3 times with exponential backoff
  - **Spec**: Plan.md line 290
  - **Test**: Mock API failure, verify 3 retries
  - **Depends**: T094

### RAG Indexing Script (90 min)

- [ ] T096 [US2] Create index_content.py script skeleton
  - **AC**: `backend/scripts/index_content.py` exists with `main()` function, CLI arg parsing for `--chapter-file`
  - **Spec**: Plan.md line 309
  - **Test**: `python index_content.py --help` shows usage
  - **Depends**: T089, T095

- [ ] T097 [US2] Implement chapter markdown loading
  - **AC**: Script reads markdown file, extracts frontmatter (chapter_id, module, title)
  - **Spec**: Plan.md line 309
  - **Test**: Load Chapter 1 markdown, output metadata
  - **Depends**: T096

- [ ] T098 [US2] Integrate ChunkingService into script
  - **AC**: Script calls `ChunkingService.chunk_chapter()`, outputs list of chunks
  - **Spec**: Plan.md line 309
  - **Test**: Process Chapter 1, output ~10-12 chunks
  - **Depends**: T097

- [ ] T099 [US2] Generate embeddings for each chunk
  - **AC**: Script calls `EmbeddingService.embed()` for each chunk's content, stores vectors
  - **Spec**: FR-021
  - **Test**: Process 1 chunk, output 1536-dim vector
  - **Depends**: T098

- [ ] T100 [US2] Store chunk vectors in Qdrant
  - **AC**: Script calls `qdrant_client.upsert()`, stores vectors with payload: {chunk_id, chapter_id, module_name, chapter_title, section_heading, content_type, token_count}
  - **Spec**: FR-021, Plan.md lines 684-694
  - **Test**: Upsert 1 chunk, verify in Qdrant dashboard
  - **Depends**: T099

- [ ] T101 [US2] Store chunk metadata in Neon
  - **AC**: Script inserts ContentChunk records into Neon with qdrant_point_id reference
  - **Spec**: FR-021, Plan.md line 680
  - **Test**: Insert 1 chunk, query Neon to verify
  - **Depends**: T100

- [ ] T102 [US2] Index all Module 1 chapters
  - **AC**: Run `index_content.py` for all 8 Module 1 chapters, ~80-120 chunks created
  - **Spec**: Plan.md line 1337
  - **Test**: Query Qdrant: `client.count('textbook_chunks')` returns 80-120
  - **Depends**: T101

### RAG Service Implementation (150 min)

- [ ] T103 [US2] Create RAGService class skeleton
  - **AC**: `backend/app/services/rag_service.py` defines `RAGService` with method `query(question: str, conversation_id: Optional[str]) -> dict`
  - **Spec**: Plan.md line 288
  - **Test**: Class imports without errors
  - **Depends**: T095, T102

- [ ] T104 [US2] Implement query embedding
  - **AC**: `query()` embeds user question using EmbeddingService
  - **Spec**: FR-023
  - **Test**: Query "What is a ROS 2 node?", output 1536-dim vector
  - **Depends**: T103

- [ ] T105 [US2] Implement vector search (top 5)
  - **AC**: `query()` searches Qdrant for top 5 most similar chunks using cosine distance
  - **Spec**: FR-023
  - **Test**: Search returns 5 chunks with scores
  - **Depends**: T104

- [ ] T106 [US2] Fetch chunk metadata from Neon
  - **AC**: For each Qdrant result, query Neon for full ContentChunk record
  - **Spec**: FR-023
  - **Test**: Top 5 chunks include chapter_title, section_heading
  - **Depends**: T105

- [ ] T107 [US2] Define RAG system prompt
  - **AC**: Create `RAG_SYSTEM_PROMPT` constant with text from Plan.md lines 1344-1364
  - **Spec**: Plan.md Phase 3 RAG Prompt Template
  - **Test**: Prompt includes instructions for citations in [^N] format
  - **Depends**: T103

- [ ] T108 [US2] Implement RAG user prompt formatting
  - **AC**: Format user prompt using template from Plan.md lines 1368-1384: Context from chunks + User question
  - **Spec**: Plan.md lines 1368-1384
  - **Test**: Format prompt with 5 chunks, verify structure
  - **Depends**: T106, T107

- [ ] T109 [US2] Implement GPT-4o-mini API call
  - **AC**: Call `openai.chat.completions.create(model='gpt-4o-mini', messages=[system_prompt, user_prompt])`
  - **Spec**: ADR-005, Plan.md line 431
  - **Test**: Send formatted prompt, receive LLM response
  - **Depends**: T108

- [ ] T110 [US2] Implement citation extraction with regex
  - **AC**: Extract citations using code from Plan.md lines 1404-1442: Parse "Citations:" section, extract [^N] references
  - **Spec**: FR-024, Plan.md lines 1404-1442
  - **Test**: Parse sample LLM response, output citation objects
  - **Depends**: T109

- [ ] T111 [US2] Implement citation URL generation
  - **AC**: Map citations to Docusaurus URLs: `/docs/{module}/{chapter}#{slugified-section}`
  - **Spec**: FR-024, Plan.md lines 1435-1446
  - **Test**: Citation for "What is ROS 2?" → URL `/docs/module1-ros2/chapter1-introduction#what-is-ros-2`
  - **Depends**: T110

- [ ] T112 [US2] Implement conversation context loading
  - **AC**: If `conversation_id` provided, load previous 5 messages from ChatMessage table
  - **Spec**: FR-026
  - **Test**: Query with conversation_id, verify previous context included
  - **Depends**: T109

- [ ] T113 [US2] Implement response time tracking
  - **AC**: `query()` measures elapsed time from start to LLM response, includes in return dict
  - **Spec**: SC-003 (<2s)
  - **Test**: Query returns `responseTime` field in milliseconds
  - **Depends**: T111

### Chat API Endpoint (60 min)

- [ ] T114 [US2] Create POST /api/chat endpoint skeleton
  - **AC**: `backend/app/api/chat.py` defines `/api/chat` route with request schema: {query, conversationId?, context?}
  - **Spec**: FR-020, Plan.md lines 876-902
  - **Test**: `curl -X POST localhost:8000/api/chat -d '{}'` returns 400 (validation error)
  - **Depends**: T113

- [ ] T115 [US2] Implement request validation
  - **AC**: Endpoint validates: `query` is non-empty string ≤500 chars, `conversationId` is valid UUID if provided
  - **Spec**: Plan.md line 892
  - **Test**: Send empty query, receive 400 error
  - **Depends**: T114

- [ ] T116 [US2] Integrate RAGService into endpoint
  - **AC**: Endpoint calls `RAGService.query()`, returns response
  - **Spec**: FR-020
  - **Test**: POST valid query, receive answer with citations
  - **Depends**: T115

- [ ] T117 [US2] Save chat messages to database
  - **AC**: Endpoint saves user message and assistant response to ChatMessage table
  - **Spec**: FR-026
  - **Test**: POST query, verify 2 rows in ChatMessage (user + assistant)
  - **Depends**: T116

- [ ] T118 [US2] Generate conversation ID if not provided
  - **AC**: If no `conversationId` in request, generate UUID and return in response
  - **Spec**: FR-026
  - **Test**: POST without conversationId, response includes new UUID
  - **Depends**: T117

- [ ] T119 [US2] Validate response time <2s
  - **AC**: Endpoint response includes `responseTime` field, value <2000ms
  - **Spec**: SC-003
  - **Test**: POST query, verify `responseTime` < 2000
  - **Depends**: T118

### Frontend Chatbot UI (180 min)

- [ ] T120 [US2] Create ChatbotButton component skeleton
  - **AC**: `docs/src/components/Chatbot/ChatbotButton.tsx` exports button component with chat icon
  - **Spec**: FR-018, Plan.md line 221
  - **Test**: Import component, renders without errors
  - **Depends**: T008

- [ ] T121 [US2] Style ChatbotButton (floating bottom-right)
  - **AC**: Button positioned `position: fixed; bottom: 20px; right: 20px; z-index: 1000`
  - **Spec**: Plan.md line 221
  - **Test**: Button visible in bottom-right corner
  - **Depends**: T120

- [ ] T122 [US2] Create ChatPanel component skeleton
  - **AC**: `docs/src/components/Chatbot/ChatPanel.tsx` exports panel component, hidden by default
  - **Spec**: FR-018, Plan.md line 222
  - **Test**: Import component, renders when `isOpen={true}`
  - **Depends**: T008

- [ ] T123 [US2] Implement ChatPanel slide-in animation
  - **AC**: Panel slides in from right when opened, slides out when closed (CSS transition)
  - **Spec**: Plan.md line 222
  - **Test**: Toggle `isOpen` prop, verify animation
  - **Depends**: T122

- [ ] T124 [US2] Add ChatPanel layout (header, messages, input)
  - **AC**: Panel contains: header with "AI Tutor" title + close button, message list area, input field + send button
  - **Spec**: Plan.md line 222
  - **Test**: Panel renders with 3 sections
  - **Depends**: T123

- [ ] T125 [US2] Create MessageList component
  - **AC**: `docs/src/components/Chatbot/MessageList.tsx` displays array of messages with scroll
  - **Spec**: Plan.md line 223
  - **Test**: Render 10 test messages, scrollable area works
  - **Depends**: T124

- [ ] T126 [US2] Style user messages (right-aligned, blue)
  - **AC**: User messages have `background: #007bff; color: white; margin-left: auto`
  - **Spec**: Plan.md line 223
  - **Test**: User message appears right-aligned with blue background
  - **Depends**: T125

- [ ] T127 [US2] Style assistant messages (left-aligned, gray)
  - **AC**: Assistant messages have `background: #f1f1f1; color: black; margin-right: auto`
  - **Spec**: Plan.md line 223
  - **Test**: Assistant message appears left-aligned with gray background
  - **Depends**: T126

- [ ] T128 [US2] Create CitationLink component
  - **AC**: `docs/src/components/Chatbot/CitationLink.tsx` renders clickable link: `[Chapter - Section]`
  - **Spec**: FR-024, Plan.md line 224
  - **Test**: Render citation, displays formatted text
  - **Depends**: T008

- [ ] T129 [US2] Implement CitationLink navigation
  - **AC**: Clicking citation navigates to chapter URL using `window.location.href`, scrolls to section anchor
  - **Spec**: FR-024
  - **Test**: Click citation, verify navigation and scroll
  - **Depends**: T128

- [ ] T130 [US2] Close ChatPanel after citation click
  - **AC**: CitationLink `onClick` closes ChatPanel before navigation
  - **Spec**: FR-024 (UX improvement)
  - **Test**: Click citation, panel closes before page navigation
  - **Depends**: T129

- [ ] T131 [US2] Create useChatAPI hook skeleton
  - **AC**: `docs/src/components/Chatbot/useChatAPI.ts` exports hook with state: `messages`, `loading`, `error`
  - **Spec**: Plan.md line 226
  - **Test**: Call hook in test component, state initialized
  - **Depends**: T008

- [ ] T132 [US2] Implement sendQuery function in hook
  - **AC**: Hook provides `sendQuery(query: string)` function that POSTs to `/api/chat`
  - **Spec**: Plan.md line 226
  - **Test**: Call sendQuery(), verify fetch() called
  - **Depends**: T131

- [ ] T133 [US2] Implement conversation ID persistence in sessionStorage
  - **AC**: Hook stores `conversationId` in sessionStorage, reuses for subsequent queries
  - **Spec**: FR-026, Plan.md line 226
  - **Test**: Send 2 queries, verify same conversationId used
  - **Depends**: T132

- [ ] T134 [US2] Integrate useChatAPI into ChatPanel
  - **AC**: ChatPanel uses hook: input field calls `sendQuery()`, message list shows `messages` array
  - **Spec**: FR-018
  - **Test**: Type query, press send, message appears in list
  - **Depends**: T125, T133

- [ ] T135 [US2] Swizzle Docusaurus DocItem wrapper
  - **AC**: `docs/src/theme/DocItemWrapper.tsx` created via swizzle, wraps default DocItem
  - **Spec**: Plan.md line 244
  - **Test**: Component file exists, pages render normally
  - **Depends**: T032

- [ ] T136 [US2] Add ChatbotButton to all doc pages
  - **AC**: DocItemWrapper includes `<ChatbotButton />`, visible on all chapter pages
  - **Spec**: FR-018
  - **Test**: Navigate to any chapter, button present
  - **Depends**: T121, T135

- [ ] T137 [US2] Connect ChatbotButton to ChatPanel
  - **AC**: Clicking ChatbotButton toggles ChatPanel visibility
  - **Spec**: FR-018
  - **Test**: Click button, panel opens; click close, panel closes
  - **Depends**: T134, T136

### Text Selection Feature (60 min)

- [ ] T138 [US2] Create TextSelectionMenu component skeleton
  - **AC**: `docs/src/components/Chatbot/TextSelectionMenu.tsx` exports component that listens for `mouseup` events
  - **Spec**: FR-025, Plan.md line 225
  - **Test**: Component imports without errors
  - **Depends**: T008

- [ ] T139 [US2] Detect text selection
  - **AC**: Component detects when user selects text using `window.getSelection()`
  - **Spec**: FR-025
  - **Test**: Select text on page, component state updates
  - **Depends**: T138

- [ ] T140 [US2] Show "Ask about this" button near selection
  - **AC**: When text selected, button appears near selection using absolute positioning
  - **Spec**: FR-025
  - **Test**: Select text, button appears next to selection
  - **Depends**: T139

- [ ] T141 [US2] Open ChatPanel with selection as context
  - **AC**: Clicking button opens ChatPanel, adds selected text to input field as context
  - **Spec**: FR-025
  - **Test**: Select "ROS 2 nodes", click button, input shows "ROS 2 nodes"
  - **Depends**: T137, T140

- [ ] T142 [US2] Add TextSelectionMenu to DocItemWrapper
  - **AC**: DocItemWrapper includes `<TextSelectionMenu />`, active on all chapter pages
  - **Spec**: FR-025
  - **Test**: Select text on any chapter, "Ask about this" button appears
  - **Depends**: T141

### RAG Quality Validation (90 min)

- [ ] T143 [US2] Create 50-query test set JSON file
  - **AC**: `tests/e2e/fixtures/test-queries.json` contains 50 queries: 20 beginner, 20 intermediate, 10 advanced
  - **Spec**: SC-004, Plan.md lines 209-211
  - **Test**: File valid JSON, 50 entries with {query, expectedChapter, difficulty}
  - **Depends**: T064

- [ ] T144 [US2] Write 20 beginner queries
  - **AC**: Test set includes queries like "What is a ROS 2 node?", "What is DDS?"
  - **Spec**: SC-004
  - **Test**: All 20 queries reference Module 1 concepts
  - **Depends**: T143

- [ ] T145 [US2] Write 20 intermediate queries
  - **AC**: Test set includes queries like "How do I create a publisher-subscriber pair?"
  - **Spec**: SC-004
  - **Test**: All 20 queries require multi-step explanations
  - **Depends**: T143

- [ ] T146 [US2] Write 10 advanced queries
  - **AC**: Test set includes queries like "How does VSLAM differ from traditional SLAM?"
  - **Spec**: SC-004
  - **Test**: All 10 queries require advanced knowledge
  - **Depends**: T143

- [ ] T147 [US2] Create citation validation script
  - **AC**: `backend/scripts/validate_citations.py` reads test set, sends queries to `/api/chat`, validates citations
  - **Spec**: SC-004
  - **Test**: Script runs, processes queries
  - **Depends**: T119, T146

- [ ] T148 [US2] Implement citation URL validation
  - **AC**: Script checks each citation URL: must resolve (not 404), must match `expectedChapter` from test set
  - **Spec**: SC-004
  - **Test**: Script detects invalid URL, reports error
  - **Depends**: T147

- [ ] T149 [US2] Run validation on all 50 queries
  - **AC**: Script processes all 50 queries, reports pass/fail count
  - **Spec**: SC-004
  - **Test**: Script output shows "X/50 queries passed"
  - **Depends**: T148

- [ ] T150 [US2] Achieve 95% citation accuracy
  - **AC**: Validation script reports ≥48/50 queries passed (95%)
  - **Spec**: SC-004
  - **Test**: Script output: "48/50 passed (96%)"
  - **Depends**: T149

**🔍 CHECKPOINT 3**: RAG Chatbot Complete (T070-T150)
**Validation**:
- ✅ Chatbot responds within 2s (SC-003)
- ✅ 95% citation accuracy (SC-004)
- ✅ Text selection feature works
- ✅ Multi-turn conversation context maintained

---

## Phase 3: User Story 3 - Authentication (Days 11-12)

**User Story**: Students create accounts and sign in
**Success Criteria**: SC-006 (signup <1 minute)
**Checkpoint**: T174 - Auth flows complete

### Backend Auth Service (120 min)

- [ ] T151 [US3] Install Better-Auth and bcrypt
  - **AC**: `backend/requirements.txt` includes `better-auth==0.3.0`, `bcrypt==4.1.1`
  - **Spec**: FR-028, Plan.md line 22
  - **Test**: `pip install` succeeds
  - **Depends**: T074

- [ ] T152 [US3] Add JWT_SECRET to .env
  - **AC**: `.env` contains `JWT_SECRET=<random-256-bit-hex>`
  - **Spec**: Assumption #16
  - **Test**: Secret is 64 hex characters
  - **Depends**: T003

- [ ] T153 [US3] Create AuthService class skeleton
  - **AC**: `backend/app/services/auth_service.py` defines `AuthService` with methods: `signup()`, `signin()`, `signout()`, `validate_session()`
  - **Spec**: Plan.md line 292
  - **Test**: Class imports without errors
  - **Depends**: T074, T151

- [ ] T154 [US3] Implement password hashing with bcrypt
  - **AC**: `signup()` hashes password using `bcrypt.hashpw()` with 10 rounds
  - **Spec**: FR-030, Assumption #16
  - **Test**: Hash "password123", verify bcrypt format
  - **Depends**: T153

- [ ] T155 [US3] Implement signup logic
  - **AC**: `signup(email, password, sw_bg, hw_bg)` creates User record, returns user dict
  - **Spec**: FR-028, FR-029
  - **Test**: Call signup(), verify User created in Neon
  - **Depends**: T154

- [ ] T156 [US3] Implement JWT token generation
  - **AC**: `signup()` generates JWT with payload: {user_id, exp: 7 days}, signed with JWT_SECRET
  - **Spec**: Assumption #16
  - **Test**: Decode JWT, verify payload
  - **Depends**: T155

- [ ] T157 [US3] Create Session record on signup
  - **AC**: `signup()` creates Session record with JWT token, expires_at = now + 7 days
  - **Spec**: Assumption #16
  - **Test**: Signup, verify Session in Neon
  - **Depends**: T156

- [ ] T158 [US3] Implement signin logic
  - **AC**: `signin(email, password)` verifies password with `bcrypt.checkpw()`, generates JWT, creates Session
  - **Spec**: FR-028
  - **Test**: Signin with valid credentials, receive JWT
  - **Depends**: T157

- [ ] T159 [US3] Implement session validation
  - **AC**: `validate_session(token)` verifies JWT signature, checks Session.expires_at, returns User or None
  - **Spec**: Assumption #16
  - **Test**: Validate valid token → User, validate expired token → None
  - **Depends**: T158

- [ ] T160 [US3] Implement auto-refresh logic
  - **AC**: `validate_session()` extends expires_at by 7 days if last_accessed_at > 5 days old
  - **Spec**: Assumption #16
  - **Test**: Mock session aged 6 days, validate → expires_at updated
  - **Depends**: T159

### Auth API Endpoints (90 min)

- [ ] T161 [US3] Create POST /api/signup endpoint skeleton
  - **AC**: `backend/app/api/auth.py` defines `/api/signup` route with request schema: {email, password, softwareBackground, hardwareBackground}
  - **Spec**: FR-028, Plan.md lines 1022-1061
  - **Test**: POST without body → 422 validation error
  - **Depends**: T160

- [ ] T162 [US3] Implement email validation
  - **AC**: Endpoint validates email format using regex `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
  - **Spec**: FR-030
  - **Test**: POST invalid email → 400 error
  - **Depends**: T161

- [ ] T163 [US3] Implement password strength validation
  - **AC**: Endpoint validates password ≥8 characters
  - **Spec**: FR-030
  - **Test**: POST password "test" → 400 error
  - **Depends**: T162

- [ ] T164 [US3] Implement background score validation
  - **AC**: Endpoint validates softwareBackground and hardwareBackground are integers 1-5
  - **Spec**: FR-029
  - **Test**: POST softwareBackground=6 → 400 error
  - **Depends**: T163

- [ ] T165 [US3] Integrate AuthService.signup()
  - **AC**: Endpoint calls `AuthService.signup()`, returns user data
  - **Spec**: FR-028
  - **Test**: POST valid data → 201, user created
  - **Depends**: T164

- [ ] T166 [US3] Set HTTP-only cookie in signup response
  - **AC**: Endpoint sets `Set-Cookie` header: `session_token=<jwt>; HttpOnly; Secure; SameSite=Strict; Max-Age=604800`
  - **Spec**: FR-031, Assumption #16
  - **Test**: Response headers include Set-Cookie with HttpOnly flag
  - **Depends**: T165

- [ ] T167 [US3] Handle duplicate email error
  - **AC**: If email exists, return 409 Conflict with message "Email already registered"
  - **Spec**: FR-028
  - **Test**: Signup twice with same email → 409 on second attempt
  - **Depends**: T166

- [ ] T168 [US3] Create POST /api/signin endpoint
  - **AC**: `/api/signin` validates credentials, calls `AuthService.signin()`, sets cookie, returns user data
  - **Spec**: FR-028, Plan.md lines 1088-1133
  - **Test**: POST valid credentials → 200 with cookie
  - **Depends**: T160

- [ ] T169 [US3] Handle invalid credentials in signin
  - **AC**: If password incorrect, return 401 Unauthorized
  - **Spec**: FR-028
  - **Test**: POST wrong password → 401
  - **Depends**: T168

- [ ] T170 [US3] Create POST /api/signout endpoint
  - **AC**: `/api/signout` deletes Session from Neon, sets cookie Max-Age=0
  - **Spec**: FR-034, Plan.md lines 1135-1150
  - **Test**: POST with valid cookie → Session deleted, cookie cleared
  - **Depends**: T168

- [ ] T171 [US3] Create GET /api/profile endpoint
  - **AC**: `/api/profile` requires session_token cookie, calls `validate_session()`, returns user data or 401
  - **Spec**: FR-033, Plan.md lines 1152-1178
  - **Test**: GET with valid cookie → user data, GET without cookie → 401
  - **Depends**: T160

### Frontend Auth UI (120 min)

- [ ] T172 [US3] Create useAuth hook skeleton
  - **AC**: `docs/src/components/Auth/useAuth.ts` exports context provider with state: `user`, `loading`, `error`
  - **Spec**: Plan.md line 232
  - **Test**: Wrap app in provider, access context
  - **Depends**: T008

- [ ] T173 [US3] Implement signUp function in hook
  - **AC**: Hook provides `signUp(email, password, swBg, hwBg)` that POSTs to `/api/signup`
  - **Spec**: Plan.md line 232
  - **Test**: Call signUp(), verify fetch() to /api/signup
  - **Depends**: T172

- [ ] T174 [US3] Implement signIn function in hook
  - **AC**: Hook provides `signIn(email, password)` that POSTs to `/api/signin`
  - **Spec**: Plan.md line 232
  - **Test**: Call signIn(), verify fetch() to /api/signin
  - **Depends**: T173

- [ ] T175 [US3] Implement signOut function in hook
  - **AC**: Hook provides `signOut()` that POSTs to `/api/signout`, clears user state
  - **Spec**: Plan.md line 232
  - **Test**: Call signOut(), verify user set to null
  - **Depends**: T174

- [ ] T176 [US3] Implement session persistence check
  - **AC**: Hook calls `/api/profile` on mount, sets user if valid session exists
  - **Spec**: FR-032
  - **Test**: Reload page with valid cookie, user state restored
  - **Depends**: T171, T175

- [ ] T177 [US3] Create SignUpForm component skeleton
  - **AC**: `docs/src/components/Auth/SignUpForm.tsx` exports form with fields: email, password, software background slider (1-5), hardware background slider (1-5)
  - **Spec**: FR-029, Plan.md line 229
  - **Test**: Component renders with 4 input fields
  - **Depends**: T008

- [ ] T178 [US3] Implement form validation (client-side)
  - **AC**: Form validates email format, password ≥8 chars before submit
  - **Spec**: FR-030
  - **Test**: Submit with invalid email → error message shown
  - **Depends**: T177

- [ ] T179 [US3] Integrate useAuth.signUp() into form
  - **AC**: Form submit calls `signUp()`, redirects to homepage on success
  - **Spec**: FR-028
  - **Test**: Fill form, submit → API called, redirect happens
  - **Depends**: T173, T178

- [ ] T180 [US3] Create SignInForm component
  - **AC**: `docs/src/components/Auth/SignInForm.tsx` exports form with email, password fields
  - **Spec**: Plan.md line 230
  - **Test**: Component renders with 2 fields
  - **Depends**: T008

- [ ] T181 [US3] Integrate useAuth.signIn() into form
  - **AC**: Form submit calls `signIn()`, redirects to homepage on success
  - **Spec**: FR-028
  - **Test**: Fill form, submit → API called, redirect happens
  - **Depends**: T174, T180

- [ ] T182 [US3] Create ProfileIndicator component
  - **AC**: `docs/src/components/Auth/ProfileIndicator.tsx` displays user email + sign out button
  - **Spec**: FR-033, Plan.md line 231
  - **Test**: Pass user prop, component shows email
  - **Depends**: T008

- [ ] T183 [US3] Integrate useAuth into ProfileIndicator
  - **AC**: ProfileIndicator uses `useAuth()` context, shows email if authenticated
  - **Spec**: FR-033
  - **Test**: Sign in, indicator shows email
  - **Depends**: T176, T182

- [ ] T184 [US3] Implement sign out button
  - **AC**: Clicking "Sign Out" calls `signOut()`, redirects to homepage
  - **Spec**: FR-034
  - **Test**: Click button, user signed out, redirect happens
  - **Depends**: T175, T183

- [ ] T185 [US3] Create signup page
  - **AC**: `docs/src/pages/signup.tsx` renders SignUpForm with "Sign Up" title
  - **Spec**: Plan.md line 250
  - **Test**: Navigate to /signup, see form
  - **Depends**: T179

- [ ] T186 [US3] Swizzle Docusaurus Navbar
  - **AC**: `docs/src/theme/Navbar/index.tsx` created via swizzle
  - **Spec**: Plan.md line 251
  - **Test**: Navbar renders normally
  - **Depends**: T032

- [ ] T187 [US3] Add ProfileIndicator to Navbar
  - **AC**: Navbar includes `<ProfileIndicator />` on right side
  - **Spec**: FR-033
  - **Test**: Sign in, see email in navbar
  - **Depends**: T183, T186

**🔍 CHECKPOINT 4**: Auth Complete (T151-T187)
**Validation**:
- ✅ Signup flow <1 minute (SC-006)
- ✅ Session persists across page reloads
- ✅ Sign out clears session

---

## Phase 4: User Story 4 - Personalization (Days 13-14)

**User Story**: Adjust content depth based on user background
**Success Criteria**: SC-007 (personalization <1s)
**Checkpoint**: T204 - Personalization functional

### Content Variant Generation (240 min = 4 hours)

**Note**: Generate variants for Module 1 Chapter 1 first (prototype), then batch Chapters 2-8

- [ ] T188 [US4] Generate beginner variant for Chapter 1
  - **AC**: `docs/docs/module1-ros2/chapter1-introduction-beginner.md` exists with more basic explanations, fewer advanced code patterns than default
  - **Spec**: FR-036, Assumption #14
  - **Test**: Compare beginner vs default, beginner has 20% more explanation text
  - **Depends**: T052

- [ ] T189 [US4] Generate advanced variant for Chapter 1
  - **AC**: `docs/docs/module1-ros2/chapter1-introduction-advanced.md` exists with more code examples, less basic explanation than default
  - **Spec**: FR-036, Assumption #14
  - **Test**: Compare advanced vs default, advanced has 30% more code
  - **Depends**: T052

- [ ] T190 [US4] Generate beginner variants for Chapters 2-8
  - **AC**: All 7 remaining chapters have `-beginner.md` variants
  - **Spec**: Assumption #14
  - **Test**: Verify 7 files exist
  - **Depends**: T188

- [ ] T191 [US4] Generate advanced variants for Chapters 2-8
  - **AC**: All 7 remaining chapters have `-advanced.md` variants
  - **Spec**: Assumption #14
  - **Test**: Verify 7 files exist
  - **Depends**: T189

### Frontend Personalization Logic (90 min)

- [ ] T192 [US4] Create usePersonalization hook skeleton
  - **AC**: `docs/src/components/Personalization/usePersonalization.ts` exports hook with function `getVariant(user: User) -> 'beginner'|'default'|'advanced'`
  - **Spec**: Plan.md line 237
  - **Test**: Hook imports without errors
  - **Depends**: T176

- [ ] T193 [US4] Implement variant calculation logic
  - **AC**: `getVariant()` calculates: sum = softwareBackground + hardwareBackground; if sum ≤3 → beginner, if sum 4-7 → default, if sum ≥8 → advanced
  - **Spec**: Assumption #14, Plan.md line 263
  - **Test**: User (sw=2, hw=2) → beginner, User (sw=3, hw=4) → default, User (sw=4, hw=5) → advanced
  - **Depends**: T192

- [ ] T194 [US4] Implement localStorage caching for variant
  - **AC**: Hook stores selected variant in localStorage: `chapter-{chapterId}-variant`
  - **Spec**: Assumption #14, Plan.md line 265
  - **Test**: Get variant, check localStorage, key exists
  - **Depends**: T193

- [ ] T195 [US4] Create PersonalizeButton component skeleton
  - **AC**: `docs/src/components/Personalization/PersonalizeButton.tsx` exports button "Personalize Content"
  - **Spec**: FR-035, Plan.md line 237
  - **Test**: Component renders button
  - **Depends**: T008

- [ ] T196 [US4] Integrate usePersonalization into button
  - **AC**: Button onClick calls `getVariant(user)`, loads variant markdown file
  - **Spec**: FR-035
  - **Test**: Click button, console logs variant type
  - **Depends**: T194, T195

- [ ] T197 [US4] Implement variant content loading
  - **AC**: Button loads variant markdown (e.g., `chapter1-introduction-beginner.md`), replaces chapter content
  - **Spec**: FR-035
  - **Test**: Click button, content changes to variant
  - **Depends**: T196

- [ ] T198 [US4] Implement content replacement in <1s
  - **AC**: Variant content loads and renders in <1000ms (use `performance.now()`)
  - **Spec**: SC-007
  - **Test**: Click button, measure time, verify <1s
  - **Depends**: T197

- [ ] T199 [US4] Add "Show Original" button
  - **AC**: After personalization, "Show Original" button appears, reloads default content
  - **Spec**: FR-038
  - **Test**: Personalize, click "Show Original", default content restored
  - **Depends**: T198

- [ ] T200 [US4] Add personalization indicator
  - **AC**: When personalized, indicator shows "Content personalized for your level"
  - **Spec**: FR-037
  - **Test**: Personalize, see indicator message
  - **Depends**: T199

- [ ] T201 [US4] Swizzle DocItem Layout component
  - **AC**: `docs/src/theme/DocItem/Layout/index.tsx` created via swizzle
  - **Spec**: Plan.md line 244
  - **Test**: Component renders chapter normally
  - **Depends**: T135

- [ ] T202 [US4] Add PersonalizeButton to chapter top
  - **AC**: DocItem Layout includes `<PersonalizeButton />` at top of content (only for authenticated users)
  - **Spec**: FR-035
  - **Test**: Sign in, navigate to chapter, see button
  - **Depends**: T200, T201

- [ ] T203 [US4] Hide button for unauthenticated users
  - **AC**: PersonalizeButton only renders if `user !== null`
  - **Spec**: FR-035 (bonus feature requires auth)
  - **Test**: Sign out, button disappears
  - **Depends**: T202

**🔍 CHECKPOINT 5**: Personalization Complete (T188-T203)
**Validation**:
- ✅ Variant selection <1s (SC-007)
- ✅ Content adapts based on user profile
- ✅ Cache hit loads instantly

---

## Phase 5: User Story 5 - Translation (Days 13-14)

**User Story**: Translate content to Urdu while preserving code
**Success Criteria**: SC-008 (translation <3s first, <0.5s cached)
**Checkpoint**: T222 - Translation functional

### Backend Translation Service (120 min)

- [ ] T204 [US5] Create TranslationService class skeleton
  - **AC**: `backend/app/services/translation_service.py` defines `TranslationService` with method `translate(content: str, target_lang: str) -> str`
  - **Spec**: Plan.md line 291
  - **Test**: Class imports without errors
  - **Depends**: T090

- [ ] T205 [US5] Define translation prompt template
  - **AC**: Create `TRANSLATION_PROMPT` constant: "Translate the following text to {target_lang}, but keep all content within triple backticks (```) unchanged"
  - **Spec**: FR-042, Plan.md line 291
  - **Test**: Prompt template includes code preservation instruction
  - **Depends**: T204

- [ ] T206 [US5] Implement code block extraction with regex
  - **AC**: `translate()` extracts code blocks using regex `r'```[\s\S]*?```'`, stores in list
  - **Spec**: FR-042
  - **Test**: Extract from sample markdown, returns list of code blocks
  - **Depends**: T205

- [ ] T207 [US5] Replace code blocks with placeholders
  - **AC**: Replace code blocks with `{{CODE_BLOCK_N}}` placeholders before translation
  - **Spec**: FR-042
  - **Test**: Markdown with 2 code blocks → text with `{{CODE_BLOCK_0}}`, `{{CODE_BLOCK_1}}`
  - **Depends**: T206

- [ ] T208 [US5] Implement OpenAI translation API call
  - **AC**: Call `openai.chat.completions.create()` with translation prompt + placeholder text
  - **Spec**: FR-041, Assumption #3
  - **Test**: Translate sample text to Urdu, receive Urdu response
  - **Depends**: T207

- [ ] T209 [US5] Re-insert original code blocks
  - **AC**: Replace `{{CODE_BLOCK_N}}` placeholders with original code blocks
  - **Spec**: FR-042
  - **Test**: Translated text + code blocks → full markdown with code intact
  - **Depends**: T208

- [ ] T210 [US5] Validate code preservation
  - **AC**: Unit test: Translate markdown with code blocks, assert code blocks identical to original
  - **Spec**: FR-042
  - **Test**: Code blocks match byte-for-byte
  - **Depends**: T209

- [ ] T211 [US5] Create translations table migration
  - **AC**: `backend/alembic/versions/002_add_translations.py` creates `translations` table: {id, chapter_id, language, translated_content, created_at}
  - **Spec**: Plan.md line 29
  - **Test**: Migration file exists with CREATE TABLE
  - **Depends**: T077

- [ ] T212 [US5] Apply translations table migration
  - **AC**: `alembic upgrade head` creates translations table in Neon
  - **Spec**: Plan.md line 29
  - **Test**: Query Neon: `\dt` shows translations table
  - **Depends**: T211

### Translation API Endpoint (60 min)

- [ ] T213 [US5] Create POST /api/translate endpoint skeleton
  - **AC**: `backend/app/api/translate.py` defines `/api/translate` route with request schema: {chapterId, content, targetLang}
  - **Spec**: FR-041, Plan.md lines 1191-1218
  - **Test**: POST without body → 422 validation error
  - **Depends**: T210

- [ ] T214 [US5] Implement target language validation
  - **AC**: Endpoint validates `targetLang` is "urdu" (only supported language)
  - **Spec**: FR-041
  - **Test**: POST targetLang="spanish" → 400 error
  - **Depends**: T213

- [ ] T215 [US5] Implement translation cache lookup
  - **AC**: Endpoint queries translations table for (chapter_id, language) match
  - **Spec**: ADR-008, Plan.md line 29
  - **Test**: Query cached translation, cache hit → instant return
  - **Depends**: T212, T214

- [ ] T216 [US5] Integrate TranslationService on cache miss
  - **AC**: If cache miss, endpoint calls `TranslationService.translate()`, stores result in translations table
  - **Spec**: FR-041
  - **Test**: Translate new chapter, verify DB insert
  - **Depends**: T215

- [ ] T217 [US5] Return cached flag in response
  - **AC**: Response includes `{translatedContent: string, cached: boolean}`
  - **Spec**: Plan.md line 1230
  - **Test**: Cache hit → cached=true, cache miss → cached=false
  - **Depends**: T216

- [ ] T218 [US5] Validate first request <3s
  - **AC**: Cache miss translation completes in <3000ms
  - **Spec**: SC-008
  - **Test**: Measure time, verify <3s
  - **Depends**: T217

- [ ] T219 [US5] Validate cached request <0.5s
  - **AC**: Cache hit returns in <500ms
  - **Spec**: SC-008
  - **Test**: Measure time, verify <0.5s
  - **Depends**: T217

### Frontend Translation UI (90 min)

- [ ] T220 [US5] Create useTranslation hook skeleton
  - **AC**: `docs/src/components/Translation/useTranslation.ts` exports hook with function `translateChapter(chapterId: string, content: string) -> Promise<string>`
  - **Spec**: Plan.md line 238
  - **Test**: Hook imports without errors
  - **Depends**: T008

- [ ] T221 [US5] Implement API call in hook
  - **AC**: `translateChapter()` POSTs to `/api/translate`, returns translated content
  - **Spec**: FR-041
  - **Test**: Call function, verify fetch() to /api/translate
  - **Depends**: T220

- [ ] T222 [US5] Implement localStorage caching
  - **AC**: Hook stores translated content in localStorage: `translation-{chapterId}-urdu`
  - **Spec**: ADR-008, Plan.md line 265
  - **Test**: Translate, check localStorage, key exists
  - **Depends**: T221

- [ ] T223 [US5] Check localStorage before API call
  - **AC**: Hook checks localStorage first, only calls API if cache miss
  - **Spec**: ADR-008
  - **Test**: Translate twice, second call uses localStorage (no API call)
  - **Depends**: T222

- [ ] T224 [US5] Create TranslateButton component skeleton
  - **AC**: `docs/src/components/Translation/TranslateButton.tsx` exports button "Translate to Urdu"
  - **Spec**: FR-040, Plan.md line 238
  - **Test**: Component renders button
  - **Depends**: T008

- [ ] T225 [US5] Integrate useTranslation into button
  - **AC**: Button onClick calls `translateChapter()`, shows loading indicator during API call
  - **Spec**: FR-040
  - **Test**: Click button, see loading spinner
  - **Depends**: T223, T224

- [ ] T226 [US5] Replace chapter content with Urdu translation
  - **AC**: Button loads translated content, replaces chapter DOM
  - **Spec**: FR-040
  - **Test**: Click button, content changes to Urdu
  - **Depends**: T225

- [ ] T227 [US5] Verify code blocks remain in English
  - **AC**: After translation, code blocks still display English code
  - **Spec**: FR-042
  - **Test**: Visual inspection: Urdu text, English code
  - **Depends**: T226

- [ ] T228 [US5] Verify Urdu Unicode rendering
  - **AC**: Urdu characters display correctly (not mojibake)
  - **Spec**: FR-044
  - **Test**: Visual inspection: Urdu text readable
  - **Depends**: T226

- [ ] T229 [US5] Add "Show Original" button
  - **AC**: After translation, "Show Original" button appears, reloads English content
  - **Spec**: FR-045
  - **Test**: Translate, click "Show Original", English content restored
  - **Depends**: T228

- [ ] T230 [US5] Add TranslateButton to DocItem Layout
  - **AC**: DocItem Layout includes `<TranslateButton />` at top of content (only for authenticated users)
  - **Spec**: FR-040
  - **Test**: Sign in, navigate to chapter, see button
  - **Depends**: T229, T202

- [ ] T231 [US5] Hide button for unauthenticated users
  - **AC**: TranslateButton only renders if `user !== null`
  - **Spec**: FR-040 (bonus feature requires auth)
  - **Test**: Sign out, button disappears
  - **Depends**: T230

**🔍 CHECKPOINT 6**: Translation Complete (T204-T231)
**Validation**:
- ✅ First request <3s (SC-008)
- ✅ Cached request <0.5s (SC-008)
- ✅ Code blocks preserved in English
- ✅ Urdu text renders correctly

---

## Phase 6: Testing, Optimization & Delivery (Days 16-18)

**Objective**: E2E testing, Lighthouse optimization, demo video
**Success Criteria**: SC-005 (Lighthouse), SC-010 (deployed), SC-011 (demo), SC-012 (git)
**Checkpoint**: T270 - Ready for submission

### Playwright E2E Testing (180 min)

- [ ] T232 [TEST] Install Playwright
  - **AC**: `package.json` (root) includes `@playwright/test@^1.40`, `npx playwright install` executed
  - **Spec**: Plan.md line 33
  - **Test**: Playwright browsers installed
  - **Depends**: T001

- [ ] T233 [TEST] Create Playwright config
  - **AC**: `tests/e2e/playwright.config.ts` configures browsers: Chromium, Firefox, WebKit; baseURL: `http://localhost:3000`
  - **Spec**: Plan.md line 331
  - **Test**: Config file valid TypeScript
  - **Depends**: T232

- [ ] T234 [TEST] Create smoke test
  - **AC**: `tests/e2e/tests/smoke.spec.ts` tests: Homepage loads, modules visible, chatbot button present
  - **Spec**: Plan.md line 338
  - **Test**: `npx playwright test smoke` passes
  - **Depends**: T137, T233

- [ ] T235 [TEST] Test sidebar navigation
  - **AC**: `tests/e2e/tests/navigation.spec.ts` tests clicking sidebar links navigates to chapters
  - **Spec**: Plan.md line 333
  - **Test**: Test passes
  - **Depends**: T069, T233

- [ ] T236 [TEST] Test search functionality
  - **AC**: navigation.spec.ts tests search for "ROS 2 node", clicks result, navigates to chapter
  - **Spec**: FR-005
  - **Test**: Test passes
  - **Depends**: T235

- [ ] T237 [TEST] Test prev/next navigation
  - **AC**: navigation.spec.ts tests clicking "Next Chapter" button navigates forward
  - **Spec**: FR-006
  - **Test**: Test passes
  - **Depends**: T236

- [ ] T238 [TEST] Test chatbot query
  - **AC**: `tests/e2e/tests/chatbot.spec.ts` tests: Open chatbot, type query, send, receive response with citation
  - **Spec**: FR-018, FR-024
  - **Test**: Test passes
  - **Depends**: T137, T233

- [ ] T239 [TEST] Test citation click
  - **AC**: chatbot.spec.ts tests clicking citation navigates to chapter, scrolls to section
  - **Spec**: FR-024
  - **Test**: Test passes
  - **Depends**: T238

- [ ] T240 [TEST] Test text selection query
  - **AC**: chatbot.spec.ts tests selecting text, clicking "Ask about this", chatbot opens with context
  - **Spec**: FR-025
  - **Test**: Test passes
  - **Depends**: T142, T233

- [ ] T241 [TEST] Test multi-turn conversation
  - **AC**: chatbot.spec.ts tests sending 2 queries, verifies conversationId persists
  - **Spec**: FR-026
  - **Test**: Test passes
  - **Depends**: T240

- [ ] T242 [TEST] Test signup flow
  - **AC**: `tests/e2e/tests/auth.spec.ts` tests: Navigate to /signup, fill form, submit, redirects to homepage, profile indicator shows email
  - **Spec**: FR-028, SC-006
  - **Test**: Test passes, measures time <1 minute
  - **Depends**: T187, T233

- [ ] T243 [TEST] Test signin flow
  - **AC**: auth.spec.ts tests signin with existing user, profile indicator appears
  - **Spec**: FR-028
  - **Test**: Test passes
  - **Depends**: T242

- [ ] T244 [TEST] Test session persistence
  - **AC**: auth.spec.ts tests: Sign in, reload page, user still authenticated
  - **Spec**: FR-032
  - **Test**: Test passes
  - **Depends**: T243

- [ ] T245 [TEST] Test personalization variant selection
  - **AC**: `tests/e2e/tests/personalization.spec.ts` tests: Sign in with beginner profile, click "Personalize", content changes
  - **Spec**: FR-035, SC-007
  - **Test**: Test passes, measures time <1s
  - **Depends**: T203, T233

- [ ] T246 [TEST] Test personalization cache hit
  - **AC**: personalization.spec.ts tests: Personalize, navigate away, return, personalized content loads instantly
  - **Spec**: SC-007
  - **Test**: Test passes, time <0.5s
  - **Depends**: T245

- [ ] T247 [TEST] Test translation to Urdu
  - **AC**: `tests/e2e/tests/translation.spec.ts` tests: Sign in, click "Translate to Urdu", Urdu text appears, code blocks in English
  - **Spec**: FR-040, FR-042, SC-008
  - **Test**: Test passes, first request <3s
  - **Depends**: T231, T233

- [ ] T248 [TEST] Test translation cache hit
  - **AC**: translation.spec.ts tests: Translate, navigate away, return, click translate again, loads <0.5s
  - **Spec**: SC-008
  - **Test**: Test passes, cached <0.5s
  - **Depends**: T247

### Lighthouse Performance Optimization (180 min)

- [ ] T249 [PERF] Install Lighthouse CI
  - **AC**: `package.json` includes `@lhci/cli@^0.13`, `lighthouserc.json` created with config from Plan.md lines 1730-1751
  - **Spec**: SC-005, Plan.md line 1720
  - **Test**: `npx lhci autorun` executes
  - **Depends**: T008

- [ ] T250 [PERF] Run baseline Lighthouse audit
  - **AC**: Audit homepage, Module 1 Chapter 1, record scores
  - **Spec**: SC-005
  - **Test**: Audit runs, outputs scores
  - **Depends**: T249

- [ ] T251 [PERF] Analyze JavaScript bundle size
  - **AC**: Run `npm run build`, check bundle size using `npx webpack-bundle-analyzer`
  - **Spec**: Plan.md line 1537
  - **Test**: Analyzer shows bundle breakdown
  - **Depends**: T250

- [ ] T252 [PERF] Implement code splitting for chatbot
  - **AC**: Lazy load ChatPanel using `React.lazy(() => import('./components/ChatPanel'))`
  - **Spec**: Plan.md line 1540
  - **Test**: Build, verify ChatPanel in separate chunk
  - **Depends**: T251

- [ ] T253 [PERF] Remove unused dependencies
  - **AC**: Run `npx depcheck`, uninstall unused packages
  - **Spec**: Plan.md line 1543
  - **Test**: depcheck reports 0 unused deps
  - **Depends**: T252

- [ ] T254 [PERF] Enable tree-shaking
  - **AC**: Add `"sideEffects": false` to docs/package.json
  - **Spec**: Plan.md line 1544
  - **Test**: Build, bundle size reduced
  - **Depends**: T253

- [ ] T255 [PERF] Verify JavaScript <300KB gzipped
  - **AC**: Total JavaScript bundle <300KB after gzip
  - **Spec**: Plan.md line 1545
  - **Test**: Build, check dist/ files, total <300KB
  - **Depends**: T254

- [ ] T256 [PERF] Optimize images to WebP
  - **AC**: Convert all PNG/JPG in static/img/ to WebP using `npx @squoosh/cli --webp auto`
  - **Spec**: Plan.md line 1551
  - **Test**: All images .webp format
  - **Depends**: T250

- [ ] T257 [PERF] Resize large images
  - **AC**: Resize hero images to max 1920px width, diagrams to max 800px width
  - **Spec**: Plan.md line 1555
  - **Test**: All images ≤specified dimensions
  - **Depends**: T256

- [ ] T258 [PERF] Add lazy loading to images
  - **AC**: Add `loading="lazy"` attribute to images below fold
  - **Spec**: Plan.md line 1556
  - **Test**: Images below fold have lazy attribute
  - **Depends**: T257

- [ ] T259 [PERF] Verify images <100KB each
  - **AC**: All images <100KB file size
  - **Spec**: Plan.md line 1557
  - **Test**: Check file sizes, all <100KB
  - **Depends**: T258

- [ ] T260 [PERF] Add ARIA labels to icon buttons
  - **AC**: ChatbotButton, close buttons have `aria-label` attributes
  - **Spec**: Plan.md line 1614
  - **Test**: Run `npx @axe-core/cli localhost:3000`, verify no missing labels
  - **Depends**: T250

- [ ] T261 [PERF] Add alt text to all images
  - **AC**: All `<img>` tags have `alt` attribute with descriptive text
  - **Spec**: Plan.md line 1619
  - **Test**: Axe audit passes, no missing alt text
  - **Depends**: T260

- [ ] T262 [PERF] Fix color contrast issues
  - **AC**: All text meets WCAG 2.1 AA contrast ratio (≥4.5:1)
  - **Spec**: Plan.md line 1625
  - **Test**: Axe audit passes, no contrast errors
  - **Depends**: T261

- [ ] T263 [PERF] Add meta description to docusaurus.config.js
  - **AC**: Config includes `metadata: [{name: 'description', content: 'Learn Physical AI and Humanoid Robotics with ROS 2'}]`
  - **Spec**: Plan.md line 1687
  - **Test**: Homepage HTML has meta description
  - **Depends**: T250

- [ ] T264 [PERF] Run final Lighthouse audit
  - **AC**: Audit homepage + 3 sample chapters (Module 1-3, Chapter 1 each)
  - **Spec**: SC-005
  - **Test**: All pages audited
  - **Depends**: T255, T259, T262, T263

- [ ] T265 [PERF] Verify Performance >90
  - **AC**: All 4 pages have Performance score >90
  - **Spec**: SC-005
  - **Test**: Audit results show Performance ≥90
  - **Depends**: T264

- [ ] T266 [PERF] Verify Accessibility >95
  - **AC**: All 4 pages have Accessibility score >95
  - **Spec**: SC-005
  - **Test**: Audit results show Accessibility ≥95
  - **Depends**: T264

- [ ] T267 [PERF] Verify Best Practices >90
  - **AC**: All 4 pages have Best Practices score >90
  - **Spec**: SC-005
  - **Test**: Audit results show Best Practices ≥90
  - **Depends**: T264

- [ ] T268 [PERF] Verify SEO >90
  - **AC**: All 4 pages have SEO score >90
  - **Spec**: SC-005
  - **Test**: Audit results show SEO ≥90
  - **Depends**: T264

- [ ] T269 [PERF] Add Lighthouse CI to GitHub Actions
  - **AC**: `.github/workflows/lighthouse.yml` runs Lighthouse on PR, fails if scores below thresholds
  - **Spec**: Plan.md line 1720
  - **Test**: Create test PR, verify Lighthouse runs
  - **Depends**: T268

### Final Deliverables (120 min)

- [ ] T270 [DOC] Complete README.md
  - **AC**: README includes: Project Overview (100 words), Features list (base + bonuses), Tech Stack (frontend/backend/DBs), Setup Instructions (10 steps), Deployment (GitHub Pages + Fly.io URLs), Demo Video link, License (MIT)
  - **Spec**: SC-010
  - **Test**: All 7 sections complete, setup instructions tested
  - **Depends**: T004

- [ ] T271 [DEMO] Record demo video: Homepage
  - **AC**: 5-second clip showing homepage with module cards
  - **Spec**: SC-011
  - **Test**: Clip recorded, 5s duration
  - **Depends**: T234

- [ ] T272 [DEMO] Record demo video: Navigation
  - **AC**: 5-second clip showing sidebar navigation to chapter
  - **Spec**: SC-011
  - **Test**: Clip recorded, 5s duration
  - **Depends**: T271

- [ ] T273 [DEMO] Record demo video: Chatbot query
  - **AC**: 20-second clip showing: Open chatbot, type "What is a ROS 2 node?", send, receive answer with citation, click citation
  - **Spec**: SC-011
  - **Test**: Clip recorded, 20s duration, shows citation navigation
  - **Depends**: T272

- [ ] T274 [DEMO] Record demo video: Signup
  - **AC**: 15-second clip showing signup form fill, submit, profile indicator appears
  - **Spec**: SC-011
  - **Test**: Clip recorded, 15s duration
  - **Depends**: T273

- [ ] T275 [DEMO] Record demo video: Personalization
  - **AC**: 15-second clip showing: Click "Personalize", content changes, indicator shows "personalized"
  - **Spec**: SC-011
  - **Test**: Clip recorded, 15s duration
  - **Depends**: T274

- [ ] T276 [DEMO] Record demo video: Translation
  - **AC**: 15-second clip showing: Click "Translate to Urdu", Urdu text appears, code blocks English
  - **Spec**: SC-011
  - **Test**: Clip recorded, 15s duration, visual proof of code preservation
  - **Depends**: T275

- [ ] T277 [DEMO] Edit demo video compilation
  - **AC**: Combine all clips into single video, add captions/voiceover, total duration <90s
  - **Spec**: SC-011
  - **Test**: Final video <90s, all features shown
  - **Depends**: T276

- [ ] T278 [DEMO] Upload demo video
  - **AC**: Video uploaded to YouTube or docs/static/demo-video.mp4, URL added to README
  - **Spec**: SC-011
  - **Test**: Video accessible via link
  - **Depends**: T277

- [ ] T279 [DEPLOY] Deploy frontend to production
  - **AC**: Push to main triggers GitHub Actions, site deploys to GitHub Pages
  - **Spec**: SC-010
  - **Test**: Visit github.io URL, site loads
  - **Depends**: T023, T268

- [ ] T280 [DEPLOY] Deploy backend to production
  - **AC**: Push to main triggers backend deploy to Fly.io
  - **Spec**: SC-010
  - **Test**: curl <fly-app-url>/health returns 200
  - **Depends**: T026, T119

- [ ] T281 [DEPLOY] Run smoke tests on production
  - **AC**: Execute smoke.spec.ts against production URLs, all tests pass
  - **Spec**: SC-010
  - **Test**: Smoke tests green
  - **Depends**: T234, T279, T280

- [ ] T282 [AUDIT] Validate conventional commit format
  - **AC**: Run `git log --oneline`, 100% commits follow `<type>(<scope>): <subject>` format
  - **Spec**: SC-012
  - **Test**: All commits valid format
  - **Depends**: T021

- [ ] T283 [AUDIT] Validate commit size <500 lines
  - **AC**: Run `git log --stat`, no commits >500 lines changed
  - **Spec**: SC-012
  - **Test**: All commits ≤500 lines
  - **Depends**: T282

- [ ] T284 [AUDIT] Validate Claude co-author footer
  - **AC**: Run `git log`, 100% of AI-assisted commits include "Co-Authored-By: Claude <noreply@anthropic.com>"
  - **Spec**: SC-012
  - **Test**: All commits have co-author
  - **Depends**: T283

- [ ] T285 [AUDIT] Create git history validation script
  - **AC**: `scripts/check-git-history.sh` automates checks from T282-T284
  - **Spec**: SC-012
  - **Test**: Script runs, reports PASS
  - **Depends**: T284

**🏁 FINAL CHECKPOINT**: Ready for Submission (T232-T285)
**Validation**:
- ✅ All E2E tests pass (20+ scenarios)
- ✅ Lighthouse scores >thresholds (SC-005)
- ✅ Demo video <90s showing all features (SC-011)
- ✅ Production deployment successful (SC-010)
- ✅ Git history quality 100% (SC-012)

---

## Task Summary

**Total Atomic Tasks**: 285 tasks
**Average Task Duration**: 15-30 minutes
**Total Estimated Effort**: ~142 hours (285 tasks × 30 min avg)

**By Phase**:
- Phase 0 (Foundation): 26 tasks (13 hours)
- Phase 1 (US1 - Content): 43 tasks (21.5 hours)
- Phase 2 (US2 - Chatbot): 81 tasks (40.5 hours)
- Phase 3 (US3 - Auth): 37 tasks (18.5 hours)
- Phase 4 (US4 - Personalization): 16 tasks (8 hours)
- Phase 5 (US5 - Translation): 28 tasks (14 hours)
- Phase 6 (Testing/Deploy): 54 tasks (27 hours)

**Dependency Chains**:
- Critical path: T001 → T005 → T027 → T040 → T070 → T103 → T114 → T232 → T279 (longest chain: ~25 tasks)
- Parallel opportunities: Content generation (T040-T059), Frontend/Backend (T030-T035 || T023-T029), Bonus features (T151-T231 all parallelizable)

**Checkpoint Gates**: 6 checkpoints require human validation before proceeding

**Success Criteria Coverage**: All 12 SCs validated through specific tasks
