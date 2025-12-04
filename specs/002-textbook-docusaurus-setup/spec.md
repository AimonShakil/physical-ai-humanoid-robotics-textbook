# Feature Specification: Physical AI & Humanoid Robotics Textbook with Docusaurus

**Feature Branch**: `002-textbook-docusaurus-setup`
**Created**: 2025-11-28
**Status**: Draft
**Input**: User description: "i am sharing here my business requirements we are creating a book for a hackathon for that we need a static site render and we set up docusaurus else than than create everything as per constitution and create 99% accurate specs"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Read Textbook Content (Priority: P1) 🎯 MVP

Students access the online textbook to learn Physical AI and Humanoid Robotics through structured, progressive content across 4 modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA).

**Why this priority**: Core value proposition - without readable content, there is no textbook. This is the foundation for all other features.

**Independent Test**: Navigate to deployed site, browse modules, read chapters, view code examples and diagrams. Success = content is accessible, readable, and properly formatted.

**Acceptance Scenarios**:

1. **Given** a student visits the textbook homepage, **When** they click on "Module 1: ROS 2", **Then** they see a list of chapters and can navigate to any chapter
2. **Given** a student is reading a chapter, **When** they scroll through content, **Then** they see properly formatted text, code blocks with syntax highlighting, diagrams, and exercises
3. **Given** a student is on a chapter page, **When** they click "Next Chapter", **Then** they navigate to the next sequential chapter
4. **Given** a student wants to find specific content, **When** they use the search function, **Then** they see relevant results from across all modules
5. **Given** a student is reading on mobile, **When** they view any chapter, **Then** the content is responsive and readable without horizontal scrolling

---

### User Story 2 - Ask Questions via RAG Chatbot (Priority: P2)

Students ask questions about textbook content using an embedded AI chatbot that retrieves relevant information from the textbook and provides contextual answers with citations.

**Why this priority**: Critical differentiator for hackathon scoring (base requirement). Transforms passive reading into active learning.

**Independent Test**: Open chatbot, ask question like "What is a ROS 2 node?", receive answer with citation linking back to relevant chapter section. Success = accurate answers with source references.

**Acceptance Scenarios**:

1. **Given** a student is reading a chapter, **When** they click the chatbot icon, **Then** a chat interface opens without disrupting reading
2. **Given** the chatbot is open, **When** a student types "What is a ROS 2 publisher?" and sends, **Then** they receive a relevant answer within 2 seconds with a citation link
3. **Given** a student clicks a citation link, **When** the link loads, **Then** they navigate to the specific section/heading referenced in the answer
4. **Given** a student selects text on a page, **When** they click "Ask about this", **Then** the chatbot opens with the selected text as context
5. **Given** a student asks a follow-up question, **When** they send it, **Then** the chatbot maintains conversation context from previous messages

---

### User Story 3 - Sign Up and Sign In (Priority: P3) ⭐ Bonus

Students create accounts and sign in to access personalized features, with profile questions about software and hardware background captured during signup.

**Why this priority**: Enables bonus features (personalization, progress tracking). Not required for basic textbook functionality.

**Independent Test**: Complete signup form with background questions, sign in, verify session persists. Success = authenticated state maintained, profile data captured.

**Acceptance Scenarios**:

1. **Given** a new user clicks "Sign Up", **When** they fill email, password, and answer background questions (software level 1-5, hardware level 1-5), **Then** their account is created and they are signed in
2. **Given** an existing user clicks "Sign In", **When** they enter valid credentials, **Then** they are authenticated and see their profile indicator
3. **Given** a user is signed in, **When** they reload the page or navigate, **Then** their session persists without re-authentication
4. **Given** a user clicks their profile icon, **When** the menu opens, **Then** they see options to view profile and sign out
5. **Given** a user clicks "Sign Out", **When** confirmed, **Then** they are signed out and redirected to public view

---

### User Story 4 - Personalize Content (Priority: P4) ⭐ Bonus

Signed-in students click a "Personalize" button at the start of each chapter to adjust content depth based on their background profile (more code for advanced, more explanation for beginners).

**Why this priority**: Bonus feature for extra points. Enhances learning experience but not core functionality.

**Independent Test**: Sign in as user with high software background, click "Personalize" on chapter, verify content adjusts (more code, less basic explanation). Success = visible content adaptation.

**Acceptance Scenarios**:

1. **Given** a signed-in user with software background ≥4, **When** they click "Personalize Content" at chapter start, **Then** the chapter shows more advanced code patterns and fewer basic explanations
2. **Given** a signed-in user with hardware background ≤2, **When** they click "Personalize Content", **Then** the chapter shows more robotics fundamentals and additional diagrams
3. **Given** a user views personalized content, **When** they see the content, **Then** there is a clear indicator that content is personalized
4. **Given** a user wants to see original content, **When** they click "Show Original", **Then** the content reverts to default version
5. **Given** personalized content is generated, **When** the user navigates away and returns, **Then** personalized version is cached and loads quickly

---

### User Story 5 - Translate to Urdu (Priority: P5) ⭐ Bonus

Signed-in students click a "Translate to Urdu" button at the start of each chapter to view content in Urdu while preserving code blocks and formatting.

**Why this priority**: Bonus feature for extra points. Supports global accessibility (Panaversity mission) but not required for hackathon base requirements.

**Independent Test**: Sign in, navigate to chapter, click "Translate to Urdu", verify text is translated while code remains in English. Success = accurate translation with preserved formatting.

**Acceptance Scenarios**:

1. **Given** a signed-in user clicks "Translate to Urdu" on a chapter, **When** translation completes, **Then** all text content is in Urdu with Urdu Unicode characters displayed correctly
2. **Given** a chapter is translated, **When** viewing code blocks, **Then** code remains in English (not translated)
3. **Given** translated content is displayed, **When** user clicks "Show Original", **Then** content reverts to English
4. **Given** a user translates content, **When** they navigate to another chapter, **Then** the translation preference persists for that session
5. **Given** translation is in progress, **When** user waits, **Then** they see a loading indicator and translated content appears within 3 seconds

---

### Edge Cases

- What happens when a user asks the chatbot a question completely unrelated to the textbook content?
- How does the system handle a user selecting code blocks for "Ask about this" text-selection queries?
- What happens when a user tries to sign up with an email that already exists?
- How does personalization handle users who haven't answered background questions (e.g., old accounts)?
- What happens when translation API fails or times out?
- How does the chatbot handle very long questions (>500 words)?
- What happens when the vector database (Qdrant) returns no relevant chunks for a query?
- How does the system handle multiple tabs with the same signed-in user?

## Requirements *(mandatory)*

### Functional Requirements

#### Docusaurus Static Site

- **FR-001**: System MUST render textbook content as a static website using Docusaurus framework
- **FR-002**: System MUST organize content into 4 modules: Module 1 (ROS 2), Module 2 (Gazebo/Unity), Module 3 (NVIDIA Isaac), Module 4 (VLA)
- **FR-003**: Each module MUST contain 8-10 chapters with progressive difficulty
- **FR-004**: System MUST provide sidebar navigation showing all modules and chapters
- **FR-005**: System MUST include search functionality across all textbook content
- **FR-006**: System MUST support "Previous Chapter" and "Next Chapter" navigation
- **FR-007**: System MUST display code blocks with syntax highlighting for Python, Bash, XML, YAML
- **FR-008**: System MUST render Mermaid diagrams and support embedded images
- **FR-009**: System MUST be fully responsive for desktop, tablet, and mobile viewports
- **FR-010**: System MUST deploy to GitHub Pages or Vercel as publicly accessible website

#### Content Structure

- **FR-011**: Each chapter MUST include: Learning Objectives, Introduction, Core Concepts, Hands-On Lab, Exercises, Summary, Further Reading
- **FR-012**: Learning objectives MUST be stated at the beginning of each chapter
- **FR-013**: Code examples MUST include comments explaining key lines
- **FR-014**: Hands-on labs MUST include step-by-step instructions with expected outputs
- **FR-015**: Exercises MUST be provided at 3 difficulty levels: beginner, intermediate, advanced
- **FR-016**: Each chapter MUST include minimum 2 diagrams or visual aids for complex concepts
- **FR-017**: Technical jargon MUST be defined on first use

#### RAG Chatbot Integration

- **FR-018**: System MUST embed an AI chatbot accessible from any page via a persistent button/icon
- **FR-019**: Chatbot MUST use OpenAI Agents SDK or ChatKit SDK for LLM interaction
- **FR-020**: Chatbot MUST use FastAPI backend for handling queries
- **FR-021**: System MUST split chapters into semantic chunks (max 1000 tokens, 100-token overlap, respecting section boundaries) and store in Qdrant vector database (Cloud Free Tier) - see Assumption #13 for detailed chunking strategy
- **FR-022**: System MUST use Neon Serverless Postgres database for chunk metadata and chat history
- **FR-023**: Chatbot MUST retrieve top 3-5 most relevant chunks for each query
- **FR-024**: Chatbot MUST provide citations linking back to source chapter sections
- **FR-025**: Chatbot MUST support text-selection-based queries (user selects text, asks question about it)
- **FR-026**: Chatbot MUST maintain multi-turn conversation context within a session
- **FR-027**: Chatbot MUST handle both beginner questions and advanced technical queries

#### Authentication (Bonus)

- **FR-028**: System MUST implement signup and signin using Better-Auth library
- **FR-029**: Signup form MUST ask: email, password, software background (scale 1-5), hardware background (scale 1-5)
- **FR-030**: System MUST validate email format and password strength (minimum 8 characters)
- **FR-031**: System MUST use secure session management with HTTP-only cookies (7-day expiry)
- **FR-032**: System MUST persist authenticated state across page reloads
- **FR-033**: System MUST display user profile indicator when signed in
- **FR-034**: System MUST provide sign-out functionality

#### Personalization (Bonus)

- **FR-035**: System MUST show "Personalize Content" button at start of each chapter for authenticated users
- **FR-036**: Personalization MUST adjust content based on user background profile:
  - Software background ≥4: Show advanced code patterns, reduce basic explanations
  - Hardware background ≤2: Expand robotics fundamentals, add more diagrams
- **FR-037**: Personalized content MUST be clearly indicated to user
- **FR-038**: System MUST provide "Show Original" button to revert to default content
- **FR-039**: System MUST cache personalized content for performance

#### Translation (Bonus)

- **FR-040**: System MUST show "Translate to Urdu" button at start of each chapter for authenticated users
- **FR-041**: Translation MUST convert English text to Urdu using translation API (Google Translate API or OpenAI)
- **FR-042**: Code blocks MUST remain in English (not translated)
- **FR-043**: System MUST preserve markdown formatting during translation
- **FR-044**: System MUST display Urdu Unicode characters correctly
- **FR-045**: System MUST provide "Show Original" button to revert to English
- **FR-046**: System MUST cache translated content for performance

#### Performance & Deployment

- **FR-047**: Docusaurus build MUST complete without errors or warnings
- **FR-048**: System MUST deploy via automated CI/CD pipeline (GitHub Actions)
- **FR-049**: System MUST serve static assets via CDN for fast delivery
- **FR-050**: System MUST implement error boundaries to handle runtime errors gracefully

### Key Entities

- **Module**: Represents a major learning unit (e.g., ROS 2, Gazebo/Unity). Attributes: module number, title, description, prerequisites, chapter list
- **Chapter**: Represents a single learning unit within a module. Attributes: chapter number, title, module, content (markdown), learning objectives, exercises, estimated reading time
- **User** (Bonus): Represents a signed-in student. Attributes: email, password hash, software background (1-5), hardware background (1-5), account creation date, session token
- **Chat Message**: Represents a single chatbot interaction. Attributes: user ID (optional), message text, role (user/assistant), timestamp, conversation ID, citations (links to chapters)
- **Content Chunk**: Represents a semantic section of textbook content indexed for RAG. Attributes: chunk ID, chapter reference, section heading, content text, content type (text/code/exercise), embedding vector, metadata

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: By Nov 30, 2025, students can navigate to any chapter and start reading within 3 seconds of page load (measured via Chrome DevTools Network tab and Lighthouse Performance score)

- **SC-002**: By Nov 30, 2025, textbook contains 32-40 complete chapters (each with Learning Objectives, Introduction, Core Concepts, Hands-On Lab, Exercises, Summary, Further Reading) across 4 modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA) covering all course topics defined in constitution

- **SC-003**: By Nov 30, 2025, chatbot responds to user questions within 2 seconds with answers that meet quality criteria:
  - Retrieve content from correct module/chapter (verified by citation accuracy in SC-004)
  - Contain at least one valid citation linking to textbook content
  - Pass manual quality review: 10 test queries rated "helpful" by >80% of acceptance testers (minimum 3 reviewers per query)

- **SC-004**: By Nov 30, 2025, 95% of a standardized test query set (minimum 50 queries) return answers with valid citations linking to correct chapter sections:
  - **Test Set Composition**: 20 beginner queries (e.g., "What is a ROS 2 node?"), 20 intermediate queries (e.g., "How do I create a publisher-subscriber pair?"), 10 advanced queries (e.g., "How does VSLAM differ from traditional SLAM?")
  - **Validation Criteria**: Citation link must resolve to valid chapter URL, linked section must contain content semantically relevant to query topic, citation format includes chapter title and section heading

- **SC-005**: By Nov 30, 2025, deployed textbook passes all Google Lighthouse scores when tested on homepage and 3 sample chapters (one from each module): Performance >90, Accessibility >95, Best Practices >90, SEO >90

- **SC-006**: By Nov 30, 2025, new students can complete entire signup process (from clicking "Sign Up" to successful authentication and landing on textbook homepage) in under 1 minute, including email/password entry, software background question (1-5 scale), hardware background question (1-5 scale), and account creation with auto-signin (bonus feature)

- **SC-007**: By Nov 30, 2025, personalized content displays within 1 second of clicking "Personalize" button, assuming pre-computed content variants cached; subsequent chapter personalizations load from cache within 0.5 seconds (bonus feature)

- **SC-008**: By Nov 30, 2025, translated Urdu content displays within 3 seconds of clicking "Translate" button on first request; subsequent requests for same chapter load from cache within 0.5 seconds (bonus feature)

- **SC-009**: By Nov 30, 2025, 100% of code examples in all chapters pass validation via code-validator subagent: syntax correctness, import validation, ROS 2 API compliance with Humble/Iron, no hardcoded secrets or security vulnerabilities

- **SC-010**: By Nov 30, 2025 at 5:00 PM (1 hour before submission deadline), site is successfully deployed and publicly accessible at a GitHub Pages or Vercel URL that loads without errors, shows all 4 modules in navigation, and passes manual smoke test (homepage, 1 chapter from each module, chatbot opens and responds)

- **SC-011**: By Nov 30, 2025, demo video (under 90 seconds) is created and uploaded, demonstrating:
  - **Base Features** (must show all): (1) Navigate to and display a textbook chapter with visible content (5-10 seconds), (2) Open chatbot, type and send question, show answer with citation, click citation to navigate to source (15-20 seconds)
  - **Bonus Features** (must show at least 2 of): (3) Complete signup with background questions (10-15 seconds), (4) Click "Personalize" and show visible content difference (10-15 seconds), (5) Click "Translate to Urdu" and show Urdu text with English code blocks preserved (10-15 seconds)
  - **Quality Requirements**: Screen recording with captions or voiceover, clearly shows feature functionality in action (not static screenshots), uploaded to accessible URL for judges

- **SC-012**: By Nov 30, 2025, GitHub repository demonstrates quality version control practices:
  - **Commit Format**: 100% of commits follow conventional commit format `<type>(<scope>): <subject>` (e.g., "feat(rag): add citation linking", "docs(module1): create ROS 2 nodes chapter")
  - **Commit Quality**: No generic messages like "fix", "update", "changes", "wip"; minimum 10 characters excluding type/scope prefix
  - **Commit Size**: No commits with >500 lines changed (indicates proper work breakdown into atomic commits)
  - **Attribution**: All AI-assisted commits include Claude Code co-author footer
  - **History Structure**: Feature branches follow naming `[number]-[feature-name]`, no force pushes to main branch, minimum 10 commits total demonstrating iterative development

## Assumptions

1. **Content Creation**: Textbook content will be generated using the chapter-generator subagent for consistency and quality
2. **Hosting**: GitHub Pages is preferred for deployment (free, integrated with GitHub, sufficient for hackathon)
3. **Translation Service**: Will use OpenAI translation API for Urdu translation (consistent with RAG LLM provider)
4. **Code Validation**: All code examples will be validated using code-validator subagent before inclusion
5. **RAG Indexing**: Content will be indexed using rag-indexer subagent after chapters are created
6. **Testing**: E2E tests will be generated using test-generator subagent for comprehensive coverage
7. **Embedding Model**: Will use OpenAI text-embedding-3-small for vector embeddings (cost-effective, high quality)
8. **Database Free Tiers**: Qdrant Cloud Free Tier (1GB) and Neon Serverless Postgres free tier are sufficient for hackathon scope
9. **Authentication Library**: Better-Auth was specified in hackathon requirements and supports custom profile questions
10. **Content Volume**: Each chapter approximately 1500-2000 words with 40% code/60% explanation ratio
11. **Mobile Support**: Docusaurus provides responsive design out of the box, no custom mobile development needed
12. **Browser Support**: Modern browsers (Chrome, Firefox, Safari, Edge) with ES6+ support

### Clarifications Added During Specification Review (2025-11-28)

13. **RAG Chunking Strategy** (addresses FR-021):
    - Chunks MUST respect semantic boundaries (H2/H3 headings, paragraph breaks)
    - Chunks MUST use 100-token overlap between adjacent chunks to preserve context
    - Code blocks that exceed 1000 tokens MUST be kept whole (not split) with surrounding context included in chunk metadata
    - Chunking priority: Full section if <1000 tokens → Sub-section at H3 level → Paragraph level → Character split as last resort

14. **Content Personalization Implementation** (addresses FR-036 and SC-007):
    - **Pre-computed approach**: Generate 3 content variants per chapter during build (beginner/intermediate/advanced)
    - Storage overhead: 32 chapters × 3 variants × ~2000 words = ~192K words total (~500KB compressed markdown)
    - Personalization MUST select variant based on user profile: software+hardware background ≤3 = beginner, 4-7 = intermediate, ≥8 = advanced
    - Variant selection happens client-side (no API call), enabling <1s display (SC-007 compliance)
    - Cache key format: `chapter-{chapterId}-{variant}` stored in browser localStorage

15. **Chapter Content Minimum Viable Structure** (addresses FR-011, FR-012, FR-014, FR-015):
    - **Target word count**: 2000-3000 words per chapter (excluding code blocks)
    - **Learning objectives**: Minimum 3, maximum 6 per chapter
    - **Hands-On Lab**: Must include 1 complete tutorial with 5-10 step-by-step instructions, expected outputs for verification, and estimated completion time (15-30 minutes)
    - **Exercises**: Exactly 3 exercises (1 beginner, 1 intermediate, 1 advanced) with clear success criteria
    - **Diagrams**: Minimum 2 visual aids (Mermaid diagrams, architecture diagrams, or flow charts)
    - **Code examples**: Minimum 3 code blocks with inline comments, maximum 50 lines each

16. **Better-Auth Session Management** (addresses FR-031 and security architecture):
    - Session storage: HTTP-only cookies (secure, httpOnly, sameSite=strict) to prevent XSS attacks
    - Session token: JWT with 7-day expiry, stored server-side in Neon Postgres sessions table
    - Token refresh: Automatic refresh on each API call if token age >5 days, transparent to user
    - API authentication for RAG calls: Include session cookie in FastAPI requests, validate JWT signature and expiry
    - CSRF protection: Use CSRF tokens for state-changing operations (signup, profile updates)
    - No localStorage for session tokens (security risk); profile data (background scores) can be cached in localStorage for personalization

17. **Demo Video Acceptance Criteria** (addresses SC-011 edge cases):
    - **Retakes allowed**: Edited compilation is acceptable; does not need to be single continuous recording
    - **Narration language**: English required (judges expect English explanations); Urdu UI text acceptable during translation demo
    - **Video format**: MP4, MOV, or WebM; uploaded to any accessible URL (YouTube, Vimeo, Google Drive with public link, or GitHub Pages)
    - **Minimum resolution**: 720p (1280×720) for readability
    - **Audio requirements**: Clear narration or on-screen captions explaining each feature; background music optional but must not overpower narration
    - **Failure handling**: If bonus feature fails during demo recording, skip and show different bonus feature (only need 2 of 5 total bonus features)

## Dependencies

### External Services
- GitHub (repository hosting, GitHub Pages deployment)
- OpenAI API (embeddings, chat completion, translation)
- Qdrant Cloud Free Tier (vector database)
- Neon Serverless Postgres Free Tier (relational database)
- Better-Auth (authentication library)

### Development Tools
- Node.js 18+ and npm (Docusaurus requirements)
- Python 3.8+ (FastAPI backend)
- Git (version control)
- Claude Code (development assistant)
- Playwright (E2E testing via test-generator subagent)

### Internal Dependencies
- Constitution v1.0.0 (governs all development)
- Subagents: chapter-generator, code-validator, rag-indexer, test-generator
- Skills: constitution-check, adr-quick

## Out of Scope

- **User Progress Tracking**: Tracking which chapters users have completed (not required for hackathon)
- **Interactive Code Execution**: Running code examples in-browser (not required, users expected to run locally)
- **User Comments/Forums**: Community features for discussion (not required)
- **Offline Access**: Progressive Web App (PWA) features for offline reading (not required)
- **Multiple Language Translations**: Only Urdu translation required (bonus), not full i18n
- **Admin Dashboard**: Content management interface (content managed via markdown files in Git)
- **Analytics**: User behavior tracking beyond basic usage (not required)
- **Quiz/Assessment System**: Automated grading (exercises provided but not auto-graded)
- **Video Content**: Embedded tutorials or lecture videos (text and diagrams only)
- **Real Hardware Integration**: Actual robot control (simulation and conceptual only)

## Risks & Mitigations

### Risk 1: Content Volume (32-40 chapters)
**Impact**: High - Incomplete content fails base requirements
**Mitigation**: Use chapter-generator subagent to accelerate content creation. Prioritize Module 1 (ROS 2) as MVP to demonstrate concept, then expand.

### Risk 2: API Costs (OpenAI, Translation)
**Impact**: Medium - Exceeding budget limits progress
**Mitigation**: Use Context7 MCP for semantic caching to reduce API calls. Set usage limits and monitor costs daily.

### Risk 3: RAG Answer Quality
**Impact**: Medium - Poor answers hurt scoring
**Mitigation**: Use rag-indexer subagent with semantic chunking. Test with diverse queries and refine chunking strategy. Ensure rich metadata for better retrieval.

### Risk 4: Tight Timeline (Deadline: Nov 30, 6 PM)
**Impact**: High - Incomplete features lose points
**Mitigation**: Prioritize P1 (textbook) and P2 (RAG) first. Treat bonus features (auth, personalization, translation) as stretch goals. Use subagents to automate repetitive work.

### Risk 5: Integration Complexity
**Impact**: Medium - Docusaurus + FastAPI + Auth + RAG integration challenges
**Mitigation**: Create ADRs for key architectural decisions. Use test-generator subagent for E2E tests. Test integrations incrementally.

## Notes

- This specification follows the Physical AI & Humanoid Robotics Textbook Constitution v1.0.0
- Hackathon submission deadline: Sunday, Nov 30, 2025 at 6:00 PM
- Required deliverables: Public GitHub repo, deployed site link, demo video (<90 seconds)
- Scoring: 100 points (base) + up to 200 bonus points (50 each for subagents, auth, personalization, translation)
- All development should follow Spec-Driven Development workflow: `/sp.specify` → `/sp.plan` → `/sp.tasks` → `/sp.implement`
- Use constitution-check skill to validate this spec before proceeding to `/sp.plan`
