<!--
Sync Impact Report:
Version: 0.0.0 → 1.0.0
Rationale: Initial constitution creation for Physical AI & Humanoid Robotics textbook project

Added Sections:
  - All core principles (I-VIII)
  - Educational Content Guidelines
  - Technical Architecture Standards
  - Governance section

Templates Requiring Updates:
  ✅ Constitution created from template
  ⚠ spec-template.md - should align with educational content principles
  ⚠ plan-template.md - should reference textbook structure and RAG integration
  ⚠ tasks-template.md - should include educational content tasks

Follow-up TODOs: None
-->

# Physical AI & Humanoid Robotics Textbook Constitution

## Core Principles

### I. Educational-First Content Design

**Every chapter and module MUST prioritize learning outcomes over feature complexity.**

- Content MUST be structured as progression from fundamentals to advanced topics
- Each module MUST include clear learning objectives stated at the beginning
- Complex concepts MUST be introduced incrementally with concrete examples
- All technical content MUST include both theoretical explanation and practical application
- Code examples MUST be runnable, tested, and include expected outputs
- Visual aids (diagrams, flowcharts, simulation screenshots) MUST accompany complex explanations

**Rationale**: This is an educational textbook for students transitioning from AI software to Physical AI. Content must scaffold learning progressively to bridge the gap between digital intelligence and embodied systems.

### II. Modular Content Architecture

**All content MUST be organized into independent, reusable modules that can stand alone or combine.**

- Each module (ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA) MUST be completable independently
- Modules MUST declare explicit prerequisites and learning dependencies
- Content MUST avoid circular dependencies between modules
- Each chapter MUST be self-contained with its own introduction, content, exercises, and summary
- Cross-module references MUST use explicit links and context
- Modules MUST be versioned independently for updates without breaking dependents

**Rationale**: Modular design enables students to learn at their own pace, instructors to customize curriculum paths, and authors to update content incrementally. It also supports the personalization features (bonus points) where content can be adapted based on user background.

### III. Interactive Learning Through RAG

**The RAG chatbot MUST be treated as a first-class learning companion, not an afterthought.**

- RAG system MUST be able to answer questions on any section of the textbook
- RAG MUST support text-selection-based queries for deep dives on specific passages
- RAG responses MUST cite specific sections, page references, or code blocks
- RAG MUST handle beginner questions ("What is ROS 2?") and advanced queries ("How does VSLAM differ from traditional SLAM?")
- RAG integration MUST be non-intrusive to reading flow but easily accessible
- RAG MUST maintain conversation context for multi-turn learning dialogues

**Rationale**: Modern learners expect interactive, AI-powered learning experiences. The RAG system transforms a static textbook into a dynamic learning environment where students can ask questions in context and receive immediate, relevant answers.

### IV. Code-First Technical Validation

**All code examples, simulations, and technical procedures MUST be executable and tested.**

- Every code snippet MUST be syntactically correct and tested in the target environment
- ROS 2 examples MUST be tested with specified ROS 2 distribution (e.g., Humble, Iron)
- Simulation examples MUST include environment setup, execution steps, and expected results
- URDF models MUST be validated for correct syntax and visualizable in RViz or Gazebo
- Python agent-ROS bridges MUST demonstrate working message passing
- All commands and CLI examples MUST be verified on the target platform (Ubuntu/Linux)

**Rationale**: Students lose trust and momentum when examples don't work. Every technical element must be validated to ensure reproducibility and learning continuity.

### V. Accessibility and Personalization

**Content MUST be accessible to learners with diverse backgrounds and support personalization.**

- Content MUST assume only basic programming knowledge (Python fundamentals)
- Hardware/robotics background MUST NOT be assumed; introduce concepts from first principles
- Technical jargon MUST be defined on first use with glossary links
- User background profiles (signup questions) MUST inform content adaptation
- Personalization features MUST adjust content depth based on user's software/hardware experience
- Translation features (Urdu translation bonus) MUST preserve technical accuracy and formatting
- UI MUST support accessibility standards (keyboard navigation, screen readers, high contrast)

**Rationale**: Panaversity aims to democratize AI education globally. Content must be accessible to students from non-robotics backgrounds and support localization for diverse learners.

### VI. Deployment and Performance Standards

**The deployed textbook MUST be fast, reliable, and production-ready.**

- Docusaurus build MUST complete without errors or warnings
- GitHub Pages deployment MUST be automated via CI/CD
- Page load time MUST be under 3 seconds on typical connections
- RAG chatbot API (FastAPI) MUST respond within 2 seconds for typical queries
- Neon Postgres database MUST handle concurrent RAG queries without degradation
- Qdrant vector search MUST return relevant results in under 500ms
- Authentication (Better-Auth bonus) MUST be secure, session-based, and protect user data
- All API endpoints MUST implement rate limiting and input validation

**Rationale**: A slow or unreliable textbook frustrates learners. Production-grade deployment demonstrates professional standards and ensures positive learning experiences.

### VII. Spec-Driven Development with Claude Code

**All development work MUST follow Spec-Kit Plus workflows and leverage Claude Code effectively.**

- Every feature (textbook chapter, RAG integration, auth system) MUST start with a specification
- Specifications MUST be reviewed and approved before implementation begins
- Implementation MUST reference the specification and track completion via tasks
- Claude Code subagents and skills (bonus points) MUST be created for reusable workflows:
  - Content generation subagent for consistent chapter structure
  - Code validation subagent for testing examples
  - RAG indexing subagent for embedding generation and vector storage
- Architectural decisions (RAG architecture, auth approach, translation strategy) MUST be documented as ADRs
- Prompt History Records MUST be created for all significant AI-assisted work
- Development MUST follow Test-First discipline where applicable (RAG tests, API tests, auth tests)

**Rationale**: Spec-driven development ensures clarity, alignment, and quality. Leveraging Claude Code's advanced features (subagents, skills) demonstrates mastery and earns bonus points in the hackathon.

### VIII. Version Control and Collaboration Hygiene

**All work MUST follow Git best practices for reproducibility and collaboration.**

- Commits MUST be atomic, well-described, and reference related specs/tasks
- Branches MUST follow naming convention: `feature/<###-feature-name>`
- Pull requests MUST include description, testing evidence, and demo (if UI change)
- Secrets (API keys for OpenAI, Neon, Qdrant) MUST NEVER be committed; use `.env` and `.gitignore`
- README MUST include setup instructions, deployment steps, and architecture overview
- Dependencies MUST be pinned with version specifications (package.json, requirements.txt)
- Documentation MUST be updated alongside code changes

**Rationale**: The hackathon submission requires a public GitHub repo. Clean Git hygiene demonstrates professionalism and makes the project accessible to judges and potential team members.

---

## Educational Content Guidelines

### Content Structure Standards

- **Module Structure**: Introduction → Core Concepts → Hands-On Labs → Exercises → Summary → Further Reading
- **Chapter Length**: Target 10-15 minutes reading time per chapter (1500-2000 words)
- **Code-to-Text Ratio**: Aim for 40% code/examples, 60% explanation for technical modules
- **Exercise Difficulty**: Include beginner (apply concepts), intermediate (combine concepts), advanced (extend/research)
- **Visual Requirements**: Minimum 2 diagrams or screenshots per chapter for complex topics

### Learning Progression

**Module 1 (ROS 2)**: Foundation for all subsequent modules
- MUST introduce pub/sub messaging before services
- MUST demonstrate rclpy before C++ (Python-first audience)
- MUST include visualization (RViz, rqt_graph) for concept clarity

**Module 2 (Gazebo/Unity)**: Build on ROS 2 knowledge
- MUST show Gazebo-ROS integration with working examples
- MUST explain physics engine parameters with visual effects
- MUST demonstrate sensor simulation before robot control

**Module 3 (NVIDIA Isaac)**: Advanced perception and training
- MUST differentiate Isaac Sim from Gazebo (photorealism, synthetic data)
- MUST explain domain randomization with concrete examples
- MUST show Isaac ROS integration with existing ROS 2 knowledge

**Module 4 (VLA)**: Culmination integrating all prior modules
- MUST require understanding of ROS 2, simulation, and perception
- MUST demonstrate LLM-to-action pipeline with working code
- Capstone project MUST integrate voice (Whisper) → planning (LLM) → execution (ROS 2) → navigation (Nav2)

---

## Technical Architecture Standards

### Technology Stack (NON-NEGOTIABLE)

- **Textbook Framework**: Docusaurus (React-based static site generator)
- **Deployment**: GitHub Pages or Vercel (must be publicly accessible)
- **RAG Backend**: FastAPI (Python async API framework)
- **RAG LLM**: OpenAI Agents SDK or ChatKit SDK
- **Vector Database**: Qdrant Cloud Free Tier
- **Relational Database**: Neon Serverless Postgres
- **Authentication** (bonus): Better-Auth
- **Content Language**: English (primary), Urdu (translation bonus)

### RAG Architecture Requirements

- **Embedding Generation**: Use OpenAI embeddings (text-embedding-3-small or ada-002)
- **Chunking Strategy**: Chunk chapters into semantically meaningful sections (paragraphs, code blocks)
- **Metadata**: Store chapter title, module, section heading, content type (text/code) with each chunk
- **Retrieval**: Hybrid search (vector similarity + keyword matching) for best results
- **Context Window**: Retrieve top 3-5 most relevant chunks for LLM context
- **Text Selection Query**: Capture selected text, embed it, find related content + enable direct Q&A on selection

### Authentication Architecture (Bonus)

- **User Profile**: Email, password, software background (scale 1-5), hardware background (scale 1-5)
- **Personalization Logic**:
  - Software background ≥4 → Show advanced code patterns, reduce basic explanations
  - Hardware background ≤2 → Expand robotics fundamentals, add more diagrams
- **Session Management**: Secure HTTP-only cookies, 7-day expiry
- **Authorization**: Public content (unauthenticated), personalized features (authenticated)

### Personalization & Translation (Bonus)

- **Personalization Button**: At chapter start, adjust content based on user profile + explicit request
- **Translation Button**: Use translation API (e.g., Google Translate API, OpenAI translation) to convert English → Urdu
- **Content Adaptation**: Store original + personalized/translated versions; cache for performance
- **UI/UX**: Clear indicators when viewing personalized or translated content

---

## Governance

### Constitution Authority

- This constitution supersedes all other project practices and conventions
- All design decisions, code reviews, and architectural choices MUST verify compliance
- Violations MUST be justified in writing with rationale for exception
- Complexity beyond these standards MUST be approved and documented via ADR

### Amendment Process

- Amendments MUST be proposed with rationale and impact analysis
- Amendments MUST increment version: MAJOR (breaking changes), MINOR (new principles), PATCH (clarifications)
- Amendments MUST update dependent templates (spec, plan, tasks, commands)
- Amendments MUST be reviewed and approved before adoption

### Compliance Review

- Every feature specification MUST include a Constitution Check section
- Every implementation plan MUST validate against core principles
- Every pull request MUST affirm adherence to standards
- Violations discovered post-merge MUST be logged and corrected promptly

### Living Documentation

- This constitution is a living document; practical feedback improves it
- Update constitution when project needs evolve or new patterns emerge
- Maintain Prompt History Records (PHRs) for all constitutional amendments
- Document significant architectural decisions as ADRs

---

**Version**: 1.0.0 | **Ratified**: 2025-11-28 | **Last Amended**: 2025-11-28
