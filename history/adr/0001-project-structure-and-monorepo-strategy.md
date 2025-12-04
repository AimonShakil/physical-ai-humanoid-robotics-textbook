# ADR-0001: Project Structure and Monorepo Strategy

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-11-28
- **Feature:** 002-textbook-docusaurus-setup
- **Context:** The Physical AI Textbook project requires integrating a Docusaurus static site (frontend) with a FastAPI backend (RAG API), shared automation scripts, and E2E tests. The project has a tight hackathon deadline (Nov 30, 2025) requiring fast development velocity and simple deployment. Decision needed: Single repository (monorepo) vs separate repositories for frontend/backend.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security? YES - affects all development workflow, CI/CD, deployment
     2) Alternatives: Multiple viable options considered with tradeoffs? YES - monorepo vs multi-repo vs polyrepo
     3) Scope: Cross-cutting concern (not an isolated detail)? YES - affects every developer interaction, deployment process
-->

## Decision

**Use a monorepo structure with the following components:**

- **Repository Structure**: Single Git repository with top-level folders
  - `docs/` - Docusaurus frontend (Node.js/TypeScript)
  - `backend/` - FastAPI backend (Python)
  - `scripts/` - Shared automation (chapter-generator, code-validator, rag-indexer, test-generator subagent invocations)
  - `tests/e2e/` - Playwright E2E tests (tests integrated system)
  - `.github/workflows/` - Unified CI/CD pipeline

- **Workspace Tooling**: npm workspaces for frontend dependencies (no Turborepo/Nx complexity for hackathon scope)

- **Deployment Strategy**: Dual deployment from single pipeline
  - Frontend: GitHub Actions → GitHub Pages (static site)
  - Backend: GitHub Actions → Fly.io (Docker container)

- **Dependency Management**:
  - Frontend: `docs/package.json` with exact versions (no `^` or `~`)
  - Backend: `backend/requirements.txt` with pinned versions (`==`)
  - Root: `package.json` with npm workspace configuration

## Consequences

### Positive

- **Simplified Development**: Developers clone one repo and have entire system (frontend + backend + tests)
- **Atomic Commits**: Can commit frontend/backend changes together when modifying API contracts
- **Shared CI/CD**: Single GitHub Actions pipeline tests and deploys both components, reducing duplication
- **API Contract Co-location**: OpenAPI specs in `specs/002-textbook-docusaurus-setup/contracts/` stay synchronized with implementations
- **Faster Iteration**: No need to version-sync across multiple repos during rapid hackathon development
- **Subagent Integration**: Automation scripts (rag-indexer, chapter-generator) can access both frontend content and backend indexing logic easily
- **Unified Documentation**: Single README, single CLAUDE.md, single constitution for all code

### Negative

- **Larger Repository**: ~4K-6K LOC total (docs + backend), longer initial clone time (~2-3x vs separate repos)
- **CI/CD Complexity**: Need to intelligently run frontend tests only on `docs/` changes, backend tests only on `backend/` changes (mitigated with path filters in GitHub Actions)
- **Build Tool Confusion**: Developers must remember `npm` for frontend, `pip` for backend (no unified build command)
- **Deployment Coupling**: Both services deploy from same repo - harder to roll back frontend independently of backend
- **Language Mixing**: Node.js + Python toolchains both required locally, increases setup complexity for new contributors

## Alternatives Considered

### Alternative 1: Multi-Repo (Separate Repositories)

**Structure**:
- `textbook-frontend` repo (Docusaurus)
- `textbook-backend` repo (FastAPI)
- `textbook-automation` repo (subagent scripts)

**Pros**:
- Independent versioning (frontend v1.2.0, backend v1.3.1)
- Cleaner separation of concerns
- Smaller individual repo sizes
- Independent CI/CD pipelines (simpler per-repo)

**Cons**:
- **Development Friction**: Developers need 3 repo clones, context switching
- **API Contract Drift**: OpenAPI specs in separate repos → easy to desync
- **Cross-Repo Changes**: Atomic frontend+backend changes require 2 PRs, coordination overhead
- **Duplicate CI/CD**: Separate workflows for each repo, more YAML maintenance
- **Subagent Complexity**: Automation scripts need to operate across repos

**Why Rejected**: Hackathon timeline (18 days) requires velocity. Multi-repo overhead (coordination, API sync, 3x PR reviews) too costly for small team.

### Alternative 2: Polyrepo with Shared Packages

**Structure**:
- Separate repos like Alternative 1
- Shared `@textbook/contracts` npm package for API types
- Shared `@textbook/subagents` package for automation

**Pros**:
- API contract as versioned package (strong typing)
- Reusable automation across projects

**Cons**:
- **Publishing Overhead**: Need to publish packages to npm registry or use local file: links
- **Version Management**: Coordinating package versions adds complexity
- **Build Pipeline**: CI/CD must build shared packages before consuming repos
- **Overkill for Scope**: Only 1 frontend + 1 backend, not building a platform

**Why Rejected**: Over-engineered for hackathon scope. Shared package versioning slows iteration. No need for external package distribution.

### Alternative 3: Monorepo with Turborepo/Nx

**Structure**:
- Monorepo like chosen decision
- Add Turborepo or Nx for caching, task orchestration, dependency graph

**Pros**:
- Intelligent caching (skip unchanged tests)
- Task parallelization (run frontend/backend tests concurrently)
- Dependency graph visualization

**Cons**:
- **Setup Complexity**: Turborepo/Nx configuration (turborepo.json, nx.json) adds learning curve
- **Build Tool Lock-in**: Tied to Turborepo/Nx ecosystem
- **Overkill for Scale**: Only 2 packages (docs, backend), caching benefits minimal
- **CI/CD Overhead**: GitHub Actions already parallelizes jobs, Turborepo adds redundancy

**Why Rejected**: Hackathon scope (2 packages, <6K LOC) doesn't justify Turborepo complexity. GitHub Actions path filters achieve same goal (skip unchanged tests) with zero config.

## References

- Feature Spec: [specs/002-textbook-docusaurus-setup/spec.md](../../specs/002-textbook-docusaurus-setup/spec.md)
- Implementation Plan: [specs/002-textbook-docusaurus-setup/plan.md](../../specs/002-textbook-docusaurus-setup/plan.md#project-structure)
- Related ADRs: ADR-0002 (Frontend Stack), ADR-0003 (Backend Hosting), ADR-0008 (Testing Stack)
- Project Structure Diagram: [plan.md lines 213-398](../../specs/002-textbook-docusaurus-setup/plan.md#L213-L398)
