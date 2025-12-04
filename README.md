# Physical AI & Humanoid Robotics Textbook

An interactive educational platform featuring a comprehensive textbook on Physical AI and Humanoid Robotics, powered by Docusaurus 3.x with an intelligent RAG-based chatbot assistant.

## Features

- **Interactive Textbook**: 32-40 chapters organized into 3 modules (ROS 2, Humanoid Robotics, Physical AI)
- **RAG Chatbot**: Context-aware AI assistant with GPT-4o-mini for answering questions with citations
- **Authentication**: Secure signup/login with Better-Auth (optional bonus feature)
- **Personalization**: Content variants for beginner/intermediate/advanced levels (optional bonus)
- **Translation**: Urdu translation with two-tier caching (optional bonus)
- **Performance**: <3s page load, <2s chatbot response, Lighthouse scores >90

## Tech Stack

### Frontend
- **Framework**: Docusaurus 3.x (React-based static site generator)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Auth**: Better-Auth (HTTP-only cookies)

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: Neon Serverless Postgres
- **Vector DB**: Qdrant Cloud (1536-dim embeddings)
- **LLM**: OpenAI GPT-4o-mini + text-embedding-3-small

### Infrastructure
- **Hosting**: Railway (frontend + backend)
- **CI/CD**: GitHub Actions
- **Testing**: Playwright (E2E), Pytest (backend)
- **Monitoring**: Lighthouse CI

## Project Structure

```
.
├── docs/                  # Docusaurus 3.x frontend
│   ├── docs/             # Markdown content (32-40 chapters)
│   ├── src/              # React components
│   └── docusaurus.config.ts
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── main.py      # FastAPI app
│   │   ├── services/    # RAG service, chunking, embeddings
│   │   └── models/      # Database models
│   └── requirements.txt
├── tests/
│   └── e2e/             # Playwright E2E tests
├── scripts/             # Deployment and utility scripts
└── .github/workflows/   # CI/CD pipelines
```

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd physical-ai-humanoid-robotics-textbook
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys (OpenAI, Qdrant, Neon Postgres)
```

3. Install frontend dependencies:
```bash
cd docs
npm install
```

4. Install backend dependencies:
```bash
cd ../backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Development

Run frontend (Docusaurus):
```bash
cd docs
npm start
# Visit http://localhost:3000
```

Run backend (FastAPI):
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### Testing

Run E2E tests:
```bash
cd tests/e2e
npx playwright test
```

Run backend tests:
```bash
cd backend
pytest
```

Run Lighthouse CI:
```bash
cd docs
npm run build
npx lhci autorun
```

## Deployment

The application is deployed on Railway:

- **Frontend**: Static site build from `docs/build`
- **Backend**: FastAPI application on port 8000
- **Database**: Neon Serverless Postgres
- **Vector DB**: Qdrant Cloud

Deployment happens automatically via GitHub Actions on push to `main`.

## Success Criteria

- [ ] SC-001: Page load time <3 seconds (p95)
- [ ] SC-002: 32-40 chapters across 3 modules
- [ ] SC-003: Chatbot response <2 seconds (p95)
- [ ] SC-004: 95%+ citation accuracy
- [ ] SC-005: Lighthouse scores >90 (Performance, Accessibility, Best Practices, SEO)
- [ ] SC-006: Signup flow <1 minute
- [ ] SC-007: Personalization load <1 second
- [ ] SC-008: Translation <3s initial, <0.5s cached
- [ ] SC-009: 100% code validation (ESLint, Ruff, TypeScript)
- [ ] SC-010: Deployed by Nov 30, 2025 5:00 PM
- [ ] SC-011: Demo video <90 seconds
- [ ] SC-012: Git history with conventional commits

## Documentation

- **Specification**: `specs/002-textbook-docusaurus-setup/spec.md`
- **Implementation Plan**: `specs/002-textbook-docusaurus-setup/plan.md`
- **Task Breakdown**: `specs/002-textbook-docusaurus-setup/tasks.md`
- **ADRs**: `history/adr/` (0001-0008)

## License

[Add your license here]

## Contributors

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
