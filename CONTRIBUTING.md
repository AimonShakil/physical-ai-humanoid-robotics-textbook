# Contributing to Physical AI & Humanoid Robotics Textbook

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Git

### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/physical-ai-humanoid-robotics-textbook.git
   cd physical-ai-humanoid-robotics-textbook
   ```

3. Copy environment template:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Start development servers:
   ```bash
   ./scripts/dev.sh
   ```

## Project Structure

```
.
├── docs/                  # Docusaurus frontend
│   ├── docs/             # Textbook content (Markdown)
│   ├── src/              # React components
│   └── docusaurus.config.ts
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── api/         # API routes
│   │   ├── services/    # Business logic (RAG, embeddings)
│   │   └── db/          # Database clients
│   └── tests/           # Backend tests
├── tests/e2e/           # Playwright E2E tests
└── scripts/             # Deployment scripts
```

## Contributing Guidelines

### Content Contributions

**Adding/Editing Chapters:**

1. Textbook content is in `docs/docs/`
2. Follow naming convention: `moduleX-topic/chapterY-title.md`
3. Include frontmatter:
   ```yaml
   ---
   sidebar_position: 1
   title: Chapter Title
   ---
   ```
4. Use Docusaurus features (admonitions, code blocks, tabs)

### Code Contributions

**Branch Naming:**
- Features: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`

**Commit Messages:**
Follow Conventional Commits format:
```
<type>(<scope>): <subject>

<body>

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Code Style:**
- Frontend: ESLint + Prettier (run `npm run lint`)
- Backend: Ruff (run `ruff check .`)
- TypeScript: Strict mode enabled

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run tests:
   ```bash
   # Frontend
   cd docs && npm test

   # Backend
   cd backend && pytest
   ```
4. Run linters:
   ```bash
   # Frontend
   cd docs && npm run lint

   # Backend
   cd backend && ruff check .
   ```
5. Update documentation if needed
6. Submit PR with clear description
7. Wait for CI checks to pass
8. Address review feedback

### Testing Requirements

**All PRs must include:**
- Unit tests for new functions/classes
- Integration tests for API endpoints
- E2E tests for user-facing features (if applicable)

**Test Coverage:**
- Backend: Minimum 80% coverage
- Frontend: Minimum 70% coverage

## Content Guidelines

### Writing Style

- **Clear and Concise**: Use simple language, avoid jargon
- **Beginner-Friendly**: Assume reader has basic programming knowledge
- **Practical Examples**: Include code snippets and real-world use cases
- **Visual Aids**: Use diagrams (Mermaid), images, and videos where helpful

### Code Examples

- Always test code examples before committing
- Include complete, runnable code (not snippets without context)
- Add comments explaining complex logic
- Follow language-specific best practices

### Diagrams

Use Mermaid for diagrams:

\`\`\`mermaid
graph LR
  A[Start] --> B[Process]
  B --> C[End]
\`\`\`

## Community

### Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

### Questions?

- Open an issue for bugs or feature requests
- Start a discussion for general questions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
