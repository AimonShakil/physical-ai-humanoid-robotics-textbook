# ADR-0008: Testing and Quality Assurance Stack

- **Status:** Accepted
- **Date:** 2025-11-28
- **Feature:** 002-textbook-docusaurus-setup
- **Context:** Comprehensive testing required for hackathon submission: E2E tests for user flows, backend unit tests for RAG/auth services, performance validation (SC-005 Lighthouse >90), code validation for examples (SC-009 100% pass rate). Must support test-generator subagent integration for automated test creation. Critical decision: E2E testing framework (Playwright vs Cypress vs Selenium), backend testing approach, and performance monitoring tools.

## Decision

**Use the following testing stack:**

- **E2E Testing**: Playwright (not Cypress or Selenium)
  - Multi-browser: Chromium, Firefox, WebKit (Safari)
  - Test generation: test-generator subagent creates `.spec.ts` files
  - CI/CD: GitHub Actions runs tests on every PR
  - Coverage: Minimum 20 scenarios (5 navigation, 5 chatbot, 3 auth, 3 personalization, 3 translation, 1 smoke)

- **Backend Unit Testing**: pytest with pytest-asyncio
  - Framework: pytest 7.x (Python standard)
  - Async support: pytest-asyncio for FastAPI async endpoints
  - Mocking: pytest fixtures for Qdrant, Neon Postgres, OpenAI API
  - Coverage: `test_rag_service.py`, `test_chunking_service.py`, `test_auth_service.py`, `test_translation_service.py`

- **Performance Testing**: Lighthouse CI
  - Runs on: Homepage + 3 sample chapters (one per module)
  - Metrics: Performance >90, Accessibility >95, Best Practices >90, SEO >90 (SC-005)
  - Enforcement: CI/CD fails if any score below threshold
  - Reports: Uploaded to GitHub Actions artifacts

- **Code Validation**: code-validator subagent
  - Languages: Python, Bash, XML (ROS 2 launch files)
  - Checks: Syntax correctness, import validation, ROS 2 API compliance, no secrets
  - Requirement: 100% pass rate before deployment (SC-009)
  - Execution: Runs during Phase 2 content creation + Phase 7 final validation

- **Test Organization**:
  - E2E: `tests/e2e/tests/*.spec.ts` (repository root)
  - Backend: `backend/tests/test_*.py`
  - Fixtures: `tests/e2e/fixtures/test-queries.json` (50-query test set for SC-004)

## Consequences

### Positive

- **Multi-Browser Coverage**: Playwright tests Chromium, Firefox, WebKit → catches browser-specific bugs
- **Fast Execution**: Playwright parallelizes tests → 20 E2E scenarios run in <5 minutes
- **Auto-Waiting**: Playwright auto-waits for elements → reduces flaky tests vs manual `wait()` calls
- **Codegen Support**: Playwright codegen generates tests from browser interactions → accelerates test creation
- **CI/CD Integration**: Playwright has official GitHub Actions support → zero config CI setup
- **TypeScript Native**: Playwright `.spec.ts` files match frontend TypeScript → consistent language across project
- **Pytest Standard**: pytest is Python industry standard → extensive documentation, community support
- **Async Testing**: pytest-asyncio supports FastAPI async endpoints → accurate latency measurement
- **Lighthouse Enforcement**: CI/CD blocks deployment if performance regresses → prevents accidental degradation

### Negative

- **Playwright Learning Curve**: Team unfamiliar with Playwright API → need 2-3 hours to learn docs (vs Cypress familiarity)
- **Test Maintenance**: E2E tests brittle → UI changes require test updates (mitigated: page object pattern)
- **CI/CD Runtime**: 20 E2E tests + Lighthouse = 10-15 minutes CI time → slows PR feedback loop
- **Mocking Complexity**: Mocking Qdrant, Neon Postgres in pytest requires custom fixtures → 1-2 days setup
- **No Visual Regression**: Playwright doesn't include screenshot diffing → need manual visual QA (acceptable: hackathon scope)
- **Lighthouse Variability**: Scores vary ±5 points per run → may need retries (mitigated: run 3x, take median)

## Alternatives Considered

### Alternative 1: Cypress + Mocha + Percy

**Stack**:
- Cypress for E2E testing (JavaScript-based)
- Mocha for test runner
- Percy for visual regression testing
- pytest for backend (same as chosen stack)

**Pros**:
- **Team Familiarity**: Some team members used Cypress before → faster onboarding
- **Visual Testing**: Percy screenshot diffing catches UI regressions → better quality assurance
- **Developer Experience**: Cypress Test Runner GUI excellent for debugging → faster test development
- **Ecosystem**: More plugins available (cypress-axe for accessibility, etc.)

**Cons**:
- **Single Browser**: Cypress only tests Chromium → misses Firefox/Safari bugs (Cypress WebKit support experimental)
- **Slower Execution**: Cypress doesn't parallelize as well as Playwright → 20 tests take 10-15 minutes
- **No Auto-Waiting**: Cypress requires explicit `cy.wait()` calls → more flaky tests
- **Percy Cost**: Percy visual regression $29/month beyond free tier → exceeds hackathon budget
- **JavaScript Only**: Cypress tests in JS, frontend in TS → language inconsistency (minor)

**Why Rejected**: Single-browser limitation is deal-breaker (SC-005 requires testing across browsers). Slower execution (10-15min vs 5min) slows development iteration. Percy cost ($29/month) not justified for hackathon scope → manual visual QA acceptable. Playwright auto-waiting more reliable than Cypress explicit waits → reduces flaky test risk under deadline pressure.

### Alternative 2: Selenium WebDriver + JUnit + BrowserStack

**Stack**:
- Selenium WebDriver for E2E (supports all browsers)
- JUnit for test framework (Java-based)
- BrowserStack for cross-browser cloud testing
- pytest for backend (same as chosen stack)

**Pros**:
- **Industry Standard**: Selenium most widely used E2E tool → extensive documentation, support
- **True Cross-Browser**: Tests real Safari on macOS, Edge on Windows via BrowserStack → most accurate browser testing
- **Mature Ecosystem**: 15+ years of development → very stable, well-tested
- **Language Flexibility**: Selenium bindings for Python, Java, C# → can match backend language

**Cons**:
- **Complexity**: Selenium requires manual WebDriver setup, explicit waits, browser driver versioning → high setup overhead
- **Slow Execution**: Selenium 5-10x slower than Playwright → 20 tests take 30-60 minutes
- **Flaky Tests**: No auto-waiting → tests fail randomly due to timing issues → high maintenance burden
- **BrowserStack Cost**: $29/month beyond free tier → exceeds budget (same as Percy)
- **Java Overhead**: JUnit tests in Java vs TypeScript frontend → language mismatch, extra toolchain

**Why Rejected**: Setup complexity (WebDriver management, explicit waits) consumes 2-3 days of hackathon timeline. Execution speed (30-60min) unacceptable for rapid iteration (need fast feedback loop). Flaky test rate with Selenium significantly higher than Playwright → wastes time debugging tests instead of building features. BrowserStack cost not justified when Playwright's local WebKit/Firefox testing sufficient for hackathon.

### Alternative 3: Manual Testing Only (No E2E Framework)

**Stack**:
- Manual test scripts (Google Docs checklist)
- Lighthouse CI for performance (same as chosen stack)
- pytest for backend (same as chosen stack)
- code-validator subagent (same as chosen stack)

**Pros**:
- **Zero Setup**: No E2E framework to configure → saves 1-2 days setup time
- **No Maintenance**: No brittle tests to update when UI changes
- **Faster Development**: No time spent writing/debugging tests → more time for features
- **Flexibility**: Humans catch edge cases automated tests miss

**Cons**:
- **No Regression Protection**: Breaking changes in chatbot/auth go undetected until manual test → high bug risk before demo
- **Inconsistent Coverage**: Manual tests skip scenarios due to time pressure → gaps in testing
- **No CI/CD Validation**: Cannot block broken PRs automatically → broken code reaches main branch
- **Subagent Integration Lost**: test-generator subagent becomes useless → violates Constitution Principle VII (SDD with subagents)
- **Hackathon Scoring**: Judges value automated testing → losing potential quality points

**Why Rejected**: Hackathon scoring explicitly rewards subagent usage (50 bonus points) → test-generator subagent mandatory. Regression risk too high (5 bonus features × 4 user flows = 20 test scenarios) → manual testing would miss bugs. CI/CD automation critical for Nov 30 deadline (need confidence code works before final deployment). Time savings (1-2 days) not worth quality risk → automated tests catch bugs early, saving debugging time later.

## References

- Feature Spec: [specs/002-textbook-docusaurus-setup/spec.md](../../specs/002-textbook-docusaurus-setup/spec.md) (SC-004 50-query test set, SC-005 Lighthouse, SC-009 code validation)
- Implementation Plan: [specs/002-textbook-docusaurus-setup/plan.md](../../specs/002-textbook-docusaurus-setup/plan.md#phase-7-testing-optimization--delivery-days-16-18)
- Related ADRs: ADR-0002 (Frontend Stack - React + TypeScript match Playwright), ADR-0004 (RAG Architecture - 50-query test set for citation accuracy)
- Success Criteria: SC-004 (95% citation accuracy on 50-query set), SC-005 (Lighthouse >90), SC-009 (100% code validation pass rate)
- Playwright Docs: https://playwright.dev/
- pytest Docs: https://docs.pytest.org/
- Lighthouse CI: https://github.com/GoogleChrome/lighthouse-ci
