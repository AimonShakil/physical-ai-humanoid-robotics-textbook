# ADR-0002: Frontend Technology Stack

- **Status:** Accepted
- **Date:** 2025-11-28
- **Feature:** 002-textbook-docusaurus-setup
- **Context:** Need to build an educational textbook site with 32-40 chapters across 4 modules, supporting features like code syntax highlighting, diagrams, responsive design, and chatbot integration. Must achieve Lighthouse Performance >90, Accessibility >95 (SC-005) and page load <3s (SC-001). Hackathon timeline requires framework with minimal setup and strong documentation.

## Decision

**Use the following integrated frontend stack:**

- **Static Site Generator**: Docusaurus 3.x (React-based, MDX support)
- **UI Framework**: React 18.x (required by Docusaurus 3.x)
- **Language**: TypeScript 5.x (type safety for components, API calls)
- **Styling**: Docusaurus default theme + custom CSS (Tailwind considered but rejected to avoid build complexity)
- **Code Highlighting**: Prism (Docusaurus built-in via prism-react-renderer)
- **Diagrams**: Mermaid.js integration via `@docusaurus/plugin-mermaid`
- **Deployment**: GitHub Pages (free, integrated with GitHub Actions)
- **Build Tool**: Docusaurus CLI (abstracts Webpack/Vite complexity)

## Consequences

### Positive

- **Optimized for Documentation**: Docusaurus designed specifically for technical documentation → excellent out-of-box experience for textbook content
- **Lighthouse Performance**: Docusaurus sites typically score 95+ on Performance (static generation, code splitting, lazy loading) → exceeds SC-005 requirement
- **MDX Support**: Write content in Markdown with embedded React components → flexible chapter structure
- **Built-in Features**: Sidebar navigation, search (Algolia DocSearch), versioning, dark mode → zero configuration
- **React Component Ecosystem**: Can integrate chatbot UI, auth forms, personalization buttons as React components seamlessly
- **GitHub Pages Integration**: One-click deployment via GitHub Actions → free hosting, automatic HTTPS, CDN distribution
- **Strong Documentation**: Docusaurus has extensive guides, active community → reduces learning curve for hackathon timeline
- **SEO Optimized**: Meta tags, sitemap generation, Open Graph support → meets SC-005 SEO >90 requirement

### Negative

- **Framework Lock-in**: Tied to Docusaurus upgrade cycle, breaking changes in major versions (3.x → 4.x)
- **React Overhead**: ~40KB bundle size for React runtime (vs vanilla HTML/CSS would be <5KB) → acceptable tradeoff for component reusability
- **Limited Styling Control**: Default theme is opinionated → customization requires swizzling (ejecting theme components)
- **Static-Only**: Cannot do server-side rendering or incremental static regeneration (ISR) like Next.js → must rebuild entire site for content updates
- **Build Time**: With 32-40 chapters, build time ~2-3 minutes → slow feedback loop during content creation (mitigated with `docusaurus start` dev server)

## Alternatives Considered

### Alternative 1: Next.js 14 + App Router + Vercel

**Stack**:
- Next.js 14 (React framework with SSR/SSG)
- Tailwind CSS (utility-first styling)
- Vercel (deployment platform)
- `next-mdx-remote` for Markdown rendering

**Pros**:
- Full-stack framework (can add API routes if needed)
- Incremental static regeneration (rebuild individual pages)
- Image optimization built-in
- Excellent performance (Lighthouse >95)

**Cons**:
- **Setup Overhead**: Need to configure MDX rendering, syntax highlighting, sidebar navigation from scratch
- **No Documentation Focus**: Next.js optimized for web apps, not technical docs → missing features like built-in search, versioning
- **Configuration Complexity**: `next.config.js`, Tailwind config, MDX plugins → slower time-to-first-content
- **Overkill for Static**: Don't need SSR (all content static) → Next.js advantages wasted

**Why Rejected**: Hackathon timeline (3 days for Phase 2) requires out-of-box documentation features. Next.js setup would consume 1-2 days configuring sidebar, search, code highlighting. Docusaurus provides these for free.

### Alternative 2: VitePress + Vue 3 + Netlify

**Stack**:
- VitePress (Vue-based static site generator)
- Vue 3 (UI framework)
- Vite (build tool)
- Netlify (deployment)

**Pros**:
- Faster build times than Docusaurus (Vite is faster than Webpack)
- Smaller bundle size (~20KB Vue runtime vs 40KB React)
- Good documentation features (sidebar, search, code highlighting)

**Cons**:
- **Ecosystem Gap**: Fewer React developers on team → Vue 3 learning curve
- **Component Ecosystem**: Cannot reuse existing React chatbot components → need to rewrite in Vue
- **Less Mature**: VitePress is younger than Docusaurus → fewer community plugins, less Stack Overflow content
- **Auth Integration**: Better-Auth (bonus feature Phase 5) has React bindings, not Vue → integration friction

**Why Rejected**: Team familiarity with React ecosystem. Chatbot UI components (ChatPanel, MessageList) easier to build in React. Constitution Principle VII emphasizes leveraging existing tools (Claude Code subagents already reference React patterns).

### Alternative 3: Gatsby + GraphQL + GitHub Pages

**Stack**:
- Gatsby (React-based static site generator)
- GraphQL (data layer for content)
- `gatsby-transformer-remark` for Markdown
- GitHub Pages

**Pros**:
- React-based like Docusaurus → familiar ecosystem
- Powerful GraphQL data layer for querying chapters, modules
- Rich plugin ecosystem (gatsby-plugin-*)
- Excellent performance (static generation + code splitting)

**Cons**:
- **GraphQL Overhead**: Must define schemas for chapters, modules → unnecessary complexity for textbook
- **Configuration Heavy**: `gatsby-config.js`, `gatsby-node.js` for page generation → steeper learning curve than Docusaurus
- **Build Performance**: Gatsby builds slower than Docusaurus for large sites (32-40 pages) → 4-5 min vs 2-3 min
- **Not Documentation-Focused**: Gatsby designed for general websites → missing sidebar navigation, search out-of-box

**Why Rejected**: GraphQL adds complexity without benefit (textbook content is simple file-based Markdown, no complex data queries needed). Docusaurus provides documentation features (sidebar, search) with zero GraphQL setup.

## References

- Feature Spec: [specs/002-textbook-docusaurus-setup/spec.md](../../specs/002-textbook-docusaurus-setup/spec.md) (FR-001 through FR-010)
- Implementation Plan: [specs/002-textbook-docusaurus-setup/plan.md](../../specs/002-textbook-docusaurus-setup/plan.md#phase-2-content-architecture--mvp-chapters-days-3-5)
- Related ADRs: ADR-0001 (Monorepo Structure - docs/ folder), ADR-0008 (Testing Stack - Playwright for React components)
- Success Criteria: SC-001 (page load <3s), SC-005 (Lighthouse >90), SC-009 (code highlighting for validated examples)
- Docusaurus Docs: https://docusaurus.io/docs
