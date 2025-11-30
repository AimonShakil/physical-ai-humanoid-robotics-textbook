# ADR-0003: Backend Hosting and Runtime

- **Status:** Accepted
- **Date:** 2025-11-28
- **Feature:** 002-textbook-docusaurus-setup
- **Context:** The RAG backend (FastAPI) must respond to chatbot queries within <2s total (SC-003), including vector search, LLM generation, and network latency. The backend handles authentication (bonus), translation (bonus), and must stay within free tier limits during hackathon development. Critical decision: hosting platform that balances cold start elimination, free tier availability, and deployment simplicity.

## Decision

**Use the following backend hosting stack:**

- **Hosting Platform**: Fly.io (not Railway, Vercel, or AWS)
  - Deployment: Fly.io VMs running Ubuntu 22.04 (Linux x86_64)
  - Free Tier: Shared CPU VM (256MB RAM), sufficient for RAG API
  - Cold Start: Zero cold starts (persistent VM, always warm)

- **Runtime**: Python 3.11+ with FastAPI
  - ASGI Server: Uvicorn with asyncio event loop
  - Containerization: Docker multi-stage build (Python 3.11-slim base)

- **Database Connections**:
  - Qdrant Cloud Free Tier (vector database, 1GB storage)
  - Neon Serverless Postgres Free Tier (metadata, chat history, sessions)

- **Deployment Pipeline**: GitHub Actions
  - Trigger: Push to main branch on backend/ path changes
  - Process: Build Docker image → Push to Fly.io registry → Deploy to Fly.io VM
  - Environment Variables: OPENAI_API_KEY, QDRANT_URL, NEON_DATABASE_URL injected via Fly.io secrets

## Consequences

### Positive

- **Zero Cold Starts**: Persistent VM stays warm → no 5-10s Lambda cold start penalty → helps meet SC-003 <2s requirement
- **Predictable Latency**: FastAPI on dedicated VM → consistent 500-800ms response times (vs serverless 500-5000ms variability)
- **Free Tier Availability**: Fly.io shared CPU VM free forever → zero ongoing costs during hackathon development
- **Simple Deployment**: `flyctl deploy` or GitHub Actions → single command deployment, no complex infrastructure
- **Full Control**: Can install system dependencies, run background workers, persistent storage → flexibility for future features
- **Docker-Based**: Dockerfile ensures dev/prod parity → "works on my machine" issues eliminated

### Negative

- **Manual Scaling**: Fly.io free tier = 1 VM → cannot auto-scale under load (mitigated: hackathon demo has <10 concurrent users)
- **VM Management**: Need to monitor uptime, restart if crashes (vs serverless auto-restart) → requires monitoring setup
- **Resource Limits**: 256MB RAM cap on free tier → need efficient memory usage (FastAPI + Qdrant client + OpenAI SDK ~150MB)
- **Regional Latency**: Fly.io VM location impacts response time → must choose region near judges (mitigated: deploy to US East)
- **No Built-in CDN**: Static assets served from VM, not CDN (vs Vercel) → not an issue since frontend on GitHub Pages CDN
- **Lock-in Risk**: Fly.io-specific config (`fly.toml`) → migration to Railway/Render requires reconfig (low risk for hackathon)

## Alternatives Considered

### Alternative 1: Railway.app + Python 3.11 + Docker

**Stack**:
- Railway.app (platform-as-a-service similar to Fly.io)
- Python 3.11 FastAPI
- Docker deployment
- PostgreSQL addon (Railway-managed)

**Pros**:
- **Free Tier**: $5 credit/month → sufficient for hackathon (500 hours runtime)
- **Zero Cold Starts**: Persistent VM like Fly.io
- **Simpler Database**: Railway-managed Postgres → one less service to configure vs Neon
- **Better Dashboard**: Railway UI more intuitive than Fly.io CLI

**Cons**:
- **Credit Expiry**: $5 credit expires after trial period → not "free forever" like Fly.io
- **Less Community Support**: Fly.io has larger user base → more Stack Overflow answers, better docs
- **No Qdrant Integration**: Would need external Qdrant Cloud anyway → no advantage over Fly.io
- **Heavier Resource Usage**: Railway default config uses 512MB RAM → overkill for our API, wastes resources

**Why Rejected**: Fly.io's true free tier (no credit expiry) better for long-term demo preservation after hackathon. Railway advantage (managed Postgres) not useful since we need Neon for specific features (serverless, generous free tier). Fly.io community support stronger for troubleshooting under deadline pressure.

### Alternative 2: Vercel Serverless Functions + Python Runtime

**Stack**:
- Vercel Serverless Functions (AWS Lambda under the hood)
- Python 3.11 runtime
- Edge deployment (multi-region)
- Vercel KV (Redis) for caching

**Pros**:
- **Auto-Scaling**: Handles traffic spikes automatically → better than single VM
- **Zero Config**: Vercel detects FastAPI automatically → fastest deployment
- **Edge Network**: Multi-region deployment → lower latency for global users
- **Integrated Frontend**: Could host Docusaurus on Vercel too → single platform

**Cons**:
- **Cold Starts**: 1-3s cold start for Python runtime → violates SC-003 <2s requirement (500ms vector + 800ms LLM + 3000ms cold = 4.3s)
- **Function Timeout**: 10s max on free tier → risk timeout for complex RAG queries
- **Limited State**: Serverless stateless → cannot keep Qdrant connection pool warm → every request pays connection overhead
- **Higher Costs**: Vercel charges for function invocations beyond free tier (100K/month) → risk exceeding limit during testing
- **No Persistent Workers**: Cannot run background jobs (e.g., periodic index updates) → would need separate cron service

**Why Rejected**: Cold start latency is deal-breaker for SC-003. RAG pipeline (retrieve + generate) already uses ~1.5s → cannot afford 1-3s cold start overhead. Serverless benefits (auto-scaling, edge) not needed for hackathon demo with <10 concurrent users. Fly.io persistent VM better fit for latency-sensitive RAG workload.

### Alternative 3: AWS EC2 t2.micro + FastAPI + Docker

**Stack**:
- AWS EC2 t2.micro (1 vCPU, 1GB RAM, free tier)
- Ubuntu 22.04 LTS
- FastAPI with Uvicorn
- Docker Compose for service orchestration
- Manual Nginx reverse proxy

**Pros**:
- **Most Control**: Full root access, install anything → maximum flexibility
- **Generous Free Tier**: 750 hours/month for 12 months → more than Fly.io's 256MB
- **Industry Standard**: AWS skills transferable, widely used in production
- **Better Resources**: 1GB RAM vs Fly.io 256MB → can run heavier workloads

**Cons**:
- **Manual Infrastructure**: Need to configure security groups, SSH keys, Nginx, SSL certificates → 1-2 days setup overhead
- **No Auto-Deploy**: GitHub Actions → EC2 requires SSH deploy scripts, not one-click like Fly.io
- **Maintenance Burden**: Must apply Ubuntu security patches, monitor disk usage, restart crashed services manually
- **Slower Iteration**: Deploy takes 5-10 minutes (SSH upload, Docker rebuild) vs Fly.io 2 minutes
- **Network Config**: Need elastic IP, load balancer setup → complexity not justified for single-user demo
- **Free Tier Expiry**: 12 months only → demo broken after trial period

**Why Rejected**: Setup complexity consumes 1-2 days of hackathon timeline (Nov 30 deadline) → unacceptable. Fly.io provides 90% of EC2 benefits with 10% of setup time. Manual infrastructure management increases bug risk under deadline pressure. AWS free tier expiry (12 months) vs Fly.io permanent free tier means demo broken in 2026.

## References

- Feature Spec: [specs/002-textbook-docusaurus-setup/spec.md](../../specs/002-textbook-docusaurus-setup/spec.md) (SC-003 <2s response requirement)
- Implementation Plan: [specs/002-textbook-docusaurus-setup/plan.md](../../specs/002-textbook-docusaurus-setup/plan.md#phase-3-rag-backend-development-days-6-8)
- Related ADRs: ADR-0004 (RAG Architecture - backend serves Qdrant queries), ADR-0005 (Authentication - backend manages sessions)
- Success Criteria: SC-003 (<2s RAG response), SC-010 (public deployment URL)
- Fly.io Docs: https://fly.io/docs/python/
- Railway Comparison: https://railway.app/pricing
- Vercel Serverless Limits: https://vercel.com/docs/functions/serverless-functions/runtimes/python
