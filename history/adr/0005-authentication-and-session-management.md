# ADR-0005: Authentication and Session Management

- **Status:** Accepted
- **Date:** 2025-11-28
- **Feature:** 002-textbook-docusaurus-setup
- **Context:** Bonus feature (Phase 5) requires user authentication for content personalization and usage tracking. Must achieve <1min signup flow (SC-006), persist sessions across page reloads (FR-032), and protect against XSS/CSRF attacks (Assumption #16). Critical decision: session token storage mechanism (HTTP-only cookies vs localStorage), authentication library integration, and JWT management strategy.

## Decision

**Use the following authentication stack:**

- **Authentication Library**: Better-Auth (TypeScript-first auth library)
  - Handles password hashing (bcrypt), JWT generation, CSRF tokens
  - Integrates with FastAPI backend + React frontend

- **Session Storage**: HTTP-only cookies (not localStorage or sessionStorage)
  - Cookie Flags: `secure`, `httpOnly`, `sameSite=strict`
  - Cookie Name: `textbook_session`
  - Expiry: 7 days (auto-refreshed on each API call if >5 days old)

- **Session Token**: JWT (JSON Web Token)
  - Payload: `{user_id, email, created_at, expires_at}`
  - Signature: HMAC-SHA256 with secret key (stored in Fly.io secrets)
  - Server-Side Storage: Neon Postgres `sessions` table (user_id, token_hash, expires_at)

- **Authentication Flow**:
  1. Signup: POST /api/signup → Validate email/password → Create user in Postgres → Issue JWT → Set HTTP-only cookie
  2. Signin: POST /api/signin → Verify password → Issue JWT → Set HTTP-only cookie
  3. API Requests: Browser auto-sends cookie → FastAPI validates JWT → Allow/deny request
  4. Signout: POST /api/signout → Delete session from Postgres → Clear cookie

- **CSRF Protection**: CSRF tokens for state-changing operations
  - Signup/profile updates: Require `X-CSRF-Token` header (generated on page load)
  - Read-only operations (chat, translate): No CSRF check (authenticated by cookie only)

- **Password Requirements** (FR-030):
  - Minimum 8 characters
  - Must contain: 1 uppercase, 1 lowercase, 1 number
  - Hashed with bcrypt (cost factor 12)

## Consequences

### Positive

- **XSS Protection**: HTTP-only cookies inaccessible to JavaScript → prevents session theft via XSS attacks
- **CSRF Protection**: CSRF tokens + `sameSite=strict` → prevents cross-site request forgery
- **Persistent Sessions**: 7-day expiry → users stay logged in across browser restarts → meets FR-032
- **Fast Signup**: Better-Auth handles complexity → <1min signup flow → meets SC-006
- **Server-Side Validation**: JWT signature verified by FastAPI → cannot be forged by client
- **Automatic Refresh**: Token auto-refreshes at >5 days → seamless UX, no sudden logouts
- **Integration Ready**: Better-Auth has React hooks (`useAuth()`) and FastAPI middleware → minimal integration code

### Negative

- **Cookie Size Limit**: JWT stored in cookie → max 4KB → limits session data (mitigated: only store user_id, not profile data)
- **Better-Auth Learning Curve**: Team unfamiliar with library → need 2-3 hours to learn docs
- **Server-Side Session Storage**: Must store session in Postgres → database dependency for auth (vs stateless JWT)
- **CORS Complexity**: Cookies require `credentials: 'include'` in fetch calls + CORS `Access-Control-Allow-Credentials` → easy to misconfigure
- **Mobile Browser Issues**: Some mobile browsers block third-party cookies → not an issue (same-origin cookies, `sameSite=strict`)
- **No localStorage Fallback**: If cookies disabled, auth fails → acceptable (cookies standard for secure auth)

## Alternatives Considered

### Alternative 1: localStorage for JWT Tokens

**Stack**:
- Better-Auth for signup/signin
- JWT token stored in `localStorage.setItem('token', jwt)`
- Manual token inclusion: `Authorization: Bearer ${token}` header on each API call

**Pros**:
- **Simpler CORS**: No `credentials: 'include'` complexity → easier cross-origin setup
- **Larger Storage**: localStorage 5-10MB limit vs cookies 4KB → can store more data
- **No Cookie Flags**: No `secure`, `httpOnly`, `sameSite` configuration needed → faster setup
- **Explicit Control**: Developer controls token sending → more predictable than auto-sent cookies

**Cons**:
- **XSS Vulnerability**: JavaScript can read localStorage → XSS attack steals token → critical security risk (FR-031 requires secure session management)
- **CSRF Still Needed**: Must implement CSRF tokens anyway for state-changing operations → no simplification
- **Manual Refresh**: Must write custom token refresh logic → Better-Auth auto-refresh only works with cookies
- **Browser Security Features**: Modern browsers recommend HTTP-only cookies for auth → going against best practices

**Why Rejected**: XSS vulnerability is deal-breaker. FR-031 explicitly requires "secure session management with HTTP-only cookies" → localStorage violates requirement. Modern auth best practices (OWASP, Auth0 recommendations) strongly favor HTTP-only cookies for tokens. Risk of hackathon demo being flagged for security flaw outweighs CORS simplicity.

### Alternative 2: sessionStorage for JWT Tokens

**Stack**:
- Better-Auth for signup/signin
- JWT in `sessionStorage.setItem('token', jwt)`
- Tokens cleared on tab close → no persistence across page reloads

**Pros**:
- **Tab Isolation**: Each tab has separate session → prevents cross-tab attacks
- **Auto-Cleanup**: Session cleared on tab close → reduces stale session risk
- **Same Storage API**: Similar to localStorage → familiar developer experience

**Cons**:
- **No Persistence**: Violates FR-032 (persist state across page reloads) → users logged out every tab close → terrible UX
- **Same XSS Vulnerability**: JavaScript can read sessionStorage → same security risk as localStorage
- **More Logins**: Users must re-login on every browser session → increases friction, violates SC-006 <1min signup goal (need to signup/signin repeatedly)
- **Better-Auth Incompatibility**: Better-Auth designed for persistent cookies, not ephemeral sessionStorage → would need custom session management

**Why Rejected**: Fails FR-032 requirement (persistent authenticated state). Ephemeral sessions acceptable for high-security apps (banking), but hackathon textbook needs frictionless UX. Same XSS vulnerability as localStorage without persistence benefit.

### Alternative 3: Server-Side Sessions with Session IDs (No JWT)

**Stack**:
- Better-Auth with server-side sessions (no JWT)
- Session ID stored in HTTP-only cookie (random UUID)
- Session data (user_id, email, backgrounds) stored entirely in Postgres `sessions` table
- Each API call queries Postgres to load session data

**Pros**:
- **Smallest Cookie**: Only session ID in cookie (~36 bytes) vs JWT (~200-300 bytes) → more room for other cookies
- **Instant Revocation**: Delete session from Postgres → immediately invalidates across all requests → better for security incidents
- **Flexible Session Data**: Can store unlimited data in Postgres (not constrained by 4KB cookie limit) → useful for complex profiles
- **Simpler Validation**: No JWT signature verification → just check session ID exists and not expired

**Cons**:
- **Database Hit on Every Request**: Every /api/chat call queries Postgres → adds 20-50ms latency → threatens SC-003 <2s requirement (already tight with 500ms vector + 800ms LLM)
- **Scaling Challenge**: More users → more session queries → Postgres becomes bottleneck (mitigated: hackathon has <10 users, but bad practice)
- **Better-Auth JWT Features Lost**: Better-Auth designed for JWT → lose automatic token refresh, signature validation, expiry handling
- **Caching Complexity**: Would need Redis to cache sessions → additional service dependency (Qdrant + Neon + Redis = 3 databases)

**Why Rejected**: Database query on every request adds unacceptable latency for latency-sensitive RAG workload. JWT validation is CPU-bound (no I/O) → 0-1ms overhead vs 20-50ms Postgres query. Better-Auth JWT features (auto-refresh, signature validation) too valuable to lose. Instant revocation not needed for hackathon (no security incident expected). KISS principle: JWT is industry standard for stateless auth.

## References

- Feature Spec: [specs/002-textbook-docusaurus-setup/spec.md](../../specs/002-textbook-docusaurus-setup/spec.md) (FR-028 through FR-034, Assumption #16)
- Implementation Plan: [specs/002-textbook-docusaurus-setup/plan.md](../../specs/002-textbook-docusaurus-setup/plan.md#phase-5-authentication-bonus-days-11-12)
- Related ADRs: ADR-0003 (Backend Hosting - FastAPI session validation), ADR-0006 (Personalization - uses background scores from user profile)
- Success Criteria: SC-006 (<1min signup flow)
- Better-Auth Docs: https://www.better-auth.com/docs
- OWASP Session Management: https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html
