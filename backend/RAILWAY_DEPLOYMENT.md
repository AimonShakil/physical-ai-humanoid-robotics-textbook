# Railway Backend Deployment Guide

## Overview
This guide walks you through deploying the FastAPI backend to Railway.

## Prerequisites
- Railway account (sign up at https://railway.app)
- GitHub account with this repository
- Required API keys:
  - OpenAI API key
  - Qdrant Cloud URL and API key

## Deployment Steps

### Step 1: Create New Railway Project
1. Go to https://railway.app and sign in with GitHub
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository: `physical-ai-humanoid-robotics-textbook`
5. Railway will detect the project structure

### Step 2: Configure Root Directory (IMPORTANT!)
Railway needs to know to deploy from the `backend` folder:

1. In your Railway project dashboard, click on your service
2. Go to **Settings** tab
3. Scroll to **"Root Directory"**
4. Set it to: `backend`
5. Click **"Save"**

### Step 3: Add Environment Variables
In the Railway dashboard, go to **Variables** tab and add:

```bash
# Required Environment Variables
OPENAI_API_KEY=sk-proj-...
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=physical_ai_textbook

# Optional (for future features)
DATABASE_URL=postgresql://user:password@host:port/database
CONTEXT7_API_KEY=your-context7-key
```

**Important**: Copy these from your local `backend/.env` file.

### Step 4: Verify Build Configuration
Railway will automatically detect:
- **Language**: Python 3.11
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

These are configured in `backend/railway.json` and `backend/Procfile`.

### Step 5: Deploy
1. Click **"Deploy"** in Railway
2. Wait for build to complete (2-3 minutes)
3. Railway will show build logs in real-time
4. Once deployed, you'll see a green ✅ status

### Step 6: Get Your Deployment URL
1. In Railway dashboard, go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Copy your URL (e.g., `https://your-backend.up.railway.app`)

### Step 7: Update Frontend Configuration
Update your frontend to use the Railway backend:

```bash
# In docs/.env.local (create if doesn't exist)
REACT_APP_BACKEND_URL=https://your-backend.up.railway.app
```

Then rebuild and redeploy your frontend:
```bash
cd docs
npm run build
git add .
git commit -m "feat: update backend URL for Railway deployment"
git push origin main
```

## Testing Your Deployment

### Test Health Endpoint
```bash
curl https://your-backend.up.railway.app/health
# Expected: {"status":"ok"}
```

### Test Chat Endpoint
```bash
curl -X POST https://your-backend.up.railway.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ROS 2?"}'
```

Expected response with answer and citations.

## Troubleshooting

### Error: "Application failed to respond"
**Cause**: Railway can't find the start command
**Fix**: Verify `backend/Procfile` exists and contains:
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Error: "Module 'app' not found"
**Cause**: Railway root directory not set correctly
**Fix**: Set **Root Directory** to `backend` in Railway Settings

### Error: "Connection timeout" or "500 Internal Server Error"
**Cause**: Missing or incorrect environment variables
**Fix**:
1. Check all required env vars are set in Railway
2. Verify Qdrant URL and API key are correct
3. Test Qdrant connection: `curl https://your-qdrant-url:6333/collections`

### Error: "CORS policy blocked"
**Cause**: Frontend URL not in CORS allowed origins
**Fix**: The backend now allows `https://aimonshakil.github.io` - verify in `backend/app/main.py`:
```python
allow_origins=[
    "http://localhost:3000",
    "https://aimonshakil.github.io",
]
```

### Checking Logs
View real-time logs in Railway:
1. Go to your service in Railway dashboard
2. Click **"Deployments"** tab
3. Click on latest deployment
4. View build and runtime logs

## Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for embeddings and chat | `sk-proj-...` |
| `QDRANT_URL` | ✅ Yes | Qdrant Cloud cluster URL | `https://xxx.cloud.qdrant.io` |
| `QDRANT_API_KEY` | ✅ Yes | Qdrant API key | `your-key-here` |
| `QDRANT_COLLECTION_NAME` | ✅ Yes | Collection name in Qdrant | `physical_ai_textbook` |
| `DATABASE_URL` | ❌ No | PostgreSQL connection (future) | `postgresql://...` |
| `CONTEXT7_API_KEY` | ❌ No | Context7 caching (optional) | `ctx7_...` |

## Performance Expectations

After deployment:
- **Cold Start**: 2-5 seconds (first request after idle)
- **Warm Response**: 8-10 seconds (OpenAI API latency)
- **Health Check**: <100ms

## Cost Estimates

Railway free tier includes:
- $5 free credits per month
- Unlimited projects
- 512MB RAM, 1 vCPU per service

Expected monthly cost:
- **Railway**: $0-5 (within free tier for hobby projects)
- **OpenAI API**: ~$2-10 (depends on usage)
- **Qdrant Cloud**: Free tier (1GB storage)

## Next Steps

1. ✅ Deploy backend to Railway
2. Update frontend environment variable
3. Test chatbot on GitHub Pages
4. Monitor Railway logs for errors
5. Consider adding Context7 MCP for caching (70% cost reduction)

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Project Issues: https://github.com/AimonShakil/physical-ai-humanoid-robotics-textbook/issues

---

**Generated with**: [Claude Code](https://claude.com/claude-code)
**Last Updated**: December 4, 2025
