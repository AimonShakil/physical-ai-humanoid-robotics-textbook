# Backend Setup Guide

## Prerequisites

- Python 3.11+
- pip

## Database Setup

### Neon Serverless Postgres

1. Go to https://neon.tech
2. Create a free account
3. Create a new project: "Physical AI Textbook"
4. Copy the connection string (format: `postgresql://user:password@host/db`)
5. Add to `.env`:
   ```
   NEON_DATABASE_URL=postgresql://user:password@ep-xxxxx.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

### Qdrant Cloud

1. Go to https://cloud.qdrant.io
2. Create a free account
3. Create a cluster: "textbook-vectors" (Free tier: 1GB)
4. Copy cluster URL and API key
5. Add to `.env`:
   ```
   QDRANT_URL=https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east.aws.cloud.qdrant.io:6333
   QDRANT_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   QDRANT_COLLECTION_NAME=textbook_chunks
   ```

### OpenAI API

1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add to `.env`:
   ```
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

## Installation

1. Create virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Test database connections:
   ```bash
   python scripts/test_db_connection.py
   python scripts/test_qdrant_connection.py
   ```

4. Run development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. Verify health endpoint:
   ```bash
   curl http://localhost:8000/health
   # Expected: {"status":"ok"}
   ```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
