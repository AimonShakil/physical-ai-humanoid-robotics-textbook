"""
Entry point for Railway deployment.
This file imports the FastAPI app from the nested app directory.
"""
from app.main import app

# Railway will auto-detect this and start with uvicorn
__all__ = ["app"]
