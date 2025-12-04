from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router, initialize_services
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Physical AI & Humanoid Robotics API",
    description="Backend API for RAG chatbot and content management",
    version="1.0.0",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Docusaurus dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api", tags=["chat"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    initialize_services()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Physical AI & Humanoid Robotics API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "chat_health": "/api/chat/health",
            "chat_stats": "/api/chat/stats"
        }
    }
