"""
RAG Chat API Endpoint

Provides intelligent Q&A with citations from the textbook.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from openai import OpenAI

# Import our services
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services.embedding_service import EmbeddingService


router = APIRouter()

# Initialize services
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
embedding_service = None

# Initialize embedding service on startup
def initialize_services():
    global embedding_service
    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if all([openai_api_key, qdrant_url, qdrant_api_key]):
        embedding_service = EmbeddingService(
            openai_api_key=openai_api_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
    else:
        print("Warning: Missing environment variables for embedding service")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    max_results: int = 5


class Citation(BaseModel):
    """Citation model for sources."""
    module: str
    chapter: str
    section: str
    content_preview: str
    score: float


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    citations: List[Citation]
    context_used: int


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG-powered chat endpoint.

    Retrieves relevant textbook content and generates an answer with citations.
    """
    if not embedding_service:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized. Check environment variables."
        )

    try:
        # Step 1: Retrieve relevant chunks
        relevant_chunks = embedding_service.search_similar(
            query=request.query,
            limit=request.max_results,
            score_threshold=0.7
        )

        if not relevant_chunks:
            return ChatResponse(
                answer="I couldn't find relevant information in the textbook to answer your question. Could you rephrase or ask about a specific topic covered in the modules (ROS 2, Humanoid Robotics, or Physical AI)?",
                citations=[],
                context_used=0
            )

        # Step 2: Build context from chunks
        context = "\n\n---\n\n".join([
            f"From {chunk['metadata']['module']} - {chunk['metadata']['chapter']} - {chunk['metadata']['section']}:\n{chunk['content']}"
            for chunk in relevant_chunks
        ])

        # Step 3: Build conversation history
        messages = []
        if request.conversation_history:
            for msg in request.conversation_history[-5:]:  # Last 5 messages
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Step 4: Build system prompt
        system_prompt = f"""You are a helpful AI teaching assistant for a robotics textbook covering ROS 2, Humanoid Robotics, and Physical AI.

Use the following textbook content to answer the user's question. Always cite which module, chapter, and section your information comes from.

If the answer isn't in the provided context, say so - don't make up information.

Textbook Context:
{context}

When answering:
1. Be clear and educational
2. Include code examples when relevant
3. Cite your sources (e.g., "According to Module 1 - Chapter 3...")
4. If asked about code, provide working examples
"""

        messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request.query})

        # Step 5: Generate response with OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        answer = response.choices[0].message.content

        # Step 6: Build citations
        citations = [
            Citation(
                module=chunk['metadata']['module'],
                chapter=chunk['metadata']['chapter'],
                section=chunk['metadata']['section'],
                content_preview=chunk['content'][:200] + "...",
                score=chunk['score']
            )
            for chunk in relevant_chunks
        ]

        return ChatResponse(
            answer=answer,
            citations=citations,
            context_used=len(relevant_chunks)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/chat/health")
async def chat_health():
    """Health check for chat service."""
    return {
        "status": "healthy" if embedding_service else "degraded",
        "embedding_service": embedding_service is not None
    }


@router.get("/chat/stats")
async def chat_stats():
    """Get statistics about the vector database."""
    if not embedding_service:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized"
        )

    try:
        info = embedding_service.get_collection_info()
        return {
            "collection_info": info,
            "status": "ready"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching stats: {str(e)}"
        )
