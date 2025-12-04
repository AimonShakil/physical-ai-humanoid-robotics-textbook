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
openai_client = None
embedding_service = None

# Initialize embedding service on startup
def initialize_services():
    global embedding_service, openai_client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "textbook_chunks")

    if all([openai_api_key, qdrant_url, qdrant_api_key]):
        # Initialize OpenAI client here, after .env is loaded
        openai_client = OpenAI(api_key=openai_api_key)
        print(f"OpenAI client initialized with API key: {openai_api_key[:20]}...")

        embedding_service = EmbeddingService(
            openai_api_key=openai_api_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=collection_name
        )
    else:
        print("Warning: Missing environment variables for embedding service")
        print(f"  OPENAI_API_KEY: {'✅' if openai_api_key else '❌'}")
        print(f"  QDRANT_URL: {'✅' if qdrant_url else '❌'}")
        print(f"  QDRANT_API_KEY: {'✅' if qdrant_api_key else '❌'}")


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
    if not embedding_service or not openai_client:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized. Check environment variables (OpenAI API key, Qdrant URL/API key)."
        )

    try:
        import time
        start_time = time.time()

        # Step 1: Retrieve relevant chunks
        embed_start = time.time()
        relevant_chunks = embedding_service.search_similar(
            query=request.query,
            limit=request.max_results,
            score_threshold=0.5
        )
        embed_time = time.time() - embed_start
        print(f"⏱️  Embedding + Search time: {embed_time:.2f}s")

        if not relevant_chunks:
            return ChatResponse(
                answer="I couldn't find relevant information in the textbook to answer your question. Could you rephrase or ask about a specific topic covered in the modules (ROS 2, Humanoid Robotics, or Physical AI)?",
                citations=[],
                context_used=0
            )

        # Step 2: Build context from chunks (limit to top 3 for faster processing)
        top_chunks = relevant_chunks[:3]  # Use only top 3 instead of 5
        context = "\n\n---\n\n".join([
            f"From {chunk['metadata']['module']} - {chunk['metadata']['chapter']} - {chunk['metadata']['section']}:\n{chunk['content'][:500]}"  # Limit content to 500 chars
            for chunk in top_chunks
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
        llm_start = time.time()
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use gpt-4o-mini for better performance and cost
            messages=messages,
            temperature=0.7,
            max_tokens=500  # Reduce from 800 to 500 for faster responses
        )
        llm_time = time.time() - llm_start
        print(f"⏱️  LLM generation time: {llm_time:.2f}s")

        answer = response.choices[0].message.content

        total_time = time.time() - start_time
        print(f"⏱️  TOTAL response time: {total_time:.2f}s")

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
        import traceback
        traceback.print_exc()
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
