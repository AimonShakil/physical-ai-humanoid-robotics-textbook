"""
Semantic Cache Service for RAG Chat

Provides intelligent caching based on semantic similarity to reduce OpenAI API calls
and improve response times.
"""

import time
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from openai import OpenAI
import numpy as np


class CacheEntry:
    """Represents a cached query-response pair with metadata."""

    def __init__(
        self,
        query: str,
        query_embedding: List[float],
        response: str,
        citations: List[Dict[str, Any]],
        context_used: int,
        ttl_seconds: int = 86400  # 24 hours default
    ):
        self.query = query
        self.query_embedding = np.array(query_embedding)
        self.response = response
        self.citations = citations
        self.context_used = context_used
        self.created_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        self.hit_count = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.expires_at

    def cosine_similarity(self, other_embedding: np.ndarray) -> float:
        """Calculate cosine similarity with another embedding."""
        # Normalize vectors
        norm_a = np.linalg.norm(self.query_embedding)
        norm_b = np.linalg.norm(other_embedding)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Compute cosine similarity
        return np.dot(self.query_embedding, other_embedding) / (norm_a * norm_b)


class SemanticCacheService:
    """
    Semantic caching service using embedding similarity.

    Reduces OpenAI API calls by caching responses and matching semantically similar queries.
    """

    def __init__(
        self,
        openai_api_key: str,
        similarity_threshold: float = 0.85,
        ttl_seconds: int = 86400
    ):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.cache: List[CacheEntry] = []

        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text query."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        self.cache = [entry for entry in self.cache if not entry.is_expired()]

    async def get_cached_response(
        self,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check cache for semantically similar queries.

        Returns cached response if similarity >= threshold, otherwise None.
        """
        self.total_queries += 1

        # Clean up expired entries
        self._cleanup_expired()

        if not self.cache:
            self.cache_misses += 1
            return None

        # Get query embedding
        start_time = time.time()
        query_embedding = np.array(self._get_embedding(query))
        embed_time = time.time() - start_time
        print(f"🔍 Cache lookup embedding time: {embed_time:.2f}s")

        # Find most similar cached entry
        best_match = None
        best_similarity = 0.0

        for entry in self.cache:
            similarity = entry.cosine_similarity(query_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        # Check if best match exceeds threshold
        if best_match and best_similarity >= self.similarity_threshold:
            self.cache_hits += 1
            best_match.hit_count += 1
            print(f"✅ Cache HIT! Similarity: {best_similarity:.3f} (threshold: {self.similarity_threshold})")
            print(f"   Cached query: '{best_match.query}'")
            print(f"   Current query: '{query}'")

            return {
                "response": best_match.response,
                "citations": best_match.citations,
                "context_used": best_match.context_used,
                "from_cache": True,
                "cache_similarity": best_similarity,
                "cached_query": best_match.query
            }
        else:
            self.cache_misses += 1
            print(f"❌ Cache MISS. Best similarity: {best_similarity:.3f} (threshold: {self.similarity_threshold})")
            return None

    async def cache_response(
        self,
        query: str,
        response: str,
        citations: List[Dict[str, Any]],
        context_used: int
    ):
        """Cache a query-response pair with semantic embedding."""
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)

            # Create cache entry
            entry = CacheEntry(
                query=query,
                query_embedding=query_embedding,
                response=response,
                citations=citations,
                context_used=context_used,
                ttl_seconds=self.ttl_seconds
            )

            # Add to cache
            self.cache.append(entry)
            print(f"💾 Cached response for query: '{query[:50]}...'")
            print(f"   Cache size: {len(self.cache)} entries")

        except Exception as e:
            print(f"⚠️  Failed to cache response: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0.0

        # Estimate cost savings (approximate)
        # Each cache hit saves ~$0.002 (1 embedding + 1 LLM call)
        cost_per_query = 0.002
        savings_usd = self.cache_hits * cost_per_query

        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self.cache),
            "cost_savings_usd": round(savings_usd, 4),
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds
        }

    def clear_cache(self):
        """Clear all cache entries."""
        self.cache = []
        print("🗑️  Cache cleared")
