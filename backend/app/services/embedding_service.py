"""
Embedding Service for RAG

Generates OpenAI embeddings and stores them in Qdrant vector database.
"""

import os
from typing import List, Dict, Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


class EmbeddingService:
    """Generates and stores embeddings for textbook content."""

    def __init__(
        self,
        openai_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "textbook_chunks"
    ):
        """
        Initialize embedding service.

        Args:
            openai_api_key: OpenAI API key
            qdrant_url: Qdrant cluster URL
            qdrant_api_key: Qdrant API key
            collection_name: Name of Qdrant collection
        """
        print("EmbeddingService: Initializing OpenAI client...")
        self.openai_client = OpenAI(api_key=openai_api_key)
        print("EmbeddingService: OpenAI client initialized.")
        print("EmbeddingService: Initializing Qdrant client...")
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60.0,  # Increase timeout to 60 seconds
            https=True,
            prefer_grpc=False  # Use REST API instead of gRPC for better reliability
        )
        print("EmbeddingService: Qdrant client initialized.")
        self.collection_name = collection_name
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536  # text-embedding-3-small dimension

    def initialize_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def store_chunks(self, chunks: List[Any]):
        """
        Generate embeddings for chunks and store in Qdrant.

        Args:
            chunks: List of DocumentChunk objects
        """
        points = []

        for chunk in chunks:
            # Generate embedding
            embedding = self.generate_embedding(chunk.content)

            # Create Qdrant point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=chunk.to_dict()
            )
            points.append(point)

            # Batch upload every 100 points
            if len(points) >= 100:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Uploaded {len(points)} points to Qdrant")
                points = []

        # Upload remaining points
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Uploaded {len(points)} points to Qdrant")

    def search_similar(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic search.

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of similar chunks with scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )

        # Format results
        results = []
        for hit in search_results:
            results.append({
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"],
                "score": hit.score
            })

        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        collection_info = self.qdrant_client.get_collection(
            collection_name=self.collection_name
        )
        return {
            "name": collection_info.config.params,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count
        }


def main():
    """Test the embedding service."""
    import sys
    from app.services.document_chunker import DocumentChunker

    if len(sys.argv) < 2:
        print("Usage: python embedding_service.py <docs_path>")
        sys.exit(1)

    # Load environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not all([openai_api_key, qdrant_url, qdrant_api_key]):
        print("Error: Missing environment variables")
        print("Required: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY")
        sys.exit(1)

    # Initialize service
    service = EmbeddingService(
        openai_api_key=openai_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )

    # Initialize collection
    service.initialize_collection()

    # Chunk documents
    docs_path = sys.argv[1]
    chunker = DocumentChunker(docs_path)
    chunks = chunker.chunk_documents()
    print(f"Created {len(chunks)} chunks")

    # Store chunks (this will take some time due to API rate limits)
    print("Generating embeddings and storing in Qdrant...")
    service.store_chunks(chunks)

    # Get collection info
    info = service.get_collection_info()
    print(f"\nCollection info: {info}")

    # Test search
    query = "How do I create a ROS 2 node?"
    print(f"\nTesting search with query: '{query}'")
    results = service.search_similar(query, limit=3)
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {result['score']:.3f}) ---")
        print(f"Module: {result['metadata']['module']}")
        print(f"Chapter: {result['metadata']['chapter']}")
        print(f"Section: {result['metadata']['section']}")
        print(f"Content: {result['content'][:200]}...")


if __name__ == "__main__":
    main()
