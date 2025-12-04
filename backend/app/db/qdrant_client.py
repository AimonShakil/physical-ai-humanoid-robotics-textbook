import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

# Get Qdrant credentials from environment
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "textbook_chunks")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")

# Initialize Qdrant client with increased timeout
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0,  # Increase timeout to 60 seconds
    https=True,
    prefer_grpc=False  # Use REST API instead of gRPC for better reliability
)


def create_collection_if_not_exists():
    """Create Qdrant collection for textbook chunks if it doesn't exist"""
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]

    if QDRANT_COLLECTION_NAME not in collection_names:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1536,  # text-embedding-3-small dimension
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {QDRANT_COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {QDRANT_COLLECTION_NAME}")


def get_qdrant_client():
    """Get Qdrant client instance"""
    return qdrant_client
