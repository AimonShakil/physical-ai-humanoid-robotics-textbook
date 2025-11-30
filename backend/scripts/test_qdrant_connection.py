"""Test Qdrant Cloud connection"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.qdrant_client import qdrant_client, QDRANT_COLLECTION_NAME, create_collection_if_not_exists


def test_connection():
    """Test Qdrant connection and list collections"""
    try:
        # Test connection by getting collections
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        print("✓ Connected to Qdrant Cloud successfully!")
        print(f"✓ Available collections: {collection_names}")

        # Create collection if it doesn't exist
        print(f"\nChecking collection: {QDRANT_COLLECTION_NAME}")
        create_collection_if_not_exists()

        # Verify collection exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if QDRANT_COLLECTION_NAME in collection_names:
            print(f"✓ Collection '{QDRANT_COLLECTION_NAME}' is ready!")

            # Get collection info
            collection_info = qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
            print(f"✓ Vector dimension: {collection_info.config.params.vectors.size}")
            print(f"✓ Distance metric: {collection_info.config.params.vectors.distance}")

            return True
        else:
            print(f"✗ Collection '{QDRANT_COLLECTION_NAME}' not found")
            return False

    except Exception as e:
        print(f"✗ Failed to connect to Qdrant Cloud: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
