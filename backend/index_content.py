"""
Index textbook content into Qdrant
"""
import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 60)
print("Starting textbook indexing process...")
print("=" * 60)

# Check environment variables
print("\n1. Checking environment variables...")
openai_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

if not all([openai_key, qdrant_url, qdrant_key]):
    print("❌ Missing environment variables!")
    print(f"  OPENAI_API_KEY: {'✅' if openai_key else '❌'}")
    print(f"  QDRANT_URL: {'✅' if qdrant_url else '❌'}")
    print(f"  QDRANT_API_KEY: {'✅' if qdrant_key else '❌'}")
    sys.exit(1)

print("✅ All environment variables found")

# Import services
print("\n2. Importing services...")
from app.services.document_chunker import DocumentChunker
from app.services.embedding_service import EmbeddingService
print("✅ Services imported")

# Initialize services
print("\n3. Initializing embedding service...")
service = EmbeddingService(
    openai_api_key=openai_key,
    qdrant_url=qdrant_url,
    qdrant_api_key=qdrant_key,
    collection_name="textbook_chunks"
)
print("✅ Embedding service initialized")

# Create collection
print("\n4. Creating Qdrant collection...")
try:
    service.initialize_collection()
    print("✅ Collection ready")
except Exception as e:
    print(f"❌ Error creating collection: {e}")
    sys.exit(1)

# Chunk documents
print("\n5. Chunking textbook documents...")
docs_path = "../docs/docs"
chunker = DocumentChunker(docs_path, max_chunk_size=1000)
chunks = chunker.chunk_documents()
print(f"✅ Created {len(chunks)} chunks")

# Show sample
if chunks:
    print(f"\n   Sample chunk:")
    print(f"   - Module: {chunks[0].module}")
    print(f"   - Chapter: {chunks[0].chapter}")
    print(f"   - Section: {chunks[0].section}")
    print(f"   - Content length: {len(chunks[0].content)} chars")

# Generate embeddings and store
print("\n6. Generating embeddings and uploading to Qdrant...")
print(f"   This will take ~{len(chunks) * 2} seconds (2s per chunk)")
print("   Progress:")

try:
    service.store_chunks(chunks)
    print("\n✅ All chunks indexed successfully!")
except Exception as e:
    print(f"\n❌ Error during indexing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get collection info
print("\n7. Verifying indexed data...")
try:
    info = service.get_collection_info()
    print(f"✅ Collection info:")
    print(f"   - Points: {info.get('points_count', 'N/A')}")
    print(f"   - Vectors: {info.get('vectors_count', 'N/A')}")
except Exception as e:
    print(f"⚠ Could not fetch collection info: {e}")

print("\n" + "=" * 60)
print("✅ Indexing complete!")
print("=" * 60)
