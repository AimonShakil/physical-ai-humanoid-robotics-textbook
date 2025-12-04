"""Test Neon Postgres database connection"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.db.postgres_client import engine


def test_connection():
    """Test database connection with simple query"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            value = result.scalar()

            if value == 1:
                print("✓ Connected to Neon Postgres successfully!")
                print(f"✓ Database URL: {engine.url.host}")
                return True
            else:
                print("✗ Unexpected result from database")
                return False

    except Exception as e:
        print(f"✗ Failed to connect to Neon Postgres: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
