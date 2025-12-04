import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()

# Get database URL from environment
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")

if not NEON_DATABASE_URL:
    raise ValueError("NEON_DATABASE_URL environment variable not set")

# Create SQLAlchemy engine
engine = create_engine(
    NEON_DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,
    max_overflow=10,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency for FastAPI routes to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
