"""
Database Connection Manager.
Handles asynchronous connections to the PostgreSQL database using SQLAlchemy.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager
from config.settings import settings
from src.database.models import Base

# Create the primary async database engine
# URL is provided via settings.database_url (requires asyncpg)
engine = create_async_engine(
    settings.database_url,
    echo=False,         # Set to True for SQL logging in development
    future=True
)

# Create a session factory for generating async database sessions
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False # Prevents closing objects after commit
)

async def init_db():
    """
    Initialize the database by creating all tables defined in models.py.
    This is called during application startup.
    """
    async with engine.begin() as conn:
        # Tables are created based on the schema in Base.metadata
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """
    Standard FastAPI dependency for database sessions.
    Returns an async generator.
    """
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()

@asynccontextmanager
async def get_db_ctx():
    """
    Asynchronous context manager for internal database usage.
    Usage:
        async with get_db_ctx() as db:
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Aliases for backward compatibility
get_db_yield = get_db
get_db_session = get_db
