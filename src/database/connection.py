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

@asynccontextmanager
async def get_db():
    """
    Provides a transactional scope around a series of operations.
    Usage:
        async with get_db() as db:
            result = await db.execute(...)
    Ensures sessions are closed properly even if exceptions occur.
    """
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()


# Alias for backward compatibility with code using get_db_session
get_db_session = get_db
