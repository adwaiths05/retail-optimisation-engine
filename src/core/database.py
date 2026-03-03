from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from src.core.config import settings

# 1. Programmatically create the Async URL for the API
# This keeps the original settings.DATABASE_URL untouched for training scripts
raw_url = settings.DATABASE_URL.replace("postgres://", "postgresql://")
ASYNC_DATABASE_URL = raw_url.replace("postgresql://", "postgresql+asyncpg://")

# 2. Create the Async Engine
engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={"ssl": True} # Neon requires SSL
)

# 3. Setup the Session Factory for FastAPI
AsyncSessionLocal = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)