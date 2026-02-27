from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from src.core.config import settings

# 1. Clean and Force the Async Driver
# We swap 'postgresql://' or 'postgres://' for 'postgresql+asyncpg://'
raw_url = settings.DATABASE_URL.replace("postgres://", "postgresql://")
ASYNC_URL = raw_url.replace("postgresql://", "postgresql+asyncpg://")

# 2. Create the Async Engine
engine = create_async_engine(
    ASYNC_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={"ssl": True} # Neon requires SSL
)

# 3. Setup the Session Factory
AsyncSessionLocal = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)