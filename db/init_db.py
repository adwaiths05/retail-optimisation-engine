import asyncio
import sys
import os

# Add the project root to the python path so it can find the 'src' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from src.core.database import engine
from src.database.models import Base

async def setup_database():
    """
    Initializes the Neon Postgres database:
    1. Enables the pgvector extension.
    2. Creates all relational tables (products, user_profiles, experiments).
    3. Builds the HNSW index for sub-100ms vector search.
    """
    print("🚀 Connecting to Neon via SQLAlchemy...")
    
    try:
        async with engine.begin() as conn:
            # 1. Prepare Vector Capabilities
            print("🔧 Enabling pgvector extension...")
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # 2. Sync Models to Neon
            print("🏗️ Creating tables defined in src/database/models.py...")
            # run_sync is used because Base.metadata.create_all is a synchronous SQLAlchemy method
            await conn.run_sync(Base.metadata.create_all)
            
            # 3. Optimize for Search Speed
            print("⚡ Building HNSW index on product embeddings...")
            # We use m=16 and ef_construction=64 as a balanced starting point for 50k items
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_products_vector 
                ON products USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """))
            
        print("✅ Database successfully initialized on Neon!")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(setup_database())