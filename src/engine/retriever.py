from sqlalchemy import text
from typing import List
from src.core.database import AsyncSessionLocal

class Retriever:
    def __init__(self, top_k: int = 100):
        self.top_k = top_k

    async def get_nearest_products(self, user_embedding: List[float]):
        """
        Performs vector similarity search in Neon Postgres using the <=> operator.
        """
        async with AsyncSessionLocal() as session:
            # We fetch a larger pool (top_k) to allow the Ranker to filter/re-order
            query = text("""
                SELECT product_id, product_name, price, margin, stock,
                       (embedding <=> :user_vec) as distance
                FROM products
                ORDER BY distance ASC
                LIMIT :limit
            """)
            
            # Convert embedding list to string format for pgvector
            result = await session.execute(query, {
                "user_vec": str(user_embedding),
                "limit": self.top_k
            })
            
            return result.fetchall()