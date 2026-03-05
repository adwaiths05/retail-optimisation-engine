import redis.asyncio as redis
import json
import logging
from src.core.config import settings

logger = logging.getLogger("session-manager")

class SessionManager:
    def __init__(self):
        # Create an async connection pool
        self.redis = redis.from_url(
            settings.REDIS_URL, 
            decode_responses=True, 
            encoding="utf-8"
        )
        self.experiments = ["control", "margin_boost"]

    async def get_user_group(self, user_id: int) -> str:
        """Step 11: Get or set the A/B test group for a user."""
        key = f"user:{user_id}:group"
        group = await self.redis.get(key)
        
        if not group:
            # Simple deterministic assignment (even/odd)
            group = self.experiments[user_id % len(self.experiments)]
            # Store it for 7 days so the user experience is consistent
            await self.redis.setex(key, 604800, group)
        return group

    async def get_cached_embedding(self, user_id: int):
        """Step 10: Retrieve embedding from cache."""
        key = f"user:{user_id}:embedding"
        data = await self.redis.get(key)
        if data:
            logger.info(f"⚡ Cache Hit for User {user_id}")
            return json.loads(data)
        return None

    async def cache_embedding(self, user_id: int, embedding: list):
        """Step 10: Store embedding in cache for 10 minutes."""
        key = f"user:{user_id}:embedding"
        # 600 seconds = 10 minutes (good for active shopping sessions)
        await self.redis.setex(key, 600, json.dumps(embedding))

    def get_ranking_weights(self, group: str):
        """Returns weights based on the A/B group."""
        if group == "margin_boost":
            return {"w_relevance": 0.6, "w_margin": 0.3, "w_inventory": 0.1}
        return {"w_relevance": 0.8, "w_margin": 0.1, "w_inventory": 0.1}