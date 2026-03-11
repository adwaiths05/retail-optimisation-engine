import redis.asyncio as redis
import json
import logging
from src.core.config import settings

logger = logging.getLogger("session-manager")

class SessionManager:
    def __init__(self):
        self.redis = redis.from_url(
            settings.REDIS_URL, 
            decode_responses=True, 
            encoding="utf-8",
            # Upstash requires these for stable cloud-to-cloud connections
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True
        )
        self.experiments = ["control", "margin_boost"]

    async def get_user_group(self, user_id: int) -> str:
        """Step 11: Get or set the A/B test group for a user."""
        key = f"user:{user_id}:group"
        try:
            group = await self.redis.get(key)
            
            if not group:
                group = self.experiments[user_id % len(self.experiments)]
                # Store for 7 days
                await self.redis.setex(key, 604800, group)
            return group
        except Exception as e:
            logger.error(f"Redis Error (Group): {e}")
            # Fallback logic so the API doesn't crash if Redis is down
            return self.experiments[user_id % len(self.experiments)]

    async def get_cached_embedding(self, user_id: int):
        """Step 10: Retrieve embedding from cache."""
        key = f"user:{user_id}:embedding"
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Redis Error (Cache Hit): {e}")
        return None

    async def cache_embedding(self, user_id: int, embedding: list):
        """Step 10: Store embedding in cache for 10 minutes."""
        key = f"user:{user_id}:embedding"
        try:
            # 600 seconds = 10 minutes
            await self.redis.setex(key, 600, json.dumps(embedding))
        except Exception as e:
            logger.error(f"Redis Error (Caching): {e}")

    def get_ranking_weights(self, group: str):
        """Returns weights based on the A/B group."""
        if group == "margin_boost":
            return {"w_relevance": 0.6, "w_margin": 0.3, "w_inventory": 0.1}
        return {"w_relevance": 0.8, "w_margin": 0.1, "w_inventory": 0.1}