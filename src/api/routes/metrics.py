from fastapi import APIRouter
import time
from src.engine.session_manager import SessionManager

router = APIRouter()
session_mgr = SessionManager()

@router.get("/system")
async def get_dynamic_metrics():
    """
    Step 15: Real-time system health.
    """
    # Simple check to see if Redis is responding
    start = time.time()
    redis_alive = await session_mgr.redis.ping()
    redis_latency = (time.time() - start) * 1000

    return {
        "status": "healthy",
        "redis_connected": redis_alive,
        "redis_latency_ms": round(redis_latency, 2),
        "uptime": "active"
    }