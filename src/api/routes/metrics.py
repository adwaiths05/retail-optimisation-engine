from fastapi import APIRouter
import time
from src.engine.session_manager import SessionManager

router = APIRouter()
session_mgr = SessionManager()

@router.get("/system")
async def get_dynamic_metrics():
    start = time.perf_counter()
    try:
        redis_alive = await session_mgr.redis.ping()
        redis_latency = (time.time() - start) * 1000
    except Exception:
        redis_alive = False
        redis_latency = 0

    return {
        "status": "healthy" if redis_alive else "degraded",
        "redis_connected": redis_alive,
        "redis_latency_ms": round(redis_latency, 2),
        "uptime": "active"
    }