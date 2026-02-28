from fastapi import APIRouter
from sqlalchemy import text
from src.core.database import AsyncSessionLocal
from src.engine.session_manager import SessionManager

router = APIRouter()
session_mgr = SessionManager()

@router.post("/assign")
async def assign_experiment(user_id: int):
    """
    Assigns user to A/B variant and persists it via Redis.
    """
    variant = await session_mgr.get_user_group(user_id)
    # Mapping internal group names to clean API labels
    return {
        "user_id": user_id, 
        "variant": "B (Margin Boost)" if variant == "margin_boost" else "A (Control)"
    }

@router.get("/results")
async def get_experiment_performance():
    """
    Step 11 & 15: Calculates real-time CTR and Revenue per group.
    Ensures Business Alignment metrics are visible.
    """
    async with AsyncSessionLocal() as session:
        # Optimized query calculating CTR: (Clicks / Views) * 100
        query = text("""
            SELECT 
                experiment_id as group_name,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(event_id) FILTER (WHERE event_type = 'click') as clicks,
                COUNT(event_id) FILTER (WHERE event_type = 'view') as views,
                SUM(revenue) as total_revenue
            FROM experiment_events
            GROUP BY experiment_id
        """)
        
        result = await session.execute(query)
        rows = result.fetchall()
        
        performance = []
        for row in rows:
            # Avoid division by zero
            ctr = (row.clicks / row.views) if row.views and row.views > 0 else 0
            
            performance.append({
                "group": row.group_name,
                "metrics": {
                    "unique_users": row.unique_users,
                    "ctr_percent": round(ctr * 100, 2),
                    "total_revenue": round(float(row.total_revenue or 0), 2),
                    "conversion_rate": round((row.clicks / row.unique_users), 4) if row.unique_users > 0 else 0
                }
            })
            
    return {
        "experiment_id": "exp_001_margin_optimization",
        "results": performance
    }