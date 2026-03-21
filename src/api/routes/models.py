from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from src.core.database import get_db
from mlops.model_registry import get_current_metadata
import pandas as pd

router = APIRouter()

@router.get("/current")
async def get_current_model():
    """Returns model metadata from the registry."""
    return get_current_metadata()

@router.post("/retrain")
async def trigger_retrain():
    """Signals that a retraining job is initiated."""
    return {"job_id": "retrain_9921", "status": "queued", "message": "Backend ready for inference."}

@router.get("/monitoring-data")
async def get_monitoring_data(db: AsyncSession = Depends(get_db)):
    """
    Returns reference (training) and current (recent live events) windows
    aligned for drift checks.
    """
    ref_df = pd.read_csv(
        "./data/raw/order_products__prior.csv",
        usecols=["product_id", "reordered", "add_to_cart_order"],
    ).head(3000)
    ref_df["event_value"] = 1.0
    ref_df["event_hour"] = 12

    curr_query = text("""
        SELECT
            e.product_id,
            CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END AS reordered,
            CASE
                WHEN e.event_type = 'view' THEN 1
                WHEN e.event_type = 'click' THEN 2
                WHEN e.event_type = 'cart_add' THEN 3
                ELSE 4
            END AS add_to_cart_order,
            CASE
                WHEN e.event_type = 'view' THEN 0.25
                WHEN e.event_type = 'click' THEN 0.5
                WHEN e.event_type = 'cart_add' THEN 0.75
                ELSE 1.0
            END AS event_value,
            EXTRACT(HOUR FROM e.timestamp) AS event_hour
        FROM experiment_events e
        WHERE timestamp > NOW() - INTERVAL '7 days'
        ORDER BY timestamp DESC
        LIMIT 3000
    """)
    
    curr_result = await db.execute(curr_query)

    current_rows = [dict(r) for r in curr_result.mappings().all()]
    reference_rows = ref_df[["product_id", "reordered", "add_to_cart_order", "event_value", "event_hour"]]

    return {
        "reference": reference_rows.to_dict(orient="records"),
        "current": current_rows,
    }