from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from src.core.database import get_db
from mlops.model_registry import get_current_metadata

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
    Lean endpoint: Returns raw data as JSON. 
    Frontend will convert this to DataFrames for Drift analysis.
    """
    # 1. Pull Reference Data (Training baseline) from products table
    # Products table has: price, aisle_id, margin
    ref_query = text("SELECT price, aisle_id, margin FROM products LIMIT 2000")
    
    # 2. Pull Current Data (Live production features) from experiment_events table
    # ExperimentEvent table has: revenue, margin, product_id, timestamp
    # We alias 'revenue' as 'price' to keep the schema consistent for the frontend drift analysis.
    curr_query = text("""
        SELECT revenue as price, product_id, margin 
        FROM experiment_events 
        WHERE timestamp > NOW() - INTERVAL '7 days'
        LIMIT 2000
    """)
    
    ref_result = await db.execute(ref_query)
    curr_result = await db.execute(curr_query)
    
    # Convert SQLAlchemy rows to plain list of dicts for JSON serialization
    return {
        "reference": [dict(r) for r in ref_result.mappings().all()],
        "current": [dict(r) for r in curr_result.mappings().all()]
    }