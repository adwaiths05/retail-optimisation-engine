from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from src.database.models import ExperimentEvent
from src.core.database import AsyncSessionLocal
import datetime

router = APIRouter()

class EventRequest(BaseModel):
    user_id: int
    product_id: int
    event_type: str  # e.g., 'view', 'click', 'purchase'
    experiment_group: str  # e.g., 'control' or 'margin_boost'
    revenue: float = 0.0

@router.post("/events")
async def log_interaction(event_data: EventRequest):
    """
    Logs user interactions to calculate CTR and Revenue Lift.
    Fulfills Step 11 & 15: Feedback Loop Tracking.
    """
    async with AsyncSessionLocal() as session:
        try:
            async with session.begin():
                event = ExperimentEvent(
                    user_id=event_data.user_id,
                    product_id=event_data.product_id,
                    event_type=event_data.event_type,
                    experiment_id=event_data.experiment_group, # Matches 'group_name' logic
                    revenue=event_data.revenue,
                    timestamp=datetime.datetime.utcnow()
                )
                session.add(event)
            return {"status": "success", "event_type": event_data.event_type}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to log event: {str(e)}")