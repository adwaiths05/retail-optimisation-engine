from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field, field_validator
from src.database.models import ExperimentEvent
from src.core.database import AsyncSessionLocal
import datetime

router = APIRouter()

class EventRequest(BaseModel):
    user_id: int
    product_id: int
    event_type: str = Field(..., description="view|click|cart_add|purchase")
    experiment_group: str = Field(..., description="control|margin_boost")
    revenue: float = 0.0
    margin: float = 0.0

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"view", "click", "cart_add", "purchase"}
        if normalized not in allowed:
            raise ValueError(f"event_type must be one of: {', '.join(sorted(allowed))}")
        return normalized

    @field_validator("experiment_group")
    @classmethod
    def validate_group(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"control", "margin_boost"}
        if normalized not in allowed:
            raise ValueError(f"experiment_group must be one of: {', '.join(sorted(allowed))}")
        return normalized

@router.post("/events")
async def log_interaction(event_data: EventRequest):
    """
    Logs normalized user interactions for feedback analytics and drift windows.
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
                    margin=event_data.margin,
                    timestamp=datetime.datetime.utcnow()
                )
                session.add(event)
            return {
                "status": "success",
                "event_type": event_data.event_type,
                "experiment_group": event_data.experiment_group,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to log event: {str(e)}")