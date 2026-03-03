from fastapi import APIRouter
from mlops.model_registry import get_current_metadata

router = APIRouter()

@router.get("/current")
async def get_current_model():
    return get_current_metadata()

@router.post("/retrain")
async def trigger_retrain():
    # In a full app, this would trigger a Celery/Arq task
    return {"job_id": "retrain_9921", "status": "queued"}