from fastapi import APIRouter

router = APIRouter()

@router.get("/current")
async def get_current_model():
    return {
        "model_version": "v1.2.0",
        "type": "two_tower_onnx",
        "trained_at": "2026-02-20",
        "auc": 0.83
    }

@router.post("/retrain")
async def trigger_retrain():
    # In a full app, this would trigger a Celery/Arq task
    return {"job_id": "retrain_9921", "status": "queued"}