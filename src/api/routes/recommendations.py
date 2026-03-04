from fastapi import APIRouter, HTTPException, Query, Body, Depends
import onnxruntime as ort
import numpy as np
import time
import json
import os
from typing import List, Dict, Any
from src.engine.retriever import Retriever
from src.engine.ranker import Ranker
from src.engine.session_manager import SessionManager
from src.core.config import settings

router = APIRouter()

class PredictionEngine:
    """
    Manages the ONNX inference session dynamically.
    Instead of hardcoding the path, it consults the MLOps registry.
    """
    _session = None
    _input_name = None
    _last_model_path = None

    @classmethod
    def get_engine(cls):
        # 1. Determine which model path to use (Registry vs Static Config)
        model_to_load = settings.MODEL_PATH
        
        if os.path.exists(settings.METADATA_PATH):
            try:
                with open(settings.METADATA_PATH, "r") as f:
                    metadata = json.load(f)
                    # Use the promoted production model path if it exists
                    model_to_load = metadata.get("artifacts", {}).get("onnx_model", settings.MODEL_PATH)
            except Exception:
                pass # Fallback to default if metadata is corrupt

        # 2. Lazy load or reload if the path has changed (Promotion happened)
        if cls._session is None or cls._last_model_path != model_to_load:
            try:
                cls._session = ort.InferenceSession(model_to_load)
                cls._input_name = cls._session.get_inputs()[0].name
                cls._last_model_path = model_to_load
                print(f"✅ Loaded Neural Engine: {model_to_load}")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize ONNX: {e}")
        
        return cls._session, cls._input_name

# --- Initialize Engine Components ---
retriever = Retriever(top_k=100) 
session_mgr = SessionManager()   
ranker = Ranker(settings.RERANKER_PATH, settings.MAPPINGS_PATH)

@router.post("/recommendations")
async def get_personalized_recommendations(
    user_id: int = Query(..., description="The ID of the user"),
    top_k: int = Query(10, gt=0, le=50),
    context: Dict[str, Any] = Body(default={})
):
    start_total = time.time()
    
    try:
        # 1. SESSION & A/B GROUP
        group = await session_mgr.get_user_group(user_id)
        weights = session_mgr.get_ranking_weights(group)
        
        # 2. USER EMBEDDING (Cache-Aside)
        user_vec = await session_mgr.get_cached_embedding(user_id)
        
        inference_source = "cache"
        inference_time = 0.0
        
        if not user_vec:
            # Get the dynamic engine session
            ort_session, input_name = PredictionEngine.get_engine()
            
            inference_start = time.time()
            inputs = {input_name: np.array([user_id], dtype=np.int64)}
            user_vec = ort_session.run(None, inputs)[0][0].tolist()
            
            # Cache for subsequent requests
            await session_mgr.cache_embedding(user_id, user_vec)
            inference_source = "model_inference"
            inference_time = (time.time() - inference_start) * 1000

        # 3. RETRIEVAL (pgvector)
        candidates = await retriever.get_nearest_products(user_vec)
        
        if not candidates:
            return {"user_id": user_id, "recommendations": [], "status": "no_candidates"}

        # 4. BUSINESS-AWARE RE-RANKING
        final_ranked_results = ranker.rank(candidates, user_id, weights=weights)

        total_latency = (time.time() - start_total) * 1000

        return {
            "user_id": user_id,
            "experiment_group": group,
            "recommendations": final_ranked_results[:top_k],
            "metadata": {
                "total_latency_ms": round(total_latency, 2),
                "inference_time_ms": round(inference_time, 2),
                "inference_source": inference_source,
                "model_path": PredictionEngine._last_model_path # Track which model is live
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")