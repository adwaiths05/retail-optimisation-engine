from fastapi import APIRouter, HTTPException, Query, Body, Depends
import onnxruntime as ort
import numpy as np
import time
from typing import List, Dict, Any
from src.engine.retriever import Retriever
from src.engine.ranker import Ranker
from src.engine.session_manager import SessionManager
from src.core.config import settings

router = APIRouter()

# --- Initialize Engine Components ---
try:
    # Load the quantized ONNX model for high-speed inference
    ort_session = ort.InferenceSession(settings.MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    print(f"❌ Failed to load ONNX model: {e}")

retriever = Retriever(top_k=100) 
session_mgr = SessionManager()   
# Initialize Ranker with XGBoost artifacts
ranker = Ranker(settings.RERANKER_PATH, settings.MAPPINGS_PATH)

@router.post("/recommendations")
async def get_personalized_recommendations(
    user_id: int = Query(..., description="The ID of the user"),
    top_k: int = Query(10, gt=0, le=50),
    context: Dict[str, Any] = Body(default={})
):
    """
    Personalization Pipeline:
    1. A/B Group Assignment (Control vs Margin Boost)
    2. User Embedding Retrieval (Cache-Aside with Redis)
    3. Neural Retrieval (Vector Search in pgvector)
    4. Business-Aware Re-Ranking (XGBoost + Margin Weights)
    """
    start_total = time.time()
    
    try:
        # 1. SESSION & A/B GROUP
        # Determines if the user is in 'control' or 'margin_boost'
        group = await session_mgr.get_user_group(user_id)
        weights = session_mgr.get_ranking_weights(group)
        
        # 2. USER EMBEDDING (Cache-Aside Pattern)
        # Check Redis first to stay within the <150ms latency budget
        user_vec = await session_mgr.get_cached_embedding(user_id)
        
        inference_source = "cache"
        inference_time = 0.0
        
        if not user_vec:
            # If cache miss, run the ONNX model
            inference_start = time.time()
            inputs = {input_name: np.array([user_id], dtype=np.int64)}
            user_vec = ort_session.run(None, inputs)[0][0].tolist()
            
            # Cache for subsequent requests (10 min TTL)
            await session_mgr.cache_embedding(user_id, user_vec)
            inference_source = "model_inference"
            inference_time = (time.time() - inference_start) * 1000

        # 3. RETRIEVAL (Vector Similarity Search)
        # Fetches candidates from Neon Postgres using pgvector
        candidates = await retriever.get_nearest_products(user_vec)
        
        if not candidates:
            return {"user_id": user_id, "recommendations": [], "status": "no_candidates"}

        # 4. BUSINESS-AWARE RE-RANKING
        # Combined ML probability and business weights to prevent zero scores
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
                "model_version": "v1.2.0-quantized"
            }
        }

    except Exception as e:
        # Global error handling for engine failures
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")